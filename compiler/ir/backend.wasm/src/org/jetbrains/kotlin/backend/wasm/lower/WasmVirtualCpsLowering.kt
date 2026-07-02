/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.expressions.impl.IrBlockImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrBranchImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrWhenImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrInstanceInitializerCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrSetValueImpl
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.util.*
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid

/**
 * Defunctionalized CPS conversion for VIRTUAL mutual recursion over a closed
 * class hierarchy (CHA-based).
 *
 * Motivation: the stdlib regex matcher (kotlin.text.regex.AbstractSet.matches)
 * recurses through virtual dispatch over the compiled pattern's object graph.
 * Recursion depth is proportional to input length, causing stack overflow on
 * real-world inputs (KT-63689, KT-61542, KT-78089). The recursion is neither
 * tail-call-shaped nor TMC-shaped (post-call state restores and result
 * inspection), so neither return_call emission nor WasmTailModConsLowering
 * rescues it. This pass converts the whole hierarchy's overrides into a
 * single trampoline function with heap-allocated frames.
 *
 * Design (see .ai/gsoc-assets/virtual-cps-design.md):
 *  - CHA enumerates all overrides of the target base method in the module
 *    (sound under wasm whole-world compilation).
 *  - Each override body is compiled into a basic-block plan, splitting at
 *    virtual call sites of the target method:
 *      * calls in tail position    -> receiver/state swap, no frame
 *      * non-tail calls            -> heap frame capturing live locals
 *    Unsupported constructs bail out: that override keeps its native body
 *    and the trampoline invokes it as an ordinary virtual call (partial
 *    conversion is always semantics-preserving).
 *  - A single `run$virtualCps` function holds the flat state machine:
 *    while(true) + when(state), state = (override, block) pairs.
 *
 * PoC scoping: the target hierarchy is currently selected by
 * [isTargetBaseClass]; this becomes a compiler flag before upstreaming.
 */
internal class WasmVirtualCpsLowering(private val context: WasmBackendContext) : ModuleLoweringPass {

    companion object {
        private const val TARGET_METHOD_NAME = "matches"

        /**
         * Native recursion budget before switching a subtree to the heap-frame
         * trampoline. Far below the wasm stack guard (~10-15K frames) even
         * with several frames per level and an already-deep caller stack, and
         * far above what everyday patterns reach.
         */
        private const val HYBRID_DEPTH_THRESHOLD = 512

        /** Hierarchies the matcher transform applies to. */
        private val TARGET_BASE_CLASSES = setOf(
            "kotlin.text.regex.AbstractSet",
            "MiniSet", // wasmVirtualCpsMatcher.kt box test (root package)
        )
    }

    private val enabled = context.configuration.get(WasmConfigurationKeys.WASM_ENABLE_STACKLESS_RECURSION) == true

    override fun lower(irModule: IrModuleFragment) {
        if (!enabled) return
        val allClasses = collectClasses(irModule)
        val baseClasses = allClasses.filter { isTargetBaseClass(it) }
        for (base in baseClasses) {
            val baseMethod = base.declarations
                .filterIsInstance<IrSimpleFunction>()
                .firstOrNull { it.name.asString() == TARGET_METHOD_NAME && it.body == null }
                ?: continue
            lowerHierarchy(irModule, base, baseMethod, allClasses)
        }
        lowerDeepRecursiveFunctions(irModule)
    }

    // ---------------- block simplification (shared by matcher and DRF codegens)

    /**
     * Collapses trivial control flow the planner produces in numbers:
     * empty forwarder blocks are skipped and single-predecessor Goto
     * successors are fused into their predecessor. Fewer states means
     * fewer dispatch round-trips per matcher step.
     */
    private fun simplifyPlan(plan: BodyPlan) {
        val resumeTargets = plan.blocks
            .mapNotNullTo(mutableSetOf()) { (it.terminator as? Terminator.SuspendCall)?.resume }

        fun resolve(b: BlockPlan): BlockPlan {
            var cur = b
            val seen = mutableSetOf<BlockPlan>()
            while (cur.statements.isEmpty() && cur !in resumeTargets && cur !== plan.entry && seen.add(cur)) {
                val t = cur.terminator as? Terminator.Goto ?: break
                cur = t.target
            }
            return cur
        }

        var changed = true
        while (changed) {
            changed = false

            for (b in plan.blocks) {
                when (val t = b.terminator) {
                    is Terminator.Goto -> {
                        val r = resolve(t.target)
                        if (r !== t.target) {
                            b.terminator = Terminator.Goto(r); changed = true
                        }
                    }
                    is Terminator.CondGoto -> {
                        val r1 = resolve(t.thenTarget)
                        val r2 = resolve(t.elseTarget)
                        if (r1 !== t.thenTarget || r2 !== t.elseTarget) {
                            b.terminator = Terminator.CondGoto(t.condition, r1, r2); changed = true
                        }
                    }
                    else -> {}
                }
            }

            val preds = mutableMapOf<BlockPlan, Int>()
            fun ref(b: BlockPlan) { preds[b] = (preds[b] ?: 0) + 1 }
            ref(plan.entry)
            for (b in plan.blocks) {
                when (val t = b.terminator) {
                    is Terminator.Goto -> ref(t.target)
                    is Terminator.CondGoto -> { ref(t.thenTarget); ref(t.elseTarget) }
                    is Terminator.SuspendCall -> ref(t.resume)
                    else -> {}
                }
            }
            for (b in plan.blocks) {
                if (b.terminator == null) continue
                val t = b.terminator
                if (t is Terminator.Goto) {
                    val c = t.target
                    if (c !== b && preds[c] == 1 && c !in resumeTargets && c !== plan.entry) {
                        b.statements += c.statements
                        b.terminator = c.terminator
                        c.statements.clear()
                        c.terminator = null
                        changed = true
                    }
                }
            }
        }
    }

    private fun reachableBlocks(plan: BodyPlan): List<BlockPlan> {
        val seen = LinkedHashSet<BlockPlan>()
        val work = ArrayDeque<BlockPlan>()
        work += plan.entry
        while (work.isNotEmpty()) {
            val b = work.removeFirst()
            if (!seen.add(b)) continue
            when (val t = b.terminator) {
                is Terminator.Goto -> work += t.target
                is Terminator.CondGoto -> { work += t.thenTarget; work += t.elseTarget }
                is Terminator.SuspendCall -> work += t.resume
                else -> {}
            }
        }
        return seen.toList()
    }


    // ================================================================ DeepRecursiveFunction acceleration

    private class DrfLiteral(
        val ctorCall: IrConstructorCall,
        val ref: IrRichFunctionReference?,
        val lambda: IrSimpleFunction,
        val holderProperty: IrProperty?,
        val directInvoke: IrCall?,
        val enclosingFunction: IrFunction?,
    )

    private fun IrCall.isDrfInvoke(): Boolean {
        val callee = symbol.owner
        return callee.name.asString() == "invoke" &&
                callee.fqNameWhenAvailable?.asString() == "kotlin.invoke" &&
                callee.parameters.firstOrNull()?.type?.classFqName?.asString() == "kotlin.DeepRecursiveFunction"
    }

    private fun isCallRecursive(call: IrCall): Boolean {
        val callee = call.symbol.owner
        return callee.name.asString() == "callRecursive" &&
                (callee.parent as? IrClass)?.fqNameWhenAvailable?.asString() == "kotlin.DeepRecursiveScope" &&
                call.arguments.size == 2
    }

    /**
     * Detection and reporting only, so the IR shapes can be verified before
     * the transform lands. Finds DeepRecursiveFunction literals bound to
     * immutable holders and the invoke sites that read those holders.
     */
    private fun lowerDeepRecursiveFunctions(irModule: IrModuleFragment) {
        val literals = mutableListOf<DrfLiteral>()
        irModule.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)

            override fun visitProperty(declaration: IrProperty) {
                val field = declaration.backingField
                val init = field?.initializer?.expression
                if (init is IrConstructorCall &&
                    init.symbol.owner.constructedClass.fqNameWhenAvailable?.asString() == "kotlin.DeepRecursiveFunction"
                ) {
                    val arg = init.arguments.firstOrNull()
                    if (arg is IrRichFunctionReference && !declaration.isVar) {
                        literals += DrfLiteral(init, arg, arg.invokeFunction, declaration, null, null)
                    } else if (arg is IrFunctionExpression && !declaration.isVar) {
                        literals += DrfLiteral(init, null, arg.function, declaration, null, null)
                    }
                }
                declaration.acceptChildrenVoid(this)
            }

            private val functionStack = ArrayDeque<IrFunction>()

            override fun visitFunction(declaration: IrFunction) {
                functionStack.addLast(declaration)
                declaration.acceptChildrenVoid(this)
                functionStack.removeLast()
            }

            override fun visitCall(expression: IrCall) {
                // Direct form: DeepRecursiveFunction { ... }.invoke(x)
                if (expression.isDrfInvoke()) {
                    val recv = expression.arguments[0]
                    if (recv is IrConstructorCall &&
                        recv.symbol.owner.constructedClass.fqNameWhenAvailable?.asString() == "kotlin.DeepRecursiveFunction"
                    ) {
                        val arg = recv.arguments.firstOrNull()
                        if (arg is IrRichFunctionReference) {
                            literals += DrfLiteral(recv, arg, arg.invokeFunction, null, expression, functionStack.lastOrNull())
                        }
                    }
                }
                expression.acceptChildrenVoid(this)
            }
        })
        if (literals.isEmpty()) return

        // Count invoke sites reading each holder through its getter or field.
        val invokeSites = mutableMapOf<IrProperty, Int>()
        var unboundInvokes = 0
        irModule.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitCall(expression: IrCall) {
                if (expression.isDrfInvoke()) {
                    val recv = expression.arguments[0]
                    val prop = when (recv) {
                        is IrCall -> recv.symbol.owner.correspondingPropertySymbol?.owner
                        is IrGetField -> recv.symbol.owner.correspondingPropertySymbol?.owner
                        else -> null
                    }
                    val lit = literals.firstOrNull { it.holderProperty == prop }
                    if (lit != null && prop != null) {
                        invokeSites[prop] = (invokeSites[prop] ?: 0) + 1
                    } else {
                        unboundInvokes++
                    }
                }
                expression.acceptChildrenVoid(this)
            }
        })

        for (lit in literals) {
            val lambda = lit.lambda
            var callRecCount = 0
            var otherSuspend = 0
            var captures = 0
            val declaredHere = mutableSetOf<Any>()
            lambda.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitValueParameter(declaration: IrValueParameter) {
                    declaredHere += declaration.symbol
                    declaration.acceptChildrenVoid(this)
                }
                override fun visitVariable(declaration: IrVariable) {
                    declaredHere += declaration.symbol
                    declaration.acceptChildrenVoid(this)
                }
                override fun visitCall(expression: IrCall) {
                    if (isCallRecursive(expression)) callRecCount++
                    else if (expression.symbol.owner.isSuspend) otherSuspend++
                    expression.acceptChildrenVoid(this)
                }
            })
            lambda.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitGetValue(expression: IrGetValue) {
                    if (expression.symbol !in declaredHere) captures++
                }
            })
            val sites = if (lit.directInvoke != null) 1 else lit.holderProperty?.let { invokeSites[it] } ?: 0

            var status = "skipped"
            if (sites > 0) {
                status = transformDrfLiteral(irModule, lit) ?: "transformed"
            }

            @OptIn(MessageCollectorAccess::class)
            context.configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)?.report(
                CompilerMessageSeverity.LOGGING,
                "[wasm-drf] literal at ${lit.holderProperty?.fqNameWhenAvailable}: " +
                        "callRecursive=$callRecCount otherSuspend=$otherSuspend captures=$captures invokeSites=$sites unbound=$unboundInvokes -> $status",
            )
        }
    }

    /**
     * Hoists every argument list containing a target call into ordered
     * temporaries, so nested shapes like `a + callRecursive(b)` become
     * `{ val t0 = a; val t1 = callRecursive(b); t0 + t1 }`, which the block
     * planner understands. Evaluation order is preserved by hoisting all
     * arguments up to and including the last target-carrying one.
     */
    private fun normalizeTargetCallArguments(body: IrBlockBody, owner: IrSimpleFunction, isTarget: (IrCall) -> Boolean) {
        fun containsTarget(e: IrElement): Boolean {
            var found = false
            e.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitFunction(declaration: IrFunction) {}

                override fun visitCall(expression: IrCall) {
                    if (isTarget(expression)) found = true
                    if (!found) expression.acceptChildrenVoid(this)
                }
            })
            return found
        }

        val b = context.createIrBuilder(owner.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        body.transform(object : IrTransformer<Nothing?>() {
            override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                expression.transformChildren(this, data)
                if (isTarget(expression)) return expression
                val lastTargetArg = (expression.arguments.indices).lastOrNull { i ->
                    expression.arguments[i]?.let { containsTarget(it) } == true
                } ?: return expression
                return b.irBlock(resultType = expression.type) {
                    for (i in 0..lastTargetArg) {
                        val arg = expression.arguments[i] ?: continue
                        val tmp = irTemporary(arg, nameHint = "anf$i")
                        expression.arguments[i] = irGet(tmp)
                    }
                    +expression
                }
            }
        }, null)
        body.patchDeclarationParents(owner)
    }

    private fun drfStableName(lit: DrfLiteral): String =
        lit.holderProperty?.name?.asString()
            ?: "drf\$${lit.ctorCall.startOffset}"

    /** Returns a bail reason, or null on success. */
    private fun transformDrfLiteral(irModule: IrModuleFragment, lit: DrfLiteral): String? {
        val lambda = lit.lambda
        val irFile = (lit.holderProperty?.file ?: lambda.fileOrNull) ?: return "no file"

        // The lowering pipeline may process a module more than once; the
        // generated twin's name marks a literal as already handled.
        val twinName = "${drfStableName(lit)}\$drfNative"
        if (irFile.declarations.any { it is IrSimpleFunction && it.name.asString() == twinName }) {
            return "already transformed"
        }
        val regulars = lambda.parameters.filter { it.kind == IrParameterKind.Regular }
        val valueParam = regulars.lastOrNull()
            ?.takeIf { it.type.classFqName?.asString() != "kotlin.DeepRecursiveScope" }
            ?: return "no value parameter"
        if (regulars.size > 2) return "bound parameters unsupported"
        val tType = valueParam.type
        val rType = lambda.returnType

        val isTarget: (IrCall) -> Boolean = { isCallRecursive(it) }

        // Plan on a normalized deep copy, after inlining suspend helpers that
        // hold callRecursive (JsonTreeReader's readObject pattern).
        val body = lambda.body as? IrBlockBody ?: return "no block body"
        val planCopy = body.deepCopyWithSymbols(lambda)
        inlineSuspendHelpers(planCopy, lambda, isTarget)?.let { return it }
        var hasTarget = false
        planCopy.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitCall(expression: IrCall) {
                if (isCallRecursive(expression)) hasTarget = true
                expression.acceptChildrenVoid(this)
            }
        })
        if (!hasTarget) return "no callRecursive after inlining"
        // Post-inline gate: any remaining foreign suspend call cannot run on
        // the trampoline.
        var foreignSuspend: String? = null
        planCopy.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitCall(expression: IrCall) {
                if (!isCallRecursive(expression) && expression.symbol.owner.isSuspend) {
                    foreignSuspend = expression.symbol.owner.name.asString()
                }
                expression.acceptChildrenVoid(this)
            }
        })
        foreignSuspend?.let { return "foreign suspend call: $it" }

        normalizeTargetCallArguments(planCopy, lambda, isTarget)

        // Free-variable closure conversion: values referenced from the lambda
        // but declared outside it (closure conversion has not run yet at this
        // stage) become plain parameters of the twin and the trampoline. The
        // direct invoke site sits inside the enclosing function, where those
        // values are in scope.
        val declared = mutableSetOf<Any>(lambda.symbol)
        lambda.parameters.forEach { declared += it.symbol }
        val freeSyms = LinkedHashSet<IrValueDeclaration>()
        fun scanFree(root: IrElement) {
            root.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitValueParameter(declaration: IrValueParameter) {
                    declared += declaration.symbol
                    declaration.acceptChildrenVoid(this)
                }

                override fun visitVariable(declaration: IrVariable) {
                    declared += declaration.symbol
                    declaration.acceptChildrenVoid(this)
                }
            })
            root.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitGetValue(expression: IrGetValue) {
                    if (expression.symbol !in declared) freeSyms += expression.symbol.owner
                }

                override fun visitSetValue(expression: IrSetValue) {
                    if (expression.symbol !in declared) freeSyms += expression.symbol.owner
                    expression.acceptChildrenVoid(this)
                }
            })
        }
        scanFree(planCopy)
        if (freeSyms.isNotEmpty()) {
            if (lit.directInvoke == null) return "free variables with property holder"
            val enclosing = lit.enclosingFunction ?: return "free variables without enclosing function"
            for (sym in freeSyms) {
                val parentFn = when (val d = sym) {
                    is IrValueParameter -> d.parent as? IrFunction
                    is IrVariable -> d.parent as? IrFunction
                    else -> null
                }
                if (parentFn != enclosing) return "free variable ${sym.name} not owned by enclosing function"
            }
            // Mutation of a free variable inside the lambda cannot be threaded
            // through plain parameters.
            var mutated = false
            planCopy.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitSetValue(expression: IrSetValue) {
                    if (freeSyms.any { it.symbol == expression.symbol }) mutated = true
                    expression.acceptChildrenVoid(this)
                }
            })
            if (mutated) return "free variable mutated in lambda"
        }

        val plan = BodyPlanner(lambda, lambda, planCopy, isTarget).plan()
            ?: return "planner: $lastBailReason"

        var result: String? = null
        context.irFactory.stageController.restrictTo(lit.holderProperty ?: lambda) {
            result = DrfCodegen(irFile, lit, plan, tType, rType, valueParam, freeSyms.toList()).generate(irModule)
        }
        return result
    }

    /**
     * Inlines private suspend helper functions whose bodies hold target calls
     * (e.g. JsonTreeReader's `DeepRecursiveScope.readObject()`). Restricted to
     * helpers with a single trailing return and side-effect-free call
     * arguments, so parameter substitution cannot duplicate effects.
     * Returns a bail reason or null.
     */
    private fun inlineSuspendHelpers(body: IrBlockBody, owner: IrSimpleFunction, isTarget: (IrCall) -> Boolean): String? {
        fun holdsTarget(fn: IrFunction): Boolean {
            var found = false
            fn.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitCall(expression: IrCall) {
                    if (isTarget(expression)) found = true
                    if (!found) expression.acceptChildrenVoid(this)
                }
            })
            return found
        }

        repeat(5) {
            var bail: String? = null
            var inlinedAny = false
            body.transform(object : IrTransformer<Nothing?>() {
                override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                    expression.transformChildren(this, data)
                    if (bail != null) return expression
                    val callee = expression.symbol.owner
                    if (isTarget(expression) || !callee.isSuspend || !holdsTarget(callee)) return expression

                    val calleeBody = callee.body as? IrBlockBody
                    if (calleeBody == null) { bail = "suspend helper ${callee.name} has no block body"; return expression }
                    val last = calleeBody.statements.lastOrNull()
                    if (last !is IrReturn || last.returnTargetSymbol != callee.symbol) {
                        bail = "suspend helper ${callee.name} lacks trailing return"; return expression
                    }
                    var returns = 0
                    calleeBody.acceptVoid(object : IrVisitorVoid() {
                        override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                        override fun visitReturn(expression: IrReturn) {
                            if (expression.returnTargetSymbol == callee.symbol) returns++
                            expression.acceptChildrenVoid(this)
                        }
                    })
                    if (returns != 1) { bail = "suspend helper ${callee.name} has multiple returns"; return expression }
                    for (a in expression.arguments) {
                        if (a != null && a !is IrGetValue && a !is IrGetField && a !is IrConst) {
                            bail = "suspend helper ${callee.name} has non-trivial argument"; return expression
                        }
                    }

                    val copied = calleeBody.deepCopyWithSymbols(callee)
                    val paramMap = mutableMapOf<Any, IrExpression>()
                    for (i in callee.parameters.indices) {
                        val arg = expression.arguments[i] ?: continue
                        paramMap[callee.parameters[i].symbol] = arg
                    }
                    copied.transform(object : IrTransformer<Nothing?>() {
                        override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
                            val repl = paramMap[expression.symbol] ?: return super.visitGetValue(expression, data)
                            return repl.deepCopyWithSymbols(owner)
                        }
                    }, null)

                    val stmts = copied.statements.toMutableList()
                    val ret = stmts.removeLast() as IrReturn
                    inlinedAny = true
                    return IrBlockImpl(
                        expression.startOffset, expression.endOffset, callee.returnType, null,
                        stmts + ret.value,
                    )
                }
            }, null)
            bail?.let { return it }
            if (!inlinedAny) return null
        }
        return "suspend helper inlining did not converge"
    }

    private inner class DrfCodegen(
        private val irFile: IrFile,
        private val lit: DrfLiteral,
        private val plan: BodyPlan,
        private val tType: IrType,
        private val rType: IrType,
        private val lambdaValueParam: IrValueParameter,
        private val freeSyms: List<IrValueDeclaration>,
    ) {
        private val builtIns = context.irBuiltIns
        private val name = drfStableName(lit)

        private val stateIds = mutableMapOf<BlockPlan, Int>()
        private val resumeInfo = mutableMapOf<BlockPlan, Terminator.SuspendCall>()

        private fun poolOf(type: IrType): IrType = when (type) {
            builtIns.intType, builtIns.booleanType, builtIns.charType,
            builtIns.longType, builtIns.floatType, builtIns.doubleType -> type
            else -> builtIns.anyNType
        }

        private val captures: List<Capture> = run {
            val counters = mutableMapOf<IrType, Int>()
            fun cap(key: CaptureKey, t: IrType): Capture {
                val pool = poolOf(t)
                val idx = counters.getOrElse(pool) { 0 }
                counters[pool] = idx + 1
                return Capture(key, t, pool, idx)
            }
            buildList {
                for (local in plan.locals) add(cap(CaptureKey.Local(local.symbol), local.type))
                add(cap(CaptureKey.Arg(0), tType))
            }
        }

        private val poolOrder: List<IrType> = listOf(
            builtIns.intType, builtIns.booleanType, builtIns.charType,
            builtIns.longType, builtIns.floatType, builtIns.doubleType, builtIns.anyNType,
        )
        private val poolSizes: Map<IrType, Int> = poolOrder.associateWith { pool -> captures.count { it.pool == pool } }

        private lateinit var frameOuterField: IrField
        private lateinit var frameResumeField: IrField
        private lateinit var framePoolFields: Map<IrType, List<IrField>>
        private lateinit var frameCtorParamOrder: List<Pair<IrType, Int>>
        private lateinit var frameCtor: IrConstructorSymbol

        fun generate(irModule: IrModuleFragment): String? {
            simplifyPlan(plan)
            var next = 1
            val live = reachableBlocks(plan)
            for (block in live) stateIds[block] = next++
            for (block in live) {
                val t = block.terminator
                if (t is Terminator.SuspendCall) resumeInfo[t.resume] = t
            }

            buildFrame()
            val runFun = buildTrampoline()
            val depthField = buildDepth()
            val twin = buildTwin(runFun, depthField)

            // Debug verification: no references to the lambda's parameters and

            rewriteInvokeSites(irModule, twin)
            return null
        }

        private fun buildFrame() {
            val cls = context.irFactory.buildClass {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("\$DrfFrame\$${this@DrfCodegen.name}")
                visibility = DescriptorVisibilities.PRIVATE
                modality = Modality.FINAL
                kind = ClassKind.CLASS
            }.apply {
                parent = irFile
                irFile.declarations += this
                createThisReceiverParameter()
                superTypes = listOf(builtIns.anyType)
            }
            fun prefix(pool: IrType): String = when (pool) {
                builtIns.intType -> "i"; builtIns.booleanType -> "z"; builtIns.charType -> "c"
                builtIns.longType -> "j"; builtIns.floatType -> "f"; builtIns.doubleType -> "d"
                else -> "r"
            }
            val fieldDefs = buildList {
                add("outer" to cls.defaultType.makeNullable())
                add("resume" to builtIns.intType)
                for (pool in poolOrder) repeat(poolSizes[pool]!!) { add("${prefix(pool)}$it" to pool) }
            }
            val fields = fieldDefs.map { def ->
                cls.addField {
                    startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                    name = Name.identifier(def.first)
                    type = def.second
                    visibility = DescriptorVisibilities.PUBLIC
                    isFinal = true
                }
            }
            frameOuterField = fields[0]
            frameResumeField = fields[1]
            val byPool = mutableMapOf<IrType, MutableList<IrField>>()
            val order = mutableListOf<Pair<IrType, Int>>()
            var fi = 2
            for (pool in poolOrder) {
                val list = mutableListOf<IrField>()
                repeat(poolSizes[pool]!!) { k -> list += fields[fi++]; order += pool to k }
                byPool[pool] = list
            }
            framePoolFields = byPool
            frameCtorParamOrder = order
            frameCtor = cls.addConstructor {
                isPrimary = true
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            }.apply {
                val params = fieldDefs.map { def ->
                    addValueParameter {
                        name = Name.identifier(def.first); type = def.second
                        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                    }
                }
                this.body = context.createIrBuilder(symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET).irBlockBody {
                    +irDelegatingConstructorCall(builtIns.anyClass.owner.constructors.single())
                    +IrInstanceInitializerCallImpl(UNDEFINED_OFFSET, UNDEFINED_OFFSET, cls.symbol, builtIns.unitType)
                    for (i in fields.indices) +irSetField(irGet(cls.thisReceiver!!), fields[i], irGet(params[i]))
                }
            }.symbol
        }

        private fun IrType.defaultValueExpr(): IrExpression = when (this) {
            builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, false)
            builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, ' ')
            builtIns.byteType -> IrConstImpl.byte(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            builtIns.shortType -> IrConstImpl.short(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0.0f)
            builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0.0)
            else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, makeNullable())
        }

        private fun buildTrampoline(): IrSimpleFunction {
            val holderName = this.name
            val runFun = context.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("$holderName\$drfRun")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = rType
            }
            val pCaptures = freeSyms.mapIndexed { i, fs -> runFun.addValueParameter("cap$i", fs.type) }
            val pArg = runFun.addValueParameter("value", tType)
            val b = context.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val frameType = frameOuterField.type

            runFun.body = b.irBlockBody {
                val vState = irTemporary(irInt(stateIds[plan.entry]!!), nameHint = "s", isMutable = true)
                val vArg = irTemporary(irGet(pArg), nameHint = "arg", isMutable = true, irType = tType)
                val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true, irType = frameType)
                val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true, irType = frameType)
                // Reference-typed results start as null before the first Ret,
                // so the slot must be nullable; the final return casts back.
                val vRetType = if (poolOf(rType) == builtIns.anyNType) rType.makeNullable() else rType
                val vRet = irTemporary(rType.defaultValueExpr(), nameHint = "ret", isMutable = true, irType = vRetType)

                val hoisted = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                for (local in plan.locals) {
                    hoisted[local.symbol] = irTemporary(
                        local.type.defaultValueExpr(), nameHint = "l_${local.name}", isMutable = true, irType = local.type,
                    )
                }

                val loop = b.irWhile().apply { condition = b.irTrue() }
                val intEq = this@WasmVirtualCpsLowering.context.wasmSymbols.equalityFunctions[builtIns.intType]
                fun stateEquals(sid: Int): IrExpression =
                    if (intEq != null) irCall(intEq).apply { arguments[0] = irGet(vState); arguments[1] = irInt(sid) }
                    else irEquals(irGet(vState), irInt(sid))

                val remapBase = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                remapBase[lambdaValueParam.symbol] = vArg
                for (i in freeSyms.indices) remapBase[freeSyms[i].symbol] = pCaptures[i]
                for (local in plan.locals) remapBase[local.symbol] = hoisted[local.symbol]!!

                fun remapAll(e: IrElement): IrElement = e.transform(object : IrTransformer<Nothing?>() {
                    override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
                        val to = remapBase[expression.symbol] ?: return super.visitGetValue(expression, data)
                        return IrGetValueImpl(expression.startOffset, expression.endOffset, to.type, to.symbol)
                    }

                    override fun visitSetValue(expression: IrSetValue, data: Nothing?): IrExpression {
                        expression.transformChildren(this, data)
                        val to = remapBase[expression.symbol] ?: return expression
                        return IrSetValueImpl(
                            expression.startOffset, expression.endOffset, builtIns.unitType,
                            to.symbol, expression.value, expression.origin,
                        )
                    }
                }, null)

                fun IrBlockBuilder.emitPush(resumeId: Int) {
                    +irSet(vTop.symbol, irCallConstructor(frameCtor, emptyList()).apply {
                        arguments[0] = irGet(vTop)
                        arguments[1] = irInt(resumeId)
                        for (ci in frameCtorParamOrder.indices) {
                            val slot = frameCtorParamOrder[ci]
                            val capture = captures.firstOrNull { it.pool == slot.first && it.poolIndex == slot.second }
                            arguments[ci + 2] = when (val k = capture?.key) {
                                null -> slot.first.defaultValueExpr()
                                is CaptureKey.Local -> irGet(hoisted[k.symbol]!!)
                                is CaptureKey.Arg -> irGet(vArg)
                                is CaptureKey.Self -> error("no self in drf")
                            }
                        }
                    })
                }

                fun IrBlockBuilder.emitRestore() {
                    for (capture in captures) {
                        val target = when (val k = capture.key) {
                            is CaptureKey.Local -> hoisted[k.symbol]!!
                            is CaptureKey.Arg -> vArg
                            is CaptureKey.Self -> error("no self in drf")
                        }
                        val field = framePoolFields[capture.pool]!![capture.poolIndex]
                        val read = irGetField(irGet(vFrame), field)
                        val value = if (capture.pool == builtIns.anyNType && target.type != builtIns.anyNType) {
                            irAs(read, target.type.makeNullable())
                        } else read
                        +irSet(target.symbol, value)
                    }
                }

                fun IrBlockBuilder.emitTransfer(call: IrCall) {
                    val newVal = remapAll(call.arguments[1]!!) as IrExpression
                    +irSet(vArg.symbol, newVal)
                    +irSet(vState.symbol, irInt(stateIds[plan.entry]!!))
                }

                loop.body = b.irBlock {
                    +irWhen(builtIns.unitType, buildList {
                        for (block in plan.blocks) {
                            val sid = stateIds[block] ?: continue
                            val branchBody = irBlock {
                                resumeInfo[block]?.let { susp ->
                                    emitRestore()
                                    val target = hoisted[susp.resultSymbol] ?: vArg.takeIf { susp.resultSymbol == lambdaValueParam.symbol }
                                    if (target != null) {
                                        val v: IrExpression = when {
                                            target.type == vRetType -> irGet(vRet)
                                            !target.type.isPrimitiveType() -> irAs(irGet(vRet), target.type.makeNullable())
                                            else -> irAs(irGet(vRet), target.type)
                                        }
                                        +irSet(target.symbol, v)
                                    }
                                }
                                for (stmt in block.statements) {
                                    when (stmt) {
                                        is IrVariable -> {
                                            val target = hoisted[stmt.symbol]!!
                                            stmt.initializer?.let { init -> +irSet(target.symbol, remapAll(init) as IrExpression) }
                                        }
                                        else -> +(remapAll(stmt) as IrStatement)
                                    }
                                }
                                when (val t = block.terminator!!) {
                                    is Terminator.Goto -> {
                                        +irSet(vState.symbol, irInt(stateIds[t.target]!!))
                                        +irContinue(loop)
                                    }
                                    is Terminator.CondGoto -> {
                                        +irIfThenElse(
                                            builtIns.unitType,
                                            remapAll(t.condition) as IrExpression,
                                            irBlock { +irSet(vState.symbol, irInt(stateIds[t.thenTarget]!!)); +irContinue(loop) },
                                            irBlock { +irSet(vState.symbol, irInt(stateIds[t.elseTarget]!!)); +irContinue(loop) },
                                        )
                                    }
                                    is Terminator.Ret -> {
                                        +irSet(vRet.symbol, remapAll(t.value) as IrExpression)
                                        +irSet(vState.symbol, irInt(0))
                                        +irContinue(loop)
                                    }
                                    is Terminator.TailCall -> {
                                        emitTransfer(t.call)
                                        +irContinue(loop)
                                    }
                                    is Terminator.SuspendCall -> {
                                        emitPush(stateIds[t.resume]!!)
                                        emitTransfer(t.call)
                                        +irContinue(loop)
                                    }
                                }
                            }
                            add(irBranch(stateEquals(sid), branchBody))
                        }
                        val applyBlock = irBlock {
                            val fTmp = irTemporary(irGet(vTop), nameHint = "f")
                            val retValue: IrExpression =
                                if (vRetType != rType) irAs(irGet(vRet), rType) else irGet(vRet)
                            +irIfThen(builtIns.unitType, irEqualsNull(irGet(fTmp)), irReturn(retValue))
                            +irSet(vTop.symbol, irGetField(irGet(fTmp), frameOuterField))
                            +irSet(vFrame.symbol, irGet(fTmp))
                            +irSet(vState.symbol, irGetField(irGet(fTmp), frameResumeField))
                            +irContinue(loop)
                        }
                        add(irBranch(irTrue(), applyBlock))
                    })
                }
                +loop
            }
            runFun.body!!.patchDeclarationParents(runFun)
            return runFun
        }

        private fun buildDepth(): IrField {
            val holderName = this.name
            return context.irFactory.buildField {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier("$holderName\$drfDepth")
            type = builtIns.intType
            visibility = DescriptorVisibilities.PRIVATE
            isStatic = true
            isFinal = false
        }.apply {
            parent = irFile
            irFile.declarations += this
            initializer = context.irFactory.createExpressionBody(
                IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, builtIns.intType, 0)
            )
        }
        }

        /** Native twin: the lambda body with callRecursive turned into hybrid self-calls. */
        private fun buildTwin(runFun: IrSimpleFunction, depthField: IrField): IrSimpleFunction {
            val holderName = this.name
            val twin = context.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("$holderName\$drfNative")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = rType
            }
            val pCaptures = freeSyms.mapIndexed { i, fs -> twin.addValueParameter("cap$i", fs.type) }
            val pArg = twin.addValueParameter("value", tType)
            val bodyCopy = (lit.lambda.body as IrBlockBody).deepCopyWithSymbols(twin)
            inlineSuspendHelpers(bodyCopy, twin) { isCallRecursive(it) }
            val b = context.createIrBuilder(twin.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val captureMap = freeSyms.indices.associate { freeSyms[it].symbol to pCaptures[it] }

            bodyCopy.transform(object : IrTransformer<Nothing?>() {
                override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
                    if (expression.symbol == lambdaValueParam.symbol) {
                        return IrGetValueImpl(expression.startOffset, expression.endOffset, pArg.type, pArg.symbol)
                    }
                    captureMap[expression.symbol]?.let { to ->
                        return IrGetValueImpl(expression.startOffset, expression.endOffset, to.type, to.symbol)
                    }
                    return super.visitGetValue(expression, data)
                }

                override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                    expression.transformChildren(this, data)
                    if (expression.returnTargetSymbol == lit.lambda.symbol) {
                        return IrReturnImpl(
                            expression.startOffset, expression.endOffset,
                            builtIns.nothingType, twin.symbol, expression.value,
                        )
                    }
                    return expression
                }

                override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                    expression.transformChildren(this, data)
                    if (!isCallRecursive(expression)) return expression
                    val argExpr = expression.arguments[1]!!
                    fun depthGet() = b.irGetField(null, depthField)
                    return b.irBlock(resultType = rType) {
                        val a = irTemporary(argExpr, nameHint = "darg")
                        +irIfThenElse(
                            rType,
                            irCall(builtIns.lessFunByOperandType[builtIns.intClass]!!).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(HYBRID_DEPTH_THRESHOLD)
                            },
                            irBlock(resultType = rType) {
                                +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                    arguments[0] = depthGet(); arguments[1] = irInt(1)
                                })
                                val r = irTemporary(irCall(twin.symbol).apply {
                                    for (i in pCaptures.indices) arguments[i] = irGet(pCaptures[i])
                                    arguments[pCaptures.size] = irGet(a)
                                }, nameHint = "dres")
                                +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                    arguments[0] = depthGet(); arguments[1] = irInt(-1)
                                })
                                +irGet(r)
                            },
                            irCall(runFun.symbol).apply {
                                for (i in pCaptures.indices) arguments[i] = irGet(pCaptures[i])
                                arguments[pCaptures.size] = irGet(a)
                            },
                        )
                    }
                }
            }, null)
            twin.body = bodyCopy
            twin.body!!.patchDeclarationParents(twin)
            return twin
        }

        private fun rewriteInvokeSites(irModule: IrModuleFragment, twin: IrSimpleFunction) {
            val prop = lit.holderProperty
            val direct = lit.directInvoke
            irModule.transform(object : IrTransformer<Nothing?>() {
                override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                    expression.transformChildren(this, data)
                    if (!expression.isDrfInvoke()) return expression
                    if (direct != null) {
                        if (expression !== direct) return expression
                        // Direct form: free variables are in scope at this site.
                        return IrCallImpl(
                            expression.startOffset, expression.endOffset, rType, twin.symbol,
                            typeArgumentsCount = 0,
                        ).apply {
                            for (i in freeSyms.indices) {
                                arguments[i] = IrGetValueImpl(
                                    expression.startOffset, expression.endOffset,
                                    freeSyms[i].type, freeSyms[i].symbol,
                                )
                            }
                            arguments[freeSyms.size] = expression.arguments[1]
                        }
                    }
                    val recv = expression.arguments[0]
                    val boundProp = when (recv) {
                        is IrCall -> recv.symbol.owner.correspondingPropertySymbol?.owner
                        is IrGetField -> recv.symbol.owner.correspondingPropertySymbol?.owner
                        else -> null
                    }
                    if (prop == null || boundProp != prop) return expression
                    return IrCallImpl(
                        expression.startOffset, expression.endOffset, rType, twin.symbol,
                        typeArgumentsCount = 0,
                    ).apply {
                        arguments[0] = expression.arguments[1]
                    }
                }
            }, null)
        }
    }

    private fun isTargetBaseClass(irClass: IrClass): Boolean {
        val fqn = irClass.fqNameWhenAvailable?.asString() ?: return false
        return fqn in TARGET_BASE_CLASSES
    }

    private fun collectClasses(irModule: IrModuleFragment): List<IrClass> {
        val result = mutableListOf<IrClass>()
        irModule.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitClass(declaration: IrClass) {
                result += declaration
                declaration.acceptChildrenVoid(this)
            }
        })
        return result
    }

    // ================================================================ CHA

    private class OverrideInfo(
        val irClass: IrClass?,
        val function: IrSimpleFunction,
    )

    private fun collectOverrides(
        base: IrClass,
        baseMethod: IrSimpleFunction,
        allClasses: List<IrClass>,
    ): List<OverrideInfo> {
        val result = mutableListOf<OverrideInfo>()
        for (cls in allClasses) {
            if (!cls.isSubclassOf(base)) continue
            val override = cls.declarations
                .filterIsInstance<IrSimpleFunction>()
                .firstOrNull { fn ->
                    fn.body != null && !fn.isFakeOverride &&
                            fn.allOverriddenIncludingSelf().any { it == baseMethod }
                }
                ?: continue
            result += OverrideInfo(cls, override)
        }
        return result
    }

    private fun IrSimpleFunction.allOverriddenIncludingSelf(): List<IrSimpleFunction> {
        val seen = mutableSetOf<IrSimpleFunction>()
        fun walk(fn: IrSimpleFunction) {
            if (!seen.add(fn)) return
            for (s in fn.overriddenSymbols) walk(s.owner)
        }
        walk(this)
        return seen.toList()
    }

    /** Is [call] a virtual dispatch of the target base method? */
    private fun isTargetCall(call: IrCall, baseMethod: IrSimpleFunction): Boolean {
        val callee = call.symbol.owner
        if (callee.name != baseMethod.name) return false
        if (callee == baseMethod) return true
        return callee.allOverriddenIncludingSelf().any { it == baseMethod }
    }

    // ================================================================ block planning

    /**
     * A basic block of the state machine: straight-line statements followed
     * by exactly one terminator. Statements reference the ORIGINAL function's
     * value symbols; remapping happens at codegen.
     */
    private class BlockPlan(val id: Int) {
        val statements = mutableListOf<IrStatement>()
        var terminator: Terminator? = null
    }

    private sealed class Terminator {
        class Goto(val target: BlockPlan) : Terminator()
        class CondGoto(
            val condition: IrExpression,
            val thenTarget: BlockPlan,
            val elseTarget: BlockPlan,
        ) : Terminator()

        /** `return recv.matches(args)` — receiver/args swap, no frame. */
        class TailCall(val call: IrCall) : Terminator()

        /** `v = recv.matches(args)` — push frame, resume at [resume] with [resultSymbol] set. */
        class SuspendCall(
            val call: IrCall,
            val resultSymbol: IrValueSymbol,
            val resume: BlockPlan,
        ) : Terminator()

        class Ret(val value: IrExpression) : Terminator()
    }

    private class BodyPlan(
        val info: OverrideInfo,
        val entry: BlockPlan,
        val blocks: List<BlockPlan>,
        /** Locals declared in the body, in declaration order (frame candidates). */
        val locals: List<IrVariable>,
    )

    /** Thrown internally to abandon planning for one override (bail out to native). */
    private class BailOut(val reason: String) : RuntimeException()

    private var lastBailReason: String = ""


    private class LoopFrame(val loop: IrLoop, val head: BlockPlan, val exit: BlockPlan)

    private class ReturnableFrame(
        val symbol: org.jetbrains.kotlin.ir.symbols.IrReturnTargetSymbol,
        val join: BlockPlan,
        val resultTarget: IrValueSymbol? = null,
        /** `return <returnable block>` position: returns to the block are returns of the function. */
        val returnThrough: Boolean = false,
    )

    private sealed class CaptureKey {
        object Self : CaptureKey()
        class Local(val symbol: IrValueSymbol) : CaptureKey()
        class Arg(val index: Int) : CaptureKey()
    }

    private class Capture(val key: CaptureKey, val declaredType: IrType, val pool: IrType, val poolIndex: Int)

    private inner class BodyPlanner(
        private val func: IrSimpleFunction,
        private val baseMethod: IrSimpleFunction,
        private val bodyToPlan: IrBlockBody,
        private val isTarget: (IrCall) -> Boolean = { isTargetCall(it, baseMethod) },
    ) {
        private val blocks = mutableListOf<BlockPlan>()
        private val locals = mutableListOf<IrVariable>()
        private val loopStack = ArrayDeque<LoopFrame>()
        private val returnableStack = ArrayDeque<ReturnableFrame>()
        private val builtIns get() = this@WasmVirtualCpsLowering.context.irBuiltIns

        private fun IrExpression.asDiscardedStatement(): IrStatement = this

        private fun IrExpression.asUnreachableDefault(): IrExpression = when (val rt = func.returnType) {
            builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
            builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, false)
            builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, ' ')
            builtIns.byteType -> IrConstImpl.byte(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
            builtIns.shortType -> IrConstImpl.short(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
            builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
            builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0f)
            builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0)
            else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt.makeNullable())
        }

        private fun newBlock(): BlockPlan = BlockPlan(blocks.size).also { blocks += it }

        fun plan(): BodyPlan? {
            val body = bodyToPlan
            return try {
                inlineLocalFunsWithTargetCalls(body)
                val entry = newBlock()
                val last = compileStatements(body.statements, entry)
                // Bodies whose every path returns or throws leave a block with
                // no terminator; the synthetic Ret is unreachable but must
                // still be well-typed for the function's return type.
                if (last.terminator == null) {
                    val rt = func.returnType
                    val zero: IrExpression = when (rt) {
                        builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
                        builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, false)
                        builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, ' ')
                        builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
                        builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0f)
                        builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0)
                        else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt.makeNullable())
                    }
                    last.terminator = Terminator.Ret(zero)
                }
                BodyPlan(OverrideInfo(func.parent as? IrClass, func), blocks.first(), blocks, locals)
            } catch (b: BailOut) {
                lastBailReason = b.reason
                null
            }
        }

        /**
         * Local funs whose body contains target calls (e.g. GroupQuantifierSet's
         * matchNext) cannot be split in place. If the fun has a single trailing
         * return and no parameters, inline it at every call site; captured
         * variables keep their symbols and are hoisted like any other local.
         */
        private fun inlineLocalFunsWithTargetCalls(body: IrBlockBody) {
            val localFuns = mutableListOf<IrSimpleFunction>()
            body.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                    if (hasTargetCallInside(declaration)) localFuns += declaration
                }
            })
            if (localFuns.isEmpty()) return

            for (lf in localFuns) {
                if (lf.parameters.isNotEmpty()) throw BailOut("local fun with parameters holds target call")
                val lfBody = lf.body as? IrBlockBody ?: throw BailOut("local fun without block body holds target call")
                val last = lfBody.statements.lastOrNull()
                if (last !is IrReturn || last.returnTargetSymbol != lf.symbol) throw BailOut("local fun without trailing return holds target call")
                var returns = 0
                lfBody.acceptVoid(object : IrVisitorVoid() {
                    override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                    override fun visitReturn(expression: IrReturn) {
                        if (expression.returnTargetSymbol == lf.symbol) returns++
                        expression.acceptChildrenVoid(this)
                    }
                })
                if (returns != 1) throw BailOut("local fun with multiple returns holds target call")

                body.transform(object : IrTransformer<Nothing?>() {
                    override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                        expression.transformChildren(this, data)
                        if (expression.symbol != lf.symbol) return expression
                        val copied = lfBody.deepCopyWithSymbols(lf)
                        val stmts = copied.statements.toMutableList()
                        val ret = stmts.removeLast() as IrReturn
                        return IrBlockImpl(
                            expression.startOffset, expression.endOffset, lf.returnType, null,
                            stmts + ret.value,
                        )
                    }
                }, null)
                // Drop the now-unused declaration.
                body.statements.removeAll { it === lf }
                body.transform(object : IrTransformer<Nothing?>() {
                    override fun visitBlock(expression: IrBlock, data: Nothing?): IrExpression {
                        expression.transformChildren(this, data)
                        expression.statements.removeAll { it === lf }
                        return expression
                    }
                }, null)
            }
        }

        /** Compiles [stmts] starting into [start]; returns the block that control ends in. */
        private fun compileStatements(stmts: List<IrStatement>, start: BlockPlan): BlockPlan {
            var current = start
            for (stmt in stmts) {
                if (current.terminator != null) {
                    // Unreachable trailing statements (e.g. code after return) — drop.
                    break
                }
                current = compileStatement(stmt, current)
            }
            return current
        }

        private fun compileStatement(stmt: IrStatement, current: BlockPlan): BlockPlan {
            when (stmt) {
                is IrVariable -> {
                    locals += stmt
                    val init = stmt.initializer
                    if (init != null && needsSplit(init)) {
                        if (init is IrWhen || init is IrContainerExpression) {
                            // val x = <complex>  ==>  var x; <complex with x = branch/last>
                            stmt.initializer = null
                            current.statements += stmt
                            return compileAssignmentOf(stmt.symbol, init, current)
                        }
                        val call = init as? IrCall ?: throw BailOut("target call or return nested in variable initializer ${init::class.simpleName}")
                        if (!isTarget(call)) throw BailOut("nested target call")
                        checkArgsHaveNoTargetCalls(call)
                        val resume = newBlock()
                        stmt.initializer = null
                        current.statements += stmt
                        current.terminator = Terminator.SuspendCall(call, stmt.symbol, resume)
                        return resume
                    }
                    current.statements += stmt
                    return current
                }

                is IrReturn -> {
                    if (stmt.returnTargetSymbol != func.symbol) {
                        // Return out of an inlined block (e.g. forEach's non-local
                        // return machinery): jump to that block's join, routing the
                        // value into the block's result slot when it has one.
                        val frame = returnableStack.lastOrNull { it.symbol == stmt.returnTargetSymbol }
                            ?: throw BailOut("non-local return")
                        if (frame.returnThrough) {
                            return compileStatement(
                                IrReturnImpl(stmt.startOffset, stmt.endOffset, builtIns.nothingType, func.symbol, stmt.value),
                                current,
                            )
                        }
                        val cur: BlockPlan
                        if (frame.resultTarget != null) {
                            cur = compileAssignmentOf(frame.resultTarget, stmt.value, current)
                        } else {
                            if (containsTargetCall(stmt.value)) throw BailOut("target call in returnable-block return value")
                            current.statements += stmt.value.asDiscardedStatement()
                            cur = current
                        }
                        cur.terminator = Terminator.Goto(frame.join)
                        return cur
                    }
                    val value = stmt.value
                    if (containsTargetCall(value)) {
                        when (value) {
                            is IrCall -> {
                                if (!isTarget(value)) throw BailOut("nested target call")
                                checkArgsHaveNoTargetCalls(value)
                                current.terminator = Terminator.TailCall(value)
                                return current
                            }
                            is IrWhen -> {
                                // return when { c1 -> e1; ... }  ==>  when { c1 -> return e1; ... }
                                val pushed = IrWhenImpl(value.startOffset, value.endOffset, builtIns.unitType, value.origin).apply {
                                    for (branch in value.branches) {
                                        val result = branch.result
                                        val branchBody = if (result.type.isNothing()) {
                                            result
                                        } else {
                                            IrReturnImpl(result.startOffset, result.endOffset, builtIns.nothingType, func.symbol, result)
                                        }
                                        branches += IrBranchImpl(branch.startOffset, branch.endOffset, branch.condition, branchBody)
                                    }
                                }
                                val end = compileStatement(pushed, current)
                                // A return-when with no else falls through: treat as unreachable end.
                                if (end.terminator == null) end.terminator = Terminator.Ret(value.asUnreachableDefault())
                                return end
                            }
                            is IrBlock -> {
                                if (value is IrReturnableBlock) {
                                    // return <RB>: every return@RB v is a plain return v.
                                    val join = newBlock()
                                    returnableStack.addLast(ReturnableFrame(value.symbol, join, null, returnThrough = true))
                                    val end = compileStatements(value.statements, current)
                                    returnableStack.removeLast()
                                    if (end.terminator == null) throw BailOut("returned returnable block falls through")
                                    return join
                                }
                                // return block { s1..sn; last } ==> s1..sn; return last
                                val stmts = value.statements
                                if (stmts.isEmpty()) throw BailOut("empty block in return")
                                var cur = current
                                for (i in 0 until stmts.size - 1) cur = compileStatement(stmts[i], cur)
                                val last = stmts.last()
                                val lastExpr = last as? IrExpression ?: throw BailOut("return block last statement is not an expression")
                                if (lastExpr.type.isNothing()) {
                                    return compileStatement(lastExpr, cur)
                                }
                                return compileStatement(
                                    IrReturnImpl(stmt.startOffset, stmt.endOffset, builtIns.nothingType, func.symbol, lastExpr),
                                    cur,
                                )
                            }
                            else -> throw BailOut("target call nested in return value ${value::class.simpleName}")
                        }
                    }
                    current.terminator = Terminator.Ret(value)
                    return current
                }

                is IrWhen -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    return compileWhen(stmt, current)
                }

                is IrWhileLoop -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    return compileWhile(stmt, current)
                }

                is IrDoWhileLoop -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    return compileDoWhile(stmt, current)
                }

                is IrBlock -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    if (stmt is IrReturnableBlock) {
                        if (!stmt.type.isUnit() && !stmt.type.isNothing()) {
                            throw BailOut("non-Unit returnable block with target call or return")
                        }
                        val join = newBlock()
                        returnableStack.addLast(ReturnableFrame(stmt.symbol, join))
                        val end = compileStatements(stmt.statements, current)
                        returnableStack.removeLast()
                        if (end.terminator == null) end.terminator = Terminator.Goto(join)
                        return join
                    }
                    return compileStatements(stmt.statements, current)
                }

                is IrSetValue -> {
                    if (!needsSplit(stmt.value)) {
                        current.statements += stmt
                        return current
                    }
                    return compileAssignmentOf(stmt.symbol, stmt.value, current)
                }

                is IrTypeOperatorCall -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    // A statement-position coercion to Unit just discards the
                    // value; compile the underlying expression as a statement.
                    if (stmt.operator == IrTypeOperator.IMPLICIT_COERCION_TO_UNIT) {
                        return compileStatement(stmt.argument, current)
                    }
                    throw BailOut("target call or return in type operator ${stmt.operator}")
                }

                is IrComposite -> {
                    if (!needsSplit(stmt)) {
                        current.statements += stmt
                        return current
                    }
                    return compileStatements(stmt.statements, current)
                }

                is IrBreak -> {
                    val frame = loopStack.lastOrNull { it.loop == stmt.loop }
                        ?: throw BailOut("break targets unknown loop")
                    current.terminator = Terminator.Goto(frame.exit)
                    return current
                }

                is IrContinue -> {
                    val frame = loopStack.lastOrNull { it.loop == stmt.loop }
                        ?: throw BailOut("continue targets unknown loop")
                    current.terminator = Terminator.Goto(frame.head)
                    return current
                }

                else -> {
                    if (needsSplit(stmt)) throw BailOut("target call or return in unsupported statement ${stmt::class.simpleName}")
                    current.statements += stmt
                    return current
                }
            }
        }

        /**
         * Compiles `target = value` where value contains target calls, outer
         * returns, or jumps to split loops. Nothing-typed positions (return /
         * break / continue) never produce a value, so they are compiled as
         * plain statements instead of being wrapped in an assignment.
         */
        private fun compileAssignmentOf(target: IrValueSymbol, value: IrExpression, current: BlockPlan): BlockPlan {
            if (value.type.isNothing()) {
                return compileStatement(value, current)
            }
            if (!needsSplit(value)) {
                current.statements += IrSetValueImpl(
                    value.startOffset, value.endOffset, builtIns.unitType, target, value, null,
                )
                return current
            }
            when (value) {
                is IrCall -> {
                    if (!isTarget(value)) throw BailOut("nested target call in assignment")
                    checkArgsHaveNoTargetCalls(value)
                    val resume = newBlock()
                    current.terminator = Terminator.SuspendCall(value, target, resume)
                    return resume
                }
                is IrWhen -> {
                    val pushed = IrWhenImpl(value.startOffset, value.endOffset, builtIns.unitType, value.origin).apply {
                        for (branch in value.branches) {
                            val result = branch.result
                            val branchBody = if (result.type.isNothing()) {
                                result
                            } else {
                                IrSetValueImpl(result.startOffset, result.endOffset, builtIns.unitType, target, result, null)
                            }
                            branches += IrBranchImpl(branch.startOffset, branch.endOffset, branch.condition, branchBody)
                        }
                    }
                    return compileStatement(pushed, current)
                }
                is IrReturnableBlock -> {
                    val join = newBlock()
                    returnableStack.addLast(ReturnableFrame(value.symbol, join, target))
                    val end = compileStatements(value.statements, current)
                    returnableStack.removeLast()
                    if (end.terminator == null) {
                        // Inlined function bodies complete through returns; a
                        // fall-through end would leave the target unassigned.
                        throw BailOut("returnable block in assignment falls through")
                    }
                    return join
                }
                is IrContainerExpression -> {
                    val stmts = value.statements
                    if (stmts.isEmpty()) throw BailOut("empty block in assignment")
                    var cur = current
                    for (i in 0 until stmts.size - 1) cur = compileStatement(stmts[i], cur)
                    val lastExpr = stmts.last() as? IrExpression ?: throw BailOut("assignment block last statement is not an expression")
                    return compileAssignmentOf(target, lastExpr, cur)
                }
                else -> throw BailOut("target call or return nested in assignment ${value::class.simpleName}")
            }
        }

        private fun compileWhen(whenExpr: IrWhen, current: BlockPlan): BlockPlan {
            val join = newBlock()
            var condBlock = current
            for (branch in whenExpr.branches) {
                if (needsSplit(branch.condition)) throw BailOut("target call or return in when condition")
                val isElse = branch.condition is IrConst && (branch.condition as IrConst).value == true
                val bodyBlock = newBlock()
                if (isElse) {
                    condBlock.terminator = Terminator.Goto(bodyBlock)
                } else {
                    val next = newBlock()
                    condBlock.terminator = Terminator.CondGoto(branch.condition, bodyBlock, next)
                    condBlock = next
                }
                val bodyEnd = compileStatement(branch.result.asStatement(), bodyBlock)
                if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(join)
                if (isElse) {
                    return joinOrContinue(join)
                }
            }
            // No else branch: fall through from the last condition block.
            condBlock.terminator = Terminator.Goto(join)
            return joinOrContinue(join)
        }

        private fun joinOrContinue(join: BlockPlan): BlockPlan = join

        private fun IrExpression.asStatement(): IrStatement = this

        private fun compileWhile(loop: IrWhileLoop, current: BlockPlan): BlockPlan {
            if (needsSplit(loop.condition)) throw BailOut("target call or return in loop condition")
            val head = newBlock()
            val bodyBlock = newBlock()
            val exit = newBlock()
            current.terminator = Terminator.Goto(head)
            head.terminator = Terminator.CondGoto(loop.condition, bodyBlock, exit)
            loopStack.addLast(LoopFrame(loop, head, exit))
            val bodyEnd = loop.body?.let { compileStatement(it, bodyBlock) } ?: bodyBlock
            if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(head)
            loopStack.removeLast()
            return exit
        }

        private fun compileDoWhile(loop: IrDoWhileLoop, current: BlockPlan): BlockPlan {
            if (needsSplit(loop.condition)) throw BailOut("target call or return in loop condition")
            val bodyBlock = newBlock()
            val condBlock = newBlock()
            val exit = newBlock()
            current.terminator = Terminator.Goto(bodyBlock)
            loopStack.addLast(LoopFrame(loop, condBlock, exit))
            val bodyEnd = loop.body?.let { compileStatement(it, bodyBlock) } ?: bodyBlock
            if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(condBlock)
            condBlock.terminator = Terminator.CondGoto(loop.condition, bodyBlock, exit)
            loopStack.removeLast()
            return exit
        }

        private fun checkArgsHaveNoTargetCalls(call: IrCall) {
            for (arg in call.arguments) {
                if (arg != null && containsTargetCall(arg)) throw BailOut("target call in argument of target call")
            }
        }

        /** Any statement holding a target call, an outer return, or a jump to
         *  a loop that is being split must itself be compiled into block
         *  structure, so those transfers route through the state machine. */
        private fun needsSplit(element: IrElement): Boolean =
            containsTargetCall(element) || containsOuterReturn(element) ||
                    containsJumpToSplitLoop(element) || containsReturnToSplitBlock(element)

        private fun containsReturnToSplitBlock(element: IrElement): Boolean {
            if (returnableStack.isEmpty()) return false
            val targets = returnableStack.mapTo(mutableSetOf<Any>()) { it.symbol }
            var found = false
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitFunction(declaration: IrFunction) {}

                override fun visitReturn(expression: IrReturn) {
                    if (expression.returnTargetSymbol in targets) {
                        found = true
                        return
                    }
                    expression.acceptChildrenVoid(this)
                }
            })
            return found
        }

        private fun containsJumpToSplitLoop(element: IrElement): Boolean {
            if (loopStack.isEmpty()) return false
            val splitLoops = loopStack.mapTo(mutableSetOf()) { it.loop }
            var found = false
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitFunction(declaration: IrFunction) {}

                override fun visitBreakContinue(jump: IrBreakContinue) {
                    if (jump.loop in splitLoops) found = true
                }
            })
            return found
        }

        private fun containsOuterReturn(element: IrElement): Boolean {
            var found = false
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitFunction(declaration: IrFunction) {}

                override fun visitReturn(expression: IrReturn) {
                    if (expression.returnTargetSymbol == func.symbol) {
                        found = true
                        return
                    }
                    expression.acceptChildrenVoid(this)
                }
            })
            return found
        }

        private fun containsTargetCall(element: IrElement): Boolean {
            var found = false
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitFunction(declaration: IrFunction) {
                    // Local functions are analyzed separately; a target call inside
                    // one cannot be split here.
                    if (hasTargetCallInside(declaration)) found = true
                }

                override fun visitCall(expression: IrCall) {
                    if (isTarget(expression)) {
                        found = true
                        return
                    }
                    expression.acceptChildrenVoid(this)
                }
            })
            return found
        }

        private fun hasTargetCallInside(declaration: IrFunction): Boolean {
            var found = false
            declaration.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }

                override fun visitCall(expression: IrCall) {
                    if (isTarget(expression)) found = true
                    if (!found) expression.acceptChildrenVoid(this)
                }
            })
            return found
        }
    }

    // ================================================================ hierarchy lowering

    private fun lowerHierarchy(
        irModule: IrModuleFragment,
        base: IrClass,
        baseMethod: IrSimpleFunction,
        allClasses: List<IrClass>,
    ) {
        val overrides = collectOverrides(base, baseMethod, allClasses)
        if (overrides.isEmpty()) return

        val bailReasons = mutableMapOf<String, String>()

        val plans = mutableListOf<BodyPlan>()
        val bailedOut = mutableListOf<OverrideInfo>()
        for (info in overrides) {
            val clsName = info.irClass!!.name.asString()
            // Plan on a deep copy: the native body stays intact for the
            // shallow path of the hybrid scheme.
            val original = info.function.body as? IrBlockBody
            if (original == null) {
                bailedOut += info
                bailReasons[clsName] = "no block body"
                continue
            }
            val copy = original.deepCopyWithSymbols(info.function)
            val plan = BodyPlanner(info.function, baseMethod, copy).plan()
            if (plan != null) plans += plan else {
                bailedOut += info
                bailReasons[clsName] = lastBailReason
            }
        }
        if (plans.isEmpty()) return

        context.irFactory.stageController.restrictTo(plans.first().info.function) {
            HierarchyCodegen(base, baseMethod, plans, bailedOut).generate()
        }
        reportPlanSummary(base, plans, bailedOut, bailReasons)
    }

    // ================================================================ codegen

    /**
     * State numbering: 0 = APPLY (pop frame / return), block states start at 1.
     * Every planned override's body is replaced with
     * `return run$vcps(<entryState>, this, <args>)`.
     *
     * Frame layout (single class, untyped Any? slots, v1): outer, resume,
     * r0..rN where the slots hold, in order: self, then the plan's hoisted
     * locals, then the base method's value arguments. Primitives are boxed by
     * the later autoboxing lowering; typed slots are a follow-up optimization.
     */
    private inner class HierarchyCodegen(
        private val base: IrClass,
        private val baseMethod: IrSimpleFunction,
        private val plans: List<BodyPlan>,
        private val bailedOut: List<OverrideInfo>,
    ) {
        private val allOverrides: List<OverrideInfo>
            get() = plans.map { it.info } + bailedOut

        private val irFile = base.file
        private val builtIns = context.irBuiltIns

        private val valueParamCount = baseMethod.parameters.count { it.kind == IrParameterKind.Regular }

        /** Global state id per (plan, block). */
        private val stateIds = mutableMapOf<BlockPlan, Int>()
        private val planEntryState = mutableMapOf<BodyPlan, Int>()

        /** resume block -> the SuspendCall that targets it. */
        private val resumeInfo = mutableMapOf<BlockPlan, Terminator.SuspendCall>()

        /**
         * Value parameters of the base method whose value is passed through
         * unchanged (as `IrGetValue` of the same parameter) at EVERY recursive
         * call site of every plan. Such parameters never change during one
         * trampoline activation, so they are neither updated at transfers nor
         * captured in frames. For the regex matcher this removes testString
         * and matchResult from every frame.
         */
        private val invariantParams: List<Boolean> = run {
            val inv = BooleanArray(valueParamCount) { true }
            for (plan in plans) {
                val params = plan.info.function.parameters.filter { it.kind == IrParameterKind.Regular }
                fun checkCall(call: IrCall) {
                    for (i in 0 until valueParamCount) {
                        val a = call.arguments[i + 1]
                        if (!(a is IrGetValue && a.symbol == params[i].symbol)) inv[i] = false
                    }
                }
                for (b in plan.blocks) {
                    when (val t = b.terminator) {
                        is Terminator.TailCall -> checkCall(t.call)
                        is Terminator.SuspendCall -> checkCall(t.call)
                        else -> {}
                    }
                }
            }
            inv.toList()
        }

        // ---------------- typed frame slot pools

        private fun poolOf(type: IrType): IrType = when (type) {
            builtIns.intType -> builtIns.intType
            builtIns.booleanType -> builtIns.booleanType
            builtIns.charType -> builtIns.charType
            builtIns.longType -> builtIns.longType
            builtIns.floatType -> builtIns.floatType
            builtIns.doubleType -> builtIns.doubleType
            else -> builtIns.anyNType
        }

        private val planCaptures: Map<BodyPlan, List<Capture>> = plans.associateWith { plan ->
            val counters = mutableMapOf<IrType, Int>()
            fun cap(key: CaptureKey, declaredType: IrType): Capture {
                val pool = poolOf(declaredType)
                val idx = counters.getOrElse(pool) { 0 }
                counters[pool] = idx + 1
                return Capture(key, declaredType, pool, idx)
            }

            val valueParams = plan.info.function.parameters.filter { it.kind == IrParameterKind.Regular }
            buildList {
                add(cap(CaptureKey.Self, plan.info.irClass!!.defaultType))
                for (local in plan.locals) add(cap(CaptureKey.Local(local.symbol), local.type))
                for (i in valueParams.indices) {
                    if (!invariantParams[i]) add(cap(CaptureKey.Arg(i), valueParams[i].type))
                }
            }
        }

        /** Pool emission order is fixed so ctor argument layout is stable. */
        private val poolOrder: List<IrType> = listOf(
            builtIns.intType, builtIns.booleanType, builtIns.charType,
            builtIns.longType, builtIns.floatType, builtIns.doubleType, builtIns.anyNType,
        )

        private val poolSizes: Map<IrType, Int> = poolOrder.associateWith { pool ->
            plans.maxOf { plan -> planCaptures[plan]!!.count { it.pool == pool } }
        }

        fun generate() {
            for (plan in plans) simplifyPlan(plan)

            var next = 1
            for (plan in plans) {
                val live = reachableBlocks(plan)
                for (block in live) {
                    stateIds[block] = next++
                }
                planEntryState[plan] = stateIds[plan.entry]!!
                for (block in live) {
                    val t = block.terminator
                    if (t is Terminator.SuspendCall) resumeInfo[t.resume] = t
                }
            }

            val frameClass = buildFrameClass()
            val stateIdFun = buildStateOfFun()
            val runFun = buildTrampoline(frameClass, stateIdFun)
            val depthField = buildDepthField()
            val enterFun = buildEnterFun(stateIdFun, runFun)
            for (info in allOverrides) instrumentNativeBody(info, depthField, enterFun)
        }

        // ---------------- frame class

        private lateinit var frameOuterField: IrField
        private lateinit var frameResumeField: IrField
        private lateinit var framePoolFields: Map<IrType, List<IrField>>
        private lateinit var frameCtorParamOrder: List<Pair<IrType, Int>>
        private lateinit var frameCtor: IrConstructorSymbol

        private fun buildFrameClass(): IrClass {
            val cls = context.irFactory.buildClass {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("\$VCpsFrame\$${base.name}")
                visibility = DescriptorVisibilities.PRIVATE
                modality = Modality.FINAL
                kind = ClassKind.CLASS
            }.apply {
                parent = irFile
                irFile.declarations += this
                createThisReceiverParameter()
                superTypes = listOf(builtIns.anyType)
            }

            fun poolPrefix(pool: IrType): String = when (pool) {
                builtIns.intType -> "i"
                builtIns.booleanType -> "z"
                builtIns.charType -> "c"
                builtIns.longType -> "j"
                builtIns.floatType -> "f"
                builtIns.doubleType -> "d"
                else -> "r"
            }

            val fieldDefs = buildList {
                add("outer" to cls.defaultType.makeNullable())
                add("resume" to builtIns.intType)
                for (pool in poolOrder) {
                    repeat(poolSizes[pool]!!) { add("${poolPrefix(pool)}$it" to pool) }
                }
            }
            val fields = fieldDefs.map { def ->
                cls.addField {
                    startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                    name = Name.identifier(def.first)
                    type = def.second
                    visibility = DescriptorVisibilities.PUBLIC
                    isFinal = true
                }
            }
            frameOuterField = fields[0]
            frameResumeField = fields[1]
            val byPool = mutableMapOf<IrType, MutableList<IrField>>()
            val ctorOrder = mutableListOf<Pair<IrType, Int>>()
            var fi = 2
            for (pool in poolOrder) {
                val list = mutableListOf<IrField>()
                repeat(poolSizes[pool]!!) { k ->
                    list += fields[fi++]
                    ctorOrder += pool to k
                }
                byPool[pool] = list
            }
            framePoolFields = byPool
            frameCtorParamOrder = ctorOrder

            frameCtor = cls.addConstructor {
                isPrimary = true
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            }.apply {
                val params = fieldDefs.map { def ->
                    addValueParameter {
                        name = Name.identifier(def.first)
                        type = def.second
                        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                    }
                }
                body = context.createIrBuilder(symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET).irBlockBody {
                    +irDelegatingConstructorCall(builtIns.anyClass.owner.constructors.single())
                    +IrInstanceInitializerCallImpl(
                        UNDEFINED_OFFSET, UNDEFINED_OFFSET, cls.symbol, builtIns.unitType
                    )
                    for (i in fields.indices) {
                        +irSetField(irGet(cls.thisReceiver!!), fields[i], irGet(params[i]))
                    }
                }
            }.symbol

            return cls
        }

        // ---------------- vcpsStateId virtual

        private fun buildStateOfFun(): IrSimpleFunction {
            // instanceof chain, most-derived classes first. Bailed-out override
            // classes map to -1 so they never get routed to an ancestor's plan.
            fun depthOf(cls: IrClass): Int {
                var d = 0
                var cur: IrClass? = cls
                while (cur != null && cur != base) {
                    cur = cur.superTypes.firstNotNullOfOrNull { it.classOrNull?.owner?.takeIf { c -> c.isClass || c.kind == ClassKind.OBJECT } }
                    d++
                }
                return d
            }

            val entries = buildList {
                for (plan in plans) {
                    val cls = plan.info.irClass!!
                    add(Triple(cls, planEntryState[plan]!!, depthOf(cls)))
                }
                for (info in bailedOut) {
                    val cls = info.irClass!!
                    add(Triple(cls, -1, depthOf(cls)))
                }
            }.sortedByDescending { it.third }

            val fn = context.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("vcps\$stateOf\$${base.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = builtIns.intType
            }
            val pRecv = fn.addValueParameter("recv", base.defaultType)
            val b = context.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body = b.irBlockBody {
                +irReturn(irWhen(builtIns.intType, buildList {
                    for (e in entries) {
                        add(irBranch(irIs(irGet(pRecv), e.first.defaultType), irInt(e.second)))
                    }
                    add(irBranch(irTrue(), irInt(-1)))
                }))
            }
            return fn
        }

        // ---------------- trampoline

        private fun buildTrampoline(frameClass: IrClass, stateIdFun: IrSimpleFunction): IrSimpleFunction {
            val runFun = context.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("run\$vcps\$${baseMethod.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = baseMethod.returnType
            }
            val pState = runFun.addValueParameter("state0", builtIns.intType)
            val pRecv = runFun.addValueParameter("recv0", base.defaultType)
            val valueParams = baseMethod.parameters.filter { it.kind == IrParameterKind.Regular }
            val pArgs = valueParams.mapIndexed { i, p -> runFun.addValueParameter("a$i", p.type) }

            val b = context.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val frameType = frameClass.defaultType.makeNullable()

            runFun.body = b.irBlockBody {
                val vState = irTemporary(irGet(pState), nameHint = "s", isMutable = true)
                val vRecv = irTemporary(irGet(pRecv), nameHint = "recv", isMutable = true)
                val vArgs = pArgs.mapIndexed { i, p ->
                    irTemporary(irGet(p), nameHint = "arg$i", isMutable = true)
                }
                val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true)
                val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true)
                val vRet = irTemporary(
                    baseMethod.returnType.defaultValue(), nameHint = "ret", isMutable = true,
                    irType = baseMethod.returnType,
                )

                // Hoisted locals + self vars, per plan.
                val hoisted = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                val selfVars = mutableMapOf<BodyPlan, IrValueDeclaration>()
                for (pi in plans.indices) {
                    val plan = plans[pi]
                    val cls = plan.info.irClass
                    selfVars[plan] = irTemporary(
                        irNull(cls!!.defaultType.makeNullable()), nameHint = "self$pi", isMutable = true,
                        irType = cls.defaultType.makeNullable(),
                    )
                    for (local in plan.locals) {
                        hoisted[local.symbol] = irTemporary(
                            local.type.defaultValue(), nameHint = "l${pi}_${local.name}", isMutable = true,
                            irType = local.type,
                        )
                    }
                }

                val loop = b.irWhile().apply { condition = b.irTrue() }

                // Helper closures used by block emission.
                val emitter = BlockEmitter(
                    b, loop, runFun, stateIdFun, frameClass,
                    vState, vRecv, vArgs, vTop, vFrame, vRet, hoisted, selfVars,
                )

                val intEq = this@WasmVirtualCpsLowering.context.wasmSymbols
                    .equalityFunctions[builtIns.intType]
                fun stateEquals(sid: Int): IrExpression =
                    if (intEq != null) {
                        irCall(intEq).apply {
                            arguments[0] = irGet(vState)
                            arguments[1] = irInt(sid)
                        }
                    } else {
                        irEquals(irGet(vState), irInt(sid))
                    }

                loop.body = b.irBlock {
                    +irWhen(builtIns.unitType, buildList {
                        for (plan in plans) {
                            for (block in plan.blocks) {
                                val sid = stateIds[block] ?: continue
                                val branchBody = irBlock {
                                    emitter.emitBlock(this, plan, block, stateIds, resumeInfo)
                                }
                                add(irBranch(stateEquals(sid), branchBody))
                            }
                        }
                        // else: APPLY state — pop frame or return.
                        val applyBlock = irBlock {
                            val fTmp = irTemporary(irGet(vTop), nameHint = "f")
                            +irIfThen(
                                builtIns.unitType,
                                irEqualsNull(irGet(fTmp)),
                                irReturn(irGet(vRet)),
                            )
                            +irSet(vTop.symbol, irGetField(irGet(fTmp), frameOuterField))
                            +irSet(vFrame.symbol, irGet(fTmp))
                            +irSet(vState.symbol, irGetField(irGet(fTmp), frameResumeField))
                            +irContinue(loop)
                        }
                        add(irBranch(irTrue(), applyBlock))
                    })
                }
                +loop
            }
            runFun.body!!.patchDeclarationParents(runFun)
            return runFun
        }

        private fun IrType.defaultValue(): IrExpression = when {
            this == builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            this == builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, false)
            this == builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, ' ')
            this == builtIns.byteType -> IrConstImpl.byte(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            this == builtIns.shortType -> IrConstImpl.short(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            this == builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0)
            this == builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0.0)
            this == builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, this, 0.0f)
            else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, makeNullable())
        }

        // ---------------- per-block emission

        private inner class BlockEmitter(
            private val b: IrBuilderWithScope,
            private val loop: IrLoop,
            private val runFun: IrSimpleFunction,
            private val stateIdFun: IrSimpleFunction,
            private val frameClass: IrClass,
            private val vState: IrValueDeclaration,
            private val vRecv: IrValueDeclaration,
            private val vArgs: List<IrValueDeclaration>,
            private val vTop: IrValueDeclaration,
            private val vFrame: IrValueDeclaration,
            private val vRet: IrValueDeclaration,
            private val hoisted: Map<IrValueSymbol, IrValueDeclaration>,
            private val selfVars: Map<BodyPlan, IrValueDeclaration>,
        ) {

            /** value-symbol remapping for statements of [plan]. */
            private fun remapperFor(plan: BodyPlan): Map<IrValueSymbol, IrValueDeclaration> {
                val map = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                val fn = plan.info.function
                var argIdx = 0
                for (p in fn.parameters) {
                    when (p.kind) {
                        IrParameterKind.DispatchReceiver -> map[p.symbol] = selfVars[plan]!!
                        IrParameterKind.Regular -> map[p.symbol] = vArgs[argIdx++]
                        else -> {}
                    }
                }
                for (local in plan.locals) map[local.symbol] = hoisted[local.symbol]!!
                return map
            }

            @Suppress("UNCHECKED_CAST")
            private fun <T : IrElement> remap(element: T, map: Map<IrValueSymbol, IrValueDeclaration>): T {
                return element.transform(object : IrTransformer<Nothing?>() {
                    override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
                        val to = map[expression.symbol] ?: return super.visitGetValue(expression, data)
                        return IrGetValueImpl(expression.startOffset, expression.endOffset, to.type, to.symbol)
                    }

                    override fun visitSetValue(expression: IrSetValue, data: Nothing?): IrExpression {
                        expression.transformChildren(this, data)
                        val to = map[expression.symbol] ?: return expression
                        return IrSetValueImpl(
                            expression.startOffset, expression.endOffset, builtIns.unitType,
                            to.symbol, expression.value, expression.origin,
                        )
                    }
                }, null) as T
            }

            fun emitBlock(
                bb: IrBlockBuilder,
                plan: BodyPlan,
                block: BlockPlan,
                stateIds: Map<BlockPlan, Int>,
                resumeInfo: Map<BlockPlan, Terminator.SuspendCall>,
            ): Unit = with(bb) {
                val map = remapperFor(plan)
                val self = selfVars[plan]!!

                if (block === plan.entry) {
                    // self = recv as C
                    +irSet(self.symbol, irAs(irGet(vRecv), plan.info.irClass!!.defaultType))
                }

                resumeInfo[block]?.let { susp ->
                    // Restore captured state from the popped frame, then bind result.
                    emitRestores(bb, plan, map)
                    val resultTarget = map[susp.resultSymbol] ?: hoisted[susp.resultSymbol]!!
                    +irSet(resultTarget.symbol, castTo(irGet(vRet), resultTarget.type))
                }

                for (stmt in block.statements) {
                    when (stmt) {
                        is IrVariable -> {
                            val target = hoisted[stmt.symbol]!!
                            val init = stmt.initializer
                            if (init != null) {
                                +irSet(target.symbol, remap(init, map))
                            }
                        }
                        else -> {
                            +remap(stmt, map)
                        }
                    }
                }

                when (val t = block.terminator!!) {
                    is Terminator.Goto -> {
                        +irSet(vState.symbol, irInt(stateIds[t.target]!!))
                        +irContinue(loop)
                    }

                    is Terminator.CondGoto -> {
                        +irIfThenElse(
                            builtIns.unitType,
                            remap(t.condition, map),
                            irBlock {
                                +irSet(vState.symbol, irInt(stateIds[t.thenTarget]!!))
                                +irContinue(loop)
                            },
                            irBlock {
                                +irSet(vState.symbol, irInt(stateIds[t.elseTarget]!!))
                                +irContinue(loop)
                            },
                        )
                    }

                    is Terminator.Ret -> {
                        +irSet(vRet.symbol, remap(t.value, map))
                        +irSet(vState.symbol, irInt(0))
                        +irContinue(loop)
                    }

                    is Terminator.TailCall -> {
                        emitCallTransfer(bb, plan, map, t.call)
                        +irContinue(loop)
                    }

                    is Terminator.SuspendCall -> {
                        // Push frame capturing self + hoisted locals + varying args
                        // into typed slots (no boxing).
                        val resumeId = stateIds[t.resume]!!
                        val captures = planCaptures[plan]!!
                        +irSet(vTop.symbol, irCallConstructor(frameCtor, emptyList()).apply {
                            arguments[0] = irGet(vTop)
                            arguments[1] = irInt(resumeId)
                            for (ci in frameCtorParamOrder.indices) {
                                val slot = frameCtorParamOrder[ci]
                                val capture = captures.firstOrNull { it.pool == slot.first && it.poolIndex == slot.second }
                                arguments[ci + 2] = when {
                                    capture == null -> slot.first.defaultValue()
                                    capture.key is CaptureKey.Self -> irGet(self)
                                    else -> irGet(resolveCapture(capture))
                                }
                            }
                        })
                        emitCallTransfer(bb, plan, map, t.call)
                        +irContinue(loop)
                    }
                }
            }

            fun resolveCapture(capture: Capture): IrValueDeclaration = when (val k = capture.key) {
                is CaptureKey.Self -> error("self capture must be resolved per plan")
                is CaptureKey.Local -> hoisted[k.symbol]!!
                is CaptureKey.Arg -> vArgs[k.index]
            }

            private fun emitRestores(bb: IrBlockBuilder, plan: BodyPlan, map: Map<IrValueSymbol, IrValueDeclaration>): Unit = with(bb) {
                for (capture in planCaptures[plan]!!) {
                    val target = when (val k = capture.key) {
                        is CaptureKey.Self -> selfVars[plan]!!
                        is CaptureKey.Local -> hoisted[k.symbol]!!
                        is CaptureKey.Arg -> vArgs[k.index]
                    }
                    val field = framePoolFields[capture.pool]!![capture.poolIndex]
                    val read = irGetField(irGet(vFrame), field)
                    val value = if (capture.pool == builtIns.anyNType) castTo(read, target.type) else read
                    +irSet(target.symbol, value)
                }
            }

            private fun IrBlockBuilder.castTo(expr: IrExpression, type: IrType): IrExpression = when {
                type == builtIns.anyNType -> expr
                // Restored slots may legitimately hold null (locals captured
                // before their first assignment); cast through the nullable
                // type so restoration never null-checks.
                !type.isPrimitiveType() -> irAs(expr, type.makeNullable())
                else -> irAs(expr, type)
            }

            /**
             * Common transfer for tail and suspend calls: load args, swap
             * receiver, dispatch via vcps$stateId; unplanned receiver falls
             * back to a native virtual call whose result goes through APPLY.
             */
            private fun emitCallTransfer(
                bb: IrBlockBuilder,
                plan: BodyPlan,
                map: Map<IrValueSymbol, IrValueDeclaration>,
                call: IrCall,
            ): Unit = with(bb) {
                val recvExpr = remap(call.arguments[0]!!, map)
                // Invariant parameters hold the same value across the whole
                // activation; neither evaluate nor update them.
                val varying = (0 until valueParamCount).filter { !invariantParams[it] }
                val argExprs = varying.associateWith { i ->
                    remap(call.arguments[i + 1]!!, map)
                }
                // Evaluate args before overwriting the shared arg vars.
                val recvTmp = irTemporary(recvExpr, nameHint = "nrecv")
                val argTmps = varying.associateWith { i -> irTemporary(argExprs[i]!!, nameHint = "narg$i") }
                for (i in varying) {
                    +irSet(vArgs[i].symbol, irGet(argTmps[i]!!))
                }
                +irSet(vRecv.symbol, irGet(recvTmp))
                +irSet(vState.symbol, irCall(stateIdFun.symbol).apply {
                    arguments[0] = irGet(vRecv)
                })
                // Unplanned class: run natively, result flows through APPLY.
                +irIfThen(
                    builtIns.unitType,
                    irEquals(irGet(vState), irInt(-1)),
                    irBlock {
                        +irSet(vRet.symbol, irCall(baseMethod.symbol).apply {
                            arguments[0] = irGet(vRecv)
                            for (i in vArgs.indices) arguments[i + 1] = irGet(vArgs[i])
                        })
                        +irSet(vState.symbol, irInt(0))
                    },
                )
            }
        }

        // ---------------- hybrid depth threshold

        /**
         * Module-level recursion depth counter for the native (shallow) path.
         * Wasm is single-threaded, so a plain mutable top-level field is safe.
         * Known limitation: an exception thrown through an instrumented call
         * leaks the increment; the matcher throws nothing on the match path.
         */
        private fun buildDepthField(): IrField {
            return context.irFactory.buildField {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("vcps\$depth\$${base.name}")
                type = builtIns.intType
                visibility = DescriptorVisibilities.PRIVATE
                isStatic = true
                isFinal = false
            }.apply {
                parent = irFile
                irFile.declarations += this
                initializer = context.irFactory.createExpressionBody(
                    IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, builtIns.intType, 0)
                )
            }
        }

        /** `fun vcps$enter(recv, args): Int` — route one subtree through the trampoline. */
        private fun buildEnterFun(stateIdFun: IrSimpleFunction, runFun: IrSimpleFunction): IrSimpleFunction {
            val fn = context.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("vcps\$enter\$${baseMethod.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = baseMethod.returnType
            }
            val pRecv = fn.addValueParameter("recv", base.defaultType)
            val valueParams = baseMethod.parameters.filter { it.kind == IrParameterKind.Regular }
            val pArgs = valueParams.mapIndexed { i, vp -> fn.addValueParameter("a$i", vp.type) }
            val b = context.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body = b.irBlockBody {
                val state = irTemporary(irCall(stateIdFun.symbol).apply {
                    arguments[0] = irGet(pRecv)
                }, nameHint = "s")
                +irReturn(
                    irIfThenElse(
                        baseMethod.returnType,
                        irEquals(irGet(state), irInt(-1)),
                        irCall(baseMethod.symbol).apply {
                            arguments[0] = irGet(pRecv)
                            for (i in pArgs.indices) arguments[i + 1] = irGet(pArgs[i])
                        },
                        irCall(runFun.symbol).apply {
                            arguments[0] = irGet(state)
                            arguments[1] = irGet(pRecv)
                            for (i in pArgs.indices) arguments[i + 2] = irGet(pArgs[i])
                        },
                    )
                )
            }
            return fn
        }

        /**
         * Rewrites every target call in the ORIGINAL body:
         * shallow (depth < threshold) keeps the native virtual call under a
         * counter; deep routes the subtree through the trampoline and gets a
         * plain value back, so no unwinding is ever needed.
         */
        private fun instrumentNativeBody(info: OverrideInfo, depthField: IrField, enterFun: IrSimpleFunction) {
            val fn = info.function
            val b = context.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body?.transform(object : IrTransformer<Nothing?>() {
                override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                    expression.transformChildren(this, data)
                    if (!isTargetCall(expression, baseMethod)) return expression
                    return b.irBlock(resultType = baseMethod.returnType) {
                        val recvTmp = irTemporary(expression.arguments[0]!!, nameHint = "hrecv")
                        val argTmps = (1 until expression.arguments.size).map { i ->
                            irTemporary(expression.arguments[i]!!, nameHint = "harg${i - 1}")
                        }
                        fun depthGet() = irGetField(null, depthField)
                        +irIfThenElse(
                            baseMethod.returnType,
                            irCall(builtIns.lessFunByOperandType[builtIns.intClass]!!).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(HYBRID_DEPTH_THRESHOLD)
                            },
                            irBlock(resultType = baseMethod.returnType) {
                                +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                    arguments[0] = depthGet()
                                    arguments[1] = irInt(1)
                                })
                                val r = irTemporary(irCall(expression.symbol).apply {
                                    arguments[0] = irGet(recvTmp)
                                    for (i in argTmps.indices) arguments[i + 1] = irGet(argTmps[i])
                                }, nameHint = "hres")
                                +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                    arguments[0] = depthGet()
                                    arguments[1] = irInt(-1)
                                })
                                +irGet(r)
                            },
                            irCall(enterFun.symbol).apply {
                                arguments[0] = irGet(recvTmp)
                                for (i in argTmps.indices) arguments[i + 1] = irGet(argTmps[i])
                            },
                        )
                    }
                }
            }, null)
            fn.body?.patchDeclarationParents(fn)
        }
    }

    private fun reportPlanSummary(
        base: IrClass,
        plans: List<BodyPlan>,
        bailedOut: List<OverrideInfo>,
        bailReasons: Map<String, String>,
    ) {
        val msg = buildString {
            append("[wasm-virtual-cps] ${base.name}: planned=${plans.size} bailout=${bailedOut.size}")
            if (bailedOut.isNotEmpty()) {
                append(" (")
                append(bailedOut.joinToString(", ") {
                    val n = it.irClass?.name?.asString() ?: it.function.name.asString()
                    "$n[${bailReasons[n] ?: "?"}]"
                })
                append(")")
            }
        }
        @OptIn(MessageCollectorAccess::class)
        context.configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)
            ?.report(CompilerMessageSeverity.LOGGING, msg)
    }
}
