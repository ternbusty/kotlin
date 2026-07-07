/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.common.lower.AbstractVariableRemapper
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.config.messageCollector
import org.jetbrains.kotlin.wasm.config.wasmEnableDrfAcceleration
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.util.*
import org.jetbrains.kotlin.ir.visitors.IrElementTransformerVoid
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.ir.visitors.transformChildrenVoid

/** Origin of every declaration this pass synthesizes; also the idempotency marker for re-runs. */
internal val DRF_SYNTHESIZED_DECLARATION by IrDeclarationOriginImpl.Regular

private val invokeName = Name.identifier("invoke")
private val callRecursiveName = Name.identifier("callRecursive")
private val drfClassName = Name.identifier("DeepRecursiveFunction")
private val drfClassFqName = FqName("kotlin.DeepRecursiveFunction")
private val drfScopeFqName = FqName("kotlin.DeepRecursiveScope")
private val kotlinPackageFqName = FqName("kotlin")

private const val MAX_SUSPEND_HELPER_INLINE_ROUNDS = 5

/**
 * Accelerates DeepRecursiveFunction on Wasm by compiling detected literals
 * into a synthesized native twin plus a heap-frame trampoline
 * ([WasmStacklessBodyPlanner] plans the lambda body; callRecursive is the
 * suspend point), and rewriting invoke sites to call the twin directly.
 * The coroutine machinery disappears from the executed path. Bodies stay
 * native below [STACKLESS_HYBRID_DEPTH_THRESHOLD]; deeper subtrees delegate
 * to the trampoline once and return by value.
 *
 * Literals with mutable free-variable captures, other suspend calls, or
 * untrackable holders bail out and keep stock DeepRecursiveFunction
 * behavior, so partial coverage never changes semantics.
 */
internal class WasmDrfAccelerationLowering(private val context: WasmBackendContext) : ModuleLoweringPass {

    private val enabled = context.configuration.wasmEnableDrfAcceleration

    override fun lower(irModule: IrModuleFragment) {
        if (!enabled) return
        lowerDeepRecursiveFunctions(irModule)
    }

    private class DrfLiteral(
        val ctorCall: IrConstructorCall,
        val lambda: IrSimpleFunction,
        val holderProperty: IrProperty?,
        val directInvoke: IrCall?,
        val enclosingFunction: IrFunction?,
    )

    private sealed class Outcome {
        class Bailed(val reason: String) : Outcome()
        class Transformed(val twin: IrSimpleFunction, val freeSyms: List<IrValueDeclaration>) : Outcome()
    }

    private class TransformedLiteral(
        val lit: DrfLiteral,
        val twin: IrSimpleFunction,
        val freeSyms: List<IrValueDeclaration>,
    )

    private fun IrCall.isDrfInvoke(): Boolean {
        val callee = symbol.owner
        if (callee.name != invokeName) return false
        if ((callee.parent as? IrPackageFragment)?.packageFqName != kotlinPackageFqName) return false
        return callee.parameters.firstOrNull()?.type?.classFqName == drfClassFqName
    }

    private fun isCallRecursive(call: IrCall): Boolean {
        val callee = call.symbol.owner
        return callee.name == callRecursiveName &&
                (callee.parent as? IrClass)?.fqNameWhenAvailable == drfScopeFqName &&
                call.arguments.size == 2
    }

    private fun IrConstructorCall.constructsDrf(): Boolean {
        val cls = symbol.owner.constructedClass
        return cls.name == drfClassName && cls.fqNameWhenAvailable == drfClassFqName
    }

    private fun boundHolderProperty(recv: IrExpression?): IrProperty? = when (recv) {
        is IrCall -> recv.symbol.owner.correspondingPropertySymbol?.owner
        is IrGetField -> recv.symbol.owner.correspondingPropertySymbol?.owner
        else -> null
    }

    /**
     * Finds DeepRecursiveFunction literals (bound to immutable holders or
     * invoked directly) and their invoke sites in one module walk,
     * transforms each literal that has at least one site, then rewrites
     * all rewritable invoke sites in one more walk.
     */
    private fun lowerDeepRecursiveFunctions(irModule: IrModuleFragment) {
        val literals = mutableListOf<DrfLiteral>()
        // Holder-reading drf invoke calls, paired with the property they
        // read (null when the receiver is not a property read).
        val invokeReads = mutableListOf<Pair<IrCall, IrProperty?>>()
        irModule.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)

            override fun visitProperty(declaration: IrProperty) {
                val init = declaration.backingField?.initializer?.expression
                if (init is IrConstructorCall && !declaration.isVar && init.constructsDrf()) {
                    when (val arg = init.arguments.firstOrNull()) {
                        is IrRichFunctionReference -> literals += DrfLiteral(init, arg.invokeFunction, declaration, null, null)
                        is IrFunctionExpression -> literals += DrfLiteral(init, arg.function, declaration, null, null)
                        else -> {}
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
                if (expression.isDrfInvoke()) {
                    val recv = expression.arguments[0]
                    // Direct form: DeepRecursiveFunction { ... }.invoke(x)
                    if (recv is IrConstructorCall && recv.constructsDrf()) {
                        val arg = recv.arguments.firstOrNull()
                        if (arg is IrRichFunctionReference) {
                            literals += DrfLiteral(recv, arg.invokeFunction, null, expression, functionStack.lastOrNull())
                        }
                    } else {
                        invokeReads += expression to boundHolderProperty(recv)
                    }
                }
                expression.acceptChildrenVoid(this)
            }
        })
        if (literals.isEmpty()) return

        val holders = literals.mapNotNullTo(mutableSetOf()) { it.holderProperty }
        val invokeSites = mutableMapOf<IrProperty, Int>()
        var unboundInvokes = 0
        for (read in invokeReads) {
            val prop = read.second
            if (prop != null && prop in holders) invokeSites.merge(prop, 1, Int::plus) else unboundInvokes++
        }

        @OptIn(MessageCollectorAccess::class)
        val collector = context.configuration.messageCollector
        val transformed = mutableListOf<TransformedLiteral>()
        for (lit in literals) {
            val sites = if (lit.directInvoke != null) 1 else lit.holderProperty?.let { invokeSites[it] } ?: 0
            var status = "skipped"
            if (sites > 0) {
                when (val outcome = transformDrfLiteral(lit)) {
                    is Outcome.Bailed -> status = outcome.reason
                    is Outcome.Transformed -> {
                        transformed += TransformedLiteral(lit, outcome.twin, outcome.freeSyms)
                        status = "transformed"
                    }
                }
            }
            collector.report(
                CompilerMessageSeverity.LOGGING,
                "[wasm-drf] literal at ${lit.holderProperty?.fqNameWhenAvailable}: " +
                        "invokeSites=$sites unbound=$unboundInvokes -> $status",
            )
        }
        if (transformed.isNotEmpty()) rewriteInvokeSites(irModule, transformed)
    }

    /**
     * Hoists every argument list containing a target call into ordered
     * temporaries, so nested shapes like `a + callRecursive(b)` become
     * `{ val t0 = a; val t1 = callRecursive(b); t0 + t1 }`, which the block
     * planner understands. Evaluation order is preserved by hoisting all
     * arguments up to and including the last target-carrying one.
     */
    private fun normalizeTargetCallArguments(body: IrBlockBody, owner: IrSimpleFunction) {
        val b = context.createIrBuilder(owner.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        body.transformChildrenVoid(object : IrElementTransformerVoid() {
            override fun visitCall(expression: IrCall): IrExpression {
                expression.transformChildrenVoid(this)
                if (isCallRecursive(expression)) return expression
                val lastTargetArg = expression.arguments.indices.lastOrNull { i ->
                    expression.arguments[i]?.let { anyCall(it, intoNestedFunctions = false, ::isCallRecursive) } == true
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
        })
        body.patchDeclarationParents(owner)
    }

    private fun drfStableName(lit: DrfLiteral): String =
        lit.holderProperty?.name?.asString()
            ?: "drf\$${lit.ctorCall.startOffset}"

    private fun transformDrfLiteral(lit: DrfLiteral): Outcome {
        val lambda = lit.lambda
        val irFile = (lit.holderProperty?.file ?: lambda.fileOrNull) ?: return Outcome.Bailed("no file")

        // The lowering pipeline may process a module more than once; a
        // synthesized twin for this literal marks it as already handled.
        val twinName = "${drfStableName(lit)}\$drfNative"
        if (irFile.declarations.any {
                it is IrSimpleFunction && it.origin == DRF_SYNTHESIZED_DECLARATION && it.name.asString() == twinName
            }
        ) {
            return Outcome.Bailed("already transformed")
        }
        val regulars = lambda.parameters.filter { it.kind == IrParameterKind.Regular }
        val valueParam = regulars.lastOrNull()
            ?.takeIf { it.type.classFqName != drfScopeFqName }
            ?: return Outcome.Bailed("no value parameter")
        if (regulars.size > 2) return Outcome.Bailed("bound parameters unsupported")
        val tType = valueParam.type
        val rType = lambda.returnType

        // Plan on a normalized deep copy, after inlining suspend helpers that
        // hold callRecursive (JsonTreeReader's readObject pattern).
        val body = lambda.body as? IrBlockBody ?: return Outcome.Bailed("no block body")
        val planCopy = body.deepCopyWithSymbols(lambda)
        inlineSuspendHelpers(planCopy, lambda)?.let { return Outcome.Bailed(it) }
        var hasTarget = false
        var foreignSuspend: String? = null
        planCopy.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitCall(expression: IrCall) {
                if (isCallRecursive(expression)) hasTarget = true
                else if (expression.symbol.owner.isSuspend) foreignSuspend = expression.symbol.owner.name.asString()
                expression.acceptChildrenVoid(this)
            }
        })
        if (!hasTarget) return Outcome.Bailed("no callRecursive after inlining")
        // Post-inline gate: any remaining foreign suspend call cannot run on
        // the trampoline.
        foreignSuspend?.let { return Outcome.Bailed("foreign suspend call: $it") }

        normalizeTargetCallArguments(planCopy, lambda)

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
            if (lit.directInvoke == null) return Outcome.Bailed("free variables with property holder")
            val enclosing = lit.enclosingFunction ?: return Outcome.Bailed("free variables without enclosing function")
            for (sym in freeSyms) {
                val parentFn = when (val d = sym) {
                    is IrValueParameter -> d.parent as? IrFunction
                    is IrVariable -> d.parent as? IrFunction
                    else -> null
                }
                if (parentFn != enclosing) return Outcome.Bailed("free variable ${sym.name} not owned by enclosing function")
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
            if (mutated) return Outcome.Bailed("free variable mutated in lambda")
        }

        val plan = when (val r = WasmStacklessBodyPlanner(context, lambda, planCopy) { isCallRecursive(it) }.plan()) {
            is PlanResult.Bailed -> return Outcome.Bailed("planner: ${r.reason}")
            is PlanResult.Planned -> r.plan
        }

        var twin: IrSimpleFunction? = null
        context.irFactory.stageController.restrictTo(lit.holderProperty ?: lambda) {
            twin = DrfCodegen(irFile, lit, plan, tType, rType, valueParam, freeSyms.toList()).generate()
        }
        return Outcome.Transformed(twin!!, freeSyms.toList())
    }

    /**
     * Inlines private suspend helper functions whose bodies hold target calls
     * (e.g. JsonTreeReader's `DeepRecursiveScope.readObject()`). Restricted to
     * helpers with a single trailing return and side-effect-free call
     * arguments, so parameter substitution cannot duplicate effects.
     * Returns a bail reason or null.
     */
    private fun inlineSuspendHelpers(body: IrBlockBody, owner: IrSimpleFunction): String? {
        // Helper bodies are only copied, never mutated, so the scan result
        // per callee is stable across rounds.
        val holdsTargetCache = mutableMapOf<IrFunction, Boolean>()
        fun holdsTarget(fn: IrFunction): Boolean =
            holdsTargetCache.getOrPut(fn) { anyCall(fn, predicate = ::isCallRecursive) }

        repeat(MAX_SUSPEND_HELPER_INLINE_ROUNDS) {
            var bail: String? = null
            var inlinedAny = false
            body.transformChildrenVoid(object : IrElementTransformerVoid() {
                override fun visitCall(expression: IrCall): IrExpression {
                    expression.transformChildrenVoid(this)
                    if (bail != null) return expression
                    val callee = expression.symbol.owner
                    if (isCallRecursive(expression) || !callee.isSuspend || !holdsTarget(callee)) return expression

                    val calleeBody = callee.body as? IrBlockBody
                    if (calleeBody == null) { bail = "suspend helper ${callee.name} has no block body"; return expression }
                    singleTrailingReturnIssue(calleeBody, callee.symbol)?.let {
                        bail = "suspend helper ${callee.name} $it"; return expression
                    }
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
                    copied.transformChildrenVoid(object : IrElementTransformerVoid() {
                        override fun visitGetValue(expression: IrGetValue): IrExpression {
                            val repl = paramMap[expression.symbol] ?: return super.visitGetValue(expression)
                            return repl.deepCopyWithSymbols(owner)
                        }
                    })

                    inlinedAny = true
                    return spliceInlineBody(expression, copied, callee.returnType)
                }
            })
            bail?.let { return it }
            if (!inlinedAny) return null
        }
        return "suspend helper inlining did not converge"
    }

    private fun rewriteInvokeSites(irModule: IrModuleFragment, transformed: List<TransformedLiteral>) {
        val byDirectCall = mutableMapOf<IrCall, TransformedLiteral>()
        val byHolder = mutableMapOf<IrProperty, TransformedLiteral>()
        for (t in transformed) {
            val direct = t.lit.directInvoke
            if (direct != null) byDirectCall[direct] = t else t.lit.holderProperty?.let { byHolder[it] = t }
        }
        irModule.transformChildrenVoid(object : IrElementTransformerVoid() {
            override fun visitCall(expression: IrCall): IrExpression {
                expression.transformChildrenVoid(this)
                if (!expression.isDrfInvoke()) return expression
                byDirectCall[expression]?.let { t ->
                    // Direct form: free variables are in scope at this site.
                    return IrCallImpl(
                        expression.startOffset, expression.endOffset, t.twin.returnType, t.twin.symbol,
                        typeArgumentsCount = 0,
                    ).apply {
                        for (i in t.freeSyms.indices) {
                            arguments[i] = IrGetValueImpl(
                                expression.startOffset, expression.endOffset,
                                t.freeSyms[i].type, t.freeSyms[i].symbol,
                            )
                        }
                        arguments[t.freeSyms.size] = expression.arguments[1]
                    }
                }
                val t = boundHolderProperty(expression.arguments[0])?.let { byHolder[it] } ?: return expression
                return IrCallImpl(
                    expression.startOffset, expression.endOffset, t.twin.returnType, t.twin.symbol,
                    typeArgumentsCount = 0,
                ).apply {
                    arguments[0] = expression.arguments[1]
                }
            }
        })
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
        private val backendContext = this@WasmDrfAccelerationLowering.context
        private val builtIns = backendContext.irBuiltIns
        private val name = drfStableName(lit)

        private val captures: List<Capture> = buildCaptures(builtIns, buildList {
            for (local in plan.locals) add(CaptureKey.Local(local.symbol) to local.type)
            add(CaptureKey.Arg(0) to tType)
        })
        private val poolSizes: Map<IrType, Int> =
            framePoolOrder(builtIns).associateWith { pool -> captures.count { it.pool == pool } }

        private lateinit var frame: FrameLayout

        fun generate(): IrSimpleFunction {
            val states = assignStates(listOf(plan))
            frame = buildFrameClass(backendContext, irFile, "\$DrfFrame\$$name", poolSizes, DRF_SYNTHESIZED_DECLARATION)
            val runFun = buildTrampoline(states)
            val depthField = buildDepthField(backendContext, irFile, "$name\$drfDepth", DRF_SYNTHESIZED_DECLARATION)
            return buildTwin(runFun, depthField)
        }

        private fun buildTrampoline(states: StateAssignment): IrSimpleFunction {
            val stateIds = states.stateIds
            val runFun = backendContext.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("${this@DrfCodegen.name}\$drfRun")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = rType
                origin = DRF_SYNTHESIZED_DECLARATION
            }
            val pCaptures = freeSyms.mapIndexed { i, fs -> runFun.addValueParameter("cap$i", fs.type) }
            val pArg = runFun.addValueParameter("value", tType)
            val b = backendContext.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val frameType = frame.outerField.type

            runFun.body = b.irBlockBody {
                val vState = irTemporary(irInt(stateIds.getValue(plan.entry)), nameHint = "s", isMutable = true)
                val vArg = irTemporary(irGet(pArg), nameHint = "arg", isMutable = true, irType = tType)
                val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true, irType = frameType)
                val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true, irType = frameType)
                // Reference-typed results start as null before the first Ret,
                // so the slot must be nullable; the final return casts back.
                val vRetType = if (framePoolOf(builtIns, rType) == builtIns.anyNType) rType.makeNullable() else rType
                val vRet = irTemporary(
                    IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, vRetType),
                    nameHint = "ret", isMutable = true, irType = vRetType,
                )

                val hoisted = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                for (local in plan.locals) {
                    hoisted[local.symbol] = irTemporary(
                        IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, local.type),
                        nameHint = "l_${local.name}", isMutable = true, irType = local.type,
                    )
                }

                val loop = b.irWhile().apply { condition = b.irTrue() }

                val remapBase = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                remapBase[lambdaValueParam.symbol] = vArg
                for (i in freeSyms.indices) remapBase[freeSyms[i].symbol] = pCaptures[i]
                for (local in plan.locals) remapBase[local.symbol] = hoisted.getValue(local.symbol)

                val remapper = object : AbstractVariableRemapper() {
                    override fun remapVariable(value: IrValueDeclaration): IrValueDeclaration? = remapBase[value.symbol]
                }

                fun remapAll(e: IrElement): IrElement = e.transform(remapper, null)

                fun resolveCapture(key: CaptureKey): IrValueDeclaration = when (key) {
                    is CaptureKey.Local -> hoisted.getValue(key.symbol)
                    is CaptureKey.Arg -> vArg
                    is CaptureKey.Self -> error("no self capture in DRF")
                }

                fun IrBlockBuilder.emitTransfer(call: IrCall) {
                    +irSet(vArg.symbol, remapAll(call.arguments[1]!!) as IrExpression)
                    +irSet(vState.symbol, irInt(stateIds.getValue(plan.entry)))
                }

                loop.body = b.irBlock {
                    +irWhen(builtIns.unitType, buildList {
                        for (block in plan.blocks) {
                            val sid = stateIds[block] ?: continue
                            val branchBody = irBlock {
                                states.resumeInfo[block]?.let { susp ->
                                    emitFrameRestore(backendContext, frame, captures, vFrame, ::resolveCapture)
                                    val target = hoisted[susp.resultSymbol]
                                        ?: vArg.takeIf { susp.resultSymbol == lambdaValueParam.symbol }
                                    if (target != null) {
                                        +irSet(target.symbol, irCastForSlot(builtIns, irGet(vRet), vRetType, target.type))
                                    }
                                }
                                emitBlockStatements(block, ::remapAll) { hoisted.getValue(it) }
                                emitTerminator(
                                    block.terminator!!, stateIds, vState, vRet, loop, ::remapAll,
                                    emitTailCall = { t ->
                                        emitTransfer(t.call)
                                        +irContinue(loop)
                                    },
                                    emitSuspendCall = { t ->
                                        emitFramePush(frame, captures, vTop, stateIds.getValue(t.resume), ::resolveCapture)
                                        emitTransfer(t.call)
                                        +irContinue(loop)
                                    },
                                )
                            }
                            add(irBranch(irIntEquals(backendContext, irGet(vState), irInt(sid)), branchBody))
                        }
                        val applyBlock = irBlock {
                            val retValue: IrExpression =
                                if (vRetType != rType) irAs(irGet(vRet), rType) else irGet(vRet)
                            emitFramePop(frame, vTop, vFrame, vState, loop, retValue)
                        }
                        add(irBranch(irTrue(), applyBlock))
                    })
                }
                +loop
            }
            runFun.body!!.patchDeclarationParents(runFun)
            return runFun
        }

        /** Native twin: the lambda body with callRecursive turned into hybrid self-calls. */
        private fun buildTwin(runFun: IrSimpleFunction, depthField: IrField): IrSimpleFunction {
            val twin = backendContext.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("${this@DrfCodegen.name}\$drfNative")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = rType
                origin = DRF_SYNTHESIZED_DECLARATION
            }
            val pCaptures = freeSyms.mapIndexed { i, fs -> twin.addValueParameter("cap$i", fs.type) }
            val pArg = twin.addValueParameter("value", tType)
            val bodyCopy = (lit.lambda.body as IrBlockBody).deepCopyWithSymbols(twin)
            inlineSuspendHelpers(bodyCopy, twin)
            val b = backendContext.createIrBuilder(twin.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val captureMap = freeSyms.indices.associate { freeSyms[it].symbol to pCaptures[it] }

            bodyCopy.transformChildrenVoid(object : AbstractVariableRemapper() {
                override fun remapVariable(value: IrValueDeclaration): IrValueDeclaration? =
                    if (value.symbol == lambdaValueParam.symbol) pArg else captureMap[value.symbol]

                override fun visitReturn(expression: IrReturn): IrExpression {
                    expression.transformChildrenVoid(this)
                    if (expression.returnTargetSymbol == lit.lambda.symbol) {
                        return IrReturnImpl(
                            expression.startOffset, expression.endOffset,
                            builtIns.nothingType, twin.symbol, expression.value,
                        )
                    }
                    return expression
                }

                override fun visitCall(expression: IrCall): IrExpression {
                    expression.transformChildrenVoid(this)
                    if (!isCallRecursive(expression)) return expression
                    val argExpr = expression.arguments[1]!!
                    return b.irBlock(resultType = rType) {
                        val a = irTemporary(argExpr, nameHint = "darg")
                        +irHybridDepthGuard(
                            backendContext, depthField, rType,
                            shallowCall = {
                                irCall(twin.symbol).apply {
                                    for (i in pCaptures.indices) arguments[i] = irGet(pCaptures[i])
                                    arguments[pCaptures.size] = irGet(a)
                                }
                            },
                            deepCall = {
                                irCall(runFun.symbol).apply {
                                    for (i in pCaptures.indices) arguments[i] = irGet(pCaptures[i])
                                    arguments[pCaptures.size] = irGet(a)
                                }
                            },
                        )
                    }
                }
            })
            twin.body = bodyCopy
            twin.body!!.patchDeclarationParents(twin)
            return twin
        }
    }

}
