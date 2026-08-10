/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.virtualcps

import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.*
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.util.*
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.name.Name

internal class WasmDrfAcceleration(private val context: WasmBackendContext) {

    companion object {
        private const val HYBRID_DEPTH_THRESHOLD = 512
    }

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
    fun lower(irModule: IrModuleFragment) {
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

            if (sites > 0) {
                transformDrfLiteral(irModule, lit)
            }
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

        val planner = BodyPlanner(lambda, lambda, planCopy, isTarget, context.irBuiltIns)
        val plan = planner.plan()
            ?: return "planner: ${planner.lastBailReason}"

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
                val intEq = this@WasmDrfAcceleration.context.wasmSymbols.equalityFunctions[builtIns.intType]
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
}
