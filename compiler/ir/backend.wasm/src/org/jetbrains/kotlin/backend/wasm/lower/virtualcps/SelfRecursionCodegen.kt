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
import org.jetbrains.kotlin.name.Name

/**
 * Native recursion budget before switching to the heap-frame trampoline.
 * Shared with [VirtualCpsHierarchyCodegen].
 */
private const val HYBRID_DEPTH_THRESHOLD = 512

/**
 * Code generator for defunctionalized CPS conversion of a single
 * concrete self-recursive function.
 *
 * This is the single-function analogue of [VirtualCpsHierarchyCodegen].
 * Where the virtual variant handles a closed class hierarchy with
 * multiple overrides, instanceof dispatch, and a shared state-id
 * function, this class handles one function calling itself. The
 * generated artifacts are simpler.
 *
 * Generated structure:
 *   1. Frame class holding an outer-frame link, resume state, and
 *      typed slots for captured locals and parameters.
 *   2. Trampoline function `<name>$selfCps` containing the flat
 *      state machine: `while(true) + when(state)`.
 *   3. Module-level depth counter for the hybrid threshold.
 *   4. Instrumented native body: each self-call is wrapped with
 *      `if (depth < 512) nativeCall else trampoline`.
 */
internal class SelfRecursionCodegen(
    private val context: WasmBackendContext,
    private val func: IrSimpleFunction,
    private val plan: BodyPlan,
) {
    private val irFile = func.file
    private val builtIns = context.irBuiltIns

    /** All parameters of the original function in declaration order. */
    private val allParams = func.parameters

    /** Indices into [allParams] for dispatch-receiver and regular params. */
    private val dispatchReceiverIndex: Int? =
        allParams.indexOfFirst { it.kind == IrParameterKind.DispatchReceiver }.takeIf { it >= 0 }
    private val regularParamIndices: List<Int> =
        allParams.indices.filter { allParams[it].kind == IrParameterKind.Regular }

    /** State id per basic block. */
    private val stateIds = mutableMapOf<BlockPlan, Int>()

    /** Resume block to the SuspendCall that targets it. */
    private val resumeInfo = mutableMapOf<BlockPlan, Terminator.SuspendCall>()

    /**
     * Regular parameters whose value is passed through unchanged at every
     * recursive call site. Such parameters never change during one
     * trampoline activation, so they need not be captured in frames.
     */
    private val invariantParamIndices: Set<Int> = run {
        val regParams = regularParamIndices.map { allParams[it] }
        val varying = mutableSetOf<Int>()
        for (b in plan.blocks) {
            fun checkCall(call: IrCall) {
                for (i in regParams.indices) {
                    val argIdx = regularParamIndices[i]
                    val a = call.arguments[argIdx]
                    if (!(a is IrGetValue && a.symbol == regParams[i].symbol)) varying += argIdx
                }
            }
            when (val t = b.terminator) {
                is Terminator.TailCall -> checkCall(t.call)
                is Terminator.SuspendCall -> checkCall(t.call)
                else -> {}
            }
        }
        regularParamIndices.toSet() - varying
    }

    // ================================================================ typed slot pools

    private fun poolOf(type: IrType): IrType = when (type) {
        builtIns.intType -> builtIns.intType
        builtIns.booleanType -> builtIns.booleanType
        builtIns.charType -> builtIns.charType
        builtIns.longType -> builtIns.longType
        builtIns.floatType -> builtIns.floatType
        builtIns.doubleType -> builtIns.doubleType
        else -> builtIns.anyNType
    }

    private val captures: List<Capture> = run {
        val counters = mutableMapOf<IrType, Int>()
        fun cap(key: CaptureKey, declaredType: IrType): Capture {
            val pool = poolOf(declaredType)
            val idx = counters.getOrElse(pool) { 0 }
            counters[pool] = idx + 1
            return Capture(key, declaredType, pool, idx)
        }
        buildList {
            // Dispatch receiver, if any
            if (dispatchReceiverIndex != null) {
                val recvParam = allParams[dispatchReceiverIndex]
                add(cap(CaptureKey.Self, recvParam.type))
            }
            // Hoisted locals
            for (local in plan.locals) {
                add(cap(CaptureKey.Local(local.symbol), local.type))
            }
            // Varying regular parameters
            for (i in regularParamIndices.indices) {
                if (regularParamIndices[i] !in invariantParamIndices) {
                    add(cap(CaptureKey.Arg(i), allParams[regularParamIndices[i]].type))
                }
            }
        }
    }

    private val poolOrder: List<IrType> = listOf(
        builtIns.intType, builtIns.booleanType, builtIns.charType,
        builtIns.longType, builtIns.floatType, builtIns.doubleType, builtIns.anyNType,
    )

    private val poolSizes: Map<IrType, Int> = poolOrder.associateWith { pool ->
        captures.count { it.pool == pool }
    }

    private val captureBySlot: Map<Pair<IrType, Int>, Capture> =
        captures.associateBy { it.pool to it.poolIndex }

    // ================================================================ entry point

    fun generate() {
        simplifyPlan(plan)

        var next = 1
        for (block in reachableBlocks(plan)) {
            stateIds[block] = next++
            val t = block.terminator
            if (t is Terminator.SuspendCall) resumeInfo[t.resume] = t
        }

        val frameClass = buildFrameClass()
        val runFun = buildTrampoline(frameClass)
        val depthField = buildDepthField()
        instrumentNativeBody(depthField, runFun)
    }

    // ================================================================ frame class

    private lateinit var frameOuterField: IrField
    private lateinit var frameResumeField: IrField
    private lateinit var framePoolFields: Map<IrType, List<IrField>>
    private lateinit var frameCtorParamOrder: List<Pair<IrType, Int>>
    private lateinit var frameCtor: IrConstructorSymbol

    private fun buildFrameClass(): IrClass {
        val baseName = func.name.asString()
        val cls = context.irFactory.buildClass {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier("\$SelfCpsFrame\$$baseName")
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

    // ================================================================ trampoline

    private fun buildTrampoline(frameClass: IrClass): IrSimpleFunction {
        val runFun = context.irFactory.addFunction(irFile) {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier("selfCps\$run\$${func.name}")
            visibility = DescriptorVisibilities.PRIVATE
            returnType = func.returnType
        }
        val pState = runFun.addValueParameter("state0", builtIns.intType)
        // Mirror ALL original parameters as value parameters of the trampoline.
        val pMirror = allParams.mapIndexed { i, p ->
            runFun.addValueParameter("p$i", p.type)
        }

        val b = context.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        val frameType = frameClass.defaultType.makeNullable()

        runFun.body = b.irBlockBody {
            val vState = irTemporary(irGet(pState), nameHint = "s", isMutable = true)
            val vArgs = pMirror.mapIndexed { i, p ->
                irTemporary(irGet(p), nameHint = "arg$i", isMutable = true)
            }
            val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true)
            val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true)
            val vRet = irTemporary(
                func.returnType.defaultValue(), nameHint = "ret", isMutable = true,
                irType = func.returnType,
            )

            // Hoisted locals
            val hoisted = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
            for (local in plan.locals) {
                hoisted[local.symbol] = irTemporary(
                    local.type.defaultValue(), nameHint = "l_${local.name}", isMutable = true,
                    irType = local.type,
                )
            }

            val loop = b.irWhile().apply { condition = b.irTrue() }

            val emitter = BlockEmitter(
                b, loop, runFun,
                vState, vArgs, vTop, vFrame, vRet, hoisted,
            )

            val intEq = this@SelfRecursionCodegen.context.wasmSymbols
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
                    for (block in plan.blocks) {
                        val sid = stateIds[block] ?: continue
                        val branchBody = irBlock {
                            emitter.emitBlock(this, block, stateIds, resumeInfo)
                        }
                        add(irBranch(stateEquals(sid), branchBody))
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

    // ================================================================ per-block emission

    private inner class BlockEmitter(
        private val b: IrBuilderWithScope,
        private val loop: IrLoop,
        private val runFun: IrSimpleFunction,
        private val vState: IrValueDeclaration,
        private val vArgs: List<IrValueDeclaration>,
        private val vTop: IrValueDeclaration,
        private val vFrame: IrValueDeclaration,
        private val vRet: IrValueDeclaration,
        private val hoisted: Map<IrValueSymbol, IrValueDeclaration>,
    ) {
        /**
         * Maps original function parameter symbols and local variable
         * symbols to the trampoline's mutable arg/local variables.
         */
        private val remapping: Map<IrValueSymbol, IrValueDeclaration> = buildMap {
            for (i in allParams.indices) {
                put(allParams[i].symbol, vArgs[i])
            }
            for (local in plan.locals) {
                put(local.symbol, hoisted[local.symbol]!!)
            }
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
            block: BlockPlan,
            stateIds: Map<BlockPlan, Int>,
            resumeInfo: Map<BlockPlan, Terminator.SuspendCall>,
        ): Unit = with(bb) {
            resumeInfo[block]?.let { susp ->
                emitRestores(bb)
                val resultTarget = remapping[susp.resultSymbol] ?: hoisted[susp.resultSymbol]!!
                +irSet(resultTarget.symbol, castTo(irGet(vRet), resultTarget.type))
            }

            for (stmt in block.statements) {
                when (stmt) {
                    is IrVariable -> {
                        val target = hoisted[stmt.symbol]!!
                        val init = stmt.initializer
                        if (init != null) {
                            +irSet(target.symbol, remap(init, remapping))
                        }
                    }
                    else -> {
                        +remap(stmt, remapping)
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
                        remap(t.condition, remapping),
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
                    +irSet(vRet.symbol, remap(t.value, remapping))
                    +irSet(vState.symbol, irInt(0))
                    +irContinue(loop)
                }

                is Terminator.TailCall -> {
                    emitCallTransfer(bb, t.call)
                    +irContinue(loop)
                }

                is Terminator.SuspendCall -> {
                    val resumeId = stateIds[t.resume]!!
                    +irSet(vTop.symbol, irCallConstructor(frameCtor, emptyList()).apply {
                        arguments[0] = irGet(vTop)
                        arguments[1] = irInt(resumeId)
                        for (ci in frameCtorParamOrder.indices) {
                            val slot = frameCtorParamOrder[ci]
                            val capture = captureBySlot[slot]
                            arguments[ci + 2] =
                                if (capture != null) irGet(resolveCapture(capture))
                                else slot.first.defaultValue()
                        }
                    })
                    emitCallTransfer(bb, t.call)
                    +irContinue(loop)
                }
            }
        }

        private fun resolveCapture(capture: Capture): IrValueDeclaration = when (val k = capture.key) {
            is CaptureKey.Self -> {
                if (dispatchReceiverIndex != null) vArgs[dispatchReceiverIndex]
                else error("Self capture but no dispatch receiver")
            }
            is CaptureKey.Local -> hoisted[k.symbol]!!
            is CaptureKey.Arg -> vArgs[regularParamIndices[k.index]]
        }

        private fun emitRestores(bb: IrBlockBuilder): Unit = with(bb) {
            for (capture in captures) {
                val target = resolveCapture(capture)
                val field = framePoolFields[capture.pool]!![capture.poolIndex]
                val read = irGetField(irGet(vFrame), field)
                val value = if (capture.pool == builtIns.anyNType) castTo(read, target.type) else read
                +irSet(target.symbol, value)
            }
        }

        private fun IrBlockBuilder.castTo(expr: IrExpression, type: IrType): IrExpression = when {
            type == builtIns.anyNType -> expr
            !type.isPrimitiveType() -> irAs(expr, type.makeNullable())
            else -> irAs(expr, type)
        }

        /**
         * Transfer at a recursive call: evaluate new arguments, swap them
         * into the shared vars, and re-enter the state machine at state 1.
         */
        private fun emitCallTransfer(
            bb: IrBlockBuilder,
            call: IrCall,
        ): Unit = with(bb) {
            // Evaluate all arguments before overwriting shared vars.
            val argExprs = allParams.indices.map { i ->
                if (isInvariantParam(i)) null
                else remap(call.arguments[i]!!, remapping)
            }
            val argTmps = argExprs.mapIndexed { i, expr ->
                expr?.let { irTemporary(it, nameHint = "narg$i") }
            }
            for (i in allParams.indices) {
                argTmps[i]?.let { tmp ->
                    +irSet(vArgs[i].symbol, irGet(tmp))
                }
            }
            // Always re-enter at the plan entry.
            +irSet(vState.symbol, irInt(stateIds[plan.entry]!!))
        }

        /** Whether parameter at index [i] is invariant and needs no transfer. */
        private fun isInvariantParam(i: Int): Boolean = i in invariantParamIndices
    }

    // ================================================================ hybrid depth threshold

    private fun buildDepthField(): IrField {
        return context.irFactory.buildField {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier("selfCps\$depth\$${func.name}")
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

    /**
     * Rewrites every self-call in the ORIGINAL body with the hybrid
     * threshold check: shallow path uses native recursion, deep path
     * routes through the trampoline.
     */
    private fun instrumentNativeBody(depthField: IrField, runFun: IrSimpleFunction) {
        val entryState = stateIds[plan.entry]!!
        val b = context.createIrBuilder(func.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        func.body?.transform(object : IrTransformer<Nothing?>() {
            override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                expression.transformChildren(this, data)
                if (expression.symbol != func.symbol) return expression
                return b.irBlock(resultType = func.returnType) {
                    // Hoist arguments into temporaries.
                    val argTmps = allParams.indices.map { i ->
                        irTemporary(expression.arguments[i]!!, nameHint = "harg$i")
                    }
                    fun depthGet() = irGetField(null, depthField)
                    +irIfThenElse(
                        func.returnType,
                        irCall(builtIns.lessFunByOperandType[builtIns.intClass]!!).apply {
                            arguments[0] = depthGet()
                            arguments[1] = irInt(HYBRID_DEPTH_THRESHOLD)
                        },
                        irBlock(resultType = func.returnType) {
                            +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(1)
                            })
                            val r = irTemporary(irCall(func.symbol).apply {
                                for (i in argTmps.indices) arguments[i] = irGet(argTmps[i])
                            }, nameHint = "hres")
                            +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(-1)
                            })
                            +irGet(r)
                        },
                        irCall(runFun.symbol).apply {
                            arguments[0] = irInt(entryState)
                            for (i in argTmps.indices) arguments[i + 1] = irGet(argTmps[i])
                        },
                    )
                }
            }
        }, null)
        func.body?.patchDeclarationParents(func)
    }

    // ================================================================ utils

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
}
