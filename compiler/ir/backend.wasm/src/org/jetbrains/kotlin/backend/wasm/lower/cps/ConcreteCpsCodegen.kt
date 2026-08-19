/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.cps

import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.addFunction
import org.jetbrains.kotlin.ir.builders.declarations.addValueParameter
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.symbols.IrFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.util.defaultType
import org.jetbrains.kotlin.ir.util.file
import org.jetbrains.kotlin.ir.util.patchDeclarationParents
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.name.Name

/**
 * Code generator for defunctionalized CPS conversion of a strongly
 * connected component of concrete recursive functions.
 *
 * For a single function this produces the same structure as the former
 * SelfRecursionCodegen. For multiple functions forming an SCC in the
 * call graph, all bodies are compiled into a single flat state machine
 * with a shared frame class and depth counter.
 *
 * Generated structure:
 *   1. Frame class holding an outer-frame link, resume state, and
 *      typed slots for captured locals and parameters.
 *   2. Trampoline function containing the flat state machine:
 *      `while(true) + when(state)` with blocks from all functions.
 *   3. Module-level depth counter shared across the SCC for the
 *      hybrid threshold.
 *   4. Instrumented native body for each function: every call to
 *      an SCC member is wrapped with `if (depth < 512) nativeCall
 *      else trampoline`.
 */
internal class ConcreteCpsCodegen(
    private val context: WasmBackendContext,
    private val functions: List<IrSimpleFunction>,
    private val plans: List<BodyPlan>,
) {
    init {
        require(functions.size == plans.size && functions.isNotEmpty())
    }

    private val irFile = functions.first().file
    private val builtIns = context.irBuiltIns

    // ================================================================ per-function metadata

    private class FuncInfo(
        val func: IrSimpleFunction,
        val plan: BodyPlan,
        val params: List<IrValueParameter>,
        val dispatchReceiverIndex: Int?,
        val regularParamIndices: List<Int>,
        /** Offset of this function's params in the trampoline's flat vArgs list. */
        val paramOffset: Int,
    )

    private val funcInfos: List<FuncInfo>
    private val funcIndexBySymbol: Map<IrFunctionSymbol, Int>

    init {
        var offset = 0
        funcInfos = functions.mapIndexed { i, func ->
            val params = func.parameters
            val info = FuncInfo(
                func = func,
                plan = plans[i],
                params = params,
                dispatchReceiverIndex = params.indexOfFirst { it.kind == IrParameterKind.DispatchReceiver }.takeIf { it >= 0 },
                regularParamIndices = params.indices.filter { params[it].kind == IrParameterKind.Regular },
                paramOffset = offset,
            )
            offset += params.size
            info
        }
        funcIndexBySymbol = buildMap {
            functions.forEachIndexed { i, f -> put(f.symbol, i) }
        }
    }

    /**
     * Return type of the trampoline. When all SCC members share the
     * same return type the trampoline keeps it; otherwise it returns
     * `Any?` and each call site casts back.
     */
    private val returnType: IrType = functions.map { it.returnType }.distinct().let { types ->
        if (types.size == 1) types.single() else builtIns.anyNType
    }

    /** State id per basic block across all functions. */
    private val stateIds = mutableMapOf<BlockPlan, Int>()

    /** Resume block to the SuspendCall that targets it. */
    private val resumeInfo = mutableMapOf<BlockPlan, Terminator.SuspendCall>()

    /** Entry state for each function (indexed by position in [funcInfos]). */
    private val entryStates = mutableMapOf<Int, Int>()

    // ================================================================ invariant params

    /**
     * Per-function set of parameter indices (into FuncInfo.params) whose
     * value is passed through unchanged at every call to that function
     * across all plans. Cross-function calls naturally mark all target
     * params as varying since the caller's param symbols do not match
     * the callee's.
     */
    private val invariantParamIndices: List<Set<Int>> = funcInfos.map { fi ->
        val regParams = fi.regularParamIndices.map { fi.params[it] }
        val varying = mutableSetOf<Int>()
        for (plan in plans) {
            for (b in plan.blocks) {
                fun checkCall(call: IrCall) {
                    if (call.symbol != fi.func.symbol) return
                    for (i in regParams.indices) {
                        val argIdx = fi.regularParamIndices[i]
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
        }
        fi.regularParamIndices.toSet() - varying
    }

    // ================================================================ typed slot pools

    private fun poolOf(type: IrType): IrType = cpsPoolOf(type, context)

    private val funcCaptures: List<List<Capture>> = funcInfos.mapIndexed { funcIdx, fi ->
        val counters = mutableMapOf<IrType, Int>()
        fun cap(key: CaptureKey, declaredType: IrType): Capture {
            val pool = poolOf(declaredType)
            val idx = counters.getOrElse(pool) { 0 }
            counters[pool] = idx + 1
            return Capture(key, declaredType, pool, idx)
        }
        buildList {
            if (fi.dispatchReceiverIndex != null) {
                add(cap(CaptureKey.Self, fi.params[fi.dispatchReceiverIndex].type))
            }
            for (local in fi.plan.locals) {
                add(cap(CaptureKey.Local(local.symbol), local.type))
            }
            for (i in fi.regularParamIndices.indices) {
                if (fi.regularParamIndices[i] !in invariantParamIndices[funcIdx]) {
                    add(cap(CaptureKey.Arg(i), fi.params[fi.regularParamIndices[i]].type))
                }
            }
        }
    }

    private val poolOrder: List<IrType> = cpsPoolOrder(context)

    private val poolSizes: Map<IrType, Int> = poolOrder.associateWith { pool ->
        funcCaptures.maxOf { captures -> captures.count { it.pool == pool } }
    }

    private val funcCaptureBySlot: List<Map<Pair<IrType, Int>, Capture>> = funcCaptures.map { captures ->
        captures.associateBy { it.pool to it.poolIndex }
    }

    // ================================================================ entry point

    fun generate() {
        var next = 1
        for (funcIdx in funcInfos.indices) {
            val fi = funcInfos[funcIdx]
            for (block in fi.plan.blocks) {
                stateIds[block] = next++
                val t = block.terminator
                if (t is Terminator.SuspendCall) resumeInfo[t.resume] = t
            }
            entryStates[funcIdx] = stateIds[fi.plan.entry]!!
        }

        val frameClass = buildFrameClass()
        val runFun = buildTrampoline(frameClass)
        val depthField = buildDepthField()
        for (funcIdx in funcInfos.indices) {
            instrumentNativeBody(funcIdx, depthField, runFun)
        }
    }

    // ================================================================ frame class

    private lateinit var frameLayout: CpsFrameLayout

    private fun buildFrameClass(): IrClass {
        frameLayout = buildCpsFrameClass(
            context, irFile,
            "\$CpsFrame\$${sccBaseName}",
            poolOrder, poolSizes,
        )
        return frameLayout.frameClass
    }

    // ================================================================ trampoline

    private fun buildTrampoline(frameClass: IrClass): IrSimpleFunction {
        val runFun = context.irFactory.addFunction(irFile) {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier("cps\$run\$$sccBaseName")
            visibility = DescriptorVisibilities.PRIVATE
            returnType = this@ConcreteCpsCodegen.returnType
        }
        val pState = runFun.addValueParameter("state0", builtIns.intType)
        // Mirror ALL original parameters from ALL functions as trampoline value parameters.
        val pMirrors = funcInfos.flatMap { fi ->
            fi.params.mapIndexed { i, p ->
                runFun.addValueParameter("p${fi.paramOffset + i}", p.type)
            }
        }

        val b = context.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        val frameType = frameClass.defaultType.makeNullable()

        runFun.body = b.irBlockBody {
            val vState = irTemporary(irGet(pState), nameHint = "s", isMutable = true)
            val vArgs = pMirrors.mapIndexed { i, p ->
                irTemporary(irGet(p), nameHint = "arg$i", isMutable = true)
            }
            val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true)
            val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true)
            val vRet = irTemporary(
                this@ConcreteCpsCodegen.returnType.defaultValue(), nameHint = "ret", isMutable = true,
                irType = this@ConcreteCpsCodegen.returnType,
            )

            // Hoisted locals for all functions.
            val hoisted = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
            for (funcIdx in funcInfos.indices) {
                val fi = funcInfos[funcIdx]
                for (local in fi.plan.locals) {
                    hoisted[local.symbol] = irTemporary(
                        local.type.defaultValue(), nameHint = "l${funcIdx}_${local.name}", isMutable = true,
                        irType = local.type,
                    )
                }
            }

            val loop = b.irWhile().apply { condition = b.irTrue() }

            val emitter = BlockEmitter(
                b, loop, runFun,
                vState, vArgs, vTop, vFrame, vRet, hoisted,
            )

            val intEq = this@ConcreteCpsCodegen.context.wasmSymbols
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
                    for (funcIdx in funcInfos.indices) {
                        val fi = funcInfos[funcIdx]
                        for (block in fi.plan.blocks) {
                            val sid = stateIds[block] ?: continue
                            val branchBody = irBlock {
                                emitter.emitBlock(this, funcIdx, block)
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
                        +irSet(vTop.symbol, irGetField(irGet(fTmp), frameLayout.outerField))
                        +irSet(vFrame.symbol, irGet(fTmp))
                        +irSet(vState.symbol, irGetField(irGet(fTmp), frameLayout.resumeField))
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
        /** Per-function symbol remapping: original params/locals → trampoline mutable vars. */
        private val remappings: List<Map<IrValueSymbol, IrValueDeclaration>> = funcInfos.map { fi ->
            buildMap {
                for (i in fi.params.indices) {
                    put(fi.params[i].symbol, vArgs[fi.paramOffset + i])
                }
                for (local in fi.plan.locals) {
                    put(local.symbol, hoisted[local.symbol]!!)
                }
            }
        }

        private fun <T : IrElement> remap(element: T, map: Map<IrValueSymbol, IrValueDeclaration>): T =
            cpsRemap(element, map, builtIns)

        fun emitBlock(
            bb: IrBlockBuilder,
            funcIdx: Int,
            block: BlockPlan,
        ): Unit = with(bb) {
            val remapping = remappings[funcIdx]

            resumeInfo[block]?.let { susp ->
                emitRestores(bb, funcIdx)
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
                    val targetIdx = funcIndexBySymbol[t.call.symbol]!!
                    emitCallTransfer(bb, funcIdx, targetIdx, t.call)
                    +irContinue(loop)
                }

                is Terminator.SuspendCall -> {
                    val targetIdx = funcIndexBySymbol[t.call.symbol]!!
                    val resumeId = stateIds[t.resume]!!
                    val captures = funcCaptureBySlot[funcIdx]
                    +irSet(vTop.symbol, irCallConstructor(frameLayout.ctor, emptyList()).apply {
                        arguments[0] = irGet(vTop)
                        arguments[1] = irInt(resumeId)
                        for (ci in frameLayout.ctorParamOrder.indices) {
                            val slot = frameLayout.ctorParamOrder[ci]
                            val capture = captures[slot]
                            arguments[ci + 2] =
                                if (capture != null) irGet(resolveCapture(funcIdx, capture))
                                else cpsDefaultValue(slot.first)
                        }
                    })
                    emitCallTransfer(bb, funcIdx, targetIdx, t.call)
                    +irContinue(loop)
                }
            }
        }

        private fun resolveCapture(funcIdx: Int, capture: Capture): IrValueDeclaration {
            val fi = funcInfos[funcIdx]
            return when (val k = capture.key) {
                is CaptureKey.Self -> {
                    if (fi.dispatchReceiverIndex != null) vArgs[fi.paramOffset + fi.dispatchReceiverIndex]
                    else error("Self capture but no dispatch receiver")
                }
                is CaptureKey.Local -> hoisted[k.symbol]!!
                is CaptureKey.Arg -> vArgs[fi.paramOffset + fi.regularParamIndices[k.index]]
            }
        }

        private fun emitRestores(bb: IrBlockBuilder, funcIdx: Int): Unit = with(bb) {
            for (capture in funcCaptures[funcIdx]) {
                val target = resolveCapture(funcIdx, capture)
                val field = frameLayout.poolFields[capture.pool]!![capture.poolIndex]
                val read = irGetField(irGet(vFrame), field)
                val value = if (capture.pool == builtIns.anyNType) castTo(read, target.type) else read
                +irSet(target.symbol, value)
            }
        }

        private fun IrBlockBuilder.castTo(expr: IrExpression, type: IrType): IrExpression =
            cpsCast(expr, type, builtIns)

        /**
         * Transfer to a target function: evaluate its arguments, write
         * them into the target's parameter slots, and set the state to
         * the target's entry. For self-calls, invariant parameters are
         * skipped.
         */
        private fun emitCallTransfer(
            bb: IrBlockBuilder,
            sourceFuncIdx: Int,
            targetFuncIdx: Int,
            call: IrCall,
        ): Unit = with(bb) {
            val targetInfo = funcInfos[targetFuncIdx]
            val sourceRemapping = remappings[sourceFuncIdx]
            val isSelfCall = sourceFuncIdx == targetFuncIdx

            val argExprs = targetInfo.params.indices.map { i ->
                if (isSelfCall && i in invariantParamIndices[targetFuncIdx]) null
                else remap(call.arguments[i]!!, sourceRemapping)
            }
            val argTmps = argExprs.mapIndexed { i, expr ->
                expr?.let { irTemporary(it, nameHint = "narg$i") }
            }
            for (i in targetInfo.params.indices) {
                argTmps[i]?.let { tmp ->
                    +irSet(vArgs[targetInfo.paramOffset + i].symbol, irGet(tmp))
                }
            }
            +irSet(vState.symbol, irInt(entryStates[targetFuncIdx]!!))
        }
    }

    // ================================================================ hybrid depth threshold

    private fun buildDepthField(): IrField =
        buildCpsDepthField(context, irFile, "cps\$depth\$${sccBaseName}")

    /**
     * Rewrites every call to an SCC member in the ORIGINAL body of
     * function [funcIdx] with the hybrid threshold check: shallow path
     * uses native recursion, deep path routes through the trampoline.
     */
    private fun instrumentNativeBody(funcIdx: Int, depthField: IrField, runFun: IrSimpleFunction) {
        val fi = funcInfos[funcIdx]
        val b = context.createIrBuilder(fi.func.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
        fi.func.body?.transform(object : IrTransformer<Nothing?>() {
            override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                expression.transformChildren(this, data)
                val targetFuncIdx = funcIndexBySymbol[expression.symbol] ?: return expression
                return b.irBlock(resultType = expression.type) {
                    // Hoist arguments into temporaries.
                    val argTmps = expression.arguments.indices.map { i ->
                        irTemporary(expression.arguments[i]!!, nameHint = "harg$i")
                    }
                    fun depthGet() = irGetField(null, depthField)
                    +irIfThenElse(
                        expression.type,
                        irCall(builtIns.lessFunByOperandType[builtIns.intClass]!!).apply {
                            arguments[0] = depthGet()
                            arguments[1] = irInt(HYBRID_DEPTH_THRESHOLD)
                        },
                        // Shallow: native call with depth tracking.
                        irBlock(resultType = expression.type) {
                            +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(1)
                            })
                            val r = irTemporary(irCall(expression.symbol).apply {
                                for (i in argTmps.indices) arguments[i] = irGet(argTmps[i])
                            }, nameHint = "hres")
                            +irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
                                arguments[0] = depthGet()
                                arguments[1] = irInt(-1)
                            })
                            +irGet(r)
                        },
                        // Deep: route through trampoline.
                        irBlock(resultType = expression.type) {
                            val trampolineCall = irCall(runFun.symbol).apply {
                                arguments[0] = irInt(entryStates[targetFuncIdx]!!)
                                for (fIdx in funcInfos.indices) {
                                    val fInfo = funcInfos[fIdx]
                                    for (j in fInfo.params.indices) {
                                        arguments[1 + fInfo.paramOffset + j] =
                                            if (fIdx == targetFuncIdx) irGet(argTmps[j])
                                            else fInfo.params[j].type.defaultValue()
                                    }
                                }
                            }
                            if (returnType != expression.type) {
                                +cpsCast(trampolineCall, expression.type, builtIns)
                            } else {
                                +trampolineCall
                            }
                        },
                    )
                }
            }
        }, null)
        fi.func.body?.patchDeclarationParents(fi.func)
    }

    // ================================================================ utils

    private val sccBaseName: String =
        if (functions.size == 1) functions.first().name.asString()
        else functions.joinToString("_") { it.name.asString() }

    private fun IrType.defaultValue(): IrExpression = cpsDefaultValue(this)
}
