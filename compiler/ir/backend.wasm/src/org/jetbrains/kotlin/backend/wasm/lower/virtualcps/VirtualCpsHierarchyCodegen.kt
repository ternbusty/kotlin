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
 * Native recursion budget before switching a subtree to the heap-frame
 * trampoline. Far below the wasm stack guard (~10-15K frames) even
 * with several frames per level and an already-deep caller stack, and
 * far above what everyday patterns reach.
 */
private const val HYBRID_DEPTH_THRESHOLD = 512

internal class VirtualCpsHierarchyCodegen(
    private val context: WasmBackendContext,
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

            val intEq = this@VirtualCpsHierarchyCodegen.context.wasmSymbols
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
