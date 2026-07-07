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
import org.jetbrains.kotlin.wasm.config.wasmEnableStacklessRecursion
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
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
internal val STACKLESS_SYNTHESIZED_DECLARATION by IrDeclarationOriginImpl.Regular

/** `stackless$stateOf` result for receivers whose class has no plan: run the native method instead. */
private const val STATE_UNPLANNED = -1

/**
 * Compiles virtual method hierarchies marked with
 * `@kotlin.internal.StacklessRecursion` into a single stackless trampoline
 * with typed heap frames, so deep virtual mutual recursion through the
 * hierarchy (the stdlib regex matcher, kotlin.text.regex.AbstractSet.matches)
 * no longer consumes the host stack. See KT-63689, KT-78089, KT-61542.
 *
 * CHA enumerates the overrides (sound under whole-world Wasm compilation),
 * [WasmStacklessBodyPlanner] splits each body at recursive call sites, and
 * dispatch is an instanceof chain ordered by inheritance depth. Overrides
 * the planner cannot express bail out to the untouched native method, so
 * partial coverage never changes semantics. Native bodies below
 * [STACKLESS_HYBRID_DEPTH_THRESHOLD] run unchanged; deeper subtrees delegate
 * to the trampoline once and return by value.
 */
internal class WasmStacklessMatcherLowering(private val context: WasmBackendContext) : ModuleLoweringPass {

    companion object {
        private val STACKLESS_ANNOTATION = FqName("kotlin.internal.StacklessRecursion")
    }

    private val enabled = context.configuration.wasmEnableStacklessRecursion

    override fun lower(irModule: IrModuleFragment) {
        if (!enabled) return
        val allClasses = collectClasses(irModule)
        for (base in allClasses) {
            val annotated = base.declarations
                .filterIsInstance<IrSimpleFunction>()
                .filter { it.hasAnnotation(STACKLESS_ANNOTATION) }
            if (annotated.isEmpty()) continue
            val baseMethod = annotated.firstOrNull { it.body == null }
            if (baseMethod == null) {
                // The hierarchy is derived from an abstract base declaration;
                // silence here would hide the optimization evaporating.
                @OptIn(MessageCollectorAccess::class)
                context.configuration.messageCollector.report(
                    CompilerMessageSeverity.WARNING,
                    "@StacklessRecursion on ${base.name} has no effect: the annotated method must be abstract",
                )
                continue
            }
            lowerHierarchy(base, baseMethod, allClasses)
        }
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
                    fn.name == baseMethod.name && fn.body != null && !fn.isFakeOverride && fn.overrides(baseMethod)
                }
                ?: continue
            result += OverrideInfo(cls, override)
        }
        return result
    }

    // ================================================================ block planning

    private fun lowerHierarchy(
        base: IrClass,
        baseMethod: IrSimpleFunction,
        allClasses: List<IrClass>,
    ) {
        // The lowering pipeline may process a module more than once; the
        // origin-marked dispatch function marks a hierarchy as already handled.
        val stateOfName = "stackless\$stateOf\$${base.name}"
        if (base.file.declarations.any {
                it is IrSimpleFunction && it.origin == STACKLESS_SYNTHESIZED_DECLARATION && it.name.asString() == stateOfName
            }
        ) {
            return
        }

        val overrides = collectOverrides(base, baseMethod, allClasses)
        if (overrides.isEmpty()) return

        // Is a call a virtual dispatch of the target base method? Memoized
        // per callee: the planner evaluates this on every name-matching call
        // once per enclosing nesting level.
        val targetCache = mutableMapOf<IrSimpleFunction, Boolean>()
        val isTarget: (IrCall) -> Boolean = { call ->
            val callee = call.symbol.owner
            callee.name == baseMethod.name && targetCache.getOrPut(callee) { callee.overrides(baseMethod) }
        }

        val plans = mutableListOf<BodyPlan>()
        val bailed = LinkedHashMap<OverrideInfo, String>()
        for (info in overrides) {
            // Plan on a deep copy: the native body stays intact for the
            // shallow path of the hybrid scheme.
            val original = info.function.body as? IrBlockBody
            if (original == null) {
                bailed[info] = "no block body"
                continue
            }
            val copy = original.deepCopyWithSymbols(info.function)
            when (val r = WasmStacklessBodyPlanner(context, info.function, copy, isTarget).plan()) {
                is PlanResult.Planned -> plans += r.plan
                is PlanResult.Bailed -> bailed[info] = r.reason
            }
        }
        if (plans.isEmpty()) return

        context.irFactory.stageController.restrictTo(plans.first().info.function) {
            HierarchyCodegen(base, baseMethod, plans, bailed.keys.toList(), isTarget).generate()
        }
        reportPlanSummary(base, plans, bailed)
    }

    // ================================================================ codegen

    /**
     * State numbering: [STATE_APPLY] pops a frame or returns,
     * [STATE_UNPLANNED] routes to the native method, block states start
     * at 1. Planned overrides keep their native bodies for the shallow
     * path; [instrumentNativeBody] reroutes deep call sites through the
     * trampoline.
     */
    private inner class HierarchyCodegen(
        private val base: IrClass,
        private val baseMethod: IrSimpleFunction,
        private val plans: List<BodyPlan>,
        private val bailedOut: List<OverrideInfo>,
        private val isTarget: (IrCall) -> Boolean,
    ) {
        private val backendContext = this@WasmStacklessMatcherLowering.context
        private val irFile = base.file
        private val builtIns = backendContext.irBuiltIns

        private val valueParams = baseMethod.parameters.filter { it.kind == IrParameterKind.Regular }
        private val valueParamCount = valueParams.size

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

        /** The complement: parameter indices that travel through arg vars and frames. */
        private val varyingParams: List<Int> = (0 until valueParamCount).filter { !invariantParams[it] }

        private val planCaptures: Map<BodyPlan, List<Capture>> = plans.associateWith { plan ->
            val params = plan.info.function.parameters.filter { it.kind == IrParameterKind.Regular }
            buildCaptures(builtIns, buildList {
                add(CaptureKey.Self to plan.info.irClass!!.defaultType)
                for (local in plan.locals) add(CaptureKey.Local(local.symbol) to local.type)
                for (i in varyingParams) add(CaptureKey.Arg(i) to params[i].type)
            })
        }

        private val poolSizes: Map<IrType, Int> = framePoolOrder(builtIns).associateWith { pool ->
            plans.maxOf { plan -> planCaptures.getValue(plan).count { it.pool == pool } }
        }

        private val states = assignStates(plans)
        private val stateIds: Map<BlockPlan, Int> get() = states.stateIds
        private val resumeInfo: Map<BlockPlan, Terminator.SuspendCall> get() = states.resumeInfo

        private val frame = buildFrameClass(
            backendContext, irFile, "\$StacklessFrame\$${base.name}", poolSizes, STACKLESS_SYNTHESIZED_DECLARATION,
        )

        fun generate() {
            val stateIdFun = buildStateOfFun()
            val runFun = buildTrampoline(stateIdFun)
            val depthField = buildDepthField(
                backendContext, irFile, "stackless\$depth\$${base.name}", STACKLESS_SYNTHESIZED_DECLARATION,
            )
            val enterFun = buildEnterFun(stateIdFun, runFun)
            for (info in plans.map { it.info } + bailedOut) instrumentNativeBody(info, depthField, enterFun)
        }

        // ---------------- receiver-to-state dispatch

        private fun buildStateOfFun(): IrSimpleFunction {
            // instanceof chain, most-derived classes first. A bailed-out
            // override class below a planned ancestor maps to STATE_UNPLANNED
            // so it is never routed to that ancestor's plan; bailed classes
            // with no planned ancestor just fall through to the default.
            fun depthOf(cls: IrClass): Int {
                var d = 0
                var cur: IrClass? = cls
                while (cur != null && cur != base) {
                    cur = cur.superClass
                    d++
                }
                return d
            }

            val entries = buildList {
                for (plan in plans) {
                    val cls = plan.info.irClass!!
                    add(Triple(cls, stateIds.getValue(plan.entry), depthOf(cls)))
                }
                for (info in bailedOut) {
                    val cls = info.irClass!!
                    val hasPlannedAncestor = plans.any { it.info.irClass != cls && cls.isSubclassOf(it.info.irClass!!) }
                    if (hasPlannedAncestor) add(Triple(cls, STATE_UNPLANNED, depthOf(cls)))
                }
            }.sortedByDescending { it.third }

            val fn = backendContext.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("stackless\$stateOf\$${base.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = builtIns.intType
                origin = STACKLESS_SYNTHESIZED_DECLARATION
            }
            val pRecv = fn.addValueParameter("recv", base.defaultType)
            val b = backendContext.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body = b.irBlockBody {
                +irReturn(irWhen(builtIns.intType, buildList {
                    for (e in entries) {
                        add(irBranch(irIs(irGet(pRecv), e.first.defaultType), irInt(e.second)))
                    }
                    add(irBranch(irTrue(), irInt(STATE_UNPLANNED)))
                }))
            }
            return fn
        }

        // ---------------- trampoline

        private fun buildTrampoline(stateIdFun: IrSimpleFunction): IrSimpleFunction {
            val runFun = backendContext.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("run\$stackless\$${baseMethod.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = baseMethod.returnType
                origin = STACKLESS_SYNTHESIZED_DECLARATION
            }
            val pState = runFun.addValueParameter("state0", builtIns.intType)
            val pRecv = runFun.addValueParameter("recv0", base.defaultType)
            val pArgs = valueParams.mapIndexed { i, p -> runFun.addValueParameter("a$i", p.type) }

            val b = backendContext.createIrBuilder(runFun.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            val frameType = frame.cls.defaultType.makeNullable()

            runFun.body = b.irBlockBody {
                val vState = irTemporary(irGet(pState), nameHint = "s", isMutable = true)
                val vRecv = irTemporary(irGet(pRecv), nameHint = "recv", isMutable = true)
                val vArgs = pArgs.mapIndexed { i, p ->
                    irTemporary(irGet(p), nameHint = "arg$i", isMutable = true)
                }
                val vTop = irTemporary(irNull(frameType), nameHint = "top", isMutable = true, irType = frameType)
                val vFrame = irTemporary(irNull(frameType), nameHint = "frame", isMutable = true, irType = frameType)
                val vRet = irTemporary(
                    IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, baseMethod.returnType),
                    nameHint = "ret", isMutable = true,
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
                            IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, local.type),
                            nameHint = "l${pi}_${local.name}", isMutable = true,
                            irType = local.type,
                        )
                    }
                }

                val loop = b.irWhile().apply { condition = b.irTrue() }

                val emitter = BlockEmitter(loop, stateIdFun, vState, vRecv, vArgs, vTop, vFrame, vRet, hoisted, selfVars)

                loop.body = b.irBlock {
                    +irWhen(builtIns.unitType, buildList {
                        for (plan in plans) {
                            for (block in plan.blocks) {
                                val sid = stateIds[block] ?: continue
                                val branchBody = irBlock {
                                    emitter.emitBlock(this, plan, block)
                                }
                                add(irBranch(irIntEquals(backendContext, irGet(vState), irInt(sid)), branchBody))
                            }
                        }
                        // else: APPLY state — pop frame or return.
                        val applyBlock = irBlock {
                            emitFramePop(frame, vTop, vFrame, vState, loop, irGet(vRet))
                        }
                        add(irBranch(irTrue(), applyBlock))
                    })
                }
                +loop
            }
            runFun.body!!.patchDeclarationParents(runFun)
            return runFun
        }

        // ---------------- per-block emission

        private inner class BlockEmitter(
            private val loop: IrLoop,
            private val stateIdFun: IrSimpleFunction,
            private val vState: IrValueDeclaration,
            private val vRecv: IrValueDeclaration,
            private val vArgs: List<IrValueDeclaration>,
            private val vTop: IrValueDeclaration,
            private val vFrame: IrValueDeclaration,
            private val vRet: IrValueDeclaration,
            private val hoisted: Map<IrValueSymbol, IrValueDeclaration>,
            private val selfVars: Map<BodyPlan, IrValueDeclaration>,
        ) {
            private val valueMaps = mutableMapOf<BodyPlan, Map<IrValueSymbol, IrValueDeclaration>>()
            private val remappers = mutableMapOf<BodyPlan, AbstractVariableRemapper>()

            /** value-symbol remapping for statements of [plan]. */
            private fun valueMapFor(plan: BodyPlan): Map<IrValueSymbol, IrValueDeclaration> = valueMaps.getOrPut(plan) {
                val map = mutableMapOf<IrValueSymbol, IrValueDeclaration>()
                val fn = plan.info.function
                var argIdx = 0
                for (p in fn.parameters) {
                    when (p.kind) {
                        IrParameterKind.DispatchReceiver -> map[p.symbol] = selfVars.getValue(plan)
                        IrParameterKind.Regular -> map[p.symbol] = vArgs[argIdx++]
                        else -> {}
                    }
                }
                for (local in plan.locals) map[local.symbol] = hoisted.getValue(local.symbol)
                map
            }

            private fun remapperFor(plan: BodyPlan): AbstractVariableRemapper = remappers.getOrPut(plan) {
                val map = valueMapFor(plan)
                object : AbstractVariableRemapper() {
                    override fun remapVariable(value: IrValueDeclaration): IrValueDeclaration? = map[value.symbol]
                }
            }

            @Suppress("UNCHECKED_CAST")
            private fun <T : IrElement> remap(element: T, plan: BodyPlan): T =
                element.transform(remapperFor(plan), null) as T

            private fun resolveKey(plan: BodyPlan, key: CaptureKey): IrValueDeclaration = when (key) {
                is CaptureKey.Self -> selfVars.getValue(plan)
                is CaptureKey.Local -> hoisted.getValue(key.symbol)
                is CaptureKey.Arg -> vArgs[key.index]
            }

            fun emitBlock(
                bb: IrBlockBuilder,
                plan: BodyPlan,
                block: BlockPlan,
            ): Unit = with(bb) {
                val self = selfVars.getValue(plan)

                if (block === plan.entry) {
                    // self = recv as C
                    +irSet(self.symbol, irAs(irGet(vRecv), plan.info.irClass!!.defaultType))
                }

                resumeInfo[block]?.let { susp ->
                    // Restore captured state from the popped frame, then bind result.
                    emitFrameRestore(backendContext, frame, planCaptures.getValue(plan), vFrame) { key ->
                        resolveKey(plan, key)
                    }
                    val resultTarget = valueMapFor(plan)[susp.resultSymbol] ?: hoisted.getValue(susp.resultSymbol)
                    +irSet(resultTarget.symbol, irCastForSlot(builtIns, irGet(vRet), baseMethod.returnType, resultTarget.type))
                }

                emitBlockStatements(block, { e -> remap(e, plan) }) { hoisted.getValue(it) }

                emitTerminator(
                    block.terminator!!, stateIds, vState, vRet, loop, { e -> remap(e, plan) },
                    emitTailCall = { t ->
                        emitCallTransfer(this, plan, t.call)
                        +irContinue(loop)
                    },
                    emitSuspendCall = { t ->
                        // Push frame capturing self + hoisted locals + varying args
                        // into typed slots (no boxing).
                        emitFramePush(frame, planCaptures.getValue(plan), vTop, stateIds.getValue(t.resume)) { key ->
                            resolveKey(plan, key)
                        }
                        emitCallTransfer(this, plan, t.call)
                        +irContinue(loop)
                    },
                )
            }

            /**
             * Common transfer for tail and suspend calls: load args, swap
             * receiver, dispatch via stackless$stateOf; unplanned receiver
             * falls back to a native virtual call whose result goes through
             * APPLY.
             */
            private fun emitCallTransfer(
                bb: IrBlockBuilder,
                plan: BodyPlan,
                call: IrCall,
            ): Unit = with(bb) {
                val recvArg = call.arguments[0]!!
                // Evaluate args before overwriting the shared arg vars.
                val recvTmp = irTemporary(remap(recvArg, plan), nameHint = "nrecv")
                val argTmps = varyingParams.associateWith { i ->
                    irTemporary(remap(call.arguments[i + 1]!!, plan), nameHint = "narg$i")
                }
                for (i in varyingParams) {
                    +irSet(vArgs[i].symbol, irGet(argTmps.getValue(i)))
                }
                +irSet(vRecv.symbol, irGet(recvTmp))
                val selfParam = plan.info.function.parameters.firstOrNull { it.kind == IrParameterKind.DispatchReceiver }
                if (recvArg is IrGetValue && recvArg.symbol == selfParam?.symbol) {
                    // Self-recursive transfer: the receiver already routed to
                    // this plan and stateOf is deterministic per object, so
                    // skip the dispatch chain.
                    +irSet(vState.symbol, irInt(stateIds.getValue(plan.entry)))
                } else {
                    +irSet(vState.symbol, irCall(stateIdFun.symbol).apply {
                        arguments[0] = irGet(vRecv)
                    })
                    // Unplanned class: run natively, result flows through APPLY.
                    +irIfThen(
                        builtIns.unitType,
                        irEquals(irGet(vState), irInt(STATE_UNPLANNED)),
                        irBlock {
                            +irSet(vRet.symbol, irCall(baseMethod.symbol).apply {
                                arguments[0] = irGet(vRecv)
                                for (i in vArgs.indices) arguments[i + 1] = irGet(vArgs[i])
                            })
                            +irSet(vState.symbol, irInt(STATE_APPLY))
                        },
                    )
                }
            }
        }

        /** `fun stackless$enter(recv, args): Int` — route one subtree through the trampoline. */
        private fun buildEnterFun(stateIdFun: IrSimpleFunction, runFun: IrSimpleFunction): IrSimpleFunction {
            val fn = backendContext.irFactory.addFunction(irFile) {
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
                name = Name.identifier("stackless\$enter\$${baseMethod.name}")
                visibility = DescriptorVisibilities.PRIVATE
                returnType = baseMethod.returnType
                origin = STACKLESS_SYNTHESIZED_DECLARATION
            }
            val pRecv = fn.addValueParameter("recv", base.defaultType)
            val pArgs = valueParams.mapIndexed { i, vp -> fn.addValueParameter("a$i", vp.type) }
            val b = backendContext.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body = b.irBlockBody {
                val state = irTemporary(irCall(stateIdFun.symbol).apply {
                    arguments[0] = irGet(pRecv)
                }, nameHint = "s")
                +irReturn(
                    irIfThenElse(
                        baseMethod.returnType,
                        irEquals(irGet(state), irInt(STATE_UNPLANNED)),
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
         *
         * The depth counter is a plain mutable static field; Wasm is
         * single-threaded, so that is safe. Known limitation: an exception
         * thrown through an instrumented call leaks the increment; the
         * matcher throws nothing on the match path.
         */
        private fun instrumentNativeBody(info: OverrideInfo, depthField: IrField, enterFun: IrSimpleFunction) {
            val fn = info.function
            val b = backendContext.createIrBuilder(fn.symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET)
            fn.body?.transformChildrenVoid(object : IrElementTransformerVoid() {
                override fun visitCall(expression: IrCall): IrExpression {
                    expression.transformChildrenVoid(this)
                    if (!isTarget(expression)) return expression
                    return b.irBlock(resultType = baseMethod.returnType) {
                        val recvTmp = irTemporary(expression.arguments[0]!!, nameHint = "hrecv")
                        val argTmps = (1 until expression.arguments.size).map { i ->
                            irTemporary(expression.arguments[i]!!, nameHint = "harg${i - 1}")
                        }
                        +irHybridDepthGuard(
                            backendContext, depthField, baseMethod.returnType,
                            shallowCall = {
                                irCall(expression.symbol).apply {
                                    arguments[0] = irGet(recvTmp)
                                    for (i in argTmps.indices) arguments[i + 1] = irGet(argTmps[i])
                                }
                            },
                            deepCall = {
                                irCall(enterFun.symbol).apply {
                                    arguments[0] = irGet(recvTmp)
                                    for (i in argTmps.indices) arguments[i + 1] = irGet(argTmps[i])
                                }
                            },
                        )
                    }
                }
            })
            fn.body?.patchDeclarationParents(fn)
        }
    }

    private fun reportPlanSummary(
        base: IrClass,
        plans: List<BodyPlan>,
        bailed: Map<OverrideInfo, String>,
    ) {
        val msg = buildString {
            append("[wasm-stackless] ${base.name}: planned=${plans.size} bailout=${bailed.size}")
            if (bailed.isNotEmpty()) {
                append(" (")
                append(bailed.entries.joinToString(", ") { "${it.key.irClass!!.name}[${it.value}]" })
                append(")")
            }
        }
        @OptIn(MessageCollectorAccess::class)
        context.configuration.messageCollector.report(CompilerMessageSeverity.LOGGING, msg)
    }
}
