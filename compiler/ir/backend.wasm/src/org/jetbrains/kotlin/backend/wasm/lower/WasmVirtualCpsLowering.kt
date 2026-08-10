/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.backend.wasm.lower.virtualcps.*
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.IrBlockBody
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.types.classOrNull
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.isSubclassOf
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys

/**
 * Defunctionalized CPS conversion for virtual mutual recursion over a closed
 * class hierarchy, gated behind `-Xwasm-enable-stackless-recursion`.
 *
 * This pass automatically detects abstract methods whose overrides contain
 * recursive virtual calls back to the same method. No annotation is required.
 * It uses class-hierarchy analysis to collect every override in the module and
 * compiles them into a single flat state machine with heap-allocated frames.
 * Overrides whose bodies the transformation cannot handle fall back to their
 * original virtual dispatch transparently.
 *
 * Design:
 *  - Auto-detection scans every abstract method in the module; if at least one
 *    concrete override body calls the method virtually, it is a candidate.
 *  - CHA enumerates all overrides of each detected method in the module
 *    (sound under Wasm whole-world compilation).
 *  - Each override body is compiled into a basic-block plan, splitting at
 *    virtual call sites of the target method:
 *      * calls in tail position    -> receiver/state swap, no frame
 *      * non-tail calls            -> heap frame capturing live locals
 *    Unsupported constructs bail out: that override keeps its native body
 *    and the trampoline invokes it as an ordinary virtual call (partial
 *    conversion is always semantics-preserving).
 *  - A single `run$virtualCps` function holds the flat state machine:
 *    while(true) + when(state), state = (override, block) pairs.
 */
internal class WasmVirtualCpsLowering(private val context: WasmBackendContext) : ModuleLoweringPass {

    private val enabled = context.configuration.get(WasmConfigurationKeys.WASM_ENABLE_STACKLESS_RECURSION) == true

    companion object {
        /**
         * Minimum number of concrete overrides with polymorphic recursive
         * calls required for auto-detection. A threshold of 2 filters out
         * single-delegation patterns while retaining genuine mutual-
         * recursion hierarchies with multiple participating subclasses.
         */
        private const val MIN_POLYMORPHIC_RECURSIVE_OVERRIDES = 2
    }

    override fun lower(irModule: IrModuleFragment) {
        if (!enabled) return
        val allClasses = collectClasses(irModule)
        val candidates = detectRecursiveVirtualMethods(allClasses)
        for (pair in candidates) {
            lowerHierarchy(irModule, pair.first, pair.second, allClasses)
        }
        WasmDrfAcceleration(context).lower(irModule)
    }

    // ================================================================ auto-detection

    /**
     * Scans abstract methods declared in abstract classes and returns those
     * whose concrete overrides contain polymorphic virtual calls back to
     * the same abstract method, forming a potential recursive cycle.
     *
     * Two filters reduce false positives.
     *
     *  1. Interface methods are excluded because the recursion-through-
     *     virtual-dispatch pattern that causes stack overflow is
     *     characteristic of abstract base classes (e.g. regex
     *     `AbstractSet.matches()`), not interface contracts.
     *
     *  2. Only calls whose dispatch receiver has a non-final static type
     *     are counted. A final receiver (e.g. `this.toInt().toShort()`
     *     where the receiver is `Int`) resolves to a single override and
     *     cannot form a cycle. A non-final receiver (e.g. `next.matches()`
     *     where `next` is typed as the abstract base) can dispatch to any
     *     override at runtime, enabling unbounded mutual recursion.
     *
     *  3. At least [MIN_POLYMORPHIC_RECURSIVE_OVERRIDES] overrides must
     *     contain such a polymorphic call. This filters out single-
     *     delegation patterns (e.g. one `AbstractMutableList.add`
     *     override forwarding to a delegate).
     */
    private fun detectRecursiveVirtualMethods(
        allClasses: List<IrClass>,
    ): List<Pair<IrClass, IrSimpleFunction>> {
        val result = mutableListOf<Pair<IrClass, IrSimpleFunction>>()
        for (cls in allClasses) {
            if (cls.kind != ClassKind.CLASS) continue
            for (fn in cls.declarations.filterIsInstance<IrSimpleFunction>()) {
                if (fn.body != null) continue
                if (fn.isFakeOverride) continue
                val count = countPolymorphicRecursiveOverrides(cls, fn, allClasses)
                if (count >= MIN_POLYMORPHIC_RECURSIVE_OVERRIDES) {
                    result += cls to fn
                }
            }
        }
        return result
    }

    /**
     * Counts concrete overrides of [baseMethod] within subclasses of
     * [base] whose bodies contain a polymorphic virtual call back to
     * [baseMethod] (dispatch receiver is non-final).
     */
    private fun countPolymorphicRecursiveOverrides(
        base: IrClass,
        baseMethod: IrSimpleFunction,
        allClasses: List<IrClass>,
    ): Int {
        var count = 0
        for (cls in allClasses) {
            if (!cls.isSubclassOf(base)) continue
            val override = cls.declarations
                .filterIsInstance<IrSimpleFunction>()
                .firstOrNull { fn ->
                    fn.body != null && !fn.isFakeOverride &&
                            fn.allOverriddenIncludingSelf().any { it == baseMethod }
                } ?: continue
            if (bodyContainsPolymorphicCallTo(override, baseMethod)) count++
        }
        return count
    }

    /**
     * Walks [function]'s body looking for a virtual call to [baseMethod]
     * whose dispatch receiver has a non-final static type, meaning the
     * call could dispatch to any override at runtime.
     */
    private fun bodyContainsPolymorphicCallTo(
        function: IrSimpleFunction,
        baseMethod: IrSimpleFunction,
    ): Boolean {
        var found = false
        function.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitCall(expression: IrCall) {
                if (isTargetCall(expression, baseMethod)) {
                    val receiverClass = expression.dispatchReceiver
                        ?.type?.classOrNull?.owner
                    if (receiverClass == null ||
                        receiverClass.modality != Modality.FINAL
                    ) {
                        found = true
                        return
                    }
                }
                expression.acceptChildrenVoid(this)
            }
        })
        return found
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
                    fn.body != null && !fn.isFakeOverride &&
                            fn.allOverriddenIncludingSelf().any { it == baseMethod }
                }
                ?: continue
            result += OverrideInfo(cls, override)
        }
        return result
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
            val planner = BodyPlanner(info.function, baseMethod, copy, irBuiltIns = context.irBuiltIns)
            val plan = planner.plan()
            if (plan != null) plans += plan else {
                bailedOut += info
                bailReasons[clsName] = planner.lastBailReason
            }
        }
        if (plans.isEmpty()) return

        context.irFactory.stageController.restrictTo(plans.first().info.function) {
            VirtualCpsHierarchyCodegen(context, base, baseMethod, plans, bailedOut).generate()
        }
        reportPlanSummary(base, plans, bailedOut, bailReasons)
    }

    // ================================================================ diagnostics

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
