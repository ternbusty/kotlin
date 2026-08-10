/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.backend.wasm.lower.virtualcps.*
import org.jetbrains.kotlin.backend.wasm.utils.hasStacklessVirtualRecursionAnnotation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.IrBlockBody
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.isSubclassOf
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys

/**
 * Defunctionalized CPS conversion for virtual mutual recursion over a closed
 * class hierarchy, driven by the `@kotlin.wasm.StacklessVirtualRecursion`
 * annotation and gated behind `-Xwasm-enable-stackless-recursion`.
 *
 * Place the annotation on an abstract function in a base class. This pass
 * uses class-hierarchy analysis to collect every override in the module and
 * compiles them into a single flat state machine with heap-allocated frames.
 * Overrides whose bodies the transformation cannot handle fall back to their
 * original virtual dispatch transparently.
 *
 * Design:
 *  - CHA enumerates all overrides of the annotated base method in the module
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

    override fun lower(irModule: IrModuleFragment) {
        val allClasses = collectClasses(irModule)
        val annotatedMethods = collectAnnotatedMethods(allClasses)
        if (!enabled) {
            if (annotatedMethods.isNotEmpty()) {
                warnAnnotationWithoutFlag(annotatedMethods)
            }
            return
        }
        for (pair in annotatedMethods) {
            lowerHierarchy(irModule, pair.first, pair.second, allClasses)
        }
        WasmDrfAcceleration(context).lower(irModule)
    }

    // ================================================================ annotation scanning

    /**
     * Collects abstract methods annotated with `@StacklessVirtualRecursion`
     * across all classes in the module.
     */
    private fun collectAnnotatedMethods(
        allClasses: List<IrClass>,
    ): List<Pair<IrClass, IrSimpleFunction>> {
        val result = mutableListOf<Pair<IrClass, IrSimpleFunction>>()
        for (cls in allClasses) {
            for (fn in cls.declarations.filterIsInstance<IrSimpleFunction>()) {
                if (fn.body == null && fn.hasStacklessVirtualRecursionAnnotation()) {
                    result += cls to fn
                }
            }
        }
        return result
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

    private fun warnAnnotationWithoutFlag(
        annotatedMethods: List<Pair<IrClass, IrSimpleFunction>>,
    ) {
        val locations = annotatedMethods.joinToString(", ") {
            "${it.first.name}.${it.second.name}"
        }
        @OptIn(MessageCollectorAccess::class)
        context.configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)
            ?.report(
                CompilerMessageSeverity.WARNING,
                "@StacklessVirtualRecursion is used on $locations but " +
                        "-Xwasm-enable-stackless-recursion is not enabled. " +
                        "The annotation will have no effect."
            )
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
