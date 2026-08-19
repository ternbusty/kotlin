/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.backend.wasm.lower.cps.*
import org.jetbrains.kotlin.backend.wasm.utils.StronglyConnectedComponents
import org.jetbrains.kotlin.config.reportLog
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.irBlock
import org.jetbrains.kotlin.ir.builders.irGet
import org.jetbrains.kotlin.ir.builders.irTemporary
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.IrBlockBody
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.symbols.IrFunctionSymbol
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.isLocal
import org.jetbrains.kotlin.ir.util.isOverridable
import org.jetbrains.kotlin.ir.util.patchDeclarationParents
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys

/**
 * Defunctionalized CPS conversion for concrete recursive functions,
 * gated behind `-Xwasm-enable-stackless-recursion`.
 *
 * Scans all effectively-final concrete functions, builds a call graph,
 * and uses Kosaraju-Sharir SCC detection to find recursive cycles.
 * Each SCC is compiled into a flat state machine with heap-allocated
 * frames and a hybrid depth threshold (native stack below 512 frames,
 * heap-frame trampoline above).
 */
internal class WasmConcreteCpsLowering(private val context: WasmBackendContext) : ModuleLoweringPass {

    private val enabled = context.configuration.get(WasmConfigurationKeys.WASM_ENABLE_STACKLESS_RECURSION) == true

    override fun lower(irModule: IrModuleFragment) {
        if (!enabled) return

        val concreteSCCs = detectConcreteRecursiveSCCs(irModule)
        for (scc in concreteSCCs) {
            lowerConcreteSCC(scc)
        }
    }

    /**
     * Builds a call graph of effectively-final concrete functions and
     * returns strongly connected components that contain at least one
     * recursive edge.
     */
    private fun detectConcreteRecursiveSCCs(
        irModule: IrModuleFragment,
    ): List<List<IrSimpleFunction>> {
        val eligible = mutableListOf<IrSimpleFunction>()
        irModule.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                declaration.acceptChildrenVoid(this)
                if (declaration.body == null) return
                if (declaration.isFakeOverride) return
                if (declaration.isLocal) return
                if (declaration.isOverridable) return
                eligible += declaration
            }
        })

        val funcBySymbol = eligible.associateBy { it.symbol }

        val callees = mutableMapOf<IrSimpleFunction, Set<IrSimpleFunction>>()
        for (func in eligible) {
            val targets = mutableSetOf<IrSimpleFunction>()
            func.body?.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitFunction(declaration: IrFunction) {
                    // Do not descend into nested function declarations.
                }
                override fun visitCall(expression: IrCall) {
                    funcBySymbol[expression.symbol]?.let { targets += it }
                    expression.acceptChildrenVoid(this)
                }
            })
            if (targets.isNotEmpty()) callees[func] = targets
        }

        val scc = StronglyConnectedComponents<IrSimpleFunction> { func ->
            (callees[func] ?: emptySet()).asSequence()
        }
        for (func in callees.keys) scc.visit(func)
        val components = scc.findComponents()

        return components.filter { component ->
            if (component.size == 1) {
                val func = component.single()
                val targets = callees[func] ?: emptySet()
                func in targets
            } else {
                true
            }
        }
    }

    private fun lowerConcreteSCC(scc: List<IrSimpleFunction>) {
        val sccSymbols: Set<IrFunctionSymbol> = scc.mapTo(mutableSetOf()) { it.symbol }
        val isTarget: (IrCall) -> Boolean = { it.symbol in sccSymbols }

        val plannedFunctions = mutableListOf<IrSimpleFunction>()
        val plannedPlans = mutableListOf<BodyPlan>()

        for (func in scc) {
            val original = func.body as? IrBlockBody
            if (original == null) {
                reportBailout(func, "no block body")
                return
            }
            val copy = original.deepCopyWithSymbols(func)
            normalizeTargetCallArguments(copy, func, isTarget)
            val planner = BodyPlanner(func, copy, isTarget = isTarget, builtIns = context.irBuiltIns)
            val plan = planner.plan()
            if (plan == null) {
                reportBailout(func, planner.lastBailReason)
                return
            }
            plannedFunctions += func
            plannedPlans += plan
        }

        context.irFactory.stageController.restrictTo(plannedFunctions.first()) {
            ConcreteCpsCodegen(context, plannedFunctions, plannedPlans).generate()
        }
        for (i in plannedFunctions.indices) {
            reportSuccess(plannedFunctions[i], plannedPlans[i])
        }
    }

    /**
     * Hoists argument lists containing a target call into ordered
     * temporaries so the body planner can split them into blocks.
     */
    private fun normalizeTargetCallArguments(
        body: IrBlockBody,
        owner: IrSimpleFunction,
        isTarget: (IrCall) -> Boolean,
    ) {
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

    private fun qualName(func: IrSimpleFunction): String = buildString {
        val parent = func.parent
        if (parent is IrClass) append("${parent.name}.")
        append(func.name)
    }

    private fun reportBailout(func: IrSimpleFunction, reason: String) {
        context.configuration.reportLog("[wasm-concrete-cps] ${qualName(func)}: bailout ($reason)")
    }

    private fun reportSuccess(func: IrSimpleFunction, plan: BodyPlan) {
        context.configuration.reportLog("[wasm-concrete-cps] ${qualName(func)}: planned (${plan.blocks.size} blocks)")
    }
}
