/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.BodyLoweringPass
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.builtins.StandardNames
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.types.isClassWithFqName
import org.jetbrains.kotlin.ir.types.isUnit
import org.jetbrains.kotlin.ir.util.isOverridable
import org.jetbrains.kotlin.ir.backend.js.utils.realOverrideTarget
import org.jetbrains.kotlin.ir.visitors.IrVisitor
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys
import org.jetbrains.kotlin.wasm.config.wasmEnableTailCalls

val WASM_TAIL_CALL by IrStatementOriginImpl

/**
 * Marks [IrCall]s in tail position with [WASM_TAIL_CALL] origin so that
 * [BodyGenerator][org.jetbrains.kotlin.backend.wasm.ir2wasm.codegenGenerators.BodyGenerator]
 * can emit `return_call` / `return_call_ref` without re-analysing the IR.
 *
 * With [WasmConfigurationKeys.WASM_ENABLE_TAIL_CALLS] every structurally
 * tail-positioned call is marked. With only [WasmConfigurationKeys.WASM_ENABLE_TMC]
 * marking is selective, because `return_call` prevents V8 from inlining the
 * callee into the caller: only self-recursive calls (inlining is impossible
 * anyway), calls from the TMC-generated DPS helpers (which rely on
 * `return_call` for bounded stack usage), and virtual/interface dispatch
 * (V8 never inlines through indirect calls) are marked.
 *
 * Must run after all lowerings that may wrap calls in blocks, try-catch, or
 * continuation machinery, so that the structural tail-position analysis sees
 * the final IR shape. Placed at the very end of the Wasm lowering pipeline.
 */
internal class WasmTailCallLowering(private val context: WasmBackendContext) : BodyLoweringPass {
    private val unrestricted = context.configuration.wasmEnableTailCalls
    private val selective = !unrestricted &&
            context.configuration.get(WasmConfigurationKeys.WASM_ENABLE_TMC) == true

    override fun lower(irBody: IrBody, container: IrDeclaration) {
        if (!unrestricted && !selective) return
        val irFunction = container as? IrFunction ?: return
        if (irFunction is IrConstructor) return
        markTailCalls(irFunction, unrestricted)
    }
}

private fun isSelectivelyMarkable(call: IrCall, caller: IrFunction): Boolean {
    if (call.symbol == caller.symbol) return true
    if (caller.origin === TMC_DPS_FUNCTION) return true
    val callee = call.symbol.owner.realOverrideTarget
    return callee.isOverridable && call.superQualifierSymbol == null
}

private fun markTailCalls(irFunction: IrFunction, unrestricted: Boolean) {
    val isUnitReturn = irFunction.returnType.isUnit()

    val visitor = object : IrVisitor<Unit, Boolean>() {
        override fun visitElement(element: IrElement, data: Boolean) {
            element.acceptChildren(this, false)
        }

        override fun visitFunction(declaration: IrFunction, data: Boolean) {}
        override fun visitClass(declaration: IrClass, data: Boolean) {}
        override fun visitTry(aTry: IrTry, data: Boolean) {}

        override fun visitReturn(expression: IrReturn, data: Boolean) {
            val isTail = expression.returnTargetSymbol == irFunction.symbol
            expression.value.accept(this, isTail)
        }

        override fun visitExpressionBody(body: IrExpressionBody, data: Boolean) =
            body.acceptChildren(this, data)

        override fun visitBlockBody(body: IrBlockBody, data: Boolean) =
            visitStatementContainer(body, data)

        override fun visitContainerExpression(expression: IrContainerExpression, data: Boolean) =
            visitStatementContainer(expression, data)

        private fun visitStatementContainer(container: IrStatementContainer, data: Boolean) {
            container.statements.forEachIndexed { index, irStatement ->
                val isTailStatement = if (index == container.statements.lastIndex) {
                    data
                } else {
                    isUnitReturn && container.statements[index + 1].let {
                        it is IrReturn && it.returnTargetSymbol == irFunction.symbol && it.value.isUnitRead()
                    }
                }
                irStatement.accept(this, isTailStatement)
            }
        }

        private fun IrExpression.isUnitRead(): Boolean =
            this is IrGetObjectValue && symbol.isClassWithFqName(StandardNames.FqNames.unit)

        override fun visitWhen(expression: IrWhen, data: Boolean) {
            expression.branches.forEach {
                it.condition.accept(this, false)
                it.result.accept(this, data)
            }
        }

        override fun visitCall(expression: IrCall, data: Boolean) {
            expression.acceptChildren(this, false)
            if (data && (unrestricted || isSelectivelyMarkable(expression, irFunction))) {
                expression.origin = WASM_TAIL_CALL
            }
        }
    }

    irFunction.body?.accept(visitor, true)
}
