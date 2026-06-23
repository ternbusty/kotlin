/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.ir2wasm

import org.jetbrains.kotlin.builtins.StandardNames
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.IrClass
import org.jetbrains.kotlin.ir.declarations.IrConstructor
import org.jetbrains.kotlin.ir.declarations.IrFunction
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.types.isClassWithFqName
import org.jetbrains.kotlin.ir.types.isUnit
import org.jetbrains.kotlin.ir.visitors.IrVisitor

/**
 * Collects every [IrCall] that lexically appears in tail position inside [irFunction].
 *
 * Tail position rules mirror [org.jetbrains.kotlin.backend.common.TailRecursionCallsCollector]:
 * - the value of `IrReturn` whose target is this function is a tail expression,
 * - the result of every branch of `IrWhen` whose enclosing context is tail is tail,
 * - the last statement of `IrBlock` / `IrContainerExpression` whose enclosing context is tail is tail,
 * - in a `Unit`-returning function, any statement immediately followed by `return Unit` is tail,
 * - calls inside `IrTry` are never considered tail (the catch handler frame must remain live).
 *
 * Unlike [TailRecursionCallsCollector] this collector does not require the call target to be
 * [irFunction] itself. Wasm's `return_call` can transfer control to any function with a compatible
 * return-type signature, so mutual recursion and virtually-dispatched tail calls are eligible.
 *
 * Constructors are excluded as callers: [BodyGenerator.visitFunctionReturn] pushes the implicit
 * dispatch receiver before `return`, which is incompatible with a tail-call frame swap.
 *
 * Eligibility filters that depend on Wasm-level information (signature equality, intrinsic
 * detection, callee-is-constructor) are applied later at the emit site in
 * [BodyGenerator.generateCall]; this collector returns a superset.
 */
internal fun collectWasmTailCallCandidates(irFunction: IrFunction): Set<IrCall> {
    if (irFunction is IrConstructor) return emptySet()

    val isUnitReturn = irFunction.returnType.isUnit()
    val result = mutableSetOf<IrCall>()

    class VisitorState(val isTailExpression: Boolean)

    val visitor = object : IrVisitor<Unit, VisitorState>() {
        override fun visitElement(element: IrElement, data: VisitorState) {
            element.acceptChildren(this, VisitorState(isTailExpression = false))
        }

        override fun visitFunction(declaration: IrFunction, data: VisitorState) {
            // Local functions have their own tail-position frame; skip.
        }

        override fun visitClass(declaration: IrClass, data: VisitorState) {
            // Local classes likewise.
        }

        override fun visitTry(aTry: IrTry, data: VisitorState) {
            // A tail call out of try-catch-finally would drop the handler frame; not supported.
        }

        override fun visitReturn(expression: IrReturn, data: VisitorState) {
            val isTail = expression.returnTargetSymbol == irFunction.symbol
            expression.value.accept(this, VisitorState(isTail))
        }

        override fun visitExpressionBody(body: IrExpressionBody, data: VisitorState) =
            body.acceptChildren(this, data)

        override fun visitBlockBody(body: IrBlockBody, data: VisitorState) =
            visitStatementContainer(body, data)

        override fun visitContainerExpression(expression: IrContainerExpression, data: VisitorState) =
            visitStatementContainer(expression, data)

        private fun visitStatementContainer(expression: IrStatementContainer, data: VisitorState) {
            expression.statements.forEachIndexed { index, irStatement ->
                val isTailStatement = if (index == expression.statements.lastIndex) {
                    data.isTailExpression
                } else {
                    isUnitReturn && expression.statements[index + 1].let {
                        it is IrReturn && it.returnTargetSymbol == irFunction.symbol && it.value.isUnitRead()
                    }
                }
                irStatement.accept(this, VisitorState(isTailStatement))
            }
        }

        private fun IrExpression.isUnitRead(): Boolean =
            this is IrGetObjectValue && symbol.isClassWithFqName(StandardNames.FqNames.unit)

        override fun visitWhen(expression: IrWhen, data: VisitorState) {
            expression.branches.forEach {
                it.condition.accept(this, VisitorState(isTailExpression = false))
                it.result.accept(this, data)
            }
        }

        override fun visitCall(expression: IrCall, data: VisitorState) {
            expression.acceptChildren(this, VisitorState(isTailExpression = false))
            if (data.isTailExpression) {
                result.add(expression)
            }
        }
    }

    irFunction.body?.accept(visitor, VisitorState(isTailExpression = true))
    return result
}
