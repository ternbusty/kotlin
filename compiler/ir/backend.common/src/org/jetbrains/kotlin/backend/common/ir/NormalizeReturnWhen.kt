/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.common.ir

import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.visitors.IrTransformer

/**
 * Transforms `return when { ... }` into per-branch returns so that each
 * branch's result is individually wrapped in an [IrReturn].
 *
 * Before:
 * ```
 *   return when {
 *       cond1 -> expr1
 *       else  -> expr2
 *   }
 * ```
 *
 * After:
 * ```
 *   when {
 *       cond1 -> return expr1
 *       else  -> return expr2
 *   }
 * ```
 *
 * Nested `when`/`if` and trailing-expression blocks are handled recursively.
 * This makes each branch's terminal value directly visible as an [IrReturn],
 * which simplifies tail-position analysis and tail-modulo-cons detection.
 */
fun normalizeReturnWhen(func: IrSimpleFunction) {
    val body = func.body as? IrBlockBody ?: return
    val funcSymbol = func.symbol
    body.transform(object : IrTransformer<Nothing?>() {
        override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
            expression.transformChildren(this, data)
            if (expression.returnTargetSymbol != funcSymbol) return expression
            val whenExpr = expression.value as? IrWhen ?: return expression
            for (branch in whenExpr.branches) {
                branch.result = distributeReturn(branch.result, expression)
            }
            return whenExpr
        }

        private fun distributeReturn(expr: IrExpression, proto: IrReturn): IrExpression {
            return when (expr) {
                is IrReturn -> expr
                is IrWhen -> {
                    for (branch in expr.branches) {
                        branch.result = distributeReturn(branch.result, proto)
                    }
                    expr
                }
                is IrBlock -> {
                    val lastIdx = expr.statements.lastIndex
                    if (lastIdx >= 0) {
                        val last = expr.statements[lastIdx]
                        if (last is IrExpression && last !is IrReturn) {
                            expr.statements[lastIdx] = distributeReturn(last, proto)
                        }
                    }
                    expr
                }
                else -> IrReturnImpl(
                    proto.startOffset, proto.endOffset,
                    proto.type,
                    proto.returnTargetSymbol,
                    expr,
                )
            }
        }
    }, null)
}
