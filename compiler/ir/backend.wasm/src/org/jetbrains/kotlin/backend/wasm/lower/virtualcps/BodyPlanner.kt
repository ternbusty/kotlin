/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.virtualcps

import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.IrBuiltIns
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.*
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid

internal class BodyPlanner(
    private val func: IrSimpleFunction,
    private val baseMethod: IrSimpleFunction,
    private val bodyToPlan: IrBlockBody,
    private val isTarget: (IrCall) -> Boolean = { isTargetCall(it, baseMethod) },
    private val irBuiltIns: IrBuiltIns,
) {
    private val blocks = mutableListOf<BlockPlan>()
    private val locals = mutableListOf<IrVariable>()
    private val loopStack = ArrayDeque<LoopFrame>()
    private val returnableStack = ArrayDeque<ReturnableFrame>()
    private val builtIns get() = irBuiltIns

    var lastBailReason: String = ""
        private set

    private fun IrExpression.asDiscardedStatement(): IrStatement = this

    private fun IrExpression.asUnreachableDefault(): IrExpression = when (val rt = func.returnType) {
        builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
        builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, false)
        builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, ' ')
        builtIns.byteType -> IrConstImpl.byte(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
        builtIns.shortType -> IrConstImpl.short(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
        builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
        builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0f)
        builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0)
        else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt.makeNullable())
    }

    private fun newBlock(): BlockPlan = BlockPlan(blocks.size).also { blocks += it }

    fun plan(): BodyPlan? {
        val body = bodyToPlan
        return try {
            inlineLocalFunsWithTargetCalls(body)
            val entry = newBlock()
            val last = compileStatements(body.statements, entry)
            // Bodies whose every path returns or throws leave a block with
            // no terminator; the synthetic Ret is unreachable but must
            // still be well-typed for the function's return type.
            if (last.terminator == null) {
                val rt = func.returnType
                val zero: IrExpression = when (rt) {
                    builtIns.intType -> IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
                    builtIns.booleanType -> IrConstImpl.boolean(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, false)
                    builtIns.charType -> IrConstImpl.char(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, ' ')
                    builtIns.longType -> IrConstImpl.long(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0)
                    builtIns.floatType -> IrConstImpl.float(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0f)
                    builtIns.doubleType -> IrConstImpl.double(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt, 0.0)
                    else -> IrConstImpl.constNull(UNDEFINED_OFFSET, UNDEFINED_OFFSET, rt.makeNullable())
                }
                last.terminator = Terminator.Ret(zero)
            }
            val plan = BodyPlan(OverrideInfo(func.parent as? IrClass, func), blocks.first(), blocks, locals)
            validateCaptures(plan)
            plan
        } catch (b: BailOut) {
            lastBailReason = b.reason
            null
        }
    }

    /**
     * Verifies that every [IrGetValue]/[IrSetValue] in the planned blocks
     * refers to a symbol declared in [BodyPlan.locals], the function
     * parameters, or the enclosing block. A missing declaration means the
     * code generator would produce IR with dangling references.
     */
    private fun validateCaptures(plan: BodyPlan) {
        val declared = mutableSetOf<IrValueSymbol>()
        for (p in func.parameters) declared += p.symbol
        for (l in plan.locals) declared += l.symbol

        val declCollector = object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitVariable(declaration: IrVariable) {
                declared += declaration.symbol
                declaration.acceptChildrenVoid(this)
            }
        }

        fun walkTerminator(term: Terminator?, visitor: IrVisitorVoid) {
            when (term) {
                is Terminator.SuspendCall -> term.call.acceptVoid(visitor)
                is Terminator.TailCall -> term.call.acceptVoid(visitor)
                is Terminator.Ret -> term.value.acceptVoid(visitor)
                is Terminator.CondGoto -> term.condition.acceptVoid(visitor)
                is Terminator.Goto, null -> {}
            }
        }

        for (block in plan.blocks) {
            for (stmt in block.statements) stmt.acceptVoid(declCollector)
            // Also collect declarations inside terminator expressions
            // (e.g. IrReturnableBlock with local variables in Ret.value).
            walkTerminator(block.terminator, declCollector)
        }

        val missing = mutableSetOf<IrValueSymbol>()
        val refChecker = object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitGetValue(expression: IrGetValue) {
                if (expression.symbol !in declared) missing += expression.symbol
            }
            override fun visitSetValue(expression: IrSetValue) {
                if (expression.symbol !in declared) missing += expression.symbol
                expression.acceptChildrenVoid(this)
            }
        }
        for (block in plan.blocks) {
            for (stmt in block.statements) stmt.acceptVoid(refChecker)
            walkTerminator(block.terminator, refChecker)
        }

        if (missing.isNotEmpty()) {
            val names = missing.joinToString {
                (it.owner as? IrVariable)?.name?.asString()
                    ?: (it.owner as? IrValueParameter)?.name?.asString()
                    ?: it.toString()
            }
            throw BailOut("undeclared variable references: $names")
        }
    }

    /**
     * Local funs whose body contains target calls (e.g. GroupQuantifierSet's
     * matchNext) cannot be split in place. If the fun has a single trailing
     * return and no parameters, inline it at every call site; captured
     * variables keep their symbols and are hoisted like any other local.
     */
    private fun inlineLocalFunsWithTargetCalls(body: IrBlockBody) {
        val localFuns = mutableListOf<IrSimpleFunction>()
        body.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                if (hasTargetCallInside(declaration)) localFuns += declaration
            }
        })
        if (localFuns.isEmpty()) return

        for (lf in localFuns) {
            if (lf.parameters.isNotEmpty()) throw BailOut("local fun with parameters holds target call")
            val lfBody = lf.body as? IrBlockBody ?: throw BailOut("local fun without block body holds target call")
            val last = lfBody.statements.lastOrNull()
            if (last !is IrReturn || last.returnTargetSymbol != lf.symbol) throw BailOut("local fun without trailing return holds target call")
            var returns = 0
            lfBody.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitReturn(expression: IrReturn) {
                    if (expression.returnTargetSymbol == lf.symbol) returns++
                    expression.acceptChildrenVoid(this)
                }
            })
            if (returns != 1) throw BailOut("local fun with multiple returns holds target call")

            body.transform(object : IrTransformer<Nothing?>() {
                override fun visitCall(expression: IrCall, data: Nothing?): IrElement {
                    expression.transformChildren(this, data)
                    if (expression.symbol != lf.symbol) return expression
                    val copied = lfBody.deepCopyWithSymbols(lf)
                    val stmts = copied.statements.toMutableList()
                    val ret = stmts.removeLast() as IrReturn
                    return IrBlockImpl(
                        expression.startOffset, expression.endOffset, lf.returnType, null,
                        stmts + ret.value,
                    )
                }
            }, null)
            // Drop the now-unused declaration.
            body.statements.removeAll { it === lf }
            body.transform(object : IrTransformer<Nothing?>() {
                override fun visitBlock(expression: IrBlock, data: Nothing?): IrExpression {
                    expression.transformChildren(this, data)
                    expression.statements.removeAll { it === lf }
                    return expression
                }
            }, null)
        }
    }

    /** Compiles [stmts] starting into [start]; returns the block that control ends in. */
    private fun compileStatements(stmts: List<IrStatement>, start: BlockPlan): BlockPlan {
        var current = start
        for (stmt in stmts) {
            if (current.terminator != null) {
                // Unreachable trailing statements (e.g. code after return) — drop.
                break
            }
            current = compileStatement(stmt, current)
        }
        return current
    }

    private fun compileStatement(stmt: IrStatement, current: BlockPlan): BlockPlan {
        when (stmt) {
            is IrVariable -> {
                locals += stmt
                val init = stmt.initializer
                if (init != null && needsSplit(init)) {
                    if (init is IrWhen || init is IrContainerExpression) {
                        // val x = <complex>  ==>  var x; <complex with x = branch/last>
                        stmt.initializer = null
                        current.statements += stmt
                        return compileAssignmentOf(stmt.symbol, init, current)
                    }
                    // Inline expansion of scoping functions (.let, .also,
                    // .run) may wrap the target call in an IMPLICIT_CAST
                    // from the erased generic type. Peel through it to
                    // reach the underlying IrCall.
                    val call = peelTransparentTypeOps(init) as? IrCall
                        ?: throw BailOut("target call or return nested in variable initializer ${init::class.simpleName}")
                    if (!isTarget(call)) throw BailOut("nested target call")
                    checkArgsHaveNoTargetCalls(call)
                    val resume = newBlock()
                    stmt.initializer = null
                    current.statements += stmt
                    current.terminator = Terminator.SuspendCall(call, stmt.symbol, resume)
                    return resume
                }
                current.statements += stmt
                return current
            }

            is IrReturn -> {
                if (stmt.returnTargetSymbol != func.symbol) {
                    // Return out of an inlined block (e.g. forEach's non-local
                    // return machinery): jump to that block's join, routing the
                    // value into the block's result slot when it has one.
                    val frame = returnableStack.lastOrNull { it.symbol == stmt.returnTargetSymbol }
                        ?: throw BailOut("non-local return")
                    if (frame.returnThrough) {
                        return compileStatement(
                            IrReturnImpl(stmt.startOffset, stmt.endOffset, builtIns.nothingType, func.symbol, stmt.value),
                            current,
                        )
                    }
                    val cur: BlockPlan
                    if (frame.resultTarget != null) {
                        cur = compileAssignmentOf(frame.resultTarget, stmt.value, current)
                    } else {
                        if (containsTargetCall(stmt.value)) throw BailOut("target call in returnable-block return value")
                        current.statements += stmt.value.asDiscardedStatement()
                        cur = current
                    }
                    cur.terminator = Terminator.Goto(frame.join)
                    return cur
                }
                val value = stmt.value
                if (containsTargetCall(value)) {
                    // Peel transparent type operators so that structural
                    // dispatch below reaches IrCall/IrWhen/IrBlock even
                    // when the inliner has wrapped them in IMPLICIT_CAST.
                    val peeled = peelTransparentTypeOps(value)
                    when (peeled) {
                        is IrCall -> {
                            if (!isTarget(peeled)) throw BailOut("nested target call")
                            checkArgsHaveNoTargetCalls(peeled)
                            current.terminator = Terminator.TailCall(peeled)
                            return current
                        }
                        is IrWhen -> {
                            // return when { c1 -> e1; ... }  ==>  when { c1 -> return e1; ... }
                            val pushed = IrWhenImpl(peeled.startOffset, peeled.endOffset, builtIns.unitType, peeled.origin).apply {
                                for (branch in peeled.branches) {
                                    val result = branch.result
                                    val branchBody = if (result.type.isNothing()) {
                                        result
                                    } else {
                                        IrReturnImpl(result.startOffset, result.endOffset, builtIns.nothingType, func.symbol, result)
                                    }
                                    branches += IrBranchImpl(branch.startOffset, branch.endOffset, branch.condition, branchBody)
                                }
                            }
                            val end = compileStatement(pushed, current)
                            // A return-when with no else falls through: treat as unreachable end.
                            if (end.terminator == null) end.terminator = Terminator.Ret(peeled.asUnreachableDefault())
                            return end
                        }
                        is IrBlock -> {
                            if (peeled is IrReturnableBlock) {
                                // return <RB>: every return@RB v is a plain return v.
                                val join = newBlock()
                                returnableStack.addLast(ReturnableFrame(peeled.symbol, join, null, returnThrough = true))
                                val end = compileStatements(peeled.statements, current)
                                returnableStack.removeLast()
                                if (end.terminator == null) throw BailOut("returned returnable block falls through")
                                return join
                            }
                            // return block { s1..sn; last } ==> s1..sn; return last
                            val stmts = peeled.statements
                            if (stmts.isEmpty()) throw BailOut("empty block in return")
                            var cur = current
                            for (i in 0 until stmts.size - 1) cur = compileStatement(stmts[i], cur)
                            val last = stmts.last()
                            val lastExpr = last as? IrExpression ?: throw BailOut("return block last statement is not an expression")
                            if (lastExpr.type.isNothing()) {
                                return compileStatement(lastExpr, cur)
                            }
                            return compileStatement(
                                IrReturnImpl(stmt.startOffset, stmt.endOffset, builtIns.nothingType, func.symbol, lastExpr),
                                cur,
                            )
                        }
                        else -> throw BailOut("target call nested in return value ${peeled::class.simpleName}")
                    }
                }
                current.terminator = Terminator.Ret(value)
                return current
            }

            is IrWhen -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                return compileWhen(stmt, current)
            }

            is IrWhileLoop -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                return compileWhile(stmt, current)
            }

            is IrDoWhileLoop -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                return compileDoWhile(stmt, current)
            }

            is IrBlock -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                if (stmt is IrReturnableBlock) {
                    if (!stmt.type.isUnit() && !stmt.type.isNothing()) {
                        throw BailOut("non-Unit returnable block with target call or return")
                    }
                    val join = newBlock()
                    returnableStack.addLast(ReturnableFrame(stmt.symbol, join))
                    val end = compileStatements(stmt.statements, current)
                    returnableStack.removeLast()
                    if (end.terminator == null) end.terminator = Terminator.Goto(join)
                    return join
                }
                return compileStatements(stmt.statements, current)
            }

            is IrSetValue -> {
                if (!needsSplit(stmt.value)) {
                    current.statements += stmt
                    return current
                }
                return compileAssignmentOf(stmt.symbol, stmt.value, current)
            }

            is IrTypeOperatorCall -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                // A statement-position coercion to Unit just discards the
                // value; compile the underlying expression as a statement.
                if (stmt.operator == IrTypeOperator.IMPLICIT_COERCION_TO_UNIT) {
                    return compileStatement(stmt.argument, current)
                }
                throw BailOut("target call or return in type operator ${stmt.operator}")
            }

            is IrComposite -> {
                if (!needsSplit(stmt)) {
                    current.statements += stmt
                    return current
                }
                return compileStatements(stmt.statements, current)
            }

            is IrBreak -> {
                val frame = loopStack.lastOrNull { it.loop == stmt.loop }
                    ?: throw BailOut("break targets unknown loop")
                current.terminator = Terminator.Goto(frame.exit)
                return current
            }

            is IrContinue -> {
                val frame = loopStack.lastOrNull { it.loop == stmt.loop }
                    ?: throw BailOut("continue targets unknown loop")
                current.terminator = Terminator.Goto(frame.head)
                return current
            }

            else -> {
                if (needsSplit(stmt)) throw BailOut("target call or return in unsupported statement ${stmt::class.simpleName}")
                current.statements += stmt
                return current
            }
        }
    }

    /**
     * Compiles `target = value` where value contains target calls, outer
     * returns, or jumps to split loops. Nothing-typed positions (return /
     * break / continue) never produce a value, so they are compiled as
     * plain statements instead of being wrapped in an assignment.
     */
    private fun compileAssignmentOf(target: IrValueSymbol, value: IrExpression, current: BlockPlan): BlockPlan {
        if (value.type.isNothing()) {
            return compileStatement(value, current)
        }
        if (!needsSplit(value)) {
            current.statements += IrSetValueImpl(
                value.startOffset, value.endOffset, builtIns.unitType, target, value, null,
            )
            return current
        }
        when (value) {
            is IrCall -> {
                if (!isTarget(value)) throw BailOut("nested target call in assignment")
                checkArgsHaveNoTargetCalls(value)
                val resume = newBlock()
                current.terminator = Terminator.SuspendCall(value, target, resume)
                return resume
            }
            is IrWhen -> {
                val pushed = IrWhenImpl(value.startOffset, value.endOffset, builtIns.unitType, value.origin).apply {
                    for (branch in value.branches) {
                        val result = branch.result
                        val branchBody = if (result.type.isNothing()) {
                            result
                        } else {
                            IrSetValueImpl(result.startOffset, result.endOffset, builtIns.unitType, target, result, null)
                        }
                        branches += IrBranchImpl(branch.startOffset, branch.endOffset, branch.condition, branchBody)
                    }
                }
                return compileStatement(pushed, current)
            }
            is IrReturnableBlock -> {
                val join = newBlock()
                returnableStack.addLast(ReturnableFrame(value.symbol, join, target))
                val end = compileStatements(value.statements, current)
                returnableStack.removeLast()
                if (end.terminator == null) {
                    // Inlined function bodies complete through returns; a
                    // fall-through end would leave the target unassigned.
                    throw BailOut("returnable block in assignment falls through")
                }
                return join
            }
            is IrContainerExpression -> {
                val stmts = value.statements
                if (stmts.isEmpty()) throw BailOut("empty block in assignment")
                var cur = current
                for (i in 0 until stmts.size - 1) cur = compileStatement(stmts[i], cur)
                val lastExpr = stmts.last() as? IrExpression ?: throw BailOut("assignment block last statement is not an expression")
                return compileAssignmentOf(target, lastExpr, cur)
            }
            else -> throw BailOut("target call or return nested in assignment ${value::class.simpleName}")
        }
    }

    private fun compileWhen(whenExpr: IrWhen, current: BlockPlan): BlockPlan {
        val join = newBlock()
        var condBlock = current
        for (branch in whenExpr.branches) {
            if (needsSplit(branch.condition)) throw BailOut("target call or return in when condition")
            val isElse = branch.condition is IrConst && (branch.condition as IrConst).value == true
            val bodyBlock = newBlock()
            if (isElse) {
                condBlock.terminator = Terminator.Goto(bodyBlock)
            } else {
                val next = newBlock()
                condBlock.terminator = Terminator.CondGoto(branch.condition, bodyBlock, next)
                condBlock = next
            }
            val bodyEnd = compileStatement(branch.result.asStatement(), bodyBlock)
            if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(join)
            if (isElse) {
                return joinOrContinue(join)
            }
        }
        // No else branch: fall through from the last condition block.
        condBlock.terminator = Terminator.Goto(join)
        return joinOrContinue(join)
    }

    private fun joinOrContinue(join: BlockPlan): BlockPlan = join

    private fun IrExpression.asStatement(): IrStatement = this

    private fun compileWhile(loop: IrWhileLoop, current: BlockPlan): BlockPlan {
        if (needsSplit(loop.condition)) throw BailOut("target call or return in loop condition")
        val head = newBlock()
        val bodyBlock = newBlock()
        val exit = newBlock()
        current.terminator = Terminator.Goto(head)
        head.terminator = Terminator.CondGoto(loop.condition, bodyBlock, exit)
        loopStack.addLast(LoopFrame(loop, head, exit))
        val bodyEnd = loop.body?.let { compileStatement(it, bodyBlock) } ?: bodyBlock
        if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(head)
        loopStack.removeLast()
        return exit
    }

    private fun compileDoWhile(loop: IrDoWhileLoop, current: BlockPlan): BlockPlan {
        if (needsSplit(loop.condition)) throw BailOut("target call or return in loop condition")
        val bodyBlock = newBlock()
        val condBlock = newBlock()
        val exit = newBlock()
        current.terminator = Terminator.Goto(bodyBlock)
        loopStack.addLast(LoopFrame(loop, condBlock, exit))
        val bodyEnd = loop.body?.let { compileStatement(it, bodyBlock) } ?: bodyBlock
        if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(condBlock)
        condBlock.terminator = Terminator.CondGoto(loop.condition, bodyBlock, exit)
        loopStack.removeLast()
        return exit
    }

    /**
     * Recursively strips semantically transparent type operators
     * (IMPLICIT_CAST, IMPLICIT_COERCION_TO_UNIT) that the inliner
     * inserts when expanding generic inline functions. Non-transparent
     * operators (CAST, SAM_CONVERSION, etc.) are left in place.
     */
    private fun peelTransparentTypeOps(expr: IrExpression): IrExpression = when {
        expr is IrTypeOperatorCall && expr.operator in transparentTypeOps -> peelTransparentTypeOps(expr.argument)
        else -> expr
    }

    private val transparentTypeOps = setOf(
        IrTypeOperator.IMPLICIT_CAST,
        IrTypeOperator.IMPLICIT_COERCION_TO_UNIT,
    )

    private fun checkArgsHaveNoTargetCalls(call: IrCall) {
        for (arg in call.arguments) {
            if (arg != null && containsTargetCall(arg)) throw BailOut("target call in argument of target call")
        }
    }

    /** Any statement holding a target call, an outer return, or a jump to
     *  a loop that is being split must itself be compiled into block
     *  structure, so those transfers route through the state machine. */
    private fun needsSplit(element: IrElement): Boolean =
        containsTargetCall(element) || containsOuterReturn(element) ||
                containsJumpToSplitLoop(element) || containsReturnToSplitBlock(element)

    private fun containsReturnToSplitBlock(element: IrElement): Boolean {
        if (returnableStack.isEmpty()) return false
        val targets = returnableStack.mapTo(mutableSetOf<Any>()) { it.symbol }
        var found = false
        element.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitFunction(declaration: IrFunction) {}

            override fun visitReturn(expression: IrReturn) {
                if (expression.returnTargetSymbol in targets) {
                    found = true
                    return
                }
                expression.acceptChildrenVoid(this)
            }
        })
        return found
    }

    private fun containsJumpToSplitLoop(element: IrElement): Boolean {
        if (loopStack.isEmpty()) return false
        val splitLoops = loopStack.mapTo(mutableSetOf()) { it.loop }
        var found = false
        element.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitFunction(declaration: IrFunction) {}

            override fun visitBreakContinue(jump: IrBreakContinue) {
                if (jump.loop in splitLoops) found = true
            }
        })
        return found
    }

    private fun containsOuterReturn(element: IrElement): Boolean {
        var found = false
        element.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitFunction(declaration: IrFunction) {}

            override fun visitReturn(expression: IrReturn) {
                if (expression.returnTargetSymbol == func.symbol) {
                    found = true
                    return
                }
                expression.acceptChildrenVoid(this)
            }
        })
        return found
    }

    private fun containsTargetCall(element: IrElement): Boolean {
        var found = false
        element.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitFunction(declaration: IrFunction) {
                // Local functions are analyzed separately; a target call inside
                // one cannot be split here.
                if (hasTargetCallInside(declaration)) found = true
            }

            override fun visitCall(expression: IrCall) {
                if (isTarget(expression)) {
                    found = true
                    return
                }
                expression.acceptChildrenVoid(this)
            }
        })
        return found
    }

    private fun hasTargetCallInside(declaration: IrFunction): Boolean {
        var found = false
        declaration.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitCall(expression: IrCall) {
                if (isTarget(expression)) found = true
                if (!found) expression.acceptChildrenVoid(this)
            }
        })
        return found
    }
}
