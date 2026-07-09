/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrBuiltIns
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.expressions.impl.IrBlockImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrBranchImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrWhenImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrInstanceInitializerCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrSetValueImpl
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrReturnTargetSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.util.*
import org.jetbrains.kotlin.ir.visitors.IrElementTransformerVoid
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.ir.visitors.transformChildrenVoid

/**
 * Shared machinery for the stackless-recursion lowerings
 * ([WasmDrfAccelerationLowering], [WasmStacklessMatcherLowering]): a body
 * planner that splits a function body into basic blocks at recursive call
 * sites, and the block/frame data model consumed by their code generators.
 *
 * Nothing here registers a compiler phase; this file has no effect until a
 * lowering builds a plan from it.
 */

/**
 * Native recursion budget before switching a subtree to the heap-frame
 * trampoline. Far below the wasm stack guard (~10-15K frames) even for
 * recursion that burns several native frames per level and starts from
 * an already-deep caller stack, and far above what everyday patterns
 * reach.
 */
internal const val STACKLESS_HYBRID_DEPTH_THRESHOLD = 512

// ---------------- block simplification (shared by matcher and DRF codegens)

/**
 * Collapses trivial control flow the planner produces in numbers:
 * empty forwarder blocks are skipped and single-predecessor Goto
 * successors are fused into their predecessor. Fewer states means
 * fewer dispatch round-trips per state-machine step.
 */
internal fun simplifyPlan(plan: BodyPlan) {
    val resumeTargets = plan.blocks
        .mapNotNullTo(mutableSetOf()) { (it.terminator as? Terminator.SuspendCall)?.resume }

    fun resolve(b: BlockPlan): BlockPlan {
        var cur = b
        val seen = mutableSetOf<BlockPlan>()
        while (cur.statements.isEmpty() && cur !in resumeTargets && cur !== plan.entry && seen.add(cur)) {
            val t = cur.terminator as? Terminator.Goto ?: break
            cur = t.target
        }
        return cur
    }

    var changed = true
    while (changed) {
        changed = false

        for (b in plan.blocks) {
            when (val t = b.terminator) {
                is Terminator.Goto -> {
                    val r = resolve(t.target)
                    if (r !== t.target) {
                        b.terminator = Terminator.Goto(r); changed = true
                    }
                }
                is Terminator.CondGoto -> {
                    val r1 = resolve(t.thenTarget)
                    val r2 = resolve(t.elseTarget)
                    if (r1 !== t.thenTarget || r2 !== t.elseTarget) {
                        b.terminator = Terminator.CondGoto(t.condition, r1, r2); changed = true
                    }
                }
                else -> {}
            }
        }

        val preds = mutableMapOf<BlockPlan, Int>()
        fun ref(b: BlockPlan) { preds[b] = (preds[b] ?: 0) + 1 }
        ref(plan.entry)
        for (b in plan.blocks) {
            when (val t = b.terminator) {
                is Terminator.Goto -> ref(t.target)
                is Terminator.CondGoto -> { ref(t.thenTarget); ref(t.elseTarget) }
                is Terminator.SuspendCall -> ref(t.resume)
                else -> {}
            }
        }
        for (b in plan.blocks) {
            val t = b.terminator as? Terminator.Goto ?: continue
            val c = t.target
            if (c !== b && preds[c] == 1 && c !in resumeTargets && c !== plan.entry) {
                b.statements += c.statements
                b.terminator = c.terminator
                c.statements.clear()
                c.terminator = null
                changed = true
            }
        }
    }
}

internal fun reachableBlocks(plan: BodyPlan): List<BlockPlan> {
    val seen = LinkedHashSet<BlockPlan>()
    val work = ArrayDeque<BlockPlan>()
    work += plan.entry
    while (work.isNotEmpty()) {
        val b = work.removeFirst()
        if (!seen.add(b)) continue
        when (val t = b.terminator) {
            is Terminator.Goto -> work += t.target
            is Terminator.CondGoto -> { work += t.thenTarget; work += t.elseTarget }
            is Terminator.SuspendCall -> work += t.resume
            else -> {}
        }
    }
    return seen.toList()
}


// ================================================================ shared plan data model


/** One override participating in a hierarchy plan (null class when the function is not a class member). */
internal class OverrideInfo(
    val irClass: IrClass?,
    val function: IrSimpleFunction,
)

internal class BlockPlan {
    val statements = mutableListOf<IrStatement>()
    var terminator: Terminator? = null
}

internal sealed class Terminator {
    class Goto(val target: BlockPlan) : Terminator()
    class CondGoto(
        val condition: IrExpression,
        val thenTarget: BlockPlan,
        val elseTarget: BlockPlan,
    ) : Terminator()

    /** Target call in return position — state swap, no frame. */
    class TailCall(val call: IrCall) : Terminator()

    /** `v = <target call>` — push frame, resume at [resume] with [resultSymbol] set. */
    class SuspendCall(
        val call: IrCall,
        val resultSymbol: IrValueSymbol,
        val resume: BlockPlan,
    ) : Terminator()

    class Ret(val value: IrExpression) : Terminator()
}

internal class BodyPlan(
    val info: OverrideInfo,
    val blocks: List<BlockPlan>,
    /** Locals declared in the body, in declaration order (frame candidates). */
    val locals: List<IrVariable>,
) {
    val entry: BlockPlan get() = blocks.first()
}

/** Result of [WasmStacklessBodyPlanner.plan]: a usable plan, or the reason there is none. */
internal sealed class PlanResult {
    class Planned(val plan: BodyPlan) : PlanResult()
    class Bailed(val reason: String) : PlanResult()
}

/** Thrown internally to abandon planning for one override (bail out to native). */
internal class BailOut(val reason: String) : RuntimeException()
internal class LoopFrame(val loop: IrLoop, val head: BlockPlan, val exit: BlockPlan)

internal class ReturnableFrame(
    val symbol: IrReturnTargetSymbol,
    val join: BlockPlan,
    val resultTarget: IrValueSymbol? = null,
    /** `return <returnable block>` position: returns to the block are returns of the function. */
    val returnThrough: Boolean = false,
)

internal sealed class CaptureKey {
    object Self : CaptureKey()
    class Local(val symbol: IrValueSymbol) : CaptureKey()
    class Arg(val index: Int) : CaptureKey()
}

internal class Capture(val key: CaptureKey, val pool: IrType, val poolIndex: Int)


internal class WasmStacklessBodyPlanner(
    private val context: WasmBackendContext,
    private val func: IrSimpleFunction,
    private val bodyToPlan: IrBlockBody,
    private val isTarget: (IrCall) -> Boolean,
) {
    private val blocks = mutableListOf<BlockPlan>()
    private val locals = mutableListOf<IrVariable>()
    private val loopStack = ArrayDeque<LoopFrame>()
    private val returnableStack = ArrayDeque<ReturnableFrame>()
    private val builtIns get() = context.irBuiltIns

    /** Well-typed placeholder for return positions that are unreachable by construction. */
    private fun unreachableDefault(): IrExpression =
        IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, func.returnType)

    private fun newBlock(): BlockPlan = BlockPlan().also { blocks += it }

    fun plan(): PlanResult {
        return try {
            inlineLocalFunsWithTargetCalls(bodyToPlan)
            val entry = newBlock()
            val last = compileStatements(bodyToPlan.statements, entry)
            // Bodies whose every path returns or throws leave a block with
            // no terminator; the synthetic Ret is unreachable but must
            // still be well-typed for the function's return type.
            if (last.terminator == null) {
                last.terminator = Terminator.Ret(unreachableDefault())
            }
            PlanResult.Planned(BodyPlan(OverrideInfo(func.parent as? IrClass, func), blocks, locals))
        } catch (b: BailOut) {
            PlanResult.Bailed(b.reason)
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
                // Unmemoized scan: bodies still change while this pass inlines.
                if (scanForTargetCall(declaration)) localFuns += declaration
            }
        })
        if (localFuns.isEmpty()) return

        for (lf in localFuns) {
            if (lf.parameters.isNotEmpty()) throw BailOut("local fun with parameters holds target call")
            val lfBody = lf.body as? IrBlockBody ?: throw BailOut("local fun holding target call has no block body")
            singleTrailingReturnIssue(lfBody, lf.symbol)?.let {
                throw BailOut("local fun holding target call $it")
            }

            body.transformChildrenVoid(object : IrElementTransformerVoid() {
                override fun visitCall(expression: IrCall): IrExpression {
                    expression.transformChildrenVoid(this)
                    if (expression.symbol != lf.symbol) return expression
                    return spliceInlineBody(expression, lfBody.deepCopyWithSymbols(lf), lf.returnType)
                }

                // Drop the now-unused declaration from nested blocks.
                override fun visitBlock(expression: IrBlock): IrExpression {
                    expression.transformChildrenVoid(this)
                    expression.statements.removeAll { it === lf }
                    return expression
                }
            })
            body.statements.removeAll { it === lf }
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
                    val call = init as? IrCall ?: throw BailOut("target call or return nested in variable initializer ${init::class.simpleName}")
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
                        current.statements += stmt.value
                        cur = current
                    }
                    cur.terminator = Terminator.Goto(frame.join)
                    return cur
                }
                val value = stmt.value
                if (containsTargetCall(value)) {
                    when (value) {
                        is IrCall -> {
                            if (!isTarget(value)) throw BailOut("nested target call")
                            checkArgsHaveNoTargetCalls(value)
                            current.terminator = Terminator.TailCall(value)
                            return current
                        }
                        is IrWhen -> {
                            // return when { c1 -> e1; ... }  ==>  when { c1 -> return e1; ... }
                            val pushed = IrWhenImpl(value.startOffset, value.endOffset, builtIns.unitType, value.origin).apply {
                                for (branch in value.branches) {
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
                            if (end.terminator == null) end.terminator = Terminator.Ret(unreachableDefault())
                            return end
                        }
                        is IrBlock -> {
                            if (value is IrReturnableBlock) {
                                // return <RB>: every return@RB v is a plain return v.
                                val join = newBlock()
                                returnableStack.addLast(ReturnableFrame(value.symbol, join, null, returnThrough = true))
                                val end = compileStatements(value.statements, current)
                                returnableStack.removeLast()
                                if (end.terminator == null) throw BailOut("returned returnable block falls through")
                                return join
                            }
                            // return block { s1..sn; last } ==> s1..sn; return last
                            val stmts = value.statements
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
                        else -> throw BailOut("target call nested in return value ${value::class.simpleName}")
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
            val isElse = isElseBranch(branch)
            val bodyBlock = newBlock()
            if (isElse) {
                condBlock.terminator = Terminator.Goto(bodyBlock)
            } else {
                val next = newBlock()
                condBlock.terminator = Terminator.CondGoto(branch.condition, bodyBlock, next)
                condBlock = next
            }
            val bodyEnd = compileStatement(branch.result, bodyBlock)
            if (bodyEnd.terminator == null) bodyEnd.terminator = Terminator.Goto(join)
            if (isElse) {
                return join
            }
        }
        // No else branch: fall through from the last condition block.
        condBlock.terminator = Terminator.Goto(join)
        return join
    }

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

    private fun checkArgsHaveNoTargetCalls(call: IrCall) {
        for (arg in call.arguments) {
            if (arg != null && containsTargetCall(arg)) throw BailOut("target call in argument of target call")
        }
    }

    /** Any statement holding a target call, an outer return, or a jump to
     *  a loop that is being split must itself be compiled into block
     *  structure, so those transfers route through the state machine.
     *  All conditions are checked in a single traversal; nested functions
     *  are opaque except for the target calls they hold (they are analyzed
     *  separately and cannot be split here). */
    private fun needsSplit(element: IrElement): Boolean {
        var found = false
        element.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }

            override fun visitFunction(declaration: IrFunction) {
                if (hasTargetCallInside(declaration)) found = true
            }

            override fun visitCall(expression: IrCall) {
                if (isTarget(expression)) found = true else expression.acceptChildrenVoid(this)
            }

            override fun visitReturn(expression: IrReturn) {
                if (expression.returnTargetSymbol == func.symbol ||
                    returnableStack.any { it.symbol == expression.returnTargetSymbol }
                ) {
                    found = true
                } else {
                    expression.acceptChildrenVoid(this)
                }
            }

            override fun visitBreakContinue(jump: IrBreakContinue) {
                if (loopStack.any { it.loop == jump.loop }) found = true
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
                if (isTarget(expression)) found = true else expression.acceptChildrenVoid(this)
            }
        })
        return found
    }

    private val localFunContainsTarget = HashMap<IrFunction, Boolean>()

    /**
     * Memoized: queried once per enclosing container level while compiling
     * the surrounding statements, and local-fun bodies no longer change
     * after [inlineLocalFunsWithTargetCalls] has run (which is why that
     * pass scans without the cache).
     */
    private fun hasTargetCallInside(declaration: IrFunction): Boolean =
        localFunContainsTarget.getOrPut(declaration) { scanForTargetCall(declaration) }

    private fun scanForTargetCall(declaration: IrFunction): Boolean = anyCall(declaration, predicate = isTarget)
}

/**
 * Whether any [IrCall] under [root] satisfies [predicate], short-circuiting
 * on the first hit. With [intoNestedFunctions] false, nested function
 * declarations are opaque.
 */
internal fun anyCall(root: IrElement, intoNestedFunctions: Boolean = true, predicate: (IrCall) -> Boolean): Boolean {
    var found = false
    root.acceptVoid(object : IrVisitorVoid() {
        override fun visitElement(element: IrElement) {
            if (!found) element.acceptChildrenVoid(this)
        }

        override fun visitFunction(declaration: IrFunction) {
            if (intoNestedFunctions && !found) declaration.acceptChildrenVoid(this)
        }

        override fun visitCall(expression: IrCall) {
            if (predicate(expression)) found = true else expression.acceptChildrenVoid(this)
        }
    })
    return found
}

/** Null when [body] ends in exactly one return targeting [symbol]; otherwise the reason it does not. */
internal fun singleTrailingReturnIssue(body: IrBlockBody, symbol: IrReturnTargetSymbol): String? {
    val last = body.statements.lastOrNull()
    if (last !is IrReturn || last.returnTargetSymbol != symbol) return "lacks trailing return"
    var returns = 0
    body.acceptVoid(object : IrVisitorVoid() {
        override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
        override fun visitReturn(expression: IrReturn) {
            if (expression.returnTargetSymbol == symbol) returns++
            expression.acceptChildrenVoid(this)
        }
    })
    if (returns != 1) return "has multiple returns"
    return null
}

/**
 * Splices a copied single-trailing-return body in place of [call]:
 * the statements run first, then the returned value is the block result.
 */
internal fun spliceInlineBody(call: IrCall, copied: IrBlockBody, resultType: IrType): IrBlock {
    val stmts = copied.statements.toMutableList()
    val ret = stmts.removeLast() as IrReturn
    return IrBlockImpl(call.startOffset, call.endOffset, resultType, null, stmts + ret.value)
}

// ================================================================ frame codegen helpers
//
// The pieces below are the consumer-invariant halves of the code
// generators in WasmDrfAccelerationLowering and WasmStacklessMatcherLowering:
// pool classification, frame class synthesis, state numbering, and the
// push/restore/pop protocol. Slot assignment and ctor argument order are
// the ABI between push and restore, so they must live in exactly one place.

/**
 * Pool type for a frame slot: the six unboxed primitives keep their own
 * pools; everything else shares the nullable-any reference pool.
 */
internal fun framePoolOf(builtIns: IrBuiltIns, type: IrType): IrType = when (type) {
    builtIns.intType, builtIns.booleanType, builtIns.charType,
    builtIns.longType, builtIns.floatType, builtIns.doubleType -> type
    else -> builtIns.anyNType
}

/** Pool emission order is fixed so the frame ctor argument layout is stable. */
internal fun framePoolOrder(builtIns: IrBuiltIns): List<IrType> = listOf(
    builtIns.intType, builtIns.booleanType, builtIns.charType,
    builtIns.longType, builtIns.floatType, builtIns.doubleType, builtIns.anyNType,
)

/** Assigns each seeded value a frame slot ([Capture.pool] plus [Capture.poolIndex]) in seed order. */
internal fun buildCaptures(builtIns: IrBuiltIns, seeds: List<Pair<CaptureKey, IrType>>): List<Capture> {
    val counters = mutableMapOf<IrType, Int>()
    return seeds.map { seed ->
        val pool = framePoolOf(builtIns, seed.second)
        val idx = counters.getOrElse(pool) { 0 }
        counters[pool] = idx + 1
        Capture(seed.first, pool, idx)
    }
}

internal class FrameLayout(
    val cls: IrClass,
    val outerField: IrField,
    val resumeField: IrField,
    val poolFields: Map<IrType, List<IrField>>,
    /** (pool, poolIndex) per ctor argument, after the leading outer and resume arguments. */
    val ctorParamOrder: List<Pair<IrType, Int>>,
    val ctor: IrConstructorSymbol,
)

/**
 * Synthesizes the heap frame class: a linked-list node holding the outer
 * frame, the resume state, and one field per pool slot.
 */
internal fun buildFrameClass(
    context: WasmBackendContext,
    irFile: IrFile,
    className: String,
    poolSizes: Map<IrType, Int>,
    origin: IrDeclarationOrigin,
): FrameLayout {
    val builtIns = context.irBuiltIns
    val poolOrder = framePoolOrder(builtIns)
    val cls = context.irFactory.buildClass {
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
        name = Name.identifier(className)
        visibility = DescriptorVisibilities.PRIVATE
        modality = Modality.FINAL
        kind = ClassKind.CLASS
        this.origin = origin
    }.apply {
        parent = irFile
        irFile.declarations += this
        createThisReceiverParameter()
        superTypes = listOf(builtIns.anyType)
    }
    fun prefix(pool: IrType): String = when (pool) {
        builtIns.intType -> "i"; builtIns.booleanType -> "z"; builtIns.charType -> "c"
        builtIns.longType -> "j"; builtIns.floatType -> "f"; builtIns.doubleType -> "d"
        else -> "r"
    }
    val fieldDefs = buildList {
        add("outer" to cls.defaultType.makeNullable())
        add("resume" to builtIns.intType)
        for (pool in poolOrder) repeat(poolSizes[pool] ?: 0) { add("${prefix(pool)}$it" to pool) }
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
    val byPool = mutableMapOf<IrType, MutableList<IrField>>()
    val order = mutableListOf<Pair<IrType, Int>>()
    var fi = 2
    for (pool in poolOrder) {
        val list = mutableListOf<IrField>()
        repeat(poolSizes[pool] ?: 0) { k -> list += fields[fi++]; order += pool to k }
        byPool[pool] = list
    }
    val ctor = cls.addConstructor {
        isPrimary = true
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
    }.apply {
        val params = fieldDefs.map { def ->
            addValueParameter {
                name = Name.identifier(def.first); type = def.second
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            }
        }
        body = context.createIrBuilder(symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET).irBlockBody {
            +irDelegatingConstructorCall(builtIns.anyClass.owner.constructors.single())
            +IrInstanceInitializerCallImpl(UNDEFINED_OFFSET, UNDEFINED_OFFSET, cls.symbol, builtIns.unitType)
            for (i in fields.indices) +irSetField(irGet(cls.thisReceiver!!), fields[i], irGet(params[i]))
        }
    }.symbol
    return FrameLayout(cls, fields[0], fields[1], byPool, order, ctor)
}

/** Dispatch state reserved for the frame-pop (APPLY) branch; block states start at 1. */
internal const val STATE_APPLY = 0

internal class StateAssignment(
    val stateIds: Map<BlockPlan, Int>,
    val resumeInfo: Map<BlockPlan, Terminator.SuspendCall>,
)

/**
 * Simplifies each plan and numbers its reachable blocks into one flat
 * state space starting at 1: [STATE_APPLY] is reserved for the frame-pop
 * dispatch branch.
 */
internal fun assignStates(plans: List<BodyPlan>): StateAssignment {
    val stateIds = mutableMapOf<BlockPlan, Int>()
    val resumeInfo = mutableMapOf<BlockPlan, Terminator.SuspendCall>()
    var next = 1
    for (plan in plans) {
        simplifyPlan(plan)
        val live = reachableBlocks(plan)
        for (block in live) stateIds[block] = next++
        for (block in live) {
            val t = block.terminator
            if (t is Terminator.SuspendCall) resumeInfo[t.resume] = t
        }
    }
    return StateAssignment(stateIds, resumeInfo)
}

/** Static int depth counter backing the hybrid native/trampoline split. */
internal fun buildDepthField(context: WasmBackendContext, irFile: IrFile, fieldName: String, origin: IrDeclarationOrigin): IrField =
    context.irFactory.buildField {
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
        name = Name.identifier(fieldName)
        type = context.irBuiltIns.intType
        visibility = DescriptorVisibilities.PRIVATE
        isStatic = true
        isFinal = false
        this.origin = origin
    }.apply {
        parent = irFile
        irFile.declarations += this
        initializer = context.irFactory.createExpressionBody(
            IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, context.irBuiltIns.intType, 0)
        )
    }

/**
 * The bracket around every recursive call site under the hybrid strategy:
 * `if (depth < threshold) { depth++; val r = shallow(); depth--; r } else deep()`.
 */
internal fun IrBuilderWithScope.irHybridDepthGuard(
    wasmContext: WasmBackendContext,
    depthField: IrField,
    resultType: IrType,
    shallowCall: () -> IrExpression,
    deepCall: () -> IrExpression,
): IrExpression {
    val builtIns = wasmContext.irBuiltIns
    fun depthGet() = irGetField(null, depthField)
    fun depthAdd(delta: Int) = irSetField(null, depthField, irCall(builtIns.intPlusSymbol).apply {
        arguments[0] = depthGet(); arguments[1] = irInt(delta)
    })
    return irIfThenElse(
        resultType,
        irCall(builtIns.lessFunByOperandType[builtIns.intClass]!!).apply {
            arguments[0] = depthGet()
            arguments[1] = irInt(STACKLESS_HYBRID_DEPTH_THRESHOLD)
        },
        irBlock(resultType = resultType) {
            +depthAdd(1)
            val r = irTemporary(shallowCall(), nameHint = "dres")
            +depthAdd(-1)
            +irGet(r)
        },
        deepCall(),
    )
}

/** i32 equality via the wasm intrinsic, shaped for the br_table when-optimizer. */
internal fun IrBuilderWithScope.irIntEquals(wasmContext: WasmBackendContext, left: IrExpression, right: IrExpression): IrExpression =
    irCall(wasmContext.wasmSymbols.equalityFunctions.getValue(wasmContext.irBuiltIns.intType)).apply {
        arguments[0] = left
        arguments[1] = right
    }

/**
 * Pushes a new frame: ctor arguments are outer and resume, then one value
 * per slot in [FrameLayout.ctorParamOrder]. Slots the resuming plan does
 * not capture (possible when several plans share one frame class) are
 * default-filled.
 */
internal fun IrBlockBuilder.emitFramePush(
    layout: FrameLayout,
    captures: List<Capture>,
    vTop: IrValueDeclaration,
    resumeId: Int,
    resolve: (CaptureKey) -> IrValueDeclaration,
) {
    val captureBySlot = captures.associateBy { it.pool to it.poolIndex }
    +irSet(vTop.symbol, irCallConstructor(layout.ctor, emptyList()).apply {
        arguments[0] = irGet(vTop)
        arguments[1] = irInt(resumeId)
        for (ci in layout.ctorParamOrder.indices) {
            val slot = layout.ctorParamOrder[ci]
            val capture = captureBySlot[slot]
            arguments[ci + 2] =
                if (capture == null) IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, slot.first)
                else irGet(resolve(capture.key))
        }
    })
}

/**
 * Cast for moving a value from a slot of static type [exprType] into a slot
 * of [targetType]. Reference values may legitimately be null in transit
 * (locals captured before their first assignment, results not yet
 * produced), so reference casts go through the nullable type and never
 * null-check.
 */
internal fun IrBuilderWithScope.irCastForSlot(
    builtIns: IrBuiltIns,
    expr: IrExpression,
    exprType: IrType,
    targetType: IrType,
): IrExpression = when {
    targetType == exprType -> expr
    targetType == builtIns.anyNType -> expr
    !targetType.isPrimitiveType() -> irAs(expr, targetType.makeNullable())
    else -> irAs(expr, targetType)
}

/** Restores every capture of the resuming plan from the current frame, downcasting reference-pool slots. */
internal fun IrBlockBuilder.emitFrameRestore(
    wasmContext: WasmBackendContext,
    layout: FrameLayout,
    captures: List<Capture>,
    vFrame: IrValueDeclaration,
    resolve: (CaptureKey) -> IrValueDeclaration,
) {
    val builtIns = wasmContext.irBuiltIns
    for (capture in captures) {
        val target = resolve(capture.key)
        val field = layout.poolFields.getValue(capture.pool)[capture.poolIndex]
        val read = irGetField(irGet(vFrame), field)
        +irSet(target.symbol, irCastForSlot(builtIns, read, capture.pool, target.type))
    }
}

/** State 0: pop a frame and resume its recorded state, or return [returnValue] when the stack is empty. */
internal fun IrBlockBuilder.emitFramePop(
    layout: FrameLayout,
    vTop: IrValueDeclaration,
    vFrame: IrValueDeclaration,
    vState: IrValueDeclaration,
    loop: IrLoop,
    returnValue: IrExpression,
) {
    val fTmp = irTemporary(irGet(vTop), nameHint = "f")
    +irIfThen(context.irBuiltIns.unitType, irEqualsNull(irGet(fTmp)), irReturn(returnValue))
    +irSet(vTop.symbol, irGetField(irGet(fTmp), layout.outerField))
    +irSet(vFrame.symbol, irGet(fTmp))
    +irSet(vState.symbol, irGetField(irGet(fTmp), layout.resumeField))
    +irContinue(loop)
}

/**
 * Emits a block's straight-line statements into the dispatch loop:
 * hoisted local declarations become assignments to their pre-declared
 * slots, everything else is emitted remapped.
 */
internal fun IrBlockBuilder.emitBlockStatements(
    block: BlockPlan,
    remap: (IrElement) -> IrElement,
    resolveLocal: (IrValueSymbol) -> IrValueDeclaration,
) {
    for (stmt in block.statements) {
        when (stmt) {
            is IrVariable -> {
                val target = resolveLocal(stmt.symbol)
                stmt.initializer?.let { init -> +irSet(target.symbol, remap(init) as IrExpression) }
            }
            else -> +(remap(stmt) as IrStatement)
        }
    }
}

/**
 * Emits the consumer-invariant terminators of the dispatch loop. The two
 * transfer terminators stay with the caller: their semantics (argument
 * swap versus receiver swap and dynamic entry-state dispatch) belong to
 * the consumer.
 */
internal fun IrBlockBuilder.emitTerminator(
    t: Terminator,
    stateIds: Map<BlockPlan, Int>,
    vState: IrValueDeclaration,
    vRet: IrValueDeclaration,
    loop: IrLoop,
    remap: (IrElement) -> IrElement,
    emitTailCall: IrBlockBuilder.(Terminator.TailCall) -> Unit,
    emitSuspendCall: IrBlockBuilder.(Terminator.SuspendCall) -> Unit,
) {
    when (t) {
        is Terminator.Goto -> {
            +irSet(vState.symbol, irInt(stateIds.getValue(t.target)))
            +irContinue(loop)
        }
        is Terminator.CondGoto -> {
            +irIfThenElse(
                context.irBuiltIns.unitType,
                remap(t.condition) as IrExpression,
                irBlock { +irSet(vState.symbol, irInt(stateIds.getValue(t.thenTarget))); +irContinue(loop) },
                irBlock { +irSet(vState.symbol, irInt(stateIds.getValue(t.elseTarget))); +irContinue(loop) },
            )
        }
        is Terminator.Ret -> {
            +irSet(vRet.symbol, remap(t.value) as IrExpression)
            +irSet(vState.symbol, irInt(STATE_APPLY))
            +irContinue(loop)
        }
        is Terminator.TailCall -> emitTailCall(t)
        is Terminator.SuspendCall -> emitSuspendCall(t)
    }
}

