/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.ModuleLoweringPass
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.expressions.impl.IrBlockImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrBranchImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrWhenImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrInstanceInitializerCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrSetValueImpl
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.*
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.util.*
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid

/**
 * Defunctionalized CPS conversion for VIRTUAL mutual recursion over a closed
 * class hierarchy (CHA-based).
 *
 * Motivation: the stdlib regex matcher (kotlin.text.regex.AbstractSet.matches)
 * recurses through virtual dispatch over the compiled pattern's object graph.
 * Recursion depth is proportional to input length, causing stack overflow on
 * real-world inputs (KT-63689, KT-61542, KT-78089). The recursion is neither
 * tail-call-shaped nor TMC-shaped (post-call state restores and result
 * inspection), so neither return_call emission nor WasmTailModConsLowering
 * rescues it. This pass converts the whole hierarchy's overrides into a
 * single trampoline function with heap-allocated frames.
 *
 * Design (see .ai/gsoc-assets/virtual-cps-design.md):
 *  - CHA enumerates all overrides of the target base method in the module
 *    (sound under wasm whole-world compilation).
 *  - Each override body is compiled into a basic-block plan, splitting at
 *    virtual call sites of the target method:
 *      * calls in tail position    -> receiver/state swap, no frame
 *      * non-tail calls            -> heap frame capturing live locals
 *    Unsupported constructs bail out: that override keeps its native body
 *    and the trampoline invokes it as an ordinary virtual call (partial
 *    conversion is always semantics-preserving).
 *  - A single `run$virtualCps` function holds the flat state machine:
 *    while(true) + when(state), state = (override, block) pairs.
 *
 * PoC scoping: the target hierarchy is currently selected by
 * [isTargetBaseClass]; this becomes a compiler flag before upstreaming.
 */
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
 * trampoline. Far below the wasm stack guard (~10-15K frames) even
 * with several frames per level and an already-deep caller stack, and
 * far above what everyday patterns reach.
 */
internal const val STACKLESS_HYBRID_DEPTH_THRESHOLD = 512

// ---------------- block simplification (shared by matcher and DRF codegens)

/**
 * Collapses trivial control flow the planner produces in numbers:
 * empty forwarder blocks are skipped and single-predecessor Goto
 * successors are fused into their predecessor. Fewer states means
 * fewer dispatch round-trips per matcher step.
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
            if (b.terminator == null) continue
            val t = b.terminator
            if (t is Terminator.Goto) {
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


// ================================================================ DeepRecursiveFunction acceleration


/** One override participating in a hierarchy plan (null class for synthetic roots). */
internal class OverrideInfo(
    val irClass: IrClass?,
    val function: IrSimpleFunction,
)

internal class BlockPlan(val id: Int) {
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

    /** `return recv.matches(args)` — receiver/args swap, no frame. */
    class TailCall(val call: IrCall) : Terminator()

    /** `v = recv.matches(args)` — push frame, resume at [resume] with [resultSymbol] set. */
    class SuspendCall(
        val call: IrCall,
        val resultSymbol: IrValueSymbol,
        val resume: BlockPlan,
    ) : Terminator()

    class Ret(val value: IrExpression) : Terminator()
}

internal class BodyPlan(
    val info: OverrideInfo,
    val entry: BlockPlan,
    val blocks: List<BlockPlan>,
    /** Locals declared in the body, in declaration order (frame candidates). */
    val locals: List<IrVariable>,
)

/** Thrown internally to abandon planning for one override (bail out to native). */
internal class BailOut(val reason: String) : RuntimeException()
internal class LoopFrame(val loop: IrLoop, val head: BlockPlan, val exit: BlockPlan)

internal class ReturnableFrame(
    val symbol: org.jetbrains.kotlin.ir.symbols.IrReturnTargetSymbol,
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

internal class Capture(val key: CaptureKey, val declaredType: IrType, val pool: IrType, val poolIndex: Int)


internal class WasmStacklessBodyPlanner(
    private val context: WasmBackendContext,
    private val func: IrSimpleFunction,
    private val bodyToPlan: IrBlockBody,
    private val isTarget: (IrCall) -> Boolean,
) {
    /** Why the last plan() returned null; for the lowerings' bail-out reporting. */
    var lastBailReason: String = ""
        private set

    private val blocks = mutableListOf<BlockPlan>()
    private val locals = mutableListOf<IrVariable>()
    private val loopStack = ArrayDeque<LoopFrame>()
    private val returnableStack = ArrayDeque<ReturnableFrame>()
    private val builtIns get() = context.irBuiltIns

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
            BodyPlan(OverrideInfo(func.parent as? IrClass, func), blocks.first(), blocks, locals)
        } catch (b: BailOut) {
            lastBailReason = b.reason
            null
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
                        current.statements += stmt.value.asDiscardedStatement()
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
                            if (end.terminator == null) end.terminator = Terminator.Ret(value.asUnreachableDefault())
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

// ================================================================ hierarchy lowering

