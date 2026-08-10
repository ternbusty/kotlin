/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.virtualcps

import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.expressions.IrCall

/**
 * Simplifies a [BodyPlan] by resolving empty goto chains and merging
 * single-predecessor blocks into their predecessor.
 *
 * This is a fixpoint algorithm that repeatedly:
 * 1. Resolves goto targets through empty blocks that are neither resume
 *    targets nor the entry block.
 * 2. Merges a block into its sole predecessor when the predecessor ends
 *    with a [Terminator.Goto] and the target has exactly one reference.
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

/**
 * Returns the blocks reachable from the entry of [plan] in BFS order.
 *
 * Unreachable blocks (dead code after simplification) are excluded from
 * the returned list.
 */
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

/**
 * Collects all functions transitively overridden by [this], including
 * [this] itself.
 */
internal fun IrSimpleFunction.allOverriddenIncludingSelf(): List<IrSimpleFunction> {
    val seen = mutableSetOf<IrSimpleFunction>()
    fun walk(fn: IrSimpleFunction) {
        if (!seen.add(fn)) return
        for (s in fn.overriddenSymbols) walk(s.owner)
    }
    walk(this)
    return seen.toList()
}

/** Is [call] a virtual dispatch of the target [baseMethod]? */
internal fun isTargetCall(call: IrCall, baseMethod: IrSimpleFunction): Boolean {
    val callee = call.symbol.owner
    if (callee.name != baseMethod.name) return false
    if (callee == baseMethod) return true
    return callee.allOverriddenIncludingSelf().any { it == baseMethod }
}
