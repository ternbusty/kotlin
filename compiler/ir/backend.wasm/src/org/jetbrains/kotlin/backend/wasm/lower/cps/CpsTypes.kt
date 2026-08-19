/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.cps

import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.declarations.IrVariable
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.symbols.IrReturnTargetSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.expressions.IrLoop

/**
 * A basic block of the state machine: straight-line statements followed
 * by exactly one terminator. Statements reference the ORIGINAL function's
 * value symbols; remapping happens at codegen.
 */
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
    val entry: BlockPlan,
    val blocks: List<BlockPlan>,
    /** Locals declared in the body, in declaration order (frame candidates). */
    val locals: List<IrVariable>,
)

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

internal class Capture(val key: CaptureKey, val declaredType: IrType, val pool: IrType, val poolIndex: Int)
