// TARGET_BACKEND: WASM

// The DPS helper's self-call must be return_call even though no tail-call
// flag is set: the helper only exists for @TailModCons functions and relies
// on it for bounded stack usage.
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=chain$tmcDps

// Tail Modulo Cons: a recursive call whose result is immediately wrapped in a constructor is not
// in tail position, so it consumes stack proportional to the recursion depth. WasmTailModConsLowering
// rewrites these into destination-passing form whose recursive calls ARE in tail position, so they
// run in constant stack. At depth 500_000 these patterns would overflow the host stack without the
// transform.

import kotlin.wasm.TailModCons

class Cell(val value: Int, val next: Cell?)

// Single-function self recursion: `return Cell(n, chain(n - 1))`.
@TailModCons
fun chain(n: Int): Cell? {
    if (n == 0) return null
    return Cell(n, chain(n - 1))
}

sealed interface IList
class ConsA(val head: Int, val tail: IList?) : IList
class ConsB(val head: Int, val tail: IList?) : IList

// Two-function mutual recursion, each wrapping the other's result in a constructor.
@TailModCons
fun mutualA(n: Int): IList? {
    if (n == 0) return null
    return ConsA(n, mutualB(n - 1))
}

@TailModCons
fun mutualB(n: Int): IList? {
    if (n == 0) return null
    return ConsB(n, mutualA(n - 1))
}

// Multi-parameter with 3-arg constructor and non-Int types.
class Triple(val tag: String, val value: Int, val next: Triple?)

@TailModCons
fun buildTriple(tag: String, n: Int): Triple? {
    if (n == 0) return null
    return Triple(tag, n, buildTriple(tag, n - 1))
}

// Recursive arg NOT at the last position (first arg is recursive).
class RevCell(val prev: RevCell?, val value: Int)

@TailModCons
fun revChain(n: Int): RevCell? {
    if (n == 0) return null
    return RevCell(revChain(n - 1), n)
}

// Saved variable between the base case check and the TMC return.
// The body structure is: [IrWhen, IrVariable, IrReturn Ctor(var, ..., recCall)].
class Computed(val hash: Int, val next: Computed?)

@TailModCons
fun buildComputed(n: Int): Computed? {
    if (n == 0) return null
    val hash = n * 31 + 17
    return Computed(hash, buildComputed(n - 1))
}

// Via-variable pattern: the recursive call is stored in a local before the constructor.
class ViaVar(val value: Int, val next: ViaVar?)

@TailModCons
fun buildViaVar(n: Int): ViaVar? {
    if (n == 0) return null
    val child = buildViaVar(n - 1)
    return ViaVar(n, child)
}

// Expression-body `return when { ... }`: the return wraps an IrWhen.
// normalizeReturnWhen distributes the return into each branch.
class WhenCell(val value: Int, val next: WhenCell?)

@TailModCons
fun buildWhenList(n: Int): WhenCell? = when {
    n <= 0 -> null
    else -> WhenCell(n, buildWhenList(n - 1))
}

// Expression-body `return if (...) ... else ...` with saved variable.
class IfCell(val label: Int, val next: IfCell?)

@TailModCons
fun buildIfList(n: Int): IfCell? {
    val label = n * 7 + 3
    return if (n <= 0) null else IfCell(label, buildIfList(n - 1))
}

fun <T> lengthOf(head: T?, next: (T) -> T?): Int {
    var k = 0
    var cur = head
    while (cur != null) { k++; cur = next(cur) }
    return k
}

fun lengthOfIList(c: IList?): Int {
    var k = 0
    var cur = c
    while (cur != null) {
        k++
        cur = when (cur) {
            is ConsA -> cur.tail
            is ConsB -> cur.tail
        }
    }
    return k
}

fun box(): String {
    val depth = 500_000

    if (lengthOf(chain(depth)) { it.next } != depth) return "fail chain"
    if (lengthOfIList(mutualA(depth)) != depth) return "fail mutual"
    if (lengthOf(buildTriple("x", depth)) { it.next } != depth) return "fail triple"
    if (lengthOf(revChain(depth)) { it.prev } != depth) return "fail revCell"
    val computed = buildComputed(depth)
    if (lengthOf(computed) { it.next } != depth) return "fail computed length"
    if (computed!!.hash != depth * 31 + 17) return "fail computed hash"
    if (lengthOf(buildViaVar(depth)) { it.next } != depth) return "fail viaVar"
    if (lengthOf(buildWhenList(depth)) { it.next } != depth) return "fail whenList"
    val ifList = buildIfList(depth)
    if (lengthOf(ifList) { it.next } != depth) return "fail ifList length"
    if (ifList!!.label != depth * 7 + 3) return "fail ifList label"

    return "OK"
}
