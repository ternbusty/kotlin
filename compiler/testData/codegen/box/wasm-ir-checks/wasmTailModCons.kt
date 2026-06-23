// TARGET_BACKEND: WASM

// Tail Modulo Cons: a recursive call whose result is immediately wrapped in a constructor is not
// in tail position, so it consumes stack proportional to the recursion depth. WasmTailModConsLowering
// rewrites these into destination-passing form whose recursive calls ARE in tail position, so they
// run in constant stack. At depth 500_000 these patterns would overflow the host stack without the
// transform.

class Cell(val value: Int, val next: Cell?)

// Single-function self recursion: `return Cell(n, chain(n - 1))`.
fun chain(n: Int): Cell? {
    if (n == 0) return null
    return Cell(n, chain(n - 1))
}

sealed interface IList
class ConsA(val head: Int, val tail: IList?) : IList
class ConsB(val head: Int, val tail: IList?) : IList

// Two-function mutual recursion, each wrapping the other's result in a constructor.
fun mutualA(n: Int): IList? {
    if (n == 0) return null
    return ConsA(n, mutualB(n - 1))
}

fun mutualB(n: Int): IList? {
    if (n == 0) return null
    return ConsB(n, mutualA(n - 1))
}

fun lengthOfChain(c: Cell?): Int {
    var k = 0
    var cur = c
    while (cur != null) {
        k++
        cur = cur.next
    }
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

    if (lengthOfChain(chain(depth)) != depth) return "fail chain"
    if (lengthOfIList(mutualA(depth)) != depth) return "fail mutual"

    return "OK"
}
