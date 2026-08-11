// TARGET_BACKEND: WASM

// WITHOUT tail calls or stackless recursion, this chain overflows.
// This test exists to prove the overflow occurs, complementing
// wasmVirtualCpsDeepOverflow.kt which shows CPS surviving the same depth.
// Expected: stack overflow (RangeError in Node, trap in Wasmtime).

abstract class Node {
    abstract fun eval(acc: Int): Int
}

class Add(val delta: Int, val next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc + delta)
}

class Mul(val factor: Int, val next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc * factor)
}

class Identity(val next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc)
}

class End : Node() {
    override fun eval(acc: Int): Int = acc
}

fun buildChain(n: Int): Node {
    var head: Node = End()
    for (i in 0 until n) {
        head = when (i % 3) {
            0 -> Add(1, head)
            1 -> Mul(1, head)
            else -> Identity(head)
        }
    }
    return head
}

fun box(): String {
    val depth = 100_000
    val chain = buildChain(depth)
    return try {
        chain.eval(0)
        "fail: expected stack overflow at depth $depth"
    } catch (e: Throwable) {
        "OK"
    }
}
