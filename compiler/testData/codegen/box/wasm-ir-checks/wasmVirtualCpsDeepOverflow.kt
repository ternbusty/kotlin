// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Verifies that the CPS trampoline handles depth 100,000 without
// stack overflow. Without CPS (or tail calls), this chain overflows
// because each eval() call pushes a wasm stack frame.

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
    val result = chain.eval(0)
    // Add nodes: indices 0, 3, 6, ... → depth/3 nodes if depth%3==0
    // Each Add has delta=1, Mul has factor=1
    // chain evaluation: Add(+1) -> Mul(*1) -> Identity -> Add(+1) -> ...
    // Only Add nodes change the accumulator, each adding 1
    // Number of Add nodes = ceil(depth / 3)
    val expectedAdds = (depth + 2) / 3
    if (result != expectedAdds) return "fail: result=$result expected=$expectedAdds"
    return "OK"
}
