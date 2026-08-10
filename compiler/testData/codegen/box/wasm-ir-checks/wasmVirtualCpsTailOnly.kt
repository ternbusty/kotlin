// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// All virtual calls are in tail position, so the trampoline uses receiver
// swaps without heap-frame allocation. Models a linked-list evaluator where
// each node delegates to the next via a tail virtual call.

abstract class Node {
    abstract fun eval(acc: Int): Int
}

class AddNode(val value: Int, var next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc + value)
}

class MulNode(val factor: Int, var next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc * factor)
}

class IdentityNode(var next: Node) : Node() {
    override fun eval(acc: Int): Int = next.eval(acc)
}

class EndNode : Node() {
    override fun eval(acc: Int): Int = acc
}

fun buildChain(n: Int): Node {
    val end = EndNode()
    var head: Node = end
    for (i in n downTo 1) {
        head = when (i % 3) {
            0 -> MulNode(1, head)
            1 -> AddNode(1, head)
            else -> IdentityNode(head)
        }
    }
    return head
}

fun box(): String {
    // Shallow correctness
    val end = EndNode()
    val chain = AddNode(10, MulNode(2, AddNode(3, end)))
    // eval(0) -> 0+10=10 -> 10*2=20 -> 20+3=23
    if (chain.eval(0) != 23) return "fail: shallow eval ${chain.eval(0)}"

    // Deep: crashes without the virtual CPS trampoline
    val n = 500_000
    val deep = buildChain(n)
    // Each AddNode adds 1, each MulNode multiplies by 1, IdentityNode passes through.
    // There are ceil(n/3) AddNodes among n nodes, each adding 1.
    val expectedAdds = (n + 2) / 3
    val result = deep.eval(0)
    if (result != expectedAdds) return "fail: deep eval $result != $expectedAdds"

    return "OK"
}
