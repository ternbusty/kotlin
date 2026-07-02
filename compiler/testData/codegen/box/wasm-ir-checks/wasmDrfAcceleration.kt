// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Exercises the DeepRecursiveFunction acceleration in WasmVirtualCpsLowering.
// The property-held literal and the direct-invoke literal with free-variable
// captures are both rewritten to a native twin plus a heap-frame trampoline.
// Depths beyond the hybrid threshold (512) run through the trampoline, so a
// miscompiled frame slot or restore path fails the value checks below.

class Node(val value: Int, val next: Node?)

val sum = DeepRecursiveFunction<Node?, Int> { n ->
    if (n == null) 0 else n.value + callRecursive(n.next)
}

class Tree(val left: Tree?, val right: Tree?)

val depth = DeepRecursiveFunction<Tree?, Int> { t ->
    if (t == null) 0
    else maxOf(callRecursive(t.left), callRecursive(t.right)) + 1
}

class Reader(val tokens: IntArray) {
    var pos = 0

    fun readDeep(): Int {
        val base = pos
        return DeepRecursiveFunction<Unit, Int> {
            // Captures `this@Reader` state through free variables.
            if (pos < tokens.size && tokens[pos] == 1) {
                pos++
                var children = 0
                while (pos < tokens.size && tokens[pos] != 2) {
                    children += callRecursive(Unit)
                }
                pos++
                children + 1
            } else {
                pos++
                1
            }
        }.invoke(Unit).also { check(pos > base) }
    }
}

fun box(): String {
    var list: Node? = null
    for (i in 1..10_000) list = Node(i, list)
    if (sum(list) != 50_005_000) return "fail sum"

    var spine: Tree? = null
    for (i in 1..5_000) spine = Tree(spine, null)
    if (depth(spine) != 5_000) return "fail depth"

    // Nested unary tree: 1 1 1 ... 0 2 2 2 ... (depth 2000)
    val d = 2_000
    val tokens = IntArray(d * 2 + 1)
    for (i in 0 until d) tokens[i] = 1
    tokens[d] = 0
    for (i in d + 1..d * 2) tokens[i] = 2
    if (Reader(tokens).readDeep() != d + 1) return "fail reader"

    return "OK"
}
