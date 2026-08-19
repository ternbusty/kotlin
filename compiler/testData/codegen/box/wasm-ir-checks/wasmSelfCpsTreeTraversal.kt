// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Verifies that the self-recursion CPS transform handles concrete class
// self-recursion at depth 100,000 without stack overflow. This models
// the compose-rich-editor RichSpan tree traversal pattern: a concrete
// class method calling itself on child nodes inside a loop.

class TreeNode(val value: Int, val children: MutableList<TreeNode> = mutableListOf()) {
    fun sum(): Int {
        var total = value
        for (child in children) {
            total += child.sum()
        }
        return total
    }

    fun depth(): Int {
        var maxChildDepth = 0
        for (child in children) {
            val d = child.depth()
            if (d > maxChildDepth) maxChildDepth = d
        }
        return maxChildDepth + 1
    }

    fun removeEmpty() {
        val iter = children.iterator()
        while (iter.hasNext()) {
            val child = iter.next()
            child.removeEmpty()
            if (child.value == 0 && child.children.isEmpty()) {
                iter.remove()
            }
        }
    }
}

fun box(): String {
    // Shallow: verify correctness at small depths
    val root = TreeNode(1, mutableListOf(
        TreeNode(2, mutableListOf(TreeNode(3))),
        TreeNode(4),
    ))
    val s = root.sum()
    if (s != 10) return "fail: shallow sum $s != 10"
    val d = root.depth()
    if (d != 3) return "fail: shallow depth $d != 3"

    // removeEmpty correctness
    val root2 = TreeNode(1, mutableListOf(
        TreeNode(0, mutableListOf(TreeNode(0))),
        TreeNode(2),
    ))
    root2.removeEmpty()
    if (root2.children.size != 1) return "fail: removeEmpty size ${root2.children.size} != 1"
    if (root2.children[0].value != 2) return "fail: removeEmpty kept wrong child"

    // Deep: linear chain of depth 100,000
    val n = 100_000
    var head = TreeNode(1)
    for (i in 1 until n) {
        val parent = TreeNode(1, mutableListOf(head))
        head = parent
    }
    val deepSum = head.sum()
    if (deepSum != n) return "fail: deep sum $deepSum != $n"

    val deepDepth = head.depth()
    if (deepDepth != n) return "fail: deep depth $deepDepth != $n"

    return "OK"
}
