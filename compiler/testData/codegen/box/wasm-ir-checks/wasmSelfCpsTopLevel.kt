// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Verifies the self-recursion CPS transform for top-level functions
// (no dispatch receiver). A top-level function is always effectively
// final, so the self-call target is statically known.

class ListNode(val value: Int, val next: ListNode?)

fun sumList(node: ListNode?): Int {
    if (node == null) return 0
    return node.value + sumList(node.next)
}

fun findMax(node: ListNode?): Int {
    if (node == null) return Int.MIN_VALUE
    val rest = findMax(node.next)
    return if (node.value > rest) node.value else rest
}

fun box(): String {
    // Shallow correctness
    val short = ListNode(3, ListNode(1, ListNode(4, ListNode(1, ListNode(5, null)))))
    val s = sumList(short)
    if (s != 14) return "fail: short sum $s != 14"
    val m = findMax(short)
    if (m != 5) return "fail: short max $m != 5"

    // Deep: 100,000 element linked list
    val n = 100_000
    var head: ListNode? = null
    for (i in n downTo 1) {
        head = ListNode(1, head)
    }
    val deepSum = sumList(head)
    if (deepSum != n) return "fail: deep sum $deepSum != $n"

    val deepMax = findMax(head)
    if (deepMax != 1) return "fail: deep max $deepMax != 1"

    return "OK"
}
