// TARGET_BACKEND: WASM

// RUN_PLAIN_BOX_FUNCTION

import kotlin.wasm.TailModCons

class Node(val tag: String, val child: Node?)

var sideEffectCounter = 0

fun sideEffect(): Int {
    sideEffectCounter++
    return sideEffectCounter
}

@TailModCons
fun buildChain(items: Array<String>, index: Int): Node? {
    if (index >= items.size) return null
    val saved = items[index]
    val rest = buildChain(items, index + 1)
    sideEffect()
    return Node(saved, rest)
}

fun collectTags(node: Node?): String {
    if (node == null) return ""
    val rest = collectTags(node.child)
    return if (rest.isEmpty()) node.tag else "${node.tag},$rest"
}

@TailModCons
fun buildDeep(depth: Int): Node? {
    if (depth <= 0) return null
    val label = "d$depth"
    val child = buildDeep(depth - 1)
    sideEffect()
    return Node(label, child)
}

fun box(): String {
    val items = arrayOf("a", "b", "c", "d")
    val chain = buildChain(items, 0) ?: return "FAIL: null chain"
    val tags = collectTags(chain)
    if (tags != "a,b,c,d") return "FAIL tags: $tags"

    if (sideEffectCounter != 4) return "FAIL sideEffects: $sideEffectCounter"

    sideEffectCounter = 0
    // Deep enough to overflow the host stack if the transform stops firing.
    val deep = buildDeep(100_000) ?: return "FAIL: null deep"
    if (deep.tag != "d100000") return "FAIL deep tag: ${deep.tag}"
    if (sideEffectCounter != 100_000) return "FAIL deep sideEffects: $sideEffectCounter"

    var node: Node? = deep
    var count = 0
    while (node != null) {
        count++
        node = node.child
    }
    if (count != 100_000) return "FAIL deep count: $count"

    return "OK"
}
