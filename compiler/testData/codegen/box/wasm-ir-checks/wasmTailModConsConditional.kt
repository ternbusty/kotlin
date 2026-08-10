// TARGET_BACKEND: WASM

// Constructor-wrapped recursion with a result-independent post-effect:
// the pos[0]++ after the recursive call (consuming the closing bracket)
// makes this NOT textbook tail-modulo-cons, which requires nothing but
// the constructor application after the call. The DPS-loop strategy
// still handles it by deferring such post-effects and replaying them at
// the base case, which is valid because they reference neither the
// recursive result nor any saved variable.

@file:OptIn(kotlin.wasm.ExperimentalWasmInterop::class)

import kotlin.wasm.TailModCons

sealed class Type
class Named(val name: String) : Type()
class ListType(val inner: Type) : Type()

@TailModCons
fun parseType(tokens: Array<String>, pos: IntArray): Type {
    if (pos[0] >= tokens.size) return Named("void")
    val type = if (tokens[pos[0]] == "[") {
        pos[0]++
        val inner = parseType(tokens, pos)
        pos[0]++
        ListType(inner)
    } else {
        val name = tokens[pos[0]]
        pos[0]++
        Named(name)
    }
    return type
}

fun buildNestedTokens(depth: Int): Array<String> {
    val tokens = mutableListOf<String>()
    repeat(depth) { tokens.add("[") }
    tokens.add("String")
    repeat(depth) { tokens.add("]") }
    return tokens.toTypedArray()
}

fun typeDepth(t: Type): Int {
    var depth = 0
    var cur = t
    while (cur is ListType) {
        depth++
        cur = cur.inner
    }
    return depth
}

fun box(): String {
    // shallow
    val shallow = parseType(buildNestedTokens(10), intArrayOf(0))
    if (typeDepth(shallow) != 10) return "FAIL shallow: ${typeDepth(shallow)}"

    // deep (would stack-overflow without the transform)
    val deep = parseType(buildNestedTokens(100_000), intArrayOf(0))
    if (typeDepth(deep) != 100_000) return "FAIL deep: ${typeDepth(deep)}"

    return "OK"
}
