// TARGET_BACKEND: WASM

// Iterative-rec TMC pattern: recursive call inside a when branch,
// with pre-statements that must be saved across recursion and
// post-effects that run after the recursive call returns.

import kotlin.wasm.TailModCons

sealed class Token
object LeftBracket : Token()
object RightBracket : Token()
class Name(val value: String) : Token()

sealed class GQLType
class GQLNamedType(val name: String) : GQLType()
class GQLListType(val depth: Int, val type: GQLType) : GQLType()

class Parser(private val tokens: Array<Token>) {
    private var pos = 0

    fun peek(): Token = if (pos < tokens.size) tokens[pos] else Name("end")

    fun advance(): Token {
        val t = peek()
        pos++
        return t
    }

    @TailModCons
    fun parseType(): GQLType {
        val start = pos

        val type = if (peek() is LeftBracket) {
            advance()
            val inner = parseType()
            advance() // consume RightBracket
            GQLListType(start, inner)
        } else {
            GQLNamedType((advance() as Name).value)
        }

        return type
    }
}

fun buildTokens(depth: Int): Array<Token> {
    val tokens = mutableListOf<Token>()
    repeat(depth) { tokens.add(LeftBracket) }
    tokens.add(Name("String"))
    repeat(depth) { tokens.add(RightBracket) }
    return tokens.toTypedArray()
}

fun typeDepth(t: GQLType): Int {
    var depth = 0
    var cur = t
    while (cur is GQLListType) {
        depth++
        cur = cur.type
    }
    return depth
}

fun verifyStartPositions(t: GQLType): Boolean {
    var cur = t
    var expectedStart = 0
    while (cur is GQLListType) {
        if (cur.depth != expectedStart) return false
        expectedStart++
        cur = cur.type
    }
    return true
}

fun box(): String {
    val shallow = Parser(buildTokens(10)).parseType()
    if (typeDepth(shallow) != 10) return "FAIL shallow depth: ${typeDepth(shallow)}"
    if (!verifyStartPositions(shallow)) return "FAIL shallow start positions"

    val deep = Parser(buildTokens(100_000)).parseType()
    if (typeDepth(deep) != 100_000) return "FAIL deep depth: ${typeDepth(deep)}"
    if (!verifyStartPositions(deep)) return "FAIL deep start positions"

    return "OK"
}
