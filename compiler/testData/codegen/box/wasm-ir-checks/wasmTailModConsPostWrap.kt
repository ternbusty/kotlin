// TARGET_BACKEND: WASM

// Iterative-rec TMC pattern with post-return conditional wrap.
// Models Apollo GraphQL's parseTypeInternal where:
//   1. A preVar (start) is saved across recursion
//   2. Post-effects run after the recursive call
//   3. The return value is conditionally wrapped (e.g. NonNull)
//   4. The wrap references the saved preVar

import kotlin.wasm.TailModCons

sealed class Token
object LeftBracket : Token()
object RightBracket : Token()
object Bang : Token()
class Name(val value: String) : Token()

sealed class GQLType
class GQLNamedType(val start: Int, val name: String) : GQLType()
class GQLListType(val start: Int, val type: GQLType) : GQLType()
class GQLNonNullType(val start: Int, val type: GQLType) : GQLType()

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
            GQLNamedType(start, (advance() as Name).value)
        }

        return if (peek() is Bang) {
            advance()
            GQLNonNullType(start, type)
        } else {
            type
        }
    }
}

fun buildTokens(depth: Int): Array<Token> {
    val tokens = mutableListOf<Token>()
    repeat(depth) {
        tokens.add(LeftBracket)
    }
    tokens.add(Name("String"))
    tokens.add(Bang)
    repeat(depth) {
        tokens.add(RightBracket)
        tokens.add(Bang)
    }
    return tokens.toTypedArray()
}

fun typeDepth(t: GQLType): Int {
    var depth = 0
    var cur = t
    while (true) {
        when (cur) {
            is GQLListType -> { depth++; cur = cur.type }
            is GQLNonNullType -> cur = cur.type
            is GQLNamedType -> break
        }
    }
    return depth
}

fun verifyStarts(t: GQLType): Boolean {
    var cur = t
    var expected = 0
    while (true) {
        when (cur) {
            is GQLNonNullType -> {
                if (cur.start != expected) return false
                cur = cur.type
            }
            is GQLListType -> {
                if (cur.start != expected) return false
                expected++
                cur = cur.type
            }
            is GQLNamedType -> {
                if (cur.start != expected) return false
                break
            }
        }
    }
    return true
}

fun box(): String {
    val shallow = Parser(buildTokens(10)).parseType()
    if (typeDepth(shallow) != 10) return "FAIL shallow depth: ${typeDepth(shallow)}"
    if (!verifyStarts(shallow)) return "FAIL shallow start positions"

    val deep = Parser(buildTokens(100_000)).parseType()
    if (typeDepth(deep) != 100_000) return "FAIL deep depth: ${typeDepth(deep)}"
    if (!verifyStarts(deep)) return "FAIL deep start positions"

    return "OK"
}
