// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Models the stdlib regex matcher hierarchy (kotlin.text.regex.AbstractSet):
// mutual recursion through VIRTUAL dispatch over an object graph with cycles.
// The recursion depth is proportional to input length, which is the root
// cause of KT-63689 / KT-61542 / KT-78089 class stack overflows.
//
// Shapes modelled after the real matcher:
//  - LeafSet:  tail virtual call to next.matches (CharSet consuming one char)
//  - QuantSet: non-tail call to inner.matches with result inspection and
//              fallback call to next.matches (GroupQuantifierSet backtracking)
//  - AltSet:   loop over children calling child.matches, first success wins,
//              with save/restore of shared state (NonCapturingJointSet)
//  - FinalSet: leaf, no recursion (FSet)

abstract class MiniSet {
    var next: MiniSet? = null
    abstract fun matches(startIndex: Int, testString: CharSequence, state: MatchState): Int
}

class MatchState {
    var consumed = 0
    var enterCount = 0
}

class FinalSet : MiniSet() {
    override fun matches(startIndex: Int, testString: CharSequence, state: MatchState): Int {
        return startIndex
    }
}

// Consumes one expected char, then tail virtual call to next.
class CharSet(val ch: Char) : MiniSet() {
    override fun matches(startIndex: Int, testString: CharSequence, state: MatchState): Int {
        if (startIndex < testString.length && testString[startIndex] == ch) {
            return next!!.matches(startIndex + 1, testString, state)
        }
        return -1
    }
}

// GroupQuantifierSet shape: try inner (which loops back to this via the object
// graph), inspect the result, backtrack to next on failure. The enterCount
// save/restore after the recursive call is the pattern that makes this
// non-tail and non-TMC.
class QuantSet(val inner: MiniSet) : MiniSet() {
    override fun matches(startIndex: Int, testString: CharSequence, state: MatchState): Int {
        val saved = state.enterCount
        state.enterCount = saved + 1
        val r = inner.matches(startIndex, testString, state)
        state.enterCount = saved
        if (r >= 0) {
            return r
        }
        return next!!.matches(startIndex, testString, state)
    }
}

// NonCapturingJointSet shape: try children in order, restore consumed on failure.
class AltSet(val children: Array<MiniSet>) : MiniSet() {
    override fun matches(startIndex: Int, testString: CharSequence, state: MatchState): Int {
        val savedConsumed = state.consumed
        state.consumed = startIndex
        for (child in children) {
            val shift = child.matches(startIndex, testString, state)
            if (shift >= 0) {
                return shift
            }
        }
        state.consumed = savedConsumed
        return -1
    }
}

fun buildMatcher(): MiniSet {
    // Graph for the pattern (a|b)* followed by end:
    //   quant.inner = alt(charA, charB); charA.next = quant; charB.next = quant
    //   quant.next = final
    // Matching "aaa...a" recurses ~3 frames per character.
    val final = FinalSet()
    val charA = CharSet('a')
    val charB = CharSet('b')
    val alt = AltSet(arrayOf(charA, charB))
    val quant = QuantSet(alt)
    charA.next = quant
    charB.next = quant
    quant.next = final
    return quant
}

fun box(): String {
    val state = MatchState()

    // Shallow correctness: matches consume the full string.
    val m = buildMatcher()
    if (m.matches(0, "ab", state) != 2) return "fail: shallow ab"
    if (m.matches(0, "ba", state) != 2) return "fail: shallow ba"
    if (m.matches(0, "", state) != 0) return "fail: empty"
    if (state.enterCount != 0) return "fail: enterCount not restored"

    // Backtracking correctness: 'c' never matches, AltSet must restore state.
    if (m.matches(0, "ac", state) != 1) return "fail: partial ac"

    // Deep input: proportional recursion depth. Crashes without the
    // virtual-dispatch CPS transform.
    val n = 500_000
    val deep = buildString(n) { repeat(n) { append(if (it % 2 == 0) 'a' else 'b') } }
    val r = m.matches(0, deep, state)
    if (r != n) return "fail: deep result $r != $n"

    return "OK"
}
