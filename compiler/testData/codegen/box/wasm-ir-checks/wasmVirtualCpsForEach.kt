// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Tests CPS planning of forEach with non-local return and target call inside,
// modelling the JointSet.matches() pattern from the stdlib regex engine.
// Also exercises .also and .let patterns.

abstract class Node {
    var next: Node? = null
    abstract fun eval(value: Int, state: EvalState): Int
}

class EvalState {
    var saved = 0
    var sideEffects = 0
}

class Terminal : Node() {
    override fun eval(value: Int, state: EvalState): Int = value
}

class Passthrough : Node() {
    override fun eval(value: Int, state: EvalState): Int {
        return next!!.eval(value, state)
    }
}

class Incrementer : Node() {
    override fun eval(value: Int, state: EvalState): Int {
        return next!!.eval(value + 1, state)
    }
}

// Models JointSet.matches(): forEach over children array with non-local
// return, guard check, state save/restore before and after the loop.
class Selector(private val children: Array<Node>) : Node() {
    override fun eval(value: Int, state: EvalState): Int {
        if (children.isEmpty()) {
            return -1
        }
        val oldSaved = state.saved
        state.saved = value
        children.forEach {
            val result = it.eval(value, state)
            if (result >= 0) {
                return result
            }
        }
        state.saved = oldSaved
        return -1
    }
}

// Models LookAroundSet.matches(): .also with side effect after a
// non-target call that indirectly dispatches to the same hierarchy.
class GuardNode(val inner: Node) : Node() {
    override fun eval(value: Int, state: EvalState): Int {
        return tryEval(value, state).also {
            if (it < 0) state.sideEffects++
        }
    }

    private fun tryEval(value: Int, state: EvalState): Int {
        return inner.eval(value, state)
    }
}

// Tests .let pattern: capture result in a new scope, then use next.
class LetNode : Node() {
    override fun eval(value: Int, state: EvalState): Int {
        return next!!.eval(value, state).let { r ->
            if (r < 0) -1 else r
        }
    }
}

fun box(): String {
    val st = EvalState()

    // Shallow: Selector picks first child that returns >= 0
    val term = Terminal()
    val inc = Incrementer()
    inc.next = term
    val sel = Selector(arrayOf(inc))
    sel.next = term
    val r1 = sel.eval(5, st)
    if (r1 != 6) return "fail: selector result $r1 != 6"
    if (st.saved != 5) return "fail: state.saved ${st.saved} != 5"

    // GuardNode wraps eval through tryEval with .also side effect
    st.sideEffects = 0
    val guard = GuardNode(inc)
    guard.next = term
    val r2 = guard.eval(10, st)
    if (r2 != 11) return "fail: guard result $r2 != 11"
    if (st.sideEffects != 0) return "fail: sideEffects should be 0 for positive result"

    // LetNode wraps eval with .let
    val letNode = LetNode()
    letNode.next = inc
    inc.next = term
    val r3 = letNode.eval(7, st)
    if (r3 != 8) return "fail: let result $r3 != 8"

    // Selector with empty children => -1, restores state
    st.saved = 42
    val emptySel = Selector(arrayOf())
    emptySel.next = term
    val r4 = emptySel.eval(99, st)
    if (r4 != -1) return "fail: empty selector $r4 != -1"
    if (st.saved != 42) return "fail: state.saved not preserved after empty ${st.saved}"

    // Selector with multiple children, first fails, second succeeds
    val failNode = object : Node() {
        override fun eval(value: Int, state: EvalState): Int = -1
    }
    val multi = Selector(arrayOf(failNode, inc))
    multi.next = term
    st.saved = 0
    val r5 = multi.eval(3, st)
    if (r5 != 4) return "fail: multi selector $r5 != 4"

    // Deep: chain of Selector nodes (forEach pattern) at 500k depth
    val n = 500_000
    val deepTerm = Terminal()
    var head: Node = deepTerm
    for (i in 0 until n) {
        val child = Incrementer()
        child.next = head
        val s = Selector(arrayOf(child))
        s.next = deepTerm
        head = s
    }
    val deep = head.eval(0, EvalState())
    if (deep != n) return "fail: deep selector $deep != $n"

    // Deep: chain of LetNode nodes (.let pattern)
    val n2 = 500_000
    val deepTerm2 = Terminal()
    var head2: Node = deepTerm2
    for (i in 0 until n2) {
        val l = LetNode()
        l.next = head2
        head2 = l
    }
    val deep2 = head2.eval(42, EvalState())
    if (deep2 != 42) return "fail: deep let $deep2 != 42"

    return "OK"
}
