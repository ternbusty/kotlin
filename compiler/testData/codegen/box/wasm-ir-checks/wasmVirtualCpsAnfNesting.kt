// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Tests that ANF normalization handles nested target calls: f(g(x)),
// where the result of one virtual call is passed as an argument to
// another. Without normalization, the body planner bails out with
// "target call in argument of target call".

abstract class Transform {
    var next: Transform? = null
    abstract fun apply(value: Int): Int
}

class End : Transform() {
    override fun apply(value: Int): Int = value
}

// Tail call to next: no nesting.
class Inc : Transform() {
    override fun apply(value: Int): Int = next!!.apply(value + 1)
}

// Nested target call: next.apply(inner.apply(value)).
// The result of inner.apply is fed directly as an argument to
// next.apply. ANF normalization lifts this into:
//   val t0 = next!!
//   val t1 = inner.apply(value)
//   t0.apply(t1)
class Pipe(val inner: Transform) : Transform() {
    override fun apply(value: Int): Int =
        next!!.apply(inner.apply(value))
}

fun box(): String {
    // Shallow correctness:
    // Pipe(Inc) -> End: Inc adds 1, Pipe pipes through, End returns.
    val end = End()
    val inc = Inc()
    inc.next = end
    val pipe = Pipe(inc)
    pipe.next = end
    // pipe.apply(10) -> end.apply(inc.apply(10)) -> end.apply(11) -> 11
    if (pipe.apply(10) != 11) return "fail: shallow ${pipe.apply(10)}"

    // Deep: chain of alternating Inc and Pipe nodes.
    // Each Inc adds 1, each Pipe applies its inner (an Inc that adds 1)
    // then passes to next. So each Pipe+Inc pair adds 2.
    val n = 100_000
    val term = End()
    var head: Transform = term
    for (i in 0 until n) {
        val inner = Inc()
        inner.next = End()
        val p = Pipe(inner)
        p.next = head
        head = p
    }
    // Each Pipe applies its inner Inc (+1) then forwards to next.
    val result = head.apply(0)
    if (result != n) return "fail: deep result $result != $n"

    return "OK"
}
