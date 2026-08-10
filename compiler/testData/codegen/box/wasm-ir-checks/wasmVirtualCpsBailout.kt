// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Tests partial conversion: one override uses try-catch wrapping a target
// call, which the planner cannot split. That override bails out and keeps
// its native virtual body, while the others are converted into the CPS
// trampoline. The hybrid dispatch must route bailed-out classes through
// native virtual call and still produce correct results.

abstract class Processor {
    var next: Processor? = null
    abstract fun process(value: Int): Int
}

class Increment : Processor() {
    override fun process(value: Int): Int {
        val n = next ?: return value + 1
        return n.process(value + 1)
    }
}

class Double : Processor() {
    override fun process(value: Int): Int {
        val n = next ?: return value * 2
        return n.process(value * 2)
    }
}

// This override wraps the virtual call in try-catch, causing bail-out.
// The trampoline will invoke it through native virtual dispatch.
class Guarded : Processor() {
    override fun process(value: Int): Int {
        val n = next ?: return value
        return try {
            n.process(value)
        } catch (e: Throwable) {
            -1
        }
    }
}

class Terminal : Processor() {
    override fun process(value: Int): Int = value
}

fun box(): String {
    // Shallow: verify correctness through bailed-out class
    val t = Terminal()
    val g = Guarded()
    val inc = Increment()
    val dbl = Double()
    g.next = t
    inc.next = g
    dbl.next = inc

    // dbl.process(5) -> Double: 5*2=10 -> Increment: 10+1=11 -> Guarded: 11 -> Terminal: 11
    val r1 = dbl.process(5)
    if (r1 != 11) return "fail: shallow result $r1 != 11"

    // Deep: chain of Increment nodes followed by Terminal.
    // The bailed-out Guarded is not in the chain, so the trampoline
    // handles everything with CPS.
    val n = 500_000
    val term = Terminal()
    var head: Processor = term
    for (i in 0 until n) {
        val inc2 = Increment()
        inc2.next = head
        head = inc2
    }
    val deep = head.process(0)
    if (deep != n) return "fail: deep increment $deep != $n"

    // Deep with Guarded in the chain: every 1000th node is Guarded (bailed out).
    // Native dispatch handles Guarded; CPS handles the rest.
    val term2 = Terminal()
    var head2: Processor = term2
    for (i in 0 until n) {
        val node: Processor = if (i % 1000 == 999) Guarded() else Increment()
        node.next = head2
        head2 = node
    }
    val deep2 = head2.process(0)
    // Increment adds 1, Guarded passes through. n/1000 = 500 Guarded nodes.
    val expectedIncrements = n - n / 1000
    if (deep2 != expectedIncrements) return "fail: mixed chain $deep2 != $expectedIncrements"

    return "OK"
}
