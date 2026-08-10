// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Tests CHA with a wider class hierarchy: an abstract base with many
// concrete subclasses at two inheritance levels. Verifies that the
// lowering correctly finds all overrides through the hierarchy.

abstract class Expr {
    var next: Expr? = null
    abstract fun eval(acc: Int): Int
}

open class AddConst(val delta: Int) : Expr() {
    override fun eval(acc: Int): Int = next!!.eval(acc + delta)
}

class MulConst(val factor: Int) : Expr() {
    override fun eval(acc: Int): Int = next!!.eval(acc * factor)
}

class Identity : Expr() {
    override fun eval(acc: Int): Int = next!!.eval(acc)
}

// Second level: extends a concrete open class
class ClampedAdd(delta: Int, val min: Int) : AddConst(delta) {
    override fun eval(acc: Int): Int {
        val v = acc + delta
        return next!!.eval(if (v < min) min else v)
    }
}

class Terminal : Expr() {
    override fun eval(acc: Int): Int = acc
}

fun box(): String {
    // Shallow correctness with all node types
    val t = Terminal()

    val add = AddConst(10)
    add.next = t
    if (add.eval(5) != 15) return "fail: add ${add.eval(5)}"

    val mul = MulConst(3)
    mul.next = t
    if (mul.eval(4) != 12) return "fail: mul ${mul.eval(4)}"

    val id = Identity()
    id.next = t
    if (id.eval(7) != 7) return "fail: identity ${id.eval(7)}"

    val clamp = ClampedAdd(-100, 0)
    clamp.next = t
    if (clamp.eval(50) != 0) return "fail: clamp negative ${clamp.eval(50)}"

    val clamp2 = ClampedAdd(10, 0)
    clamp2.next = t
    if (clamp2.eval(50) != 60) return "fail: clamp positive ${clamp2.eval(50)}"

    // Deep chain mixing node types. All calls are in tail position.
    // The CHA must find all 5 concrete types across 2 inheritance
    // levels.
    val n = 500_000
    val term = Terminal()
    var head: Expr = term
    for (i in n downTo 1) {
        val node: Expr = when (i % 3) {
            0 -> AddConst(1)
            1 -> MulConst(1)
            else -> Identity()
        }
        node.next = head
        head = node
    }
    // AddConst(1) appears floor(n/3) times, the rest are identity.
    val expectedAdds = n / 3
    val result = head.eval(0)
    if (result != expectedAdds) return "fail: deep result $result != $expectedAdds"

    return "OK"
}
