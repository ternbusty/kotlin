// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// FILE: entry.mjs
export function jsBox() { return box(); }

// FILE: main.kt
// Mutual recursion: two effectively-final functions calling each other.
// At depth 100_000 this overflows a native Wasm stack without CPS.

fun isEvenCount(n: Int): Boolean {
    if (n == 0) return true
    return isOddCount(n - 1)
}

fun isOddCount(n: Int): Boolean {
    if (n == 0) return false
    return isEvenCount(n - 1)
}

// Three-function mutual recursion cycle with a non-tail call pattern.
// fizz -> buzz -> fazz -> fizz, with accumulator threading.
fun fizz(n: Int, acc: Int): Int {
    if (n == 0) return acc
    val step = buzz(n - 1, acc + 1)
    return step
}

fun buzz(n: Int, acc: Int): Int {
    if (n == 0) return acc
    return fazz(n - 1, acc + 2)
}

fun fazz(n: Int, acc: Int): Int {
    if (n == 0) return acc
    return fizz(n - 1, acc + 3)
}

fun box(): String {
    // Two-function tail-call mutual recursion.
    if (!isEvenCount(100_000)) return "FAIL: 100000 should be even"
    if (isEvenCount(99_999)) return "FAIL: 99999 should be odd"
    if (isOddCount(100_000)) return "FAIL: 100000 should not be odd"
    if (!isOddCount(99_999)) return "FAIL: 99999 should be odd (via isOddCount)"

    // Three-function mutual recursion with accumulator.
    // fizz(6, 0) -> buzz(5, 1) -> fazz(4, 3) -> fizz(3, 6)
    //            -> buzz(2, 7) -> fazz(1, 9) -> fizz(0, 12) = 12
    if (fizz(6, 0) != 12) return "FAIL: fizz(6,0) = ${fizz(6, 0)}, expected 12"

    // Deep three-function recursion.
    val deep = fizz(99_999, 0)
    // At depth 99999 with 3-function cycle: each cycle of 3 steps adds 1+2+3=6.
    // 99999 = 33333 * 3, so result = 33333 * 6 = 199998.
    if (deep != 199998) return "FAIL: fizz(99999,0) = $deep, expected 199998"

    return "OK"
}
