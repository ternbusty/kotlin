// TARGET_BACKEND: WASM
// ENABLE_TAIL_CALLS

// The accumulator helper's self-call must be emitted as return_call.
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=countDown$accum
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=sumTo$accum
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=maskChain$accum
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=sumEvens$accum
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=repeatStr$accum

// `1 + f(n - 1)`: counting, the canonical accumulator shape.
fun countDown(n: Int): Int {
    if (n == 0) return 0
    return 1 + countDown(n - 1)
}

// `n + f(n - 1)` with a Long accumulator and the operand taken from a parameter.
fun sumTo(n: Long): Long {
    if (n == 0L) return 0L
    return n + sumTo(n - 1L)
}

// Recursion on the left with a pure operand, and a bitwise operator.
fun maskChain(n: Int, bit: Int): Int {
    if (n == 0) return 0
    return maskChain(n - 1, bit) or bit
}

// Multiple return sites sharing the same operator.
fun sumEvens(n: Int): Int {
    if (n == 0) return 0
    if (n % 2 == 0) return n + sumEvens(n - 1)
    return 0 + sumEvens(n - 1)
}

// String concatenation (associative, not commutative).
fun repeatStr(s: String, n: Int): String {
    if (n == 0) return ""
    return s + repeatStr(s, n - 1)
}

fun box(): String {
    if (countDown(1_000_000) != 1_000_000) return "fail countDown"

    // 1M-term sum; Long avoids overflow so the closed form checks the fold order end to end.
    if (sumTo(1_000_000L) != 500_000_500_000L) return "fail sumTo"

    if (maskChain(1_000_000, 0b101) != 0b101) return "fail maskChain"

    // sumEvens adds 2 + 4 + ... + 1_000_000 with Int wrap-around; the expected
    // value is the same sum computed in Long and truncated, because Int `+` is
    // associative under wrap-around and the fold order does not change the result.
    val expectedEvens = ((2L + 1_000_000L) * 500_000L / 2L).toInt()
    if (sumEvens(1_000_000) != expectedEvens) return "fail sumEvens"

    val expected = "ab".repeat(10_000)
    if (repeatStr("ab", 10_000) != expected) return "fail repeatStr"

    return "OK"
}
