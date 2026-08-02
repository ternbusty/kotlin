// TARGET_BACKEND: WASM

// The accumulator helper's self-call must be return_call even though no
// tail-call flag is set, exactly like the $tmcDps helpers.
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=countDown$tmcAcc
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=sumTo$tmcAcc
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=maskChain$tmcAcc

// Accumulator form of Tail Modulo Cons: a recursive call combined with an
// associative Int/Long operator (`return g + f(...)`) is not in tail position,
// so it consumes stack proportional to the recursion depth. The lowering
// rewrites it into an accumulator-passing sibling whose recursive call IS in
// tail position. At depth 1_000_000 these would overflow the host stack
// without the transform.

import kotlin.wasm.TailModCons

// `1 + f(n - 1)`: counting, the canonical shape.
@TailModCons
fun countDown(n: Int): Int {
    if (n == 0) return 0
    return 1 + countDown(n - 1)
}

// `n + f(n - 1)` with a Long accumulator and the operand taken from a parameter.
@TailModCons
fun sumTo(n: Long): Long {
    if (n == 0L) return 0L
    return n + sumTo(n - 1L)
}

// Recursion on the left with a pure operand, and a bitwise operator.
@TailModCons
fun maskChain(n: Int, bit: Int): Int {
    if (n == 0) return 0
    return maskChain(n - 1, bit) or bit
}

// `return when` branches are normalised before detection.
@TailModCons
fun sumEvens(n: Int): Int {
    return when {
        n == 0 -> 0
        n % 2 == 0 -> n + sumEvens(n - 1)
        else -> 0 + sumEvens(n - 1)
    }
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

    return "OK"
}
