// TARGET_BACKEND: WASM
// ENABLE_TAIL_CALLS

// CPS-style lambda dispatch. Each continuation is a Function1<Int, Int> called
// via invoke(), which goes through a bridged function and a callRef intrinsic.
// Without tail-call support for lambda dispatch, this overflows at depth 1M.

fun cps(n: Int, k: (Int) -> Int): Int =
    if (n == 0) k(0)
    else cps(n - 1) { x: Int -> k(x + 1) }

fun box(): String {
    val depth = 1_000_000
    val result = cps(depth) { it }
    if (result != depth) return "fail cps: expected $depth, got $result"
    return "OK"
}
