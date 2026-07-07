// TARGET_BACKEND: WASM
// ENABLE_STACKLESS_RECURSION

// Exercises WasmStacklessMatcherLowering end to end through the
// @kotlin.internal.StacklessRecursion annotation on the stdlib's
// AbstractSet.matches. The inputs below drive the matcher recursion far
// beyond any engine's native stack budget, so this test fails with a stack
// overflow if the transform silently stops firing.

fun box(): String {
    // KT-63689 shape: alternation with a variable-length branch. Stock stdlib
    // overflows around 2,000 input characters; 20,000 requires the trampoline.
    val kt63689 = Regex("(?:\\\\,|[^,])+")
    val long = "x".repeat(20_000) + "," + "y".repeat(5)
    val pieces = kt63689.findAll(long).map { it.value.length }.toList()
    if (pieces != listOf(20_000, 5)) return "fail kt63689: $pieces"

    // Backtracking correctness at depth: the first alternative is tried and
    // rejected per position, so consumed-position save/restore must survive
    // the frame conversion.
    val backtrack = Regex("(?:ab|a)+c")
    val bt = "ab".repeat(10_000) + "c"
    if (backtrack.matchEntire(bt) == null) return "fail backtrack match"
    if (backtrack.matchEntire(bt.dropLast(1)) != null) return "fail backtrack reject"

    // KT-78089 shape: single-char group under star, deep query string.
    val navlink = Regex("http://example\\.com/(.)*")
    val nav = "http://example.com/" + "q".repeat(20_000)
    if (navlink.matchEntire(nav) == null) return "fail navlink"

    return "OK"
}
