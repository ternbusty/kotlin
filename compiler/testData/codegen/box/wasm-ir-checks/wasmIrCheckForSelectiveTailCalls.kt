// TARGET_BACKEND: WASM
// ENABLE_TMC

// Selective emission mode (TMC enabled, unrestricted tail calls not requested):
// only self-recursive calls, TMC DPS helpers, and virtual/interface dispatch
// are emitted as tail calls, so V8 keeps inlining small static callees.

// Self-recursive tail call is still emitted
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call inFunction=selfRecTailCaller

// Non-recursive static call must NOT emit return_call (preserves V8 inlining)
// WASM_CHECK_INSTRUCTION_NOT_IN_FUNCTION: instruction=return_call inFunction=staticTailCaller

// Virtual dispatch tail call still produces return_call_ref
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call_ref inFunction=virtualTailCaller

// Interface dispatch tail call still produces return_call_ref
// WASM_CHECK_INSTRUCTION_IN_FUNCTION: instruction=return_call_ref inFunction=interfaceTailCaller

fun staticCallee(x: Int): Int = x + 1

fun staticTailCaller(x: Int): Int = staticCallee(x)

fun selfRecTailCaller(n: Int, acc: Int): Int =
    if (n == 0) acc else selfRecTailCaller(n - 1, acc + 1)

open class Base {
    open fun step(n: Int, acc: Int): Int = acc
}

class Derived : Base() {
    override fun step(n: Int, acc: Int): Int =
        if (n == 0) acc else step(n - 1, acc + 1)
}

fun virtualTailCaller(b: Base, n: Int): Int = b.step(n, 0)

interface Stepper {
    fun istep(n: Int, acc: Int): Int
}

class StepperImpl : Stepper {
    override fun istep(n: Int, acc: Int): Int =
        if (n == 0) acc else istep(n - 1, acc + 1)
}

fun interfaceTailCaller(s: Stepper, n: Int): Int = s.istep(n, 0)

fun box(): String {
    if (staticTailCaller(41) != 42) return "fail static"
    if (selfRecTailCaller(100_000, 0) != 100_000) return "fail selfRec"
    if (virtualTailCaller(Derived(), 100_000) != 100_000) return "fail virtual"
    if (interfaceTailCaller(StepperImpl(), 100_000) != 100_000) return "fail interface"
    return "OK"
}
