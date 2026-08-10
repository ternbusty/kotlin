/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package kotlin.wasm

/**
 * Instructs the Kotlin/Wasm compiler to convert the virtual recursion
 * rooted at this abstract function into a heap-allocated trampoline, so
 * that override chains recurse in constant stack space regardless of
 * depth.
 *
 * Place on an abstract function in a base class. The compiler uses
 * class-hierarchy analysis to collect every override and compiles them
 * into a single flat state machine with heap-allocated frames. Overrides
 * whose bodies the transformation cannot handle fall back to their
 * original virtual dispatch transparently.
 *
 * Requires `-Xwasm-enable-stackless-recursion`.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.BINARY)
@OptionalExpectation
public expect annotation class StacklessVirtualRecursion()
