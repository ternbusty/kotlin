/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package kotlin.internal

/**
 * Marks an abstract method whose override hierarchy the Kotlin/Wasm backend may compile
 * into a stackless state machine with heap frames, so that deep virtual mutual recursion
 * through the hierarchy does not consume the host stack.
 *
 * Honored by the Wasm backend when `-Xwasm-enable-stackless-recursion` is set; other
 * backends ignore the annotation. The annotation has no effect on a method with a
 * body: the override hierarchy is derived from the abstract base declaration, and
 * the backend warns when the annotated method is not abstract.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.BINARY)
internal annotation class StacklessRecursion
