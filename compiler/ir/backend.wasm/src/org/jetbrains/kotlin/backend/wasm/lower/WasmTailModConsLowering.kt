/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.FileLoweringPass
import org.jetbrains.kotlin.backend.common.ir.ValueRemapper
import org.jetbrains.kotlin.backend.common.ir.normalizeReturnWhen
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.utils.StronglyConnectedComponents
import org.jetbrains.kotlin.backend.wasm.utils.hasTailModConsAnnotation
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.backend.wasm.WasmBackendErrors
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classifierOrFail
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.patchDeclarationParents
import org.jetbrains.kotlin.ir.util.parentClassOrNull
import org.jetbrains.kotlin.ir.util.properties
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitor
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid

/**
 * Tail Modulo Cons (TMC) lowering for Kotlin/Wasm.
 *
 * Rewrites functions annotated with `kotlin.wasm.TailModCons` whose return
 * statement wraps a recursive call inside a constructor (directly or via a
 * local variable):
 *
 *     return Ctor(c(p), f(next(p)))
 *     val v = f(next(p)); return Ctor(c(p), v)
 *
 * `return when/if { ... }` expressions are first normalised into per-branch
 * returns so that constructor-wrapping returns inside branches are detected.
 *
 * Detected functions are grouped into strongly-connected components of the
 * call graph. SCCs of size 1 or 2 are rewritten using destination-passing
 * style (DPS). For each member f a sibling `f$tmcDps(args..., dst)` is
 * synthesised that writes its result into `dst`'s recursive field (via
 * IrSetField on the backing field, sound because Kotlin/Wasm declares all
 * instance fields with `isMutable = true`; see TypeGenerator.kt) and
 * tail-calls the peer's DPS. The original `f` becomes
 * `f(args) = allocate head; peer_dps(args', head); return head`.
 *
 * The DPS bodies are produced by deep-copying the original function body and
 * transforming all IrReturn nodes in place, which preserves arbitrary control
 * flow, saved variables, and via-variable patterns without rebuilding the body
 * from scratch.
 *
 * The annotation is a checked contract. An annotated function that the
 * transformation cannot handle (e.g. recursion cycles larger than two
 * functions, or cycles spanning multiple files or declaration containers)
 * is a [WasmBackendErrors.TAIL_MOD_CONS_NOT_APPLICABLE] compilation error,
 * never a silent fall-through to stack-consuming recursion.
 */
internal class WasmTailModConsLowering(private val context: WasmBackendContext) : FileLoweringPass {

    override fun lower(irFile: IrFile) {
        val annotated = mutableListOf<IrSimpleFunction>()
        // The walk must descend into bodies: local functions can carry the
        // annotation too, and the checked contract owes them an error as well.
        irFile.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                element.acceptChildrenVoid(this)
            }
            override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                if (declaration.hasTailModConsAnnotation()) {
                    annotated += declaration
                }
                declaration.acceptChildrenVoid(this)
            }
        })
        if (annotated.isEmpty()) return

        val allFunctions = mutableListOf<IrSimpleFunction>()
        for (f in annotated) {
            when {
                f.isTailrec -> reportNotApplicable(irFile, f, "the function is already tailrec; the annotation has no effect there")
                f.body !is IrBlockBody -> reportNotApplicable(irFile, f, "the function has no block body")
                else -> allFunctions += f
            }
        }

        val transformed = mutableSetOf<IrSimpleFunction>()

        // `return when/if` bodies are normalised into per-branch returns so that
        // constructor-wrapping returns inside branches are detected. Annotated
        // functions that still end up untransformed are a compilation error, so
        // normalising unconditionally never churns IR that ships.
        for (f in allFunctions) {
            normalizeReturnWhen(f)
        }

        // Only calls to functions of this file can form transformable cycles.
        val fileFunctions = allFunctions.toHashSet()

        // Collect TMC sites per function and build edges (caller -> callee) within the file.
        val sitesByFunc = mutableMapOf<IrSimpleFunction, List<TmcSite>>()
        for (f in allFunctions) {
            val sites = collectTmcSites(f).filter { it.recursiveCall.symbol.owner in fileFunctions }
            if (sites.isNotEmpty()) sitesByFunc[f] = sites
        }

        val edges: Map<IrSimpleFunction, List<IrSimpleFunction>> = sitesByFunc.mapValues { entry ->
            entry.value.mapNotNull { s ->
                val callee = s.recursiveCall.symbol.owner
                if (callee in sitesByFunc) callee else null
            }.distinct()
        }
        val sccs = computeSccs(sitesByFunc.keys.toList(), edges)

        for (scc in sccs) {
            when (scc.size) {
                1 -> {
                    val f = scc.single()
                    val selfSite = sitesByFunc.getValue(f).firstOrNull { it.recursiveCall.symbol == f.symbol }
                    if (selfSite != null && trySelfRecDpsTransform(f, selfSite)) transformed += f
                }
                2 -> if (tryPairwiseMutualRecTransform(scc, sitesByFunc)) transformed += scc
            }
        }

        // The annotation is a checked contract: an annotated function that the
        // transformation cannot handle is a compilation error, never a silent
        // fall-through to stack-consuming recursion.
        for (f in allFunctions) {
            if (f in transformed) continue
            val reason = when {
                f !in sitesByFunc -> "no recursive call wrapped in a constructor was found " +
                        "(for mutual recursion, every function in the cycle needs the annotation)"
                sitesByFunc.getValue(f).none { it.recursiveCall.symbol == f.symbol } &&
                        sccs.none { f in it && it.size == 2 } ->
                    "the recursion cycle through this function is larger than two functions, which is not supported yet"
                else -> "the function's shape is not supported by the transformation"
            }
            reportNotApplicable(irFile, f, reason)
        }
    }

    private fun reportNotApplicable(irFile: IrFile, func: IrSimpleFunction, reason: String) {
        context.diagnosticReporter.at(func, irFile)
            .report(WasmBackendErrors.TAIL_MOD_CONS_NOT_APPLICABLE, reason)
    }

    // -------------------------------------------------------------- self-rec (DPS, body-transforming)

    /** Constructor metadata validated for a DPS rewrite; null when the site is not eligible. */
    private class DpsPrep(val ctorClass: IrClass, val recField: IrField)

    private fun prepareDps(func: IrSimpleFunction, site: TmcSite): DpsPrep? {
        val ctorClass = site.ctorCall.symbol.owner.parentClassOrNull ?: return null
        val recParam = site.ctorCall.symbol.owner.parameters.getOrNull(site.recursiveArgIndex) ?: return null
        if (recParam.type.classifierOrFail != func.returnType.classifierOrFail) return null
        val recField = findRecursiveBackingField(site.ctorCall, site.recursiveArgIndex) ?: return null
        return DpsPrep(ctorClass, recField)
    }

    private fun trySelfRecDpsTransform(func: IrSimpleFunction, site: TmcSite): Boolean {
        val container = func.parent as? IrDeclarationContainer ?: return false
        val prep = prepareDps(func, site) ?: return false
        val bodyCopy = (func.body as IrBlockBody).deepCopyWithSymbols()

        val fDps = context.irFactory.stageController.restrictTo(func) {
            createDpsSibling(container, func, dstType = prep.ctorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(func, site, fDps)
        buildDpsBodyFromCopy(fDps, func, bodyCopy, site, prep.recField)
        markTailCalls(fDps)
        return true
    }

    private fun transformOriginalBodyInPlace(
        func: IrSimpleFunction,
        site: TmcSite,
        fDps: IrSimpleFunction,
    ) {
        val body = func.body as IrBlockBody
        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)

        if (site.viaLocalVariable) {
            val arg = site.ctorCall.arguments[site.recursiveArgIndex]
            if (arg is IrGetValue) {
                val variable = arg.symbol.owner
                if (variable is IrVariable) removeStatementFromTree(body, variable)
            }
        }

        body.transform(object : IrTransformer<Nothing?>() {
            override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                if (expression !== site.returnExpr) return super.visitReturn(expression, data)
                return builder.irBlock {
                    val head = createTmpVariable(
                        builder.irCtorWithNullHole(site.ctorCall, site.recursiveArgIndex),
                        nameHint = "tmcHead",
                    )
                    +builder.irCall(fDps.symbol).apply {
                        for (i in site.recursiveCall.arguments.indices) {
                            arguments[i] = site.recursiveCall.arguments[i]
                        }
                        arguments[site.recursiveCall.arguments.size] = builder.irGet(head)
                    }
                    +builder.irReturn(builder.irGet(head))
                }
            }
        }, null)
    }

    private fun buildDpsBodyFromCopy(
        dps: IrSimpleFunction,
        original: IrSimpleFunction,
        bodyCopy: IrBlockBody,
        site: TmcSite,
        recField: IrField,
        calleeSymbol: IrSimpleFunctionSymbol = original.symbol,
        peerDps: IrSimpleFunction = dps,
    ) {
        val builder = context.createIrBuilder(dps.symbol, dps.startOffset, dps.endOffset)
        val dstParam = dps.parameters.last()
        val origFuncSymbol = original.symbol
        val ctorSymbol = site.ctorCall.symbol
        val recArgIndex = site.recursiveArgIndex

        val paramMapping: Map<IrValueSymbol, IrValueSymbol> = original.parameters.withIndex().associate { iv ->
            iv.value.symbol to dps.parameters[iv.index].symbol
        }
        remapSymbols(bodyCopy, paramMapping)

        val viaVarCalls = mutableMapOf<IrValueSymbol, IrCall>()

        bodyCopy.transform(object : IrTransformer<Nothing?>() {
            override fun visitVariable(declaration: IrVariable, data: Nothing?): IrStatement {
                val init = declaration.initializer
                if (init is IrCall && init.symbol == calleeSymbol) {
                    viaVarCalls[declaration.symbol] = init
                    declaration.initializer = builder.irNull()
                }
                return super.visitVariable(declaration, data)
            }

            override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                if (expression.returnTargetSymbol != origFuncSymbol)
                    return super.visitReturn(expression, data)

                val value = expression.value
                if (value is IrConstructorCall && value.symbol == ctorSymbol) {
                    val arg = value.arguments.getOrNull(recArgIndex)
                    val recCall = when {
                        arg is IrCall && arg.symbol == calleeSymbol -> arg
                        arg is IrGetValue && arg.symbol in viaVarCalls -> viaVarCalls[arg.symbol]!!
                        else -> null
                    }
                    if (recCall != null) {
                        return builder.irBlock {
                            val cell = createTmpVariable(
                                builder.irCtorWithNullHole(value, recArgIndex),
                                nameHint = "tmcCell",
                            )
                            +builder.irSetField(builder.irGet(dstParam), recField, builder.irGet(cell))
                            +builder.irReturn(
                                builder.irCall(peerDps.symbol).apply {
                                    for (i in recCall.arguments.indices) {
                                        arguments[i] = recCall.arguments[i]
                                    }
                                    arguments[recCall.arguments.size] = builder.irGet(cell)
                                },
                            )
                        }
                    }
                }

                return builder.irBlock {
                    +builder.irSetField(builder.irGet(dstParam), recField, value)
                    +builder.irReturn(builder.irGetObject(context.irBuiltIns.unitClass))
                }
            }
        }, null)

        dps.body = bodyCopy
        bodyCopy.patchDeclarationParents(dps)
    }

    private fun removeStatementFromTree(root: IrElement, target: IrStatement) {
        root.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitBlockBody(body: IrBlockBody) {
                body.statements.removeAll { it === target }
                super.visitBlockBody(body)
            }
            override fun visitBlock(expression: IrBlock) {
                expression.statements.removeAll { it === target }
                super.visitBlock(expression)
            }
        })
    }

    // -------------------------------------------------------------- 2-function mutual-rec

    /**
     * For an SCC {A, B} where both A and B match the SimpleChainShape and refer to each other,
     * synthesise A_dps and B_dps with destination-passing style, rewrite the originals to
     * allocate the head cell and kick off DPS, and use IrSetField on the recursive field's
     * backing field to mutate.
     */
    private fun tryPairwiseMutualRecTransform(
        scc: List<IrSimpleFunction>,
        sitesByFunc: Map<IrSimpleFunction, List<TmcSite>>,
    ): Boolean {
        val a = scc[0]
        val b = scc[1]
        val siteA = sitesByFunc[a]?.firstOrNull { it.recursiveCall.symbol == b.symbol } ?: return false
        val siteB = sitesByFunc[b]?.firstOrNull { it.recursiveCall.symbol == a.symbol } ?: return false

        val container = a.parent as? IrDeclarationContainer ?: return false
        if (b.parent !== container) return false

        // Validate both sites cheaply before the expensive deep copies.
        val prepA = prepareDps(a, siteA) ?: return false
        val prepB = prepareDps(b, siteB) ?: return false
        val bodyCopyA = (a.body as IrBlockBody).deepCopyWithSymbols()
        val bodyCopyB = (b.body as IrBlockBody).deepCopyWithSymbols()

        val aDps = context.irFactory.stageController.restrictTo(a) {
            createDpsSibling(container, a, dstType = prepB.ctorClass.defaultTypeNullable())
        }
        val bDps = context.irFactory.stageController.restrictTo(b) {
            createDpsSibling(container, b, dstType = prepA.ctorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(a, siteA, bDps)
        transformOriginalBodyInPlace(b, siteB, aDps)
        buildDpsBodyFromCopy(aDps, a, bodyCopyA, siteA, prepB.recField, calleeSymbol = b.symbol, peerDps = bDps)
        buildDpsBodyFromCopy(bDps, b, bodyCopyB, siteB, prepA.recField, calleeSymbol = a.symbol, peerDps = aDps)
        markTailCalls(aDps)
        markTailCalls(bDps)

        return true
    }

    private fun IrClass.defaultTypeNullable(): IrType = symbol.defaultType.makeNullable()

    /** Allocates [src]'s constructor with a null hole at [holeIndex]; the other arguments are taken from [src] as-is. */
    private fun IrBuilderWithScope.irCtorWithNullHole(src: IrConstructorCall, holeIndex: Int): IrConstructorCall =
        irCallConstructor(src.symbol, emptyList()).apply {
            for (i in 0 until src.arguments.size) {
                arguments[i] = src.arguments[i]
            }
            arguments[holeIndex] = irNull()
        }

    private fun createDpsSibling(
        container: IrDeclarationContainer,
        original: IrSimpleFunction,
        dstType: IrType,
    ): IrSimpleFunction {
        return context.irFactory.addFunction(container) {
            // The "$tmcDps" suffix is asserted by wasm-ir-checks testdata (WASM_CHECK_INSTRUCTION_IN_FUNCTION).
            name = Name.identifier(original.name.asString() + "\$tmcDps")
            visibility = DescriptorVisibilities.PRIVATE
            modality = Modality.FINAL
            returnType = context.irBuiltIns.unitType
            origin = TMC_DPS_FUNCTION
            startOffset = original.startOffset
            endOffset = original.endOffset
        }.apply {
            // Parameter order must match the original so DPS call sites copy arguments positionally.
            for (origParam in original.parameters) {
                addValueParameter(origParam.name, origParam.type)
            }
            addValueParameter("\$tmcDst", dstType)
        }
    }

    private fun findRecursiveBackingField(ctor: IrConstructorCall, argIndex: Int): IrField? {
        val cls = ctor.symbol.owner.parentClassOrNull ?: return null
        val ctorParams = ctor.symbol.owner.parameters
        val paramName = ctorParams.getOrNull(argIndex)?.name?.asString() ?: return null
        val prop = cls.properties.firstOrNull { it.name.asString() == paramName } ?: return null
        return prop.backingField
    }

    // -------------------------------------------------------------- detection

    private fun collectTmcSites(irFunction: IrSimpleFunction): List<TmcSite> {
        val results = mutableListOf<TmcSite>()
        val selfSymbol = irFunction.symbol

        val visitor = object : IrVisitor<Unit, Unit>() {
            override fun visitElement(element: IrElement, data: Unit) {
                element.acceptChildren(this, Unit)
            }

            override fun visitFunction(declaration: IrFunction, data: Unit) {
                // Don't descend into nested local functions.
            }

            override fun visitReturn(expression: IrReturn, data: Unit) {
                if (expression.returnTargetSymbol != selfSymbol) {
                    expression.acceptChildren(this, Unit)
                    return
                }
                val value = expression.value
                if (value is IrConstructorCall) {
                    findTmcCandidateInCtor(expression, value)
                }
                expression.acceptChildren(this, Unit)
            }

            private fun findTmcCandidateInCtor(returnExpr: IrReturn, ctor: IrConstructorCall) {
                for (i in ctor.arguments.size - 1 downTo 0) {
                    val arg = ctor.arguments[i] ?: continue
                    effectiveCall(arg)?.let {
                        results += TmcSite(returnExpr, ctor, it.first, i, it.second)
                        return
                    }
                    if (arg !is IrConst && arg !is IrGetValue && arg !is IrGetField && arg !is IrGetObjectValue) return
                }
            }
        }

        irFunction.body?.accept(visitor, Unit)
        return results
    }

    // -------------------------------------------------------------- SCC (Tarjan)

    private fun computeSccs(
        nodes: List<IrSimpleFunction>,
        edges: Map<IrSimpleFunction, List<IrSimpleFunction>>,
    ): List<List<IrSimpleFunction>> {
        val components = StronglyConnectedComponents<IrSimpleFunction> { edges[it].orEmpty().asSequence() }
        for (v in nodes) components.visit(v)
        return components.findComponents()
    }

    private fun remapSymbols(element: IrElement, mapping: Map<IrValueSymbol, IrValueSymbol>) {
        if (mapping.isEmpty()) return
        element.transform(ValueRemapper(mapping), null)
    }

}

/** Marks the synthesized `f\$tmcDps` destination-passing helpers. */
private val TMC_DPS_FUNCTION by IrDeclarationOriginImpl.Regular

private data class TmcSite(
    val returnExpr: IrReturn,
    val ctorCall: IrConstructorCall,
    val recursiveCall: IrCall,
    val recursiveArgIndex: Int,
    val viaLocalVariable: Boolean,
)

private fun effectiveCall(arg: IrExpression): Pair<IrCall, Boolean>? {
    if (arg is IrCall) return arg to false
    if (arg is IrGetValue) {
        val owner = arg.symbol.owner
        if (owner is IrVariable) {
            val init = owner.initializer
            if (init is IrCall) return init to true
        }
    }
    return null
}

