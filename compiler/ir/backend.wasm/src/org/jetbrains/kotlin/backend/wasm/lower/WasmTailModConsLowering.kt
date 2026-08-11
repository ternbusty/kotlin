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
 * Rewrites `@TailModCons` functions that return a constructor wrapping
 * a recursive call into destination-passing style (DPS), turning the
 * recursive call into a tail call. Functions are grouped into SCCs of
 * the call graph so both self-recursion and mutual recursion are
 * handled. Untransformable annotated functions are a
 * [WasmBackendErrors.TAIL_MOD_CONS_NOT_APPLICABLE] compilation error.
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
                f.isTailrec -> reportNotApplicable(irFile, f, "the function is already tailrec, the annotation has no effect there")
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
                else -> if (tryMutualRecTransform(scc, sitesByFunc)) transformed += scc
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

    /** Constructor metadata validated for a DPS rewrite. Null when the site is not eligible. */
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
                        builder.irCtorWithNullPlaceholder(site.ctorCall, site.recursiveArgIndex),
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
                                builder.irCtorWithNullPlaceholder(value, recArgIndex),
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

    // -------------------------------------------------------------- N-function mutual-rec

    /**
     * For an SCC of N functions where each member has a TMC site calling
     * another member, synthesise a DPS sibling for each function and
     * rewrite the originals. Each function's DPS helper tail-calls the
     * callee's DPS helper, forming a cycle that runs in constant stack.
     */
    private fun tryMutualRecTransform(
        scc: List<IrSimpleFunction>,
        sitesByFunc: Map<IrSimpleFunction, List<TmcSite>>,
    ): Boolean {
        val sccSet = scc.toHashSet()

        // For each function in the SCC, find a TMC site calling another SCC member.
        val siteOf = mutableMapOf<IrSimpleFunction, TmcSite>()
        val calleeOf = mutableMapOf<IrSimpleFunction, IrSimpleFunction>()
        for (f in scc) {
            val site = sitesByFunc[f]?.firstOrNull { it.recursiveCall.symbol.owner in sccSet && it.recursiveCall.symbol.owner != f }
                ?: return false
            siteOf[f] = site
            calleeOf[f] = site.recursiveCall.symbol.owner
        }

        // All functions must share the same declaration container.
        val container = scc[0].parent as? IrDeclarationContainer ?: return false
        if (scc.any { it.parent !== container }) return false

        // Validate all sites and prepare DPS metadata.
        val prepOf = mutableMapOf<IrSimpleFunction, DpsPrep>()
        for (f in scc) {
            prepOf[f] = prepareDps(f, siteOf.getValue(f)) ?: return false
        }

        // Deep-copy all bodies before any mutation.
        val bodyCopies = scc.associateWith { (it.body as IrBlockBody).deepCopyWithSymbols() }

        // Compute callerOf (inverse of calleeOf in the cycle). Each function
        // has exactly one outgoing TMC edge, so the SCC is a simple cycle and
        // every node has exactly one predecessor.
        val callerOf = mutableMapOf<IrSimpleFunction, IrSimpleFunction>()
        for (f in scc) {
            callerOf[calleeOf.getValue(f)] = f
        }

        // Create DPS siblings. f$tmcDps receives the cell allocated by the
        // function that calls f, so its dst type is callerOf[f]'s constructor.
        val dpsOf = mutableMapOf<IrSimpleFunction, IrSimpleFunction>()
        for (f in scc) {
            val caller = callerOf.getValue(f)
            val callerPrep = prepOf.getValue(caller)
            dpsOf[f] = context.irFactory.stageController.restrictTo(f) {
                createDpsSibling(container, f, dstType = callerPrep.ctorClass.defaultTypeNullable())
            }
        }

        // Rewrite original bodies and build DPS bodies.
        for (f in scc) {
            val callee = calleeOf.getValue(f)
            val caller = callerOf.getValue(f)
            val site = siteOf.getValue(f)
            val callerPrep = prepOf.getValue(caller)
            val calleeDps = dpsOf.getValue(callee)

            transformOriginalBodyInPlace(f, site, calleeDps)
            buildDpsBodyFromCopy(
                dpsOf.getValue(f), f, bodyCopies.getValue(f), site,
                callerPrep.recField, calleeSymbol = callee.symbol, peerDps = calleeDps,
            )
            markTailCalls(dpsOf.getValue(f))
        }

        return true
    }

    private fun IrClass.defaultTypeNullable(): IrType = symbol.defaultType.makeNullable()

    /** Allocates [src]'s constructor with a null placeholder at [placeholderIndex]. The other arguments are taken from [src] as-is. */
    private fun IrBuilderWithScope.irCtorWithNullPlaceholder(src: IrConstructorCall, placeholderIndex: Int): IrConstructorCall =
        irCallConstructor(src.symbol, emptyList()).apply {
            for (i in 0 until src.arguments.size) {
                arguments[i] = src.arguments[i]
            }
            arguments[placeholderIndex] = irNull()
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

