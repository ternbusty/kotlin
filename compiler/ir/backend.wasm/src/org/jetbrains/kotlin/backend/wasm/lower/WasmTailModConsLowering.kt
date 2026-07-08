/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.FileLoweringPass
import org.jetbrains.kotlin.backend.common.ir.ValueRemapper
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.utils.StronglyConnectedComponents
import org.jetbrains.kotlin.backend.wasm.utils.hasTailModConsAnnotation
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.backend.wasm.WasmBackendErrors
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.expressions.impl.IrBlockImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrBreakImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrSetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrInstanceInitializerCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classifierOrFail
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.util.constructors
import org.jetbrains.kotlin.ir.util.createThisReceiverParameter
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.defaultValueForType
import org.jetbrains.kotlin.ir.util.patchDeclarationParents
import org.jetbrains.kotlin.ir.util.parentClassOrNull
import org.jetbrains.kotlin.ir.util.properties
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.ir.visitors.IrElementTransformerVoid
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.transformChildrenVoid
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
 * The annotation is a checked contract. An annotated function that no strategy
 * can transform (e.g. post-effects referencing the recursive result, recursion
 * cycles larger than two functions, or cycles spanning multiple files or
 * declaration containers) is a [WasmBackendErrors.TAIL_MOD_CONS_NOT_APPLICABLE]
 * compilation error, never a silent fall-through to stack-consuming recursion.
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

        // DPS-loop and iterative-rec consume the same matched shape
        // (`val r = when { branch(f(args)) }; return expr`, which
        // normalizeReturnWhen would break by distributing returns into
        // branches), so the matcher runs once per function. DPS-loop is
        // tried first: where post-effects reference neither the pre-loop
        // variables nor the recursive result, the recursion becomes
        // destination-passing style with mutable struct fields inside a
        // while(true) loop, with the deferred post-effects replayed at the
        // base case. No separate DPS function or return_call, so V8 loop
        // optimizations apply. Iterative-rec is the explicit-frame-stack
        // fallback for the rest.
        for (f in allFunctions) {
            val m = matchIterativeRecShape(f) ?: continue
            if (tryDpsLoopTransform(f, m) || tryIterativeRecTransform(f, irFile, m)) {
                transformed += f
            }
        }

        // `return when/if` bodies are normalised into per-branch returns so that
        // constructor-wrapping returns inside branches are detected. Annotated
        // functions that still end up untransformed are a compilation error, so
        // normalising unconditionally never churns IR that ships.
        for (f in allFunctions) {
            if (f !in transformed) normalizeReturnWhen(f)
        }

        // Only calls to functions of this file can form transformable cycles.
        val fileFunctions = allFunctions.toHashSet()

        // Collect TMC sites per function and build edges (caller -> callee) within the file.
        val sitesByFunc = mutableMapOf<IrSimpleFunction, List<TmcSite>>()
        for (f in allFunctions) {
            if (f in transformed) continue
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

    /** Everything a DPS rewrite needs from an eligible [site]; null when the site is not eligible. */
    private class DpsPrep(val ctorClass: IrClass, val recField: IrField, val bodyCopy: IrBlockBody)

    private fun prepareDps(func: IrSimpleFunction, site: TmcSite): DpsPrep? {
        val ctorClass = site.ctorCall.symbol.owner.parentClassOrNull ?: return null
        val recParam = site.ctorCall.symbol.owner.parameters.getOrNull(site.recursiveArgIndex) ?: return null
        if (recParam.type.classifierOrFail != func.returnType.classifierOrFail) return null
        val recField = findRecursiveBackingField(site.ctorCall, site.recursiveArgIndex) ?: return null
        val bodyCopy = (func.body as? IrBlockBody ?: return null).deepCopyWithSymbols()
        return DpsPrep(ctorClass, recField, bodyCopy)
    }

    private fun trySelfRecDpsTransform(func: IrSimpleFunction, site: TmcSite): Boolean {
        val container = func.parent as? IrDeclarationContainer ?: return false
        val prep = prepareDps(func, site) ?: return false

        val fDps = context.irFactory.stageController.restrictTo(func) {
            createDpsSibling(container, func, dstType = prep.ctorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(func, site, fDps)
        buildDpsBodyFromCopy(fDps, func, prep.bodyCopy, site, prep.recField)
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

    // -------------------------------------------------------------- return-when normalization

    private fun normalizeReturnWhen(func: IrSimpleFunction) {
        val body = func.body as? IrBlockBody ?: return
        val funcSymbol = func.symbol
        body.transform(object : IrTransformer<Nothing?>() {
            override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                expression.transformChildren(this, data)
                if (expression.returnTargetSymbol != funcSymbol) return expression
                val whenExpr = expression.value as? IrWhen ?: return expression
                for (branch in whenExpr.branches) {
                    branch.result = distributeReturn(branch.result, expression)
                }
                return whenExpr
            }

            private fun distributeReturn(expr: IrExpression, proto: IrReturn): IrExpression {
                return when (expr) {
                    is IrReturn -> expr
                    is IrWhen -> {
                        for (branch in expr.branches) {
                            branch.result = distributeReturn(branch.result, proto)
                        }
                        expr
                    }
                    is IrBlock -> {
                        val lastIdx = expr.statements.lastIndex
                        if (lastIdx >= 0) {
                            val last = expr.statements[lastIdx]
                            if (last is IrExpression && last !is IrReturn) {
                                expr.statements[lastIdx] = distributeReturn(last, proto)
                            }
                        }
                        expr
                    }
                    else -> IrReturnImpl(
                        proto.startOffset, proto.endOffset,
                        proto.type,
                        proto.returnTargetSymbol,
                        expr,
                    )
                }
            }
        }, null)
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

        val prepA = prepareDps(a, siteA) ?: return false
        val prepB = prepareDps(b, siteB) ?: return false

        val aDps = context.irFactory.stageController.restrictTo(a) {
            createDpsSibling(container, a, dstType = prepB.ctorClass.defaultTypeNullable())
        }
        val bDps = context.irFactory.stageController.restrictTo(b) {
            createDpsSibling(container, b, dstType = prepA.ctorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(a, siteA, bDps)
        transformOriginalBodyInPlace(b, siteB, aDps)
        buildDpsBodyFromCopy(aDps, a, prepA.bodyCopy, siteA, prepB.recField, calleeSymbol = b.symbol, peerDps = bDps)
        buildDpsBodyFromCopy(bDps, b, prepB.bodyCopy, siteB, prepA.recField, calleeSymbol = a.symbol, peerDps = aDps)

        return true
    }

    private fun IrClass.defaultTypeNullable(): IrType = symbol.defaultType.makeNullable()

    /** Allocates [src]'s constructor with a null hole at [holeIndex]; the other arguments are taken from [src] as-is. */
    private fun IrBuilderWithScope.irCtorWithNullHole(src: IrConstructorCall, holeIndex: Int): IrConstructorCall =
        irCallConstructor(src.symbol, emptyList()).apply {
            for (i in 0 until src.arguments.size) {
                arguments[i] = if (i == holeIndex) irNull() else src.arguments[i]
            }
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
                val argCount = ctor.arguments.size
                for (i in argCount - 1 downTo 0) {
                    val arg = ctor.arguments[i] ?: continue
                    val effective = effectiveCall(arg) ?: run {
                        if (arg is IrConst || arg is IrGetValue || arg is IrGetField || arg is IrGetObjectValue) {
                            continue
                        }
                        return
                    }
                    results += TmcSite(returnExpr, ctor, effective.first, i, effective.second)
                    return
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

    // -------------------------------------------------------------- DPS-loop (no return_call)

    private fun tryDpsLoopTransform(func: IrSimpleFunction, m: IterativeRecMatch): Boolean {
        // DPS-loop returns the head cell directly, so it cannot replay a wrapped
        // return; those shapes fall through to iterative-rec.
        val retVal = m.returnStmt.value
        if (retVal !is IrGetValue || retVal.symbol != m.resultVar.symbol) return false
        if (m.recPostEffects.isEmpty()) return false
        if (referencesSymbol(m.recPostEffects, m.recCallVar.symbol)) return false
        if (m.preVars.any { referencesSymbol(m.recPostEffects, it.symbol) }) return false

        val ctorCall = m.recFinalExpr as? IrConstructorCall ?: return false
        val recArgIndex = ctorCall.arguments.indexOfFirst { arg ->
            arg is IrGetValue && arg.symbol == m.recCallVar.symbol
        }
        if (recArgIndex < 0) return false
        val ctorClass = ctorCall.symbol.owner.parentClassOrNull ?: return false
        val ctorParams = ctorCall.symbol.owner.parameters
        val recParam = ctorParams.getOrNull(recArgIndex) ?: return false
        if (recParam.type.classifierOrFail != func.returnType.classifierOrFail) return false
        val recField = findRecursiveBackingField(ctorCall, recArgIndex) ?: return false

        buildDpsLoopBody(func, m, ctorCall, recArgIndex, ctorClass, recField)
        return true
    }

    private fun referencesSymbol(elements: List<IrStatement>, symbol: IrValueSymbol): Boolean {
        for (element in elements) {
            var found = false
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    if (!found) element.acceptChildrenVoid(this)
                }
                override fun visitGetValue(expression: IrGetValue) {
                    if (expression.symbol == symbol) found = true
                }
            })
            if (found) return true
        }
        return false
    }

    private fun buildDpsLoopBody(
        func: IrSimpleFunction,
        m: IterativeRecMatch,
        ctorCall: IrConstructorCall,
        recArgIndex: Int,
        ctorClass: IrClass,
        recField: IrField,
    ) {
        val body = func.body as IrBlockBody
        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)

        val recCall = m.recCallVar.initializer as IrCall

        val mutableParamVars = mutableListOf<Pair<IrValueParameter, IrVariable>>()

        val recCallArgSymbols = mutableSetOf<IrValueSymbol>()
        for (arg in recCall.arguments) {
            if (arg is IrGetValue) recCallArgSymbols.add(arg.symbol)
        }

        body.statements.clear()
        body.statements += builder.irBlockBody {
            for (w in m.earlyReturnWhens) {
                +w
            }

            for (v in m.preVars) {
                +v
            }

            val headVar = createTmpVariable(
                builder.irCtorWithNullHole(ctorCall, recArgIndex),
                nameHint = "\$head",
            )
            val dstVar = createTmpVariable(
                builder.irGet(headVar),
                nameHint = "\$dst",
                isMutable = true,
                irType = ctorClass.symbol.defaultType.makeNullable(),
            )
            val depthVar = createTmpVariable(builder.irInt(0), nameHint = "\$depth", isMutable = true)

            for (param in func.parameters) {
                if (param.symbol in recCallArgSymbols) continue
                val localCopy = createTmpVariable(
                    builder.irGet(param), nameHint = param.name.asString(), isMutable = true, irType = param.type,
                )
                mutableParamVars.add(param to localCopy)
            }

            val paramToLocal: Map<IrValueSymbol, IrValueSymbol> =
                mutableParamVars.associate { pv -> pv.first.symbol to pv.second.symbol }

            val mainLoop = builder.irWhile().apply {
                condition = builder.irTrue()
            }

            mainLoop.body = builder.irBlock {
                for (w in m.earlyReturnWhens) {
                    val wCopy = w.copyRemapped(paramToLocal)
                    wCopy.transform(object : IrTransformer<Nothing?>() {
                        override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                            if (expression.returnTargetSymbol != func.symbol)
                                return super.visitReturn(expression, data)
                            return builder.irBlock {
                                +builder.irSetField(builder.irGet(dstVar), recField, expression.value)
                                buildReplayLoop(this, builder, depthVar, m.recPostEffects, paramToLocal)
                                +builder.irReturn(builder.irGetField(builder.irGet(headVar), recField))
                            }
                        }
                    }, null)
                    +wCopy
                }

                val loopPreVarMap = mutableMapOf<IrValueSymbol, IrValueSymbol>()
                for (v in m.preVars) {
                    val vCopy = v.copyRemapped(paramToLocal)
                    loopPreVarMap[v.symbol] = vCopy.symbol
                    +vCopy
                }

                val loopLocalMap = paramToLocal + loopPreVarMap

                val condCopy = m.recursiveBranchCondition.copyRemapped(loopLocalMap)

                +irIfThenElse(
                    context.irBuiltIns.unitType,
                    condCopy,
                    irBlock {
                        val preEffectsCopy = deepCopyStatements(m.recPreEffects, loopLocalMap)
                        for (e in preEffectsCopy) +e

                        val ctorCallCopy = ctorCall.copyRemapped(loopLocalMap)
                        val cell = createTmpVariable(
                            irCtorWithNullHole(ctorCallCopy, recArgIndex),
                            nameHint = "\$cell",
                        )

                        +irSetField(irGet(dstVar), recField, irGet(cell))
                        +irSet(dstVar.symbol, irGet(cell))
                        +irSet(
                            depthVar.symbol,
                            irCallOp(context.irBuiltIns.intPlusSymbol, context.irBuiltIns.intType, irGet(depthVar), irInt(1)),
                        )

                        val recCallCopy = recCall.copyRemapped(loopLocalMap)
                        for (pv in mutableParamVars) {
                            val param = pv.first
                            val localVar = pv.second
                            val argIdx = func.parameters.indexOf(param)
                            val newArg = recCallCopy.arguments[argIdx]
                            if (newArg != null) {
                                +irSet(localVar.symbol, newArg)
                            }
                        }

                        +builder.irContinue(mainLoop)
                    },
                    irBlock {
                        for (branch in m.baseBranches) {
                            val brCopy = branch.copyRemapped(loopLocalMap)
                            val baseExpr = brCopy.result
                            +irSetField(irGet(dstVar), recField, baseExpr)
                        }
                    },
                )
                +builder.irBreak(mainLoop)
            }

            +mainLoop

            buildReplayLoop(this, builder, depthVar, m.recPostEffects, paramToLocal)
            +builder.irReturn(builder.irGetField(builder.irGet(headVar), recField))
        }.statements

        body.patchDeclarationParents(func)
    }

    private fun deepCopyStatements(
        stmts: List<IrStatement>,
        paramMapping: Map<IrValueSymbol, IrValueSymbol>,
    ): List<IrStatement> {
        val wrapper = IrBlockImpl(0, 0, context.irBuiltIns.unitType)
        wrapper.statements.addAll(stmts)
        val copy = wrapper.deepCopyWithSymbols() as IrBlock
        remapSymbols(copy, paramMapping)
        return copy.statements
    }

    private fun buildReplayLoop(
        blockBuilder: IrStatementsBuilder<*>,
        irBuilder: IrBuilderWithScope,
        depthVar: IrVariable,
        postEffects: List<IrStatement>,
        paramMapping: Map<IrValueSymbol, IrValueSymbol>,
    ) {
        val ri = blockBuilder.createTmpVariable(irBuilder.irInt(0), nameHint = "\$ri", isMutable = true)
        blockBuilder.apply {
            +irBuilder.irWhile().apply {
                condition = irBuilder.irNotEquals(irBuilder.irGet(ri), irBuilder.irGet(depthVar))
                this.body = irBuilder.irBlock {
                    for (stmt in deepCopyStatements(postEffects, paramMapping)) +stmt
                    +irBuilder.irSet(
                        ri.symbol,
                        irBuilder.irCallOp(context.irBuiltIns.intPlusSymbol, context.irBuiltIns.intType, irBuilder.irGet(ri), irBuilder.irInt(1)),
                    )
                }
            }
        }
    }

    // -------------------------------------------------------------- recursive-shape matching

    private data class IterativeRecMatch(
        val earlyReturnWhens: List<IrWhen>,
        val preVars: List<IrVariable>,
        val resultVar: IrVariable,
        val recursiveBranchCondition: IrExpression,
        val recPreEffects: List<IrStatement>,
        val recCallVar: IrVariable,
        val recPostEffects: List<IrStatement>,
        val recFinalExpr: IrExpression,
        val baseBranches: List<IrBranch>,
        val returnStmt: IrReturn,
        /** Pre-loop variables the unwind pass must restore per level (referenced by the post region). */
        val savedVars: List<IrVariable>,
    )

    private fun countSelfCalls(func: IrSimpleFunction): Int {
        var count = 0
        func.body?.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (count > 1) return
                element.acceptChildrenVoid(this)
            }
            override fun visitCall(expression: IrCall) {
                if (expression.symbol == func.symbol) count++
                if (count > 1) return
                expression.acceptChildrenVoid(this)
            }
        })
        return count
    }

    private fun matchIterativeRecShape(func: IrSimpleFunction): IterativeRecMatch? {
        // Cheap structural rejections first; the self-call count at the end needs a body traversal.
        val body = func.body as? IrBlockBody ?: return null
        val stmts = body.statements
        if (stmts.size < 2) return null

        val returnStmt = stmts.last() as? IrReturn ?: return null
        if (returnStmt.returnTargetSymbol != func.symbol) return null

        // The result variable must directly precede the trailing return; the return
        // itself may read it directly or wrap it (iterative-rec replays the wrap).
        val resultVarIdx = stmts.size - 2
        val resultVar = stmts[resultVarIdx] as? IrVariable ?: return null
        val whenExpr = resultVar.initializer as? IrWhen ?: return null

        var recBranchIdx = -1
        var matchedRecCallVar: IrVariable? = null
        var matchedPreEffects: List<IrStatement>? = null
        var matchedPostEffects: List<IrStatement>? = null
        var matchedRecFinalExpr: IrExpression? = null

        for (branchIv in whenExpr.branches.withIndex()) {
            val idx = branchIv.index
            val branch = branchIv.value
            val block = branch.result as? IrBlock ?: continue
            val blockStmts = block.statements
            if (blockStmts.isEmpty()) continue

            for (stmtIv in blockStmts.withIndex()) {
                val j = stmtIv.index
                val s = stmtIv.value
                val v = s as? IrVariable ?: continue
                val call = v.initializer as? IrCall ?: continue
                if (call.symbol != func.symbol) continue

                recBranchIdx = idx
                matchedRecCallVar = v
                matchedPreEffects = if (j > 0) blockStmts.subList(0, j).toList() else emptyList()
                matchedPostEffects = if (j + 1 < blockStmts.size - 1)
                    blockStmts.subList(j + 1, blockStmts.size - 1).toList()
                else emptyList()
                matchedRecFinalExpr = blockStmts.last() as? IrExpression
                break
            }
            if (recBranchIdx >= 0) break
        }

        if (recBranchIdx < 0 || matchedRecFinalExpr == null) return null
        if (countSelfCalls(func) != 1) return null

        val allPreStmts = if (resultVarIdx > 0) stmts.subList(0, resultVarIdx).toList() else emptyList()
        val baseBranches = whenExpr.branches.filterIndexed { i, _ -> i != recBranchIdx }
        if (baseBranches.isEmpty()) return null

        val earlyReturnWhens = mutableListOf<IrWhen>()
        val preVars = mutableListOf<IrVariable>()
        for (stmt in allPreStmts) {
            when {
                stmt is IrWhen && stmt.branches.all { b ->
                    b.result is IrReturn && (b.result as IrReturn).returnTargetSymbol == func.symbol
                } -> earlyReturnWhens += stmt
                stmt is IrVariable -> preVars += stmt
                else -> return null
            }
        }

        val postRegion = matchedPostEffects!! + listOf<IrStatement>(matchedRecFinalExpr, returnStmt)
        val savedVars = preVars.filter { referencesSymbol(postRegion, it.symbol) }

        return IterativeRecMatch(
            earlyReturnWhens = earlyReturnWhens,
            preVars = preVars,
            resultVar = resultVar,
            recursiveBranchCondition = whenExpr.branches[recBranchIdx].condition,
            recPreEffects = matchedPreEffects!!,
            recCallVar = matchedRecCallVar!!,
            recPostEffects = matchedPostEffects,
            recFinalExpr = matchedRecFinalExpr,
            baseBranches = baseBranches,
            returnStmt = returnStmt,
            savedVars = savedVars,
        )
    }

    private fun tryIterativeRecTransform(func: IrSimpleFunction, irFile: IrFile, m: IterativeRecMatch): Boolean {
        // Post-effects run in the unwind pass before the recursive result is
        // rebound; a reference to it would survive as a dangling symbol
        // (recFinalExpr is the only reader remapped to the result variable).
        if (referencesSymbol(m.recPostEffects, m.recCallVar.symbol)) return false
        // The unwind pass replays the post region against the final parameter
        // values; per-frame parameter values are not saved. Reject shapes
        // whose post region reads a parameter when the recursion rebinds one.
        val recCall = m.recCallVar.initializer as IrCall
        val sameParams = func.parameters.indices.all { i ->
            val arg = recCall.arguments[i]
            arg is IrGetValue && arg.symbol == func.parameters[i].symbol
        }
        if (!sameParams) {
            val postRegion = m.recPostEffects + listOf<IrStatement>(m.recFinalExpr, m.returnStmt.value)
            if (func.parameters.any { p -> referencesSymbol(postRegion, p.symbol) }) return false
        }
        context.irFactory.stageController.restrictTo(func) {
            doIterativeRecTransform(func, irFile, m)
        }
        return true
    }

    /**
     * Rewrites the matched recursion as two loops. The forward loop
     * evaluates the original branches once per level, in branch order:
     * the recursive branch descends (frame push, parameter rebind), any
     * other branch binds the result, applies the return wrap, and breaks.
     * Early returns bind the result and break without the wrap, as in the
     * original. The unwind loop then replays the post region once per
     * saved level. Nothing is evaluated more often than the recursion
     * it replaces.
     */
    private fun doIterativeRecTransform(func: IrSimpleFunction, irFile: IrFile, m: IterativeRecMatch) {
        val body = func.body as IrBlockBody

        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)

        val recCall = m.recCallVar.initializer as IrCall
        val sameParams = func.parameters.indices.all { i ->
            val arg = recCall.arguments[i]
            arg is IrGetValue && arg.symbol == func.parameters[i].symbol
        }

        val frameInfo = if (m.savedVars.isNotEmpty()) buildSavedFrameClass(irFile, func, m.savedVars) else null

        val newBody = builder.irBlockBody {
            val paramVars = if (!sameParams) {
                func.parameters.map { p ->
                    createTmpVariable(irGet(p), nameHint = "\$${p.name}", isMutable = true)
                }
            } else null
            val paramMap: Map<IrValueSymbol, IrValueSymbol> = if (paramVars != null) {
                func.parameters.withIndex().associate { iv -> iv.value.symbol to paramVars[iv.index].symbol }
            } else emptyMap()

            val depthVar = createTmpVariable(irInt(0), nameHint = "\$depth", isMutable = true)

            val frameStackVar: IrVariable? = frameInfo?.let { fi ->
                createTmpVariable(
                    irNull(fi.cls.defaultTypeNullable()),
                    nameHint = "\$frames",
                    isMutable = true,
                    irType = fi.cls.defaultTypeNullable(),
                )
            }

            val resultVar = createTmpVariable(
                IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, func.returnType),
                nameHint = "\$result",
                isMutable = true,
                irType = func.returnType,
            )

            // Remapped once in place here; every wrap emission below copies from it.
            remapSymbols(m.returnStmt, paramMap + mapOf(m.resultVar.symbol to resultVar.symbol))

            // Applies the return statement's wrap (anything beyond a plain
            // return of the result variable) to the result variable.
            fun IrStatementsBuilder<*>.emitReturnWrap(extraRemap: Map<IrValueSymbol, IrValueSymbol>) {
                val wrap = m.returnStmt.value
                if (wrap is IrGetValue && wrap.symbol == resultVar.symbol) return
                +irSet(resultVar.symbol, wrap.copyRemapped(extraRemap))
            }

            +irWhile().apply {
                condition = irTrue()
                this.body = irBlock {
                    for (w in m.earlyReturnWhens) {
                        remapSymbols(w, paramMap)
                        transformReturnsToResultBreaks(w, func.symbol, resultVar, this@apply)
                        +w
                    }
                    for (v in m.preVars) {
                        remapSymbols(v, paramMap)
                        +v
                    }

                    // The original `when`, in branch order: the recursive
                    // branch becomes the level descent, every other branch
                    // binds the result and leaves the loop.
                    val whenExpr = m.resultVar.initializer as IrWhen
                    +irWhen(context.irBuiltIns.unitType, whenExpr.branches.map { branch ->
                        val isRec = branch.condition === m.recursiveBranchCondition
                        remapSymbols(branch, paramMap)
                        val branchBody = if (isRec) {
                            irBlock {
                                for (effect in m.recPreEffects) +effect

                                if (frameInfo != null && frameStackVar != null) {
                                    +irSet(
                                        frameStackVar.symbol,
                                        irNewFrame(frameInfo, m.savedVars.map { irGet(it) }, irGet(frameStackVar)),
                                    )
                                }

                                if (paramVars != null) {
                                    val argTmps = func.parameters.mapIndexed { i, _ ->
                                        createTmpVariable(recCall.arguments[i]!!, nameHint = "next$i")
                                    }
                                    for (tmpIv in argTmps.withIndex()) {
                                        +irSet(paramVars[tmpIv.index].symbol, irGet(tmpIv.value))
                                    }
                                }

                                +irSet(depthVar.symbol, irCallOp(context.irBuiltIns.intPlusSymbol, context.irBuiltIns.intType, irGet(depthVar), irInt(1)))
                            }
                        } else {
                            irBlock {
                                +irSet(resultVar.symbol, branch.result)
                                emitReturnWrap(emptyMap())
                                +irBreak(this@apply)
                            }
                        }
                        irBranch(branch.condition, branchBody)
                    })
                }
            }

            +irWhile().apply {
                condition = irNotEquals(irGet(depthVar), irInt(0))
                this.body = irBlock {
                    +irSet(depthVar.symbol, irCallOp(context.irBuiltIns.intPlusSymbol, context.irBuiltIns.intType, irGet(depthVar), irInt(-1)))

                    val savedVarMap = mutableMapOf<IrValueSymbol, IrValueSymbol>()
                    if (frameInfo != null && frameStackVar != null) {
                        val frameTmp = createTmpVariable(
                            irImplicitCast(irGet(frameStackVar), frameInfo.cls.symbol.defaultType),
                            nameHint = "\$frame",
                        )
                        for (svIv in m.savedVars.withIndex()) {
                            val sv = svIv.value
                            val restoredVarCopy = sv.deepCopyWithSymbols()
                            restoredVarCopy.initializer = irGetField(irGet(frameTmp), frameInfo.fields[svIv.index])
                            +restoredVarCopy
                            savedVarMap[sv.symbol] = restoredVarCopy.symbol
                        }
                        +irSet(frameStackVar.symbol, irGetField(irGet(frameTmp), frameInfo.prevField))
                    }

                    val backwardRemap = savedVarMap + paramMap
                    for (effect in m.recPostEffects) {
                        remapSymbols(effect, backwardRemap)
                        +effect
                    }

                    +irSet(
                        resultVar.symbol,
                        m.recFinalExpr.copyRemapped(savedVarMap + mapOf(m.recCallVar.symbol to resultVar.symbol) + paramMap),
                    )

                    emitReturnWrap(savedVarMap)
                }
            }

            +irReturn(irGet(resultVar))
        }

        body.statements.clear()
        body.statements += newBody.statements
        body.patchDeclarationParents(func)
    }

    private data class SavedFrameInfo(
        val cls: IrClass,
        val fields: List<IrField>,
        val prevField: IrField,
        val ctorSymbol: IrConstructorSymbol,
    )

    /** Distinguishes frame classes when several same-named functions in one file transform. */
    private var frameClassIndex = 0

    private fun buildSavedFrameClass(
        irFile: IrFile,
        func: IrSimpleFunction,
        savedVars: List<IrVariable>,
    ): SavedFrameInfo {
        val cls = context.irFactory.buildClass {
            name = Name.identifier("\$TmcFrame_${func.name}_${frameClassIndex++}")
            visibility = DescriptorVisibilities.PRIVATE
            modality = Modality.FINAL
            kind = ClassKind.CLASS
        }.apply {
            parent = irFile
            irFile.declarations += this
            createThisReceiverParameter()
            superTypes = listOf(context.irBuiltIns.anyType)
        }

        val fields = savedVars.map { sv ->
            cls.addField {
                name = sv.name
                type = sv.type
                visibility = DescriptorVisibilities.PUBLIC
                isFinal = true
            }
        }

        val prevName = Name.identifier("\$prev")
        val prevField = cls.addField {
            name = prevName
            type = cls.defaultTypeNullable()
            visibility = DescriptorVisibilities.PUBLIC
            isFinal = true
        }

        val ctor = cls.addConstructor {
            isPrimary = true
        }.apply {
            val params = savedVars.map { sv -> addValueParameter(sv.name, sv.type) }
            val prevParam = addValueParameter(prevName, cls.defaultTypeNullable())
            body = context.createIrBuilder(symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET).irBlockBody {
                +irDelegatingConstructorCall(context.irBuiltIns.anyClass.owner.constructors.single())
                +IrInstanceInitializerCallImpl(
                    UNDEFINED_OFFSET, UNDEFINED_OFFSET,
                    cls.symbol, context.irBuiltIns.unitType
                )
                for (fieldIv in fields.withIndex()) {
                    +irSetField(irGet(cls.thisReceiver!!), fieldIv.value, irGet(params[fieldIv.index]))
                }
                +irSetField(irGet(cls.thisReceiver!!), prevField, irGet(prevParam))
            }
        }

        return SavedFrameInfo(cls, fields, prevField, ctor.symbol)
    }

    /** Owns the frame ctor argument layout: saved fields in order, then the previous frame. */
    private fun IrBuilderWithScope.irNewFrame(
        info: SavedFrameInfo,
        savedValues: List<IrExpression>,
        prev: IrExpression,
    ): IrExpression = irCallConstructor(info.ctorSymbol, emptyList()).apply {
        for (i in savedValues.indices) arguments[i] = savedValues[i]
        arguments[savedValues.size] = prev
    }

    private fun remapSymbols(element: IrElement, mapping: Map<IrValueSymbol, IrValueSymbol>) {
        if (mapping.isEmpty()) return
        element.transform(ValueRemapper(mapping), null)
    }

    private inline fun <reified T : IrElement> T.copyRemapped(mapping: Map<IrValueSymbol, IrValueSymbol>): T {
        val copy = deepCopyWithSymbols()
        if (mapping.isEmpty()) return copy
        return copy.transform(ValueRemapper(mapping), null) as T
    }

    /** `return v` targeting [funcSymbol] becomes `{ result = v; break }`. */
    private fun transformReturnsToResultBreaks(
        element: IrElement,
        funcSymbol: IrSimpleFunctionSymbol,
        resultVar: IrVariable,
        loop: IrLoop,
    ) {
        element.transform(object : IrElementTransformerVoid() {
            override fun visitReturn(expression: IrReturn): IrExpression {
                expression.transformChildrenVoid(this)
                if (expression.returnTargetSymbol != funcSymbol) return expression
                return IrBlockImpl(
                    expression.startOffset, expression.endOffset, context.irBuiltIns.nothingType, null,
                    listOf(
                        IrSetValueImpl(
                            expression.startOffset, expression.endOffset, context.irBuiltIns.unitType,
                            resultVar.symbol, expression.value, null,
                        ),
                        IrBreakImpl(expression.startOffset, expression.endOffset, context.irBuiltIns.nothingType, loop),
                    ),
                )
            }
        }, null)
    }

}

/** Marks the synthesized `f\$tmcDps` destination-passing helpers. */
internal val TMC_DPS_FUNCTION by IrDeclarationOriginImpl.Regular

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

