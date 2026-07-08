/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.FileLoweringPass
import org.jetbrains.kotlin.backend.common.lower.AbstractVariableRemapper
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.utils.StronglyConnectedComponents
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.wasm.config.WasmConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.IrBlockImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classifierOrFail
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.fqNameWhenAvailable
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
 * Detects functions whose return statement wraps a recursive call inside a
 * constructor (directly or via a local variable):
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
 * Functions not eligible for DPS (e.g. those whose post-effects reference the
 * recursive result, or SCCs of size >= 3) are left untouched.
 *
 * Candidate detection is file-local, and the pairwise rewrite requires both
 * members in the same declaration container; mutual recursion across files or
 * containers is reported as a candidate but not transformed.
 */
internal class WasmTailModConsLowering(private val context: WasmBackendContext) : FileLoweringPass {

    @OptIn(MessageCollectorAccess::class)
    private val messageCollector: MessageCollector? =
        context.configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)

    private val enabled = context.configuration.get(WasmConfigurationKeys.WASM_ENABLE_TMC) == true

    override fun lower(irFile: IrFile) {
        if (!enabled) return
        val allFunctions = mutableListOf<IrSimpleFunction>()
        irFile.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                element.acceptChildrenVoid(this)
            }
            override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                if (!declaration.isTailrec && declaration.body is IrBlockBody) {
                    allFunctions += declaration
                }
                declaration.acceptChildrenVoid(this)
            }
        })

        val transformed = mutableSetOf<IrSimpleFunction>()

        // DPS-loop runs first: for functions matching the iterative-rec shape where
        // post-effects don't reference savedVars or the recursive result, use
        // destination-passing style with mutable struct fields inside a while(true) loop.
        // The loop allocates a cell, fills the previous hole (dst.field = cell),
        // advances dst, and re-evaluates the branch condition.  At the base case the
        // deferred post-effects are replayed `depth` times in a simple loop.
        // No separate DPS function or return_call, so V8 loop optimizations apply.
        for (f in allFunctions) {
            if (tryDpsLoopTransform(f)) {
                transformed += f
            }
        }

        // Only calls to functions of this file can form transformable cycles, so
        // gate normalization and site collection on file-local callees to avoid
        // churning IR that can never become a candidate.
        val fileFunctions = allFunctions.toHashSet()

        for (f in allFunctions) {
            if (f in transformed) continue
            if (hasReturnWhenWithCtorCall(f, fileFunctions)) {
                normalizeReturnWhen(f)
            }
        }

        // Collect TMC sites per function and build edges (caller -> callee) within the file.
        val sitesByFunc = mutableMapOf<IrSimpleFunction, List<TmcSite>>()
        for (f in allFunctions) {
            if (f in transformed) continue
            val sites = collectTmcSites(f).filter { it.recursiveCall.symbol.owner in fileFunctions }
            if (sites.isNotEmpty()) sitesByFunc[f] = sites
        }

        if (sitesByFunc.isNotEmpty()) {
            val candidateSet = sitesByFunc.keys
            val edges: Map<IrSimpleFunction, List<IrSimpleFunction>> = sitesByFunc.mapValues { entry ->
                entry.value.mapNotNull { s ->
                    val callee = s.recursiveCall.symbol.owner
                    if (callee in candidateSet) callee else null
                }.distinct()
            }
            val sccs = computeSccs(candidateSet.toList(), edges)

            val mc = messageCollector
            val nonTrivialSccs = sccs.filter { it.size >= 2 }
            if (mc != null && nonTrivialSccs.isNotEmpty()) {
                mc.report(
                    CompilerMessageSeverity.STRONG_WARNING,
                    "[wasm-tmc] file=${irFile.fileEntry.name} non-trivial SCCs: ${nonTrivialSccs.size}",
                )
                for (scc in nonTrivialSccs) {
                    val members = scc.joinToString(", ") { it.fqNameWhenAvailable?.asString() ?: it.name.asString() }
                    mc.report(CompilerMessageSeverity.STRONG_WARNING, "[wasm-tmc]   SCC of ${scc.size}: $members")
                }
            }

            for (scc in sccs) {
                when {
                    scc.size == 1 -> {
                        val f = scc.single()
                        val sites = sitesByFunc[f]!!
                        val selfSite = sites.firstOrNull { it.recursiveCall.symbol == f.symbol }
                        if (selfSite != null) {
                            val ok = trySelfRecDpsTransform(f, selfSite)
                            reportSites(f, sites, ok)
                            if (ok) transformed += f
                        } else {
                            reportSites(f, sites, transformed = false)
                        }
                    }
                    scc.size == 2 -> {
                        val ok = tryPairwiseMutualRecTransform(scc, sitesByFunc)
                        for (f in scc) {
                            reportSites(f, sitesByFunc[f]!!, ok)
                            if (ok) transformed += f
                        }
                    }
                    else -> {
                        for (f in scc) reportSites(f, sitesByFunc[f]!!, transformed = false)
                    }
                }
            }
        }

    }

    private fun reportSites(func: IrSimpleFunction, sites: List<TmcSite>, transformed: Boolean) {
        val mc = messageCollector ?: return
        val fqn = func.fqNameWhenAvailable?.asString() ?: func.name.asString()
        val severity = if (transformed) CompilerMessageSeverity.STRONG_WARNING else CompilerMessageSeverity.INFO
        val tag = if (transformed) "transformed" else "candidate"
        for (s in sites) {
            val callee = s.recursiveCall.symbol.owner.name.asString()
            val ctor = s.ctorCall.symbol.owner.parentClassOrNull?.name?.asString() ?: "?"
            val viaVar = if (s.viaLocalVariable) " (via local var)" else ""
            val selfFlag = if (s.recursiveCall.symbol == func.symbol) " (SELF-RECURSE)" else ""
            mc.report(severity, "[wasm-tmc] $tag: $fqn -> $callee, wrapped in $ctor$viaVar$selfFlag")
        }
    }

    // -------------------------------------------------------------- self-rec (DPS, body-transforming)

    private fun trySelfRecDpsTransform(func: IrSimpleFunction, site: TmcSite): Boolean {
        val ctorClass = site.ctorCall.symbol.owner.parentClassOrNull ?: return false
        val recField = findRecursiveBackingField(site.ctorCall, site.recursiveArgIndex) ?: return false
        val container = func.parent as? IrDeclarationContainer ?: return false
        val ctorParams = site.ctorCall.symbol.owner.parameters
        val recParam = ctorParams.getOrNull(site.recursiveArgIndex) ?: return false
        if (recParam.type.classifierOrFail != func.returnType.classifierOrFail) return false

        val bodyCopy = (func.body as? IrBlockBody ?: return false).deepCopyWithSymbols()

        val fDps = context.irFactory.stageController.restrictTo(func) {
            createDpsSibling(container, func, dstType = ctorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(func, site, fDps)
        buildDpsBodyFromCopy(fDps, func, bodyCopy, site, recField)
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

    private fun hasReturnWhenWithCtorCall(func: IrSimpleFunction, candidates: Set<IrSimpleFunction>): Boolean {
        val funcSymbol = func.symbol
        var found = false
        func.body?.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                if (!found) element.acceptChildrenVoid(this)
            }
            override fun visitFunction(declaration: IrFunction) {}
            override fun visitReturn(expression: IrReturn) {
                if (found) return
                if (expression.returnTargetSymbol != funcSymbol) {
                    expression.acceptChildrenVoid(this)
                    return
                }
                val value = expression.value
                if (value is IrWhen) {
                    for (branch in value.branches) {
                        if (branchHasCtorWithCall(branch.result, candidates)) {
                            found = true
                            return
                        }
                    }
                }
            }
        })
        return found
    }

    private fun branchHasCtorWithCall(expr: IrExpression, candidates: Set<IrSimpleFunction>): Boolean = when (expr) {
        is IrConstructorCall -> expr.arguments.any { it is IrCall && it.symbol.owner in candidates }
        is IrBlock -> {
            val last = expr.statements.lastOrNull()
            last is IrConstructorCall && last.arguments.any { it is IrCall && it.symbol.owner in candidates }
        }
        is IrWhen -> expr.branches.any { branchHasCtorWithCall(it.result, candidates) }
        else -> false
    }

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

        val aCtorClass = siteA.ctorCall.symbol.owner.parentClassOrNull ?: return false
        val bCtorClass = siteB.ctorCall.symbol.owner.parentClassOrNull ?: return false

        val aCtorParams = siteA.ctorCall.symbol.owner.parameters
        val aRecParam = aCtorParams.getOrNull(siteA.recursiveArgIndex) ?: return false
        if (aRecParam.type.classifierOrFail != a.returnType.classifierOrFail) return false
        val bCtorParams = siteB.ctorCall.symbol.owner.parameters
        val bRecParam = bCtorParams.getOrNull(siteB.recursiveArgIndex) ?: return false
        if (bRecParam.type.classifierOrFail != b.returnType.classifierOrFail) return false

        val aRecField = findRecursiveBackingField(siteA.ctorCall, siteA.recursiveArgIndex) ?: return false
        val bRecField = findRecursiveBackingField(siteB.ctorCall, siteB.recursiveArgIndex) ?: return false

        val bodyCopyA = (a.body as? IrBlockBody ?: return false).deepCopyWithSymbols()
        val bodyCopyB = (b.body as? IrBlockBody ?: return false).deepCopyWithSymbols()

        val aDps = context.irFactory.stageController.restrictTo(a) {
            createDpsSibling(container, a, dstType = bCtorClass.defaultTypeNullable())
        }
        val bDps = context.irFactory.stageController.restrictTo(b) {
            createDpsSibling(container, b, dstType = aCtorClass.defaultTypeNullable())
        }

        transformOriginalBodyInPlace(a, siteA, bDps)
        transformOriginalBodyInPlace(b, siteB, aDps)
        buildDpsBodyFromCopy(aDps, a, bodyCopyA, siteA, bRecField, calleeSymbol = b.symbol, peerDps = bDps)
        buildDpsBodyFromCopy(bDps, b, bodyCopyB, siteB, aRecField, calleeSymbol = a.symbol, peerDps = aDps)

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

    private fun tryDpsLoopTransform(func: IrSimpleFunction): Boolean {
        val m = matchIterativeRecShape(func) ?: return false
        if (m.savedVars.isNotEmpty()) {
            for (sv in m.savedVars) {
                if (referencesSymbol(m.recPostEffects, sv.symbol)) return false
            }
        }
        if (m.recPostEffects.isEmpty()) return false
        if (referencesSymbol(m.recPostEffects, m.recCallVar.symbol)) return false

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
        val retVal = m.returnStmt.value
        if (retVal !is IrGetValue || retVal.symbol != m.resultVar.symbol) return false
        val body = func.body as IrBlockBody
        val stmts = body.statements
        val resultVarIdx = stmts.indexOf(m.resultVar)
        if (resultVarIdx < 0 || resultVarIdx + 1 != stmts.size - 1) return false

        buildDpsLoopBody(func, m, ctorCall, recArgIndex, ctorClass, recField)

        messageCollector?.report(
            CompilerMessageSeverity.STRONG_WARNING,
            "[wasm-tmc] DPS-loop transformed: ${func.fqNameWhenAvailable ?: func.name}",
        )
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
        for (stmt in copy.statements) {
            remapSymbols(stmt, paramMapping)
        }
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
        // Cheap structural rejections first; the self-call count needs a body traversal.
        val body = func.body as? IrBlockBody ?: return null
        val stmts = body.statements
        if (stmts.size < 2) return null

        val returnStmt = stmts.last() as? IrReturn ?: return null
        if (returnStmt.returnTargetSymbol != func.symbol) return null

        var resultVarIdx = -1
        for (i in stmts.size - 2 downTo 0) {
            val v = stmts[i] as? IrVariable ?: continue
            if (v.initializer is IrWhen) { resultVarIdx = i; break }
        }
        if (resultVarIdx < 0) return null
        if (countSelfCalls(func) != 1) return null

        val resultVar = stmts[resultVarIdx] as IrVariable
        val whenExpr = resultVar.initializer as IrWhen

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

        val postSymbols = mutableSetOf<IrValueSymbol>()
        val postRegion = matchedPostEffects!! + listOf(matchedRecFinalExpr) +
                stmts.subList(resultVarIdx + 1, stmts.size)
        for (element in postRegion) {
            element.acceptVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
                override fun visitGetValue(expression: IrGetValue) {
                    postSymbols += expression.symbol
                }
            })
        }

        val savedVars = preVars.filter { it.symbol in postSymbols }

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


    private fun remapSymbols(element: IrElement, mapping: Map<IrValueSymbol, IrValueSymbol>) {
        if (mapping.isEmpty()) return
        element.transform(object : AbstractVariableRemapper() {
            override fun remapVariable(value: IrValueDeclaration): IrValueDeclaration? = mapping[value.symbol]?.owner
        }, null)
    }

    private inline fun <reified T : IrElement> T.copyRemapped(mapping: Map<IrValueSymbol, IrValueSymbol>): T {
        val copy = deepCopyWithSymbols()
        remapSymbols(copy, mapping)
        return copy
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



