/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower

import org.jetbrains.kotlin.backend.common.FileLoweringPass
import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.addFunction
import org.jetbrains.kotlin.ir.builders.declarations.addValueParameter
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.IrBreakImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrReturnImpl
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classifierOrFail
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.types.typeWith
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.fqNameWhenAvailable
import org.jetbrains.kotlin.ir.util.patchDeclarationParents
import org.jetbrains.kotlin.ir.util.parentClassOrNull
import org.jetbrains.kotlin.ir.util.properties
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitor
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid

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
 * Functions not eligible for DPS (e.g. those with post-effects after the
 * recursive call, or SCCs of size >= 3) fall through to an iterative rewrite
 * using a forward/backward loop with an explicit stack.
 */
internal class WasmTailModConsLowering(private val context: WasmBackendContext) : FileLoweringPass {

    private val messageCollector: MessageCollector?
        get() = context.configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)

    override fun lower(irFile: IrFile) {
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

        for (f in allFunctions) {
            normalizeReturnWhen(f)
        }

        // Collect TMC sites per function and build edges (caller -> callee) within the file.
        val sitesByFunc = mutableMapOf<IrSimpleFunction, List<TmcSite>>()
        for (f in allFunctions) {
            val sites = collectTmcSites(f)
            if (sites.isNotEmpty()) sitesByFunc[f] = sites
        }

        if (sitesByFunc.isNotEmpty()) {
            val candidateSet = sitesByFunc.keys
            val edges: Map<IrSimpleFunction, List<IrSimpleFunction>> = sitesByFunc.mapValues { (_, sites) ->
                sites.mapNotNull { s ->
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

        for (f in allFunctions) {
            if (f in transformed) continue
            tryIterativeRecTransform(f)
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
                        builder.irCallConstructor(site.ctorCall.symbol, emptyList()).apply {
                            for (i in 0 until site.ctorCall.arguments.size) {
                                arguments[i] = if (i == site.recursiveArgIndex) builder.irNull() else site.ctorCall.arguments[i]
                            }
                        },
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

        val paramMapping: Map<IrValueSymbol, IrValueSymbol> = original.parameters.withIndex().associate { (i, p) ->
            p.symbol to dps.parameters[i].symbol
        }
        if (paramMapping.isNotEmpty()) remapSymbols(bodyCopy, paramMapping)

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
                                builder.irCallConstructor(ctorSymbol, emptyList()).apply {
                                    for (i in 0 until value.arguments.size) {
                                        arguments[i] = if (i == recArgIndex) builder.irNull() else value.arguments[i]
                                    }
                                },
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

    private fun createDpsSibling(
        container: IrDeclarationContainer,
        original: IrSimpleFunction,
        dstType: IrType,
    ): IrSimpleFunction {
        return context.irFactory.addFunction(container) {
            name = org.jetbrains.kotlin.name.Name.identifier(original.name.asString() + "\$tmcDps")
            visibility = DescriptorVisibilities.PRIVATE
            modality = Modality.FINAL
            returnType = context.irBuiltIns.unitType
            origin = IrDeclarationOrigin.DEFINED
            startOffset = original.startOffset
            endOffset = original.endOffset
        }.apply {
            // Copy original's value parameters then add `dst`.
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
                    val (call, viaVar) = effectiveCall(arg) ?: run {
                        if (arg is IrConst || arg is IrGetValue || arg is IrGetField || arg is IrGetObjectValue) {
                            continue
                        }
                        return
                    }
                    results += TmcSite(returnExpr, ctor, call, i, viaVar)
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
        val index = mutableMapOf<IrSimpleFunction, Int>()
        val lowLink = mutableMapOf<IrSimpleFunction, Int>()
        val onStack = mutableSetOf<IrSimpleFunction>()
        val stack = ArrayDeque<IrSimpleFunction>()
        var counter = 0
        val sccs = mutableListOf<List<IrSimpleFunction>>()

        fun strongConnect(v: IrSimpleFunction) {
            // Iterative Tarjan would be cleaner but the call graph here is small.
            // Recursion depth here is bounded by the number of TMC candidates in the file.
            val workStack = ArrayDeque<Pair<IrSimpleFunction, Iterator<IrSimpleFunction>>>()
            index[v] = counter
            lowLink[v] = counter
            counter++
            stack.addLast(v)
            onStack += v
            workStack.addLast(v to (edges[v]?.iterator() ?: emptyList<IrSimpleFunction>().iterator()))

            while (workStack.isNotEmpty()) {
                val (cur, it) = workStack.last()
                var descended = false
                while (it.hasNext()) {
                    val w = it.next()
                    if (w !in index) {
                        index[w] = counter
                        lowLink[w] = counter
                        counter++
                        stack.addLast(w)
                        onStack += w
                        workStack.addLast(w to (edges[w]?.iterator() ?: emptyList<IrSimpleFunction>().iterator()))
                        descended = true
                        break
                    } else if (w in onStack) {
                        lowLink[cur] = minOf(lowLink[cur]!!, index[w]!!)
                    }
                }
                if (descended) continue

                if (lowLink[cur] == index[cur]) {
                    val scc = mutableListOf<IrSimpleFunction>()
                    while (true) {
                        val w = stack.removeLast()
                        onStack -= w
                        scc += w
                        if (w === cur) break
                    }
                    sccs += scc
                }
                workStack.removeLast()
                if (workStack.isNotEmpty()) {
                    val parent = workStack.last().first
                    lowLink[parent] = minOf(lowLink[parent]!!, lowLink[cur]!!)
                }
            }
        }

        for (v in nodes) {
            if (v !in index) strongConnect(v)
        }
        return sccs
    }

    // -------------------------------------------------------------- helpers

    // -------------------------------------------------------------- general iterative rec

    private data class IterativeRecMatch(
        val earlyReturnWhens: List<IrWhen>,
        val preVars: List<IrVariable>,
        val resultVar: IrVariable,
        val recursiveBranchIdx: Int,
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
            override fun visitElement(element: IrElement) = element.acceptChildrenVoid(this)
            override fun visitCall(expression: IrCall) {
                if (expression.symbol == func.symbol) count++
                expression.acceptChildrenVoid(this)
            }
        })
        return count
    }

    private fun matchIterativeRecShape(func: IrSimpleFunction): IterativeRecMatch? {
        if (countSelfCalls(func) != 1) return null
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

        val resultVar = stmts[resultVarIdx] as IrVariable
        val whenExpr = resultVar.initializer as IrWhen

        var recBranchIdx = -1
        var matchedRecCallVar: IrVariable? = null
        var matchedPreEffects: List<IrStatement>? = null
        var matchedPostEffects: List<IrStatement>? = null
        var matchedRecFinalExpr: IrExpression? = null

        for ((idx, branch) in whenExpr.branches.withIndex()) {
            val block = branch.result as? IrBlock ?: continue
            val blockStmts = block.statements
            if (blockStmts.isEmpty()) continue

            for ((j, s) in blockStmts.withIndex()) {
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
            recursiveBranchIdx = recBranchIdx,
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

    private fun tryIterativeRecTransform(func: IrSimpleFunction) {
        val m = matchIterativeRecShape(func) ?: return
        val body = func.body as IrBlockBody
        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)

        messageCollector?.report(
            CompilerMessageSeverity.STRONG_WARNING,
            "[wasm-tmc] match: func=${func.name}, earlyReturnWhens=${m.earlyReturnWhens.size}, " +
                    "preVars=${m.preVars.size}, savedVars=${m.savedVars.size}, " +
                    "recPreEffects=${m.recPreEffects.size}, recPostEffects=${m.recPostEffects.size}",
        )

        val recCall = m.recCallVar.initializer as IrCall
        val sameParams = func.parameters.indices.all { i ->
            val arg = recCall.arguments[i]
            arg is IrGetValue && arg.symbol == func.parameters[i].symbol
        }

        val funcCopy = func.deepCopyWithSymbols()
        val mCopy = matchIterativeRecShape(funcCopy)!!
        val copyParamMap: Map<IrValueSymbol, IrValueSymbol> =
            funcCopy.parameters.withIndex().associate { (i, cp) -> cp.symbol to func.parameters[i].symbol }

        val needsStack = m.savedVars.isNotEmpty()
        val anyNType = context.irBuiltIns.anyNType

        val newBody = builder.irBlockBody {
            val paramVars = if (!sameParams) {
                func.parameters.map { p ->
                    createTmpVariable(irGet(p), nameHint = "\$${p.name}", isMutable = true)
                }
            } else null
            val paramMap: Map<IrValueSymbol, IrValueSymbol> = if (paramVars != null) {
                func.parameters.withIndex().associate { (i, p) -> p.symbol to paramVars[i].symbol }
            } else emptyMap()

            val combinedMap = copyParamMap + paramMap

            val depthVar = createTmpVariable(irInt(0), nameHint = "\$depth", isMutable = true)

            val stackVar: IrVariable?
            val stackCapVar: IrVariable?
            val arrayType = context.irBuiltIns.arrayClass.typeWith(anyNType)
            if (needsStack) {
                stackCapVar = createTmpVariable(irInt(16), nameHint = "\$stackCap", isMutable = true)
                stackVar = createTmpVariable(
                    irCall(context.irBuiltIns.arrayOfNulls, arrayType).apply {
                        typeArguments[0] = anyNType
                        arguments[0] = irInt(16 * m.savedVars.size)
                    },
                    nameHint = "\$stack",
                    isMutable = true,
                )
            } else {
                stackVar = null
                stackCapVar = null
            }

            +irWhile().apply {
                condition = irTrue()
                this.body = irBlock {
                    for (w in m.earlyReturnWhens) {
                        transformReturnsToBreaks(w, func.symbol, this@apply)
                        if (paramMap.isNotEmpty()) remapSymbols(w, paramMap)
                        +w
                    }
                    for (v in m.preVars) {
                        if (paramMap.isNotEmpty()) remapSymbols(v, paramMap)
                        +v
                    }

                    if (paramMap.isNotEmpty()) remapSymbols(m.recursiveBranchCondition, paramMap)
                    +irIfThenElse(
                        context.irBuiltIns.unitType,
                        m.recursiveBranchCondition,
                        irBlock {},
                        irBreak(this@apply),
                    )

                    for (effect in m.recPreEffects) {
                        if (paramMap.isNotEmpty()) remapSymbols(effect, paramMap)
                        +effect
                    }

                    if (needsStack && stackVar != null && stackCapVar != null) {
                        +irIfThen(
                            context.irBuiltIns.unitType,
                            irEquals(irGet(depthVar), irGet(stackCapVar)),
                            irBlock {
                                val newCap = createTmpVariable(
                                    irCallOp(intTimesSymbol, context.irBuiltIns.intType,
                                        irGet(stackCapVar), irInt(2)),
                                    nameHint = "\$newCap",
                                )
                                val newArr = createTmpVariable(
                                    irCall(context.irBuiltIns.arrayOfNulls, arrayType).apply {
                                        typeArguments[0] = anyNType
                                        arguments[0] = irCallOp(intTimesSymbol, context.irBuiltIns.intType,
                                            irGet(newCap), irInt(m.savedVars.size))
                                    },
                                    nameHint = "\$newArr",
                                )
                                val copyIdx = createTmpVariable(irInt(0), nameHint = "\$ci", isMutable = true)
                                val oldLen = irCallOp(intTimesSymbol, context.irBuiltIns.intType,
                                    irGet(stackCapVar), irInt(m.savedVars.size))
                                +irWhile().apply {
                                    condition = irNotEquals(irGet(copyIdx), oldLen)
                                    this.body = irBlock {
                                        +irCall(arraySetSymbol).apply {
                                            arguments[0] = irGet(newArr)
                                            arguments[1] = irGet(copyIdx)
                                            arguments[2] = irCall(arrayGetSymbol).apply {
                                                arguments[0] = irGet(stackVar)
                                                arguments[1] = irGet(copyIdx)
                                            }
                                        }
                                        +irSet(copyIdx.symbol, irCallOp(intPlusSymbol,
                                            context.irBuiltIns.intType, irGet(copyIdx), irInt(1)))
                                    }
                                }
                                +irSet(stackVar.symbol, irGet(newArr))
                                +irSet(stackCapVar.symbol, irGet(newCap))
                            }
                        )
                        for ((si, sv) in m.savedVars.withIndex()) {
                            val idx = irCallOp(intPlusSymbol, context.irBuiltIns.intType,
                                irCallOp(intTimesSymbol, context.irBuiltIns.intType,
                                    irGet(depthVar), irInt(m.savedVars.size)),
                                irInt(si))
                            +irCall(arraySetSymbol).apply {
                                arguments[0] = irGet(stackVar)
                                arguments[1] = idx
                                arguments[2] = irGet(sv)
                            }
                        }
                    }

                    if (paramVars != null) {
                        val recCallCopy = mCopy.recCallVar.initializer as IrCall
                        val argTmps = func.parameters.mapIndexed { i, _ ->
                            val arg = recCallCopy.arguments[i]!!
                            remapSymbols(arg, combinedMap)
                            createTmpVariable(arg, nameHint = "next$i")
                        }
                        for ((i, tmp) in argTmps.withIndex()) {
                            +irSet(paramVars[i].symbol, irGet(tmp))
                        }
                    }

                    +irSet(depthVar.symbol, irCallOp(intPlusSymbol, context.irBuiltIns.intType, irGet(depthVar), irInt(1)))
                }
            }

            for (v in mCopy.preVars) {
                remapSymbols(v, combinedMap)
                +v
            }

            val baseCaseExpr = buildGeneralBaseCaseWhen(mCopy, combinedMap)
            val resultVar = createTmpVariable(
                baseCaseExpr,
                nameHint = "\$result",
                isMutable = true,
                irType = func.returnType,
            )

            val postStmtsCopy = mCopy.returnStmt
            val resultVarCopySymbol = mCopy.resultVar.symbol
            remapSymbols(postStmtsCopy, combinedMap + mapOf(resultVarCopySymbol to resultVar.symbol))
            if (postStmtsCopy.value is IrGetValue &&
                (postStmtsCopy.value as IrGetValue).symbol == resultVar.symbol
            ) {
                // no-op: simple return of result
            } else {
                val postWrapCopy = postStmtsCopy.value.deepCopyWithSymbols()
                remapSymbols(postWrapCopy, mapOf(resultVarCopySymbol to resultVar.symbol) + combinedMap)
                +irSet(resultVar.symbol, postWrapCopy)
            }

            +irWhile().apply {
                condition = irNotEquals(irGet(depthVar), irInt(0))
                this.body = irBlock {
                    +irSet(depthVar.symbol, irCallOp(intMinusSymbol, context.irBuiltIns.intType, irGet(depthVar), irInt(1)))

                    val savedVarMap = mutableMapOf<IrValueSymbol, IrValueSymbol>()
                    if (needsStack && stackVar != null) {
                        for ((si, sv) in m.savedVars.withIndex()) {
                            val restoredVarCopy = sv.deepCopyWithSymbols()
                            val idx = irCallOp(intPlusSymbol, context.irBuiltIns.intType,
                                irCallOp(intTimesSymbol, context.irBuiltIns.intType,
                                    irGet(depthVar), irInt(m.savedVars.size)),
                                irInt(si))
                            restoredVarCopy.initializer = irImplicitCast(
                                irCall(arrayGetSymbol).apply {
                                    arguments[0] = irGet(stackVar)
                                    arguments[1] = idx
                                },
                                sv.type,
                            )
                            +restoredVarCopy
                            savedVarMap[sv.symbol] = restoredVarCopy.symbol
                            savedVarMap[mCopy.savedVars[si].symbol] = restoredVarCopy.symbol
                        }
                    }

                    val backwardRemap = savedVarMap + paramMap
                    for (effect in m.recPostEffects) {
                        if (backwardRemap.isNotEmpty()) {
                            val effectCopy = (effect as IrElement).deepCopyWithSymbols() as IrStatement
                            remapSymbols(effectCopy, backwardRemap)
                            +effectCopy
                        } else {
                            +effect
                        }
                    }

                    val recVarSymbol = m.recCallVar.symbol
                    val finalExprCopy = m.recFinalExpr.deepCopyWithSymbols()
                    remapSymbols(finalExprCopy, savedVarMap + mapOf(recVarSymbol to resultVar.symbol))
                    if (paramMap.isNotEmpty()) remapSymbols(finalExprCopy, paramMap)
                    +irSet(resultVar.symbol, finalExprCopy)

                    val returnExprCopy2 = mCopy.returnStmt.value.deepCopyWithSymbols()
                    val returnRemap = combinedMap + mapOf(resultVarCopySymbol to resultVar.symbol) + savedVarMap
                    remapSymbols(returnExprCopy2, returnRemap)
                    if (!(returnExprCopy2 is IrGetValue && returnExprCopy2.symbol == resultVar.symbol)) {
                        +irSet(resultVar.symbol, returnExprCopy2)
                    }
                }
            }

            +irReturn(irGet(resultVar))
        }

        body.statements.clear()
        body.statements += newBody.statements
        body.patchDeclarationParents(func)

        messageCollector?.report(
            CompilerMessageSeverity.STRONG_WARNING,
            "[wasm-tmc] iterative-rec transformed: ${func.fqNameWhenAvailable ?: func.name}",
        )
    }

    private val intPlusSymbol: IrSimpleFunctionSymbol by lazy {
        context.irBuiltIns.intClass.owner.declarations
            .filterIsInstance<IrSimpleFunction>()
            .single {
                it.name.asString() == "plus"
                        && it.returnType == context.irBuiltIns.intType
                        && it.parameters.last().type == context.irBuiltIns.intType
            }.symbol
    }

    private val intMinusSymbol: IrSimpleFunctionSymbol by lazy {
        context.irBuiltIns.intClass.owner.declarations
            .filterIsInstance<IrSimpleFunction>()
            .single {
                it.name.asString() == "minus"
                        && it.returnType == context.irBuiltIns.intType
                        && it.parameters.last().type == context.irBuiltIns.intType
            }.symbol
    }

    private val intTimesSymbol: IrSimpleFunctionSymbol by lazy {
        context.irBuiltIns.intClass.owner.declarations
            .filterIsInstance<IrSimpleFunction>()
            .single {
                it.name.asString() == "times"
                        && it.returnType == context.irBuiltIns.intType
                        && it.parameters.last().type == context.irBuiltIns.intType
            }.symbol
    }

    private val arrayGetSymbol: IrSimpleFunctionSymbol by lazy {
        context.irBuiltIns.arrayClass.owner.declarations
            .filterIsInstance<IrSimpleFunction>()
            .first { it.name.asString() == "get" }
            .symbol
    }

    private val arraySetSymbol: IrSimpleFunctionSymbol by lazy {
        context.irBuiltIns.arrayClass.owner.declarations
            .filterIsInstance<IrSimpleFunction>()
            .first { it.name.asString() == "set" }
            .symbol
    }

    private fun IrBlockBodyBuilder.buildGeneralBaseCaseWhen(
        mCopy: IterativeRecMatch,
        combinedMap: Map<IrValueSymbol, IrValueSymbol>,
    ): IrExpression {
        val branches = mutableListOf<IrBranch>()
        for (w in mCopy.earlyReturnWhens) {
            for (branch in w.branches) {
                remapSymbols(branch, combinedMap)
                val value = (branch.result as IrReturn).value
                branches += irBranch(branch.condition, value)
            }
        }
        for (branch in mCopy.baseBranches) {
            remapSymbols(branch, combinedMap)
            branches += branch
        }
        return irWhen(context.irBuiltIns.anyNType, branches)
    }

    private fun remapSymbols(element: IrElement, mapping: Map<IrValueSymbol, IrValueSymbol>) {
        element.transform(object : IrTransformer<Nothing?>() {
            override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
                val newSym = mapping[expression.symbol] ?: return super.visitGetValue(expression, data)
                return IrGetValueImpl(expression.startOffset, expression.endOffset, expression.type, newSym)
            }
        }, null)
    }

    private fun transformReturnsToBreaks(
        element: IrElement,
        funcSymbol: IrSimpleFunctionSymbol,
        loop: IrLoop,
    ) {
        element.transform(object : IrTransformer<Nothing?>() {
            override fun visitReturn(expression: IrReturn, data: Nothing?): IrExpression {
                if (expression.returnTargetSymbol != funcSymbol) return super.visitReturn(expression, data)
                return IrBreakImpl(expression.startOffset, expression.endOffset, context.irBuiltIns.nothingType, loop)
            }
        }, null)
    }

}

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

private fun IrElement.acceptVoid(visitor: IrVisitorVoid) = accept(visitor, null)
