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
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.*
import org.jetbrains.kotlin.ir.builders.declarations.addFunction
import org.jetbrains.kotlin.ir.builders.declarations.addValueParameter
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.*
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.symbols.IrSimpleFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueParameterSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classifierOrFail
import org.jetbrains.kotlin.ir.types.defaultType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.util.deepCopyWithSymbols
import org.jetbrains.kotlin.ir.util.fqNameWhenAvailable
import org.jetbrains.kotlin.ir.util.getPropertyGetter
import org.jetbrains.kotlin.ir.util.parentClassOrNull
import org.jetbrains.kotlin.ir.util.properties
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.ir.visitors.IrVisitor
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid

/**
 * Tail Modulo Cons (TMC) lowering for Kotlin/Wasm.
 *
 * Detects functions whose return statement matches one of
 *
 *     return Ctor(c(p), f(next(p)))
 *     val v = f(next(p)); return Ctor(c(p), v)
 *
 * (Allain et al. POPL'25 [@tail_mod_cons], adapted from OCaml's Lambda IR to
 * Kotlin/Wasm IR), groups them into strongly-connected components of the
 * call graph, and applies one of two rewrites:
 *
 *   - SCC of size 1 (self-recursive): a two-phase iterative rewrite that uses
 *     the function's own Ctor as a reversed accumulator and then walks it
 *     back, reusing exactly the input's heap budget.
 *
 *   - SCC of size N >= 2 (mutual-recursive), where every member has a TMC
 *     site whose callee is in the same SCC: a destination-passing-style
 *     rewrite. For each member f we synthesise a sibling `f_dps(args..., dst)`
 *     that writes its result into `dst`'s recursive field (via IrSetField on
 *     the backing field — sound because Kotlin/Wasm declares all instance
 *     fields with `isMutable = true` at the wasm level; see TypeGenerator.kt)
 *     and tail-calls the other member's _dps. The original `f` becomes
 *     `f(args) = allocate head; f_or_other_dps(args', head); return head`.
 *     This matches the OCaml paper's pseudo-code on p.3 (one partial cell
 *     allocated per recursion level, each mutated exactly once).
 *
 * Both rewrites preserve construction order, so they are a sound replacement
 * for the original direct-style functions.
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

        // Collect TMC sites per function and build edges (caller -> callee) within the file.
        val sitesByFunc = mutableMapOf<IrSimpleFunction, List<TmcSite>>()
        for (f in allFunctions) {
            val sites = collectTmcSites(f)
            if (sites.isNotEmpty()) sitesByFunc[f] = sites
        }
        if (sitesByFunc.isEmpty()) return

        val candidateSet = sitesByFunc.keys
        // edges: for each candidate, list of callees that are also candidates (intra-file SCC)
        val edges: Map<IrSimpleFunction, List<IrSimpleFunction>> = sitesByFunc.mapValues { (_, sites) ->
            sites.mapNotNull { s ->
                val callee = s.recursiveCall.symbol.owner
                if (callee in candidateSet) callee else null
            }.distinct()
        }
        val sccs = computeSccs(candidateSet.toList(), edges)

        // Report any non-singleton SCC so we know what real-world code actually contains.
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
                        val transformed = trySelfRecTransform(f)
                        reportSites(f, sites, transformed)
                    } else {
                        reportSites(f, sites, transformed = false)
                    }
                }
                scc.size == 2 -> {
                    val transformed = tryPairwiseMutualRecTransform(scc, sitesByFunc, irFile)
                    for (f in scc) reportSites(f, sitesByFunc[f]!!, transformed)
                }
                else -> {
                    // N>2 not yet supported — report only.
                    for (f in scc) reportSites(f, sitesByFunc[f]!!, transformed = false)
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

    // -------------------------------------------------------------- self-rec

    /** Returns true on a successful transform. */
    private fun trySelfRecTransform(func: IrSimpleFunction): Boolean {
        val match = matchSimpleChainShape(func) ?: return false
        val body = func.body as? IrBlockBody ?: return false
        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)
        val newStatements = builder.irBlockBody { buildSelfRecBody(func, match) }.statements
        body.statements.clear()
        body.statements += newStatements
        return true
    }

    private fun IrBlockBodyBuilder.buildSelfRecBody(func: IrSimpleFunction, m: SimpleChainMatch) {
        val retType = func.returnType
        val accVar = createTmpVariable(irNull(), nameHint = "tmcAcc", isMutable = true, irType = retType)
        val curVar = createTmpVariable(irGet(m.paramSymbol.owner), nameHint = "tmcCur", isMutable = true)
        +irWhile().apply {
            condition = irTrue()
            body = irBlock {
                +irIfThen(
                    context.irBuiltIns.unitType,
                    rebindParamUsage(m.baseCondition, m.paramSymbol, curVar.symbol),
                    irBreak(this@apply),
                )
                +irSet(
                    accVar.symbol,
                    irCallConstructor(m.ctorCall.symbol, emptyList()).apply {
                        arguments[0] = rebindParamUsage(m.cArg, m.paramSymbol, curVar.symbol)
                        arguments[1] = irGet(accVar)
                    },
                )
                +irSet(
                    curVar.symbol,
                    rebindParamUsage(m.nextArg, m.paramSymbol, curVar.symbol),
                )
            }
        }
        val resultVar = createTmpVariable(
            rebindParamUsage(m.baseReturnValue, m.paramSymbol, curVar.symbol),
            nameHint = "tmcResult",
            isMutable = true,
            irType = retType,
        )
        val walkerVar = createTmpVariable(irGet(accVar), nameHint = "tmcWalker", isMutable = true, irType = retType)
        val cArgGetter = m.cArgGetter ?: error("expected c-arg getter for ${func.name}")
        val nextGetter = m.nextGetter ?: error("expected next-getter for ${func.name}")

        +irWhile().apply {
            condition = irNotEquals(irGet(walkerVar), irNull())
            body = irBlock {
                +irSet(
                    resultVar.symbol,
                    irCallConstructor(m.ctorCall.symbol, emptyList()).apply {
                        arguments[0] = irCall(cArgGetter).apply { arguments[0] = irGet(walkerVar) }
                        arguments[1] = irGet(resultVar)
                    },
                )
                +irSet(
                    walkerVar.symbol,
                    irCall(nextGetter).apply { arguments[0] = irGet(walkerVar) },
                )
            }
        }
        +irReturn(irGet(resultVar))
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
        irFile: IrFile,
    ): Boolean {
        val a = scc[0]
        val b = scc[1]
        val ma = matchSimpleChainShape(a) ?: return false
        val mb = matchSimpleChainShape(b) ?: return false
        // Verify cross-reference: A's recursive call must target B, and vice versa.
        val aSites = sitesByFunc[a].orEmpty()
        val bSites = sitesByFunc[b].orEmpty()
        if (aSites.none { it.recursiveCall.symbol == b.symbol }) return false
        if (bSites.none { it.recursiveCall.symbol == a.symbol }) return false
        // Both functions must live in the same IrDeclarationContainer (IrFile) so we can
        // add sibling _dps declarations.
        val container = a.parent as? IrDeclarationContainer ?: return false
        if (b.parent !== container) return false

        // Find backing field for each Ctor's recursive slot. If either is missing
        // (e.g. computed property), bail out.
        val aRecField = findRecursiveBackingField(ma.ctorCall) ?: return false
        val bRecField = findRecursiveBackingField(mb.ctorCall) ?: return false

        // Build placeholder DPS functions first so we have symbols available for cross-references.
        // Use `restrictTo` so that the IC IrFactory can compute a parent signature for the new
        // declarations from the original function's scope.
        val aDps = context.irFactory.stageController.restrictTo(a) {
            createDpsSibling(container, a, dstType = mb.ctorClass.defaultTypeNullable())
        }
        val bDps = context.irFactory.stageController.restrictTo(b) {
            createDpsSibling(container, b, dstType = ma.ctorClass.defaultTypeNullable())
        }

        rewriteOriginalDirect(a, ma, peerDps = bDps)
        rewriteOriginalDirect(b, mb, peerDps = aDps)
        fillDpsBody(aDps, ma, peerDps = bDps, dstField = bRecField, originalParam = a.parameters[0])
        fillDpsBody(bDps, mb, peerDps = aDps, dstField = aRecField, originalParam = b.parameters[0])

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

    private fun findRecursiveBackingField(ctor: IrConstructorCall): IrField? {
        val cls = ctor.symbol.owner.parentClassOrNull ?: return null
        val ctorParams = ctor.symbol.owner.parameters
        val lastParamName = ctorParams.lastOrNull()?.name?.asString() ?: return null
        val prop = cls.properties.firstOrNull { it.name.asString() == lastParamName } ?: return null
        return prop.backingField
    }

    /**
     * Rewrites the original direct-style function so that it allocates the head cell and
     * kicks off the DPS chain by calling `peerDps`.
     */
    private fun rewriteOriginalDirect(
        func: IrSimpleFunction,
        m: SimpleChainMatch,
        peerDps: IrSimpleFunction,
    ) {
        val body = func.body as? IrBlockBody ?: return
        val builder = context.createIrBuilder(func.symbol, body.startOffset, body.endOffset)
        val newStatements = builder.irBlockBody {
            +irIfThen(
                context.irBuiltIns.unitType,
                m.baseCondition,
                irReturn(m.baseReturnValue),
            )
            val head = createTmpVariable(
                irCallConstructor(m.ctorCall.symbol, emptyList()).apply {
                    arguments[0] = m.cArg
                    arguments[1] = irNull()
                },
                nameHint = "tmcHead",
            )
            +irCall(peerDps.symbol).apply {
                // first arg: next(p)
                arguments[0] = m.nextArg
                // second arg: head
                arguments[1] = irGet(head)
            }
            +irReturn(irGet(head))
        }.statements
        body.statements.clear()
        body.statements += newStatements
    }

    /**
     * Fills in the body of the DPS sibling: writes either the base value or a fresh cell
     * into `dst.recursiveField`, then either returns (base case) or tail-calls the peer DPS.
     */
    private fun fillDpsBody(
        dps: IrSimpleFunction,
        m: SimpleChainMatch,
        peerDps: IrSimpleFunction,
        dstField: IrField,
        originalParam: IrValueParameter,
    ) {
        val builder = context.createIrBuilder(dps.symbol, dps.startOffset, dps.endOffset)
        val newParam = dps.parameters[0]
        val dstParam = dps.parameters[1]
        // Re-bind the matcher's expressions (which reference the ORIGINAL function's parameter)
        // to the new DPS function's parameter.
        val baseCondition = rebindParamUsage(m.baseCondition, originalParam.symbol, newParam.symbol)
        val baseReturnValue = rebindParamUsage(m.baseReturnValue, originalParam.symbol, newParam.symbol)
        val cArg = rebindParamUsage(m.cArg, originalParam.symbol, newParam.symbol)
        val nextArg = rebindParamUsage(m.nextArg, originalParam.symbol, newParam.symbol)

        dps.body = builder.irBlockBody {
            // Base case: dst.recField = base; return Unit
            +irIfThen(
                context.irBuiltIns.unitType,
                baseCondition,
                irBlock {
                    +irSetField(irGet(dstParam), dstField, baseReturnValue)
                    +irReturn(irGet(dstParam).let { irGetObject(context.irBuiltIns.unitClass) })
                },
            )
            // Recursive step: allocate cell, dst.recField = cell, tail-call peer.
            val cell = createTmpVariable(
                irCallConstructor(m.ctorCall.symbol, emptyList()).apply {
                    arguments[0] = cArg
                    arguments[1] = irNull()
                },
                nameHint = "tmcCell",
            )
            +irSetField(irGet(dstParam), dstField, irGet(cell))
            +irReturn(
                irCall(peerDps.symbol).apply {
                    arguments[0] = nextArg
                    arguments[1] = irGet(cell)
                },
            )
        }
    }

    // -------------------------------------------------------------- shape matcher

    private fun matchSimpleChainShape(func: IrSimpleFunction): SimpleChainMatch? {
        val intType = context.irBuiltIns.intType
        if (func.parameters.size != 1) return null
        val param = func.parameters[0]
        if (param.type != intType) return null

        val body = func.body as? IrBlockBody ?: return null
        if (body.statements.isEmpty()) return null
        val ifStmt = body.statements[0] as? IrWhen ?: return null
        val ifBranch = ifStmt.branches.firstOrNull() ?: return null
        val baseReturn = ifBranch.result as? IrReturn ?: return null
        if (baseReturn.returnTargetSymbol != func.symbol) return null

        val tail = body.statements.drop(1)
        val rec = matchRecursiveTail(tail, func) ?: return null

        val ctorClass = rec.ctor.symbol.owner.parentClassOrNull ?: return null
        val ctorParams = rec.ctor.symbol.owner.parameters
        if (ctorParams.size != 2) return null
        val lastParam = ctorParams[1]
        if (lastParam.type.classifierOrFail != func.returnType.classifierOrFail) return null

        val valueGetter = ctorClass.symbol.getPropertyGetter(ctorParams[0].name.asString())
        val nextGetter = ctorClass.symbol.getPropertyGetter(ctorParams[1].name.asString())

        return SimpleChainMatch(
            paramSymbol = param.symbol,
            baseCondition = ifBranch.condition,
            baseReturnValue = baseReturn.value,
            cArg = rec.cArg,
            nextArg = rec.nextArg,
            ctorCall = rec.ctor,
            ctorClass = ctorClass,
            cArgGetter = valueGetter,
            nextGetter = nextGetter,
        )
    }

    private fun matchRecursiveTail(
        tail: List<org.jetbrains.kotlin.ir.IrStatement>,
        func: IrSimpleFunction,
    ): RecursiveSite? {
        if (tail.size == 1) {
            val ret = tail[0] as? IrReturn ?: return null
            if (ret.returnTargetSymbol != func.symbol) return null
            val ctor = ret.value as? IrConstructorCall ?: return null
            if (ctor.arguments.size != 2) return null
            val recCall = ctor.arguments[1] as? IrCall ?: return null
            val nextArg = recCall.arguments.getOrNull(0) ?: return null
            return RecursiveSite(recCall, ctor, ctor.arguments[0]!!, nextArg)
        }
        if (tail.size == 2) {
            val varDecl = tail[0] as? IrVariable ?: return null
            val recCall = varDecl.initializer as? IrCall ?: return null
            val ret = tail[1] as? IrReturn ?: return null
            if (ret.returnTargetSymbol != func.symbol) return null
            val ctor = ret.value as? IrConstructorCall ?: return null
            if (ctor.arguments.size != 2) return null
            val lastArg = ctor.arguments[1] as? IrGetValue ?: return null
            if (lastArg.symbol.owner != varDecl) return null
            val nextArg = recCall.arguments.getOrNull(0) ?: return null
            return RecursiveSite(recCall, ctor, ctor.arguments[0]!!, nextArg)
        }
        return null
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

    private fun rebindParamUsage(
        expr: IrExpression,
        paramSymbol: IrValueParameterSymbol,
        curVarSymbol: IrValueSymbol,
    ): IrExpression {
        val copy = expr.deepCopyWithSymbols()
        val transformer = ParamRebinder(paramSymbol, curVarSymbol)
        return copy.transform(transformer, null)
    }

    private class ParamRebinder(
        private val paramSymbol: IrValueParameterSymbol,
        private val curVarSymbol: IrValueSymbol,
    ) : IrTransformer<Nothing?>() {
        override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
            if (expression.symbol == paramSymbol) {
                return IrGetValueImpl(
                    expression.startOffset, expression.endOffset,
                    expression.type, curVarSymbol,
                )
            }
            return super.visitGetValue(expression, data)
        }
    }

    private data class SimpleChainMatch(
        val paramSymbol: IrValueParameterSymbol,
        val baseCondition: IrExpression,
        val baseReturnValue: IrExpression,
        val cArg: IrExpression,
        val nextArg: IrExpression,
        val ctorCall: IrConstructorCall,
        val ctorClass: IrClass,
        val cArgGetter: IrSimpleFunctionSymbol?,
        val nextGetter: IrSimpleFunctionSymbol?,
    )

    private data class RecursiveSite(
        val recCall: IrCall,
        val ctor: IrConstructorCall,
        val cArg: IrExpression,
        val nextArg: IrExpression,
    )
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
