/*
 * Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.backend.wasm.lower.cps

import org.jetbrains.kotlin.backend.common.lower.createIrBuilder
import org.jetbrains.kotlin.backend.wasm.WasmBackendContext
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibilities
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.ir.UNDEFINED_OFFSET
import org.jetbrains.kotlin.ir.builders.declarations.*
import org.jetbrains.kotlin.ir.builders.irBlockBody
import org.jetbrains.kotlin.ir.builders.irDelegatingConstructorCall
import org.jetbrains.kotlin.ir.builders.irGet
import org.jetbrains.kotlin.ir.builders.irSetField
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.builders.IrBlockBuilder
import org.jetbrains.kotlin.ir.builders.irAs
import org.jetbrains.kotlin.ir.IrBuiltIns
import org.jetbrains.kotlin.ir.declarations.*
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.IrGetValue
import org.jetbrains.kotlin.ir.expressions.IrSetValue
import org.jetbrains.kotlin.ir.expressions.impl.IrConstImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrGetValueImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrInstanceInitializerCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.IrSetValueImpl
import org.jetbrains.kotlin.ir.symbols.IrConstructorSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueSymbol
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.isPrimitiveType
import org.jetbrains.kotlin.ir.types.makeNullable
import org.jetbrains.kotlin.ir.util.constructors
import org.jetbrains.kotlin.ir.util.createThisReceiverParameter
import org.jetbrains.kotlin.ir.util.defaultType
import org.jetbrains.kotlin.ir.util.defaultValueForType
import org.jetbrains.kotlin.ir.visitors.IrTransformer
import org.jetbrains.kotlin.name.Name

/**
 * Native recursion budget before switching to the heap-frame trampoline.
 */
internal const val HYBRID_DEPTH_THRESHOLD = 512

/**
 * Layout of a CPS frame class produced by [buildCpsFrameClass].
 */
internal class CpsFrameLayout(
    val frameClass: IrClass,
    val outerField: IrField,
    val resumeField: IrField,
    val poolFields: Map<IrType, List<IrField>>,
    val ctorParamOrder: List<Pair<IrType, Int>>,
    val ctor: IrConstructorSymbol,
)

/**
 * Maps an IR type to the typed slot pool it belongs to.
 */
internal fun cpsPoolOf(type: IrType, context: WasmBackendContext): IrType {
    val builtIns = context.irBuiltIns
    return when (type) {
        builtIns.intType -> builtIns.intType
        builtIns.booleanType -> builtIns.booleanType
        builtIns.charType -> builtIns.charType
        builtIns.longType -> builtIns.longType
        builtIns.floatType -> builtIns.floatType
        builtIns.doubleType -> builtIns.doubleType
        else -> builtIns.anyNType
    }
}

/**
 * The fixed pool emission order, ensuring constructor argument layout
 * is stable.
 */
internal fun cpsPoolOrder(context: WasmBackendContext): List<IrType> {
    val builtIns = context.irBuiltIns
    return listOf(
        builtIns.intType, builtIns.booleanType, builtIns.charType,
        builtIns.longType, builtIns.floatType, builtIns.doubleType, builtIns.anyNType,
    )
}

/**
 * Returns a zero/null default value expression for a given IR type.
 * Delegates to [IrConstImpl.defaultValueForType].
 */
internal fun cpsDefaultValue(type: IrType): IrExpression =
    IrConstImpl.defaultValueForType(UNDEFINED_OFFSET, UNDEFINED_OFFSET, type)

/**
 * Single-letter prefix for pool field naming.
 */
internal fun cpsPoolPrefix(pool: IrType, context: WasmBackendContext): String {
    val builtIns = context.irBuiltIns
    return when (pool) {
        builtIns.intType -> "i"
        builtIns.booleanType -> "z"
        builtIns.charType -> "c"
        builtIns.longType -> "j"
        builtIns.floatType -> "f"
        builtIns.doubleType -> "d"
        else -> "r"
    }
}

/**
 * Builds a CPS frame class with an outer-link field, a resume-state
 * field, and typed slots organized by [poolSizes]. Returns a
 * [CpsFrameLayout] describing the generated structure.
 */
internal fun buildCpsFrameClass(
    context: WasmBackendContext,
    irFile: IrFile,
    baseName: String,
    poolOrder: List<IrType>,
    poolSizes: Map<IrType, Int>,
): CpsFrameLayout {
    val builtIns = context.irBuiltIns
    val cls = context.irFactory.buildClass {
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
        name = Name.identifier(baseName)
        visibility = DescriptorVisibilities.PRIVATE
        modality = Modality.FINAL
        kind = ClassKind.CLASS
    }.apply {
        parent = irFile
        irFile.declarations += this
        createThisReceiverParameter()
        superTypes = listOf(builtIns.anyType)
    }

    val fieldDefs = buildList {
        add("outer" to cls.defaultType.makeNullable())
        add("resume" to builtIns.intType)
        for (pool in poolOrder) {
            repeat(poolSizes[pool]!!) { add("${cpsPoolPrefix(pool, context)}$it" to pool) }
        }
    }
    val fields = fieldDefs.map { def ->
        cls.addField {
            startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            name = Name.identifier(def.first)
            type = def.second
            visibility = DescriptorVisibilities.PUBLIC
            isFinal = true
        }
    }

    val outerField = fields[0]
    val resumeField = fields[1]
    val byPool = mutableMapOf<IrType, MutableList<IrField>>()
    val ctorOrder = mutableListOf<Pair<IrType, Int>>()
    var fi = 2
    for (pool in poolOrder) {
        val list = mutableListOf<IrField>()
        repeat(poolSizes[pool]!!) { k ->
            list += fields[fi++]
            ctorOrder += pool to k
        }
        byPool[pool] = list
    }

    val ctor = cls.addConstructor {
        isPrimary = true
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
    }.apply {
        val params = fieldDefs.map { def ->
            addValueParameter {
                name = Name.identifier(def.first)
                type = def.second
                startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
            }
        }
        body = context.createIrBuilder(symbol, UNDEFINED_OFFSET, UNDEFINED_OFFSET).irBlockBody {
            +irDelegatingConstructorCall(builtIns.anyClass.owner.constructors.single())
            +IrInstanceInitializerCallImpl(
                UNDEFINED_OFFSET, UNDEFINED_OFFSET, cls.symbol, builtIns.unitType
            )
            for (i in fields.indices) {
                +irSetField(irGet(cls.thisReceiver!!), fields[i], irGet(params[i]))
            }
        }
    }.symbol

    return CpsFrameLayout(
        frameClass = cls,
        outerField = outerField,
        resumeField = resumeField,
        poolFields = byPool,
        ctorParamOrder = ctorOrder,
        ctor = ctor,
    )
}

/**
 * Builds a module-level mutable depth counter field for the hybrid
 * threshold mechanism.
 */
internal fun buildCpsDepthField(
    context: WasmBackendContext,
    irFile: IrFile,
    baseName: String,
): IrField {
    val builtIns = context.irBuiltIns
    return context.irFactory.buildField {
        startOffset = UNDEFINED_OFFSET; endOffset = UNDEFINED_OFFSET
        name = Name.identifier(baseName)
        type = builtIns.intType
        visibility = DescriptorVisibilities.PRIVATE
        isStatic = true
        isFinal = false
    }.apply {
        parent = irFile
        irFile.declarations += this
        initializer = context.irFactory.createExpressionBody(
            IrConstImpl.int(UNDEFINED_OFFSET, UNDEFINED_OFFSET, builtIns.intType, 0)
        )
    }
}

/**
 * Rewrites [IrGetValue]/[IrSetValue] nodes whose symbols appear in
 * [map], substituting the mapped declaration. Used by both codegen
 * classes to remap original function parameters and locals to
 * trampoline mutable variables.
 */
@Suppress("UNCHECKED_CAST")
internal fun <T : IrElement> cpsRemap(
    element: T,
    map: Map<IrValueSymbol, IrValueDeclaration>,
    builtIns: IrBuiltIns,
): T {
    return element.transform(object : IrTransformer<Nothing?>() {
        override fun visitGetValue(expression: IrGetValue, data: Nothing?): IrExpression {
            val to = map[expression.symbol] ?: return super.visitGetValue(expression, data)
            return IrGetValueImpl(expression.startOffset, expression.endOffset, to.type, to.symbol)
        }

        override fun visitSetValue(expression: IrSetValue, data: Nothing?): IrExpression {
            expression.transformChildren(this, data)
            val to = map[expression.symbol] ?: return expression
            return IrSetValueImpl(
                expression.startOffset, expression.endOffset, builtIns.unitType,
                to.symbol, expression.value, expression.origin,
            )
        }
    }, null) as T
}

/**
 * Casts a restored or trampoline-returned expression from `Any?` to
 * the expected [targetType]. Reference types go through a nullable
 * cast, primitive types through a direct cast.
 */
internal fun IrBlockBuilder.cpsCast(
    expr: IrExpression,
    targetType: IrType,
    builtIns: IrBuiltIns,
): IrExpression = when {
    targetType == builtIns.anyNType -> expr
    !targetType.isPrimitiveType() -> irAs(expr, targetType.makeNullable())
    else -> irAs(expr, targetType)
}
