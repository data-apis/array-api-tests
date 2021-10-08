from . import _array_module as xp


__all__ = [
    'dtypes_to_scalars',
    'input_types',
    'promotion_table',
    'dtype_nbits',
    'dtype_signed',
    'binary_operators',
    'unary_operators',
    'operators_to_functions',
    'elementwise_function_input_types',
    'elementwise_function_output_types',
]


int_dtypes = (xp.int8, xp.int16, xp.int32, xp.int64)
uint_dtypes = (xp.uint8, xp.uint16, xp.uint32, xp.uint64)
all_int_dtypes = int_dtypes + uint_dtypes
float_dtypes = (xp.float32, xp.float64)
numeric_dtypes = all_int_dtypes + float_dtypes
all_dtypes = (xp.bool,) + numeric_dtypes


dtypes_to_scalars = {
    xp.bool: [bool],
    **{d: [int] for d in all_int_dtypes},
    **{d: [int, float] for d in float_dtypes},
}


input_types = {
    'any': all_dtypes,
    'boolean': (xp.bool,),
    'floating': float_dtypes,
    'integer': all_int_dtypes,
    'integer_or_boolean': (xp.bool,) + uint_dtypes + int_dtypes,
    'numeric': numeric_dtypes,
}


_numeric_promotions = {
    # ints
    (xp.int8, xp.int8): xp.int8,
    (xp.int8, xp.int16): xp.int16,
    (xp.int8, xp.int32): xp.int32,
    (xp.int8, xp.int64): xp.int64,
    (xp.int16, xp.int16): xp.int16,
    (xp.int16, xp.int32): xp.int32,
    (xp.int16, xp.int64): xp.int64,
    (xp.int32, xp.int32): xp.int32,
    (xp.int32, xp.int64): xp.int64,
    (xp.int64, xp.int64): xp.int64,
    # uints
    (xp.uint8, xp.uint8): xp.uint8,
    (xp.uint8, xp.uint16): xp.uint16,
    (xp.uint8, xp.uint32): xp.uint32,
    (xp.uint8, xp.uint64): xp.uint64,
    (xp.uint16, xp.uint16): xp.uint16,
    (xp.uint16, xp.uint32): xp.uint32,
    (xp.uint16, xp.uint64): xp.uint64,
    (xp.uint32, xp.uint32): xp.uint32,
    (xp.uint32, xp.uint64): xp.uint64,
    (xp.uint64, xp.uint64): xp.uint64,
    # ints and uints (mixed sign)
    (xp.int8, xp.uint8): xp.int16,
    (xp.int8, xp.uint16): xp.int32,
    (xp.int8, xp.uint32): xp.int64,
    (xp.int16, xp.uint8): xp.int16,
    (xp.int16, xp.uint16): xp.int32,
    (xp.int16, xp.uint32): xp.int64,
    (xp.int32, xp.uint8): xp.int32,
    (xp.int32, xp.uint16): xp.int32,
    (xp.int32, xp.uint32): xp.int64,
    (xp.int64, xp.uint8): xp.int64,
    (xp.int64, xp.uint16): xp.int64,
    (xp.int64, xp.uint32): xp.int64,
    # floats
    (xp.float32, xp.float32): xp.float32,
    (xp.float32, xp.float64): xp.float64,
    (xp.float64, xp.float64): xp.float64,
}
promotion_table = {
    (xp.bool, xp.bool): xp.bool,
    **_numeric_promotions,
    **{(d2, d1): res for (d1, d2), res in _numeric_promotions.items()},
}


dtype_nbits = {
    **{d: 8 for d in [xp.int8, xp.uint8]},
    **{d: 16 for d in [xp.int16, xp.uint16]},
    **{d: 32 for d in [xp.int32, xp.uint32, xp.float32]},
    **{d: 64 for d in [xp.int64, xp.uint64, xp.float64]},
}


dtype_signed = {
    **{d: True for d in int_dtypes},
    **{d: False for d in uint_dtypes},
}


elementwise_function_input_types = {
    'abs': 'numeric',
    'acos': 'floating',
    'acosh': 'floating',
    'add': 'numeric',
    'asin': 'floating',
    'asinh': 'floating',
    'atan': 'floating',
    'atan2': 'floating',
    'atanh': 'floating',
    'bitwise_and': 'integer_or_boolean',
    'bitwise_invert': 'integer_or_boolean',
    'bitwise_left_shift': 'integer',
    'bitwise_or': 'integer_or_boolean',
    'bitwise_right_shift': 'integer',
    'bitwise_xor': 'integer_or_boolean',
    'ceil': 'numeric',
    'cos': 'floating',
    'cosh': 'floating',
    'divide': 'floating',
    'equal': 'any',
    'exp': 'floating',
    'expm1': 'floating',
    'floor': 'numeric',
    'floor_divide': 'numeric',
    'greater': 'numeric',
    'greater_equal': 'numeric',
    'isfinite': 'numeric',
    'isinf': 'numeric',
    'isnan': 'numeric',
    'less': 'numeric',
    'less_equal': 'numeric',
    'log': 'floating',
    'logaddexp': 'floating',
    'log10': 'floating',
    'log1p': 'floating',
    'log2': 'floating',
    'logical_and': 'boolean',
    'logical_not': 'boolean',
    'logical_or': 'boolean',
    'logical_xor': 'boolean',
    'multiply': 'numeric',
    'negative': 'numeric',
    'not_equal': 'any',
    'positive': 'numeric',
    'pow': 'floating',
    'remainder': 'numeric',
    'round': 'numeric',
    'sign': 'numeric',
    'sin': 'floating',
    'sinh': 'floating',
    'sqrt': 'floating',
    'square': 'numeric',
    'subtract': 'numeric',
    'tan': 'floating',
    'tanh': 'floating',
    'trunc': 'numeric',
}


elementwise_function_output_types = {
    'abs': 'promoted',
    'acos': 'promoted',
    'acosh': 'promoted',
    'add': 'promoted',
    'asin': 'promoted',
    'asinh': 'promoted',
    'atan': 'promoted',
    'atan2': 'promoted',
    'atanh': 'promoted',
    'bitwise_and': 'promoted',
    'bitwise_invert': 'promoted',
    'bitwise_left_shift': 'promoted',
    'bitwise_or': 'promoted',
    'bitwise_right_shift': 'promoted',
    'bitwise_xor': 'promoted',
    'ceil': 'promoted',
    'cos': 'promoted',
    'cosh': 'promoted',
    'divide': 'promoted',
    'equal': 'bool',
    'exp': 'promoted',
    'expm1': 'promoted',
    'floor': 'promoted',
    'floor_divide': 'promoted',
    'greater': 'bool',
    'greater_equal': 'bool',
    'isfinite': 'bool',
    'isinf': 'bool',
    'isnan': 'bool',
    'less': 'bool',
    'less_equal': 'bool',
    'log': 'promoted',
    'logaddexp': 'promoted',
    'log10': 'promoted',
    'log1p': 'promoted',
    'log2': 'promoted',
    'logical_and': 'bool',
    'logical_not': 'bool',
    'logical_or': 'bool',
    'logical_xor': 'bool',
    'multiply': 'promoted',
    'negative': 'promoted',
    'not_equal': 'bool',
    'positive': 'promoted',
    'pow': 'promoted',
    'remainder': 'promoted',
    'round': 'promoted',
    'sign': 'promoted',
    'sin': 'promoted',
    'sinh': 'promoted',
    'sqrt': 'promoted',
    'square': 'promoted',
    'subtract': 'promoted',
    'tan': 'promoted',
    'tanh': 'promoted',
    'trunc': 'promoted',
}


binary_operators = {
    '__add__': '+',
    '__and__': '&',
    '__eq__': '==',
    '__floordiv__': '//',
    '__ge__': '>=',
    '__gt__': '>',
    '__le__': '<=',
    '__lshift__': '<<',
    '__lt__': '<',
    '__matmul__': '@',
    '__mod__': '%',
    '__mul__': '*',
    '__ne__': '!=',
    '__or__': '|',
    '__pow__': '**',
    '__rshift__': '>>',
    '__sub__': '-',
    '__truediv__': '/',
    '__xor__': '^',
}


unary_operators = {
    '__abs__': 'abs()',
    '__invert__': '~',
    '__neg__': '-',
    '__pos__': '+',
}


operators_to_functions = {
    '__abs__': 'abs',
    '__add__': 'add',
    '__and__': 'bitwise_and',
    '__eq__': 'equal',
    '__floordiv__': 'floor_divide',
    '__ge__': 'greater_equal',
    '__gt__': 'greater',
    '__le__': 'less_equal',
    '__lshift__': 'bitwise_left_shift',
    '__lt__': 'less',
    '__matmul__': 'matmul',
    '__mod__': 'remainder',
    '__mul__': 'multiply',
    '__ne__': 'not_equal',
    '__or__': 'bitwise_or',
    '__pow__': 'pow',
    '__rshift__': 'bitwise_right_shift',
    '__sub__': 'subtract',
    '__truediv__': 'divide',
    '__xor__': 'bitwise_xor',
    '__invert__': 'bitwise_invert',
    '__neg__': 'negative',
    '__pos__': 'positive',
}
