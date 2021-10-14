from typing import NamedTuple

from . import _array_module as xp


__all__ = [
    'int_dtypes',
    'uint_dtypes',
    'all_int_dtypes',
    'float_dtypes',
    'numeric_dtypes',
    'all_dtypes',
    'bool_and_all_int_dtypes',
    'dtypes_to_scalars',
    'is_int_dtype',
    'is_float_dtype',
    'dtype_ranges',
    'category_to_dtypes',
    'promotion_table',
    'dtype_nbits',
    'dtype_signed',
    'func_in_categories',
    'func_out_categories',
    'op_in_categories',
    'op_out_categories',
    'op_to_func',
    'binary_op_to_symbol',
    'unary_op_to_symbol',
    'inplace_op_to_symbol',
]


int_dtypes = (xp.int8, xp.int16, xp.int32, xp.int64)
uint_dtypes = (xp.uint8, xp.uint16, xp.uint32, xp.uint64)
all_int_dtypes = int_dtypes + uint_dtypes
float_dtypes = (xp.float32, xp.float64)
numeric_dtypes = all_int_dtypes + float_dtypes
all_dtypes = (xp.bool,) + numeric_dtypes
bool_and_all_int_dtypes = (xp.bool,) + all_int_dtypes


dtypes_to_scalars = {
    xp.bool: [bool],
    **{d: [int] for d in all_int_dtypes},
    **{d: [int, float] for d in float_dtypes},
}


def is_int_dtype(dtype):
    return dtype in all_int_dtypes


def is_float_dtype(dtype):
    # None equals NumPy's xp.float64 object, so we specifically check it here.
    # xp.float64 is in fact an alias of np.dtype('float64'), and its equality
    # with None is meant to be deprecated at some point.
    # See https://github.com/numpy/numpy/issues/18434
    if dtype is None:
        return False
    # TODO: Return True for float dtypes that aren't part of the spec e.g. np.float16
    return dtype in float_dtypes


class MinMax(NamedTuple):
    min: int
    max: int


dtype_ranges = {
    xp.int8: MinMax(-128, +127),
    xp.int16: MinMax(-32_768, +32_767),
    xp.int32: MinMax(-2_147_483_648, +2_147_483_647),
    xp.int64: MinMax(-9_223_372_036_854_775_808, +9_223_372_036_854_775_807),
    xp.uint8: MinMax(0, +255),
    xp.uint16: MinMax(0, +65_535),
    xp.uint32: MinMax(0, +4_294_967_295),
    xp.uint64: MinMax(0, +18_446_744_073_709_551_615),
}


category_to_dtypes = {
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


func_in_categories = {
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


func_out_categories = {
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


unary_op_to_symbol = {
    '__invert__': '~',
    '__neg__': '-',
    '__pos__': '+',
}


binary_op_to_symbol = {
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


op_to_func = {
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
    # '__matmul__': 'matmul',  # TODO: support matmul
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


op_in_categories = {}
op_out_categories = {}
for op, elwise_func in op_to_func.items():
    op_in_categories[op] = func_in_categories[elwise_func]
    op_out_categories[op] = func_out_categories[elwise_func]


inplace_op_to_symbol = {}
for op, symbol in binary_op_to_symbol.items():
    if op == '__matmul__' or op_out_categories[op] == 'bool':
        continue
    iop = f'__i{op[2:]}'
    inplace_op_to_symbol[iop] = f'{symbol}='
    op_in_categories[iop] = op_in_categories[op]
    op_out_categories[iop] = op_out_categories[op]
