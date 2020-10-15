"""
https://github.com/data-apis/array-api/blob/master/spec/API_specification/type_promotion.md
"""

import pytest

from hypothesis import given, example

from .hypothesis_helpers import shapes
from .pytest_helpers import nargs

from .function_stubs import elementwise_functions
from ._array_module import (ones, int8, int16, int32, int64, uint8,
                            uint16, uint32, uint64, float32, float64)
from . import _array_module

dtype_mapping = {
    'i1': int8,
    'i2': int16,
    'i4': int32,
    'i8': int64,
    'u1': uint8,
    'u2': uint16,
    'u4': uint32,
    'u8': uint64,
    'f4': float32,
    'f8': float64,
}

signed_integer_promotion_table = {
    ('i1', 'i1'): 'i1',
    ('i1', 'i2'): 'i2',
    ('i1', 'i4'): 'i4',
    ('i1', 'i8'): 'i8',
    ('i2', 'i1'): 'i2',
    ('i2', 'i2'): 'i2',
    ('i2', 'i4'): 'i4',
    ('i2', 'i8'): 'i8',
    ('i4', 'i1'): 'i4',
    ('i4', 'i2'): 'i4',
    ('i4', 'i4'): 'i4',
    ('i4', 'i8'): 'i8',
    ('i8', 'i1'): 'i8',
    ('i8', 'i2'): 'i8',
    ('i8', 'i4'): 'i8',
    ('i8', 'i8'): 'i8',
}

unsigned_integer_promotion_table = {
    ('u1', 'u1'): 'u1',
    ('u1', 'u2'): 'u2',
    ('u1', 'u4'): 'u4',
    ('u1', 'u8'): 'u8',
    ('u2', 'u1'): 'u2',
    ('u2', 'u2'): 'u2',
    ('u2', 'u4'): 'u4',
    ('u2', 'u8'): 'u8',
    ('u4', 'u1'): 'u4',
    ('u4', 'u2'): 'u4',
    ('u4', 'u4'): 'u4',
    ('u4', 'u8'): 'u8',
    ('u8', 'u1'): 'u8',
    ('u8', 'u2'): 'u8',
    ('u8', 'u4'): 'u8',
    ('u8', 'u8'): 'u8',
}

mixed_signed_unsigned_promotion_table = {
    ('i1', 'u1'): 'i2',
    ('i1', 'u2'): 'i4',
    ('i1', 'u4'): 'i8',
    ('i2', 'u1'): 'i2',
    ('i2', 'u2'): 'i4',
    ('i2', 'u4'): 'i8',
    ('i4', 'u1'): 'i4',
    ('i4', 'u2'): 'i4',
    ('i4', 'u4'): 'i8',
}

flipped_mixed_signed_unsigned_promotion_table = {(u, i): p for (i, u), p in mixed_signed_unsigned_promotion_table.items()}

float_promotion_table = {
    ('f4', 'f4'): 'f4',
    ('f4', 'f8'): 'f8',
    ('f8', 'f4'): 'f8',
    ('f8', 'f8'): 'f8',
}

promotion_table = {
    **signed_integer_promotion_table,
    **unsigned_integer_promotion_table,
    **mixed_signed_unsigned_promotion_table,
    **flipped_mixed_signed_unsigned_promotion_table,
    **float_promotion_table,
}

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func_name', [i for i in
                                       elementwise_functions.__all__ if
                                       nargs(i) > 1])
@pytest.mark.parametrize('dtypes', promotion_table.items())
# The spec explicitly requires type promotion to work for shape 0
@example(shape=(0,))
@given(shape=shapes)
def test_promotion(func_name, shape, dtypes):
    assert nargs(func_name) == 2
    func = getattr(_array_module, func_name)


    (type1, type2), res_type = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]
    res_dtype = dtype_mapping[res_type]

    for i in [func, dtype1, dtype2, res_dtype]:
        if isinstance(i, _array_module._UndefinedStub):
            func._raise()

    a1 = ones(shape, dtype=dtype1)
    a2 = ones(shape, dtype=dtype2)
    res = func(a1, a2)

    assert res.dtype == res_dtype, f"{func_name}({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to {res_dtype} (shape={shape})"

if __name__ == '__main__':
    for (i, j), p in promotion_table.items():
        print(f"({i}, {j}) -> {p}")
