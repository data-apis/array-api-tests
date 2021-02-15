"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""

import pytest

from hypothesis import given, example
from hypothesis.strategies import from_type, data

from .hypothesis_helpers import shapes
from .pytest_helpers import nargs
from .array_helpers import assert_exactly_equal

from .function_stubs import elementwise_functions
from ._array_module import (ones, int8, int16, int32, int64, uint8,
                            uint16, uint32, uint64, float32, float64, bool as bool_dtype)
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
    'b': bool_dtype,
}

def dtype_nbits(dtype):
    if dtype == int8:
        return 8
    elif dtype == int16:
        return 16
    elif dtype == int32:
        return 32
    elif dtype == int64:
        return 64
    elif dtype == uint8:
        return 8
    elif dtype == uint16:
        return 16
    elif dtype == uint32:
        return 32
    elif dtype == uint64:
        return 64
    elif dtype == float32:
        return 32
    elif dtype == float64:
        return 64
    else:
        raise ValueError(f"dtype_nbits is not defined for {dtype}")

def dtype_signed(dtype):
    if dtype in [int8, int16, int32, int64]:
        return True
    elif dtype in [uint8, uint16, uint32, uint64]:
        return False
    raise ValueError("dtype_signed is only defined for integer dtypes")

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

boolean_promotion_table = {
    ('b', 'b'): 'b',
}

promotion_table = {
    **signed_integer_promotion_table,
    **unsigned_integer_promotion_table,
    **mixed_signed_unsigned_promotion_table,
    **flipped_mixed_signed_unsigned_promotion_table,
    **float_promotion_table,
    **boolean_promotion_table,
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
    '__invert__': '~',
    '__neg__': '-',
    '__pos__': '+',
}

dtypes_to_scalar = {
    _array_module.bool: bool,
    _array_module.int8: int,
    _array_module.int16: int,
    _array_module.int32: int,
    _array_module.int64: int,
    _array_module.uint8: int,
    _array_module.uint16: int,
    _array_module.uint32: int,
    _array_module.uint64: int,
    _array_module.float32: float,
    _array_module.float64: float,
}

scalar_to_dtype = {s: [d for d, _s in dtypes_to_scalar.items() if _s == s] for
                   s in dtypes_to_scalar.values()}

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

@pytest.mark.parametrize('binary_op', sorted(set(binary_operators.values()) - {'@'}))
@pytest.mark.parametrize('scalar_type,dtype', [(s, d) for s in scalar_to_dtype
                                               for d in scalar_to_dtype[s]])
@given(shape=shapes, scalars=data())
def test_operator_scalar_promotion(binary_op, scalar_type, dtype, shape, scalars):
    """
    See https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars
    """
    if binary_op == '@':
        pytest.skip("matmul (@) is not supported for scalars")
    a = ones(shape, dtype=dtype)
    s = scalars.draw(from_type(scalar_type))
    scalar_as_array = _array_module.full((), s, dtype=dtype)
    get_locals = lambda: dict(a=a, s=s, scalar_as_array=scalar_as_array)

    # As per the spec:

    # The expected behavior is then equivalent to:
    #
    # 1. Convert the scalar to a 0-D array with the same dtype as that of the
    #    array used in the expression.
    #
    # 2. Execute the operation for `array <op> 0-D array` (or `0-D array <op>
    #    array` if `scalar` was the left-hand argument).

    array_scalar = f'a {binary_op} s'
    array_scalar_expected = f'a {binary_op} scalar_as_array'
    res = eval(array_scalar, get_locals())
    expected = eval(array_scalar_expected, get_locals())
    assert_exactly_equal(res, expected)

    scalar_array = f's {binary_op} a'
    scalar_array_expected = f'scalar_as_array {binary_op} a'
    res = eval(scalar_array, get_locals())
    expected = eval(scalar_array_expected, get_locals())
    assert_exactly_equal(res, expected)

if __name__ == '__main__':
    for (i, j), p in promotion_table.items():
        print(f"({i}, {j}) -> {p}")
