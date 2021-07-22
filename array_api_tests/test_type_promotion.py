"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""

import pytest

from hypothesis import given
from hypothesis.strategies import from_type, data, integers, just

from .hypothesis_helpers import (shapes, two_mutually_broadcastable_shapes,
                                 two_broadcastable_shapes, scalars)
from .pytest_helpers import nargs
from .array_helpers import assert_exactly_equal, dtype_ranges

from .function_stubs import elementwise_functions
from ._array_module import (full, int8, int16, int32, int64, uint8,
                            uint16, uint32, uint64, float32, float64, bool as
                            bool_dtype)
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

reverse_dtype_mapping = {v: k for k, v in dtype_mapping.items()}

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
    ('i8', 'u1'): 'i8',
    ('i8', 'u2'): 'i8',
    ('i8', 'u4'): 'i8',
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

input_types = {
    'any': sorted(set(promotion_table.values())),
    'boolean': sorted(set(boolean_promotion_table.values())),
    'floating': sorted(set(float_promotion_table.values())),
    'integer': sorted(set({**signed_integer_promotion_table,
                           **unsigned_integer_promotion_table}.values())),
    'integer_or_boolean': sorted(set({**signed_integer_promotion_table,
                                      **unsigned_integer_promotion_table,
                                      **boolean_promotion_table}.values())),
    'numeric': sorted(set({**float_promotion_table,
                           **signed_integer_promotion_table,
                           **unsigned_integer_promotion_table}.values())),
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

dtypes_to_scalars = {
    'b': [bool],
    'i1': [int],
    'i2': [int],
    'i4': [int],
    'i8': [int],
    # Note: unsigned int dtypes only correspond to positive integers
    'u1': [int],
    'u2': [int],
    'u4': [int],
    'u8': [int],
    'f4': [int, float],
    'f8': [int, float],
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

elementwise_function_two_arg_func_names = [func_name for func_name in
                                           elementwise_functions.__all__ if
                                           nargs(func_name) > 1]

elementwise_function_two_arg_func_names_promoted = [func_name for func_name in
                                                    elementwise_function_two_arg_func_names
                                                    if
                                                    elementwise_function_output_types[func_name]
                                                    == 'promoted']

elementwise_function_two_arg_promoted_parametrize_inputs = [(func_name, dtypes)
               for func_name in elementwise_function_two_arg_func_names_promoted
               for dtypes in promotion_table.items() if all(d in
                                                            input_types[elementwise_function_input_types[func_name]]
                                                            for d in dtypes[0])
               ]

elementwise_function_two_arg_promoted_parametrize_ids = ['-'.join((n, d1, d2)) for n, ((d1, d2), _)
                                            in elementwise_function_two_arg_promoted_parametrize_inputs]

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func_name,dtypes',
                         elementwise_function_two_arg_promoted_parametrize_inputs,
                         ids=elementwise_function_two_arg_promoted_parametrize_ids)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(two_shapes=two_mutually_broadcastable_shapes, fillvalues=data())
def test_elementwise_function_two_arg_promoted_type_promotion(func_name,
                                                              two_shapes, dtypes,
                                                              fillvalues):
    assert nargs(func_name) == 2
    func = getattr(_array_module, func_name)

    (type1, type2), res_type = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]
    res_dtype = dtype_mapping[res_type]
    fillvalue1 = fillvalues.draw(scalars(just(dtype1)))
    if func_name in ['bitwise_left_shift', 'bitwise_right_shift']:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)))


    for i in [func, dtype1, dtype2, res_dtype]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = full(shape1, fillvalue1, dtype=dtype1)
    a2 = full(shape2, fillvalue2, dtype=dtype2)
    res = func(a1, a2)

    assert res.dtype == res_dtype, f"{func_name}({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to {res_dtype} (shapes={shape1, shape2})"


elementwise_function_one_arg_func_names = [func_name for func_name in
                                           elementwise_functions.__all__ if
                                           nargs(func_name) == 1]
elementwise_function_one_arg_func_names_promoted = [func_name for func_name in
                                                    elementwise_function_one_arg_func_names
                                                    if
                                                    elementwise_function_output_types[func_name]
                                                    == 'promoted']

elementwise_function_one_arg_promoted_parametrize_inputs = [(func_name, dtypes)
               for func_name in elementwise_function_one_arg_func_names_promoted
               for dtypes in input_types[elementwise_function_input_types[func_name]]]
elementwise_function_one_arg_promoted_parametrize_ids = ['-'.join((n, d)) for n, d
                                            in elementwise_function_one_arg_promoted_parametrize_inputs]

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func_name,dtype_name',
                         elementwise_function_one_arg_promoted_parametrize_inputs,
                         ids=elementwise_function_one_arg_promoted_parametrize_ids)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=shapes, fillvalues=data())
def test_elementwise_function_one_arg_type_promotion(func_name, shape,
                                                     dtype_name, fillvalues):
    assert nargs(func_name) == 1
    func = getattr(_array_module, func_name)

    dtype = dtype_mapping[dtype_name]
    fillvalue = fillvalues.draw(scalars(just(dtype)))

    for i in [func, dtype]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    x = full(shape, fillvalue, dtype=dtype)
    res = func(x)

    assert res.dtype == dtype, f"{func_name}({dtype}) returned to {res.dtype}, should have promoted to {dtype} (shape={shape})"

binary_operators_promoted = [binary_op_name for binary_op_name in sorted(set(binary_operators) - {'__matmul__'})
                             if elementwise_function_output_types[operators_to_functions[binary_op_name]] == 'promoted']
operator_two_arg_promoted_parametrize_inputs = [(binary_op_name, dtypes)
                                       for binary_op_name in binary_operators_promoted
                                       for dtypes in promotion_table.items()
                                       if all(d in input_types[elementwise_function_input_types[operators_to_functions[binary_op_name]]] for d in dtypes[0])
                                       ]
operator_two_arg_promoted_parametrize_ids = ['-'.join((n, d1, d2)) for n, ((d1, d2), _)
                                            in operator_two_arg_promoted_parametrize_inputs]

@pytest.mark.parametrize('binary_op_name,dtypes',
                         operator_two_arg_promoted_parametrize_inputs,
                         ids=operator_two_arg_promoted_parametrize_ids)
@given(two_shapes=two_mutually_broadcastable_shapes, fillvalues=data())
def test_operator_two_arg_promoted_promotion(binary_op_name, dtypes, two_shapes,
                                    fillvalues):
    binary_op = binary_operators[binary_op_name]

    (type1, type2), res_type = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]
    res_dtype = dtype_mapping[res_type]
    fillvalue1 = fillvalues.draw(scalars(just(dtype1)))
    if binary_op_name in ['>>', '<<']:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)))


    for i in [dtype1, dtype2, res_dtype]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = full(shape1, fillvalue1, dtype=dtype1)
    a2 = full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)
    expression = f'a1 {binary_op} a2'
    res = eval(expression, get_locals())

    assert res.dtype == res_dtype, f"{dtype1} {binary_op} {dtype2} promoted to {res.dtype}, should have promoted to {res_dtype} (shape={shape1, shape2})"

operator_inplace_two_arg_promoted_parametrize_inputs = [(binary_op, dtypes) for binary_op, dtypes in operator_two_arg_promoted_parametrize_inputs
                                                        if dtypes[0][0] == dtypes[1]]
operator_inplace_two_arg_promoted_parametrize_ids = ['-'.join((n[:2] + 'i' + n[2:], d1, d2)) for n, ((d1, d2), _)
                                            in operator_inplace_two_arg_promoted_parametrize_inputs]

@pytest.mark.parametrize('binary_op_name,dtypes',
                         operator_inplace_two_arg_promoted_parametrize_inputs,
                         ids=operator_inplace_two_arg_promoted_parametrize_ids)
@given(two_shapes=two_broadcastable_shapes(), fillvalues=data())
def test_operator_inplace_two_arg_promoted_promotion(binary_op_name, dtypes, two_shapes,
                                    fillvalues):
    binary_op = binary_operators[binary_op_name]

    (type1, type2), res_type = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]
    res_dtype = dtype_mapping[res_type]
    fillvalue1 = fillvalues.draw(scalars(just(dtype1)))
    if binary_op_name in ['>>', '<<']:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)))

    for i in [dtype1, dtype2, res_dtype]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = full(shape1, fillvalue1, dtype=dtype1)
    a2 = full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)

    res_locals = get_locals()
    expression = f'a1 {binary_op}= a2'
    exec(expression, res_locals)
    res = res_locals['a1']

    assert res.dtype == res_dtype, f"{dtype1} {binary_op}= {dtype2} promoted to {res.dtype}, should have promoted to {res_dtype} (shape={shape1, shape2})"


scalar_promotion_parametrize_inputs = [(binary_op_name, dtype_name, scalar_type)
                                       for binary_op_name in sorted(set(binary_operators) - {'__matmul__'})
                                       for dtype_name in input_types[elementwise_function_input_types[operators_to_functions[binary_op_name]]]
                                       for scalar_type in dtypes_to_scalars[dtype_name]]


@pytest.mark.parametrize('binary_op_name,dtype_name,scalar_type',
                         scalar_promotion_parametrize_inputs)
@given(shape=shapes, python_scalars=data(), fillvalues=data())
def test_operator_scalar_promotion(binary_op_name, dtype_name, scalar_type,
                                   shape, python_scalars, fillvalues):
    """
    See https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars
    """
    binary_op = binary_operators[binary_op_name]
    if binary_op == '@':
        pytest.skip("matmul (@) is not supported for scalars")
    dtype = dtype_mapping[dtype_name]

    if dtype_name in input_types['integer']:
        s = python_scalars.draw(integers(*dtype_ranges[dtype]))
    else:
        s = python_scalars.draw(from_type(scalar_type))
    scalar_as_array = _array_module.asarray(s, dtype=dtype)
    get_locals = lambda: dict(a=a, s=s, scalar_as_array=scalar_as_array)

    fillvalue = fillvalues.draw(scalars(just(dtype)))
    a = full(shape, fillvalue, dtype=dtype)

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

    # Test in-place operators
    if binary_op in ['==', '!=', '<', '>', '<=', '>=']:
        return
    array_scalar = f'a {binary_op}= s'
    array_scalar_expected = f'a {binary_op}= scalar_as_array'
    a = full(shape, fillvalue, dtype=dtype)
    res_locals = get_locals()
    exec(array_scalar, get_locals())
    res = res_locals['a']
    a = full(shape, fillvalue, dtype=dtype)
    expected_locals = get_locals()
    exec(array_scalar_expected, get_locals())
    expected = expected_locals['a']
    assert_exactly_equal(res, expected)


if __name__ == '__main__':
    for (i, j), p in promotion_table.items():
        print(f"({i}, {j}) -> {p}")
