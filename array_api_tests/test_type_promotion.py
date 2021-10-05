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
from ._array_module import (full, bool as bool_dtype)
from . import _array_module
from .dtype_helpers import (
    dtype_mapping,
    promotion_table,
    input_types,
    dtypes_to_scalars,
    elementwise_function_input_types,
    elementwise_function_output_types,
    binary_operators,
    unary_operators,
    operators_to_functions,
)



elementwise_function_two_arg_func_names = [func_name for func_name in
                                           elementwise_functions.__all__ if
                                           nargs(func_name) > 1]

elementwise_function_two_arg_func_names_bool = [func_name for func_name in
                                                    elementwise_function_two_arg_func_names
                                                    if
                                                    elementwise_function_output_types[func_name]
                                                    == 'bool']

elementwise_function_two_arg_bool_parametrize_inputs = [(func_name, dtypes)
               for func_name in elementwise_function_two_arg_func_names_bool
               for dtypes in promotion_table.keys() if all(d in
                                                            input_types[elementwise_function_input_types[func_name]]
                                                            for d in dtypes)
               ]

elementwise_function_two_arg_bool_parametrize_ids = ['-'.join((n, d1, d2)) for n, (d1, d2)
                                            in elementwise_function_two_arg_bool_parametrize_inputs]

# TODO: These functions should still do type promotion internally, but we do
# not test this here (it is tested in the corresponding tests in
# test_elementwise_functions.py). This can affect the resulting values if not
# done correctly. For example, greater_equal(array(1.0, dtype=float32),
# array(1.00000001, dtype=float64)) will be wrong if the float64 array is
# downcast to float32. See for instance
# https://github.com/numpy/numpy/issues/10322.
@pytest.mark.parametrize('func_name,dtypes',
                         elementwise_function_two_arg_bool_parametrize_inputs,
                         ids=elementwise_function_two_arg_bool_parametrize_ids)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(two_shapes=two_mutually_broadcastable_shapes, fillvalues=data())
def test_elementwise_function_two_arg_bool_type_promotion(func_name,
                                                              two_shapes, dtypes,
                                                              fillvalues):
    assert nargs(func_name) == 2
    func = getattr(_array_module, func_name)

    type1, type2 = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]

    fillvalue1 = fillvalues.draw(scalars(just(dtype1)))
    if func_name in ['bitwise_left_shift', 'bitwise_right_shift']:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = fillvalues.draw(scalars(just(dtype2)))


    for i in [func, dtype1, dtype2]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = full(shape1, fillvalue1, dtype=dtype1)
    a2 = full(shape2, fillvalue2, dtype=dtype2)
    res = func(a1, a2)

    assert res.dtype == bool_dtype, f"{func_name}({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to bool (shapes={shape1, shape2})"

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

elementwise_function_one_arg_func_names_bool = [func_name for func_name in
                                                    elementwise_function_one_arg_func_names
                                                    if
                                                    elementwise_function_output_types[func_name]
                                                    == 'bool']

elementwise_function_one_arg_bool_parametrize_inputs = [(func_name, dtypes)
               for func_name in elementwise_function_one_arg_func_names_bool
               for dtypes in input_types[elementwise_function_input_types[func_name]]]
elementwise_function_one_arg_bool_parametrize_ids = ['-'.join((n, d)) for n, d
                                            in elementwise_function_one_arg_bool_parametrize_inputs]

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func_name,dtype_name',
                         elementwise_function_one_arg_bool_parametrize_inputs,
                         ids=elementwise_function_one_arg_bool_parametrize_ids)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=shapes, fillvalues=data())
def test_elementwise_function_one_arg_bool(func_name, shape,
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

    assert res.dtype == bool_dtype, f"{func_name}({dtype}) returned to {res.dtype}, should have promoted to bool (shape={shape})"

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

unary_operators_promoted = [unary_op_name for unary_op_name in sorted(unary_operators)
                             if elementwise_function_output_types[operators_to_functions[unary_op_name]] == 'promoted']
operator_one_arg_promoted_parametrize_inputs = [(unary_op_name, dtypes)
                                       for unary_op_name in unary_operators_promoted
                                       for dtypes in input_types[elementwise_function_input_types[operators_to_functions[unary_op_name]]]
                                       ]
operator_one_arg_promoted_parametrize_ids = ['-'.join((n, d)) for n, d
                                            in operator_one_arg_promoted_parametrize_inputs]


# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('unary_op_name,dtype_name',
                         operator_one_arg_promoted_parametrize_inputs,
                         ids=operator_one_arg_promoted_parametrize_ids)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=shapes, fillvalues=data())
def test_operator_one_arg_type_promotion(unary_op_name, shape, dtype_name, fillvalues):
    unary_op = unary_operators[unary_op_name]

    dtype = dtype_mapping[dtype_name]
    fillvalue = fillvalues.draw(scalars(just(dtype)))

    if isinstance(dtype, _array_module._UndefinedStub):
        dtype._raise()

    a = full(shape, fillvalue, dtype=dtype)

    get_locals = lambda: dict(a=a)

    if unary_op_name == '__abs__':
        # This is the built-in abs(), not the array module's abs()
        expression = 'abs(a)'
    else:
        expression = f'{unary_op} a'
    res = eval(expression, get_locals())

    assert res.dtype == dtype, f"{unary_op}({dtype}) returned to {res.dtype}, should have promoted to {dtype} (shape={shape})"

# Note: the boolean binary operators do not have reversed or in-place variants
binary_operators_bool = [binary_op_name for binary_op_name in sorted(set(binary_operators) - {'__matmul__'})
                             if elementwise_function_output_types[operators_to_functions[binary_op_name]] == 'bool']
operator_two_arg_bool_parametrize_inputs = [(binary_op_name, dtypes)
                                       for binary_op_name in binary_operators_bool
                                       for dtypes in promotion_table.keys()
                                       if all(d in input_types[elementwise_function_input_types[operators_to_functions[binary_op_name]]] for d in dtypes)
                                       ]
operator_two_arg_bool_parametrize_ids = ['-'.join((n, d1, d2)) for n, (d1, d2)
                                            in operator_two_arg_bool_parametrize_inputs]

@pytest.mark.parametrize('binary_op_name,dtypes',
                         operator_two_arg_bool_parametrize_inputs,
                         ids=operator_two_arg_bool_parametrize_ids)
@given(two_shapes=two_mutually_broadcastable_shapes, fillvalues=data())
def test_operator_two_arg_bool_promotion(binary_op_name, dtypes, two_shapes,
                                    fillvalues):
    binary_op = binary_operators[binary_op_name]

    type1, type2 = dtypes
    dtype1 = dtype_mapping[type1]
    dtype2 = dtype_mapping[type2]
    fillvalue1 = fillvalues.draw(scalars(just(dtype1)))
    fillvalue2 = fillvalues.draw(scalars(just(dtype2)))

    for i in [dtype1, dtype2]:
        if isinstance(i, _array_module._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = full(shape1, fillvalue1, dtype=dtype1)
    a2 = full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)
    expression = f'a1 {binary_op} a2'
    res = eval(expression, get_locals())

    assert res.dtype == bool_dtype, f"{dtype1} {binary_op} {dtype2} promoted to {res.dtype}, should have promoted to bool (shape={shape1, shape2})"

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
