"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""

from itertools import product
from typing import Iterator, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from .function_stubs import elementwise_functions
from .pytest_helpers import nargs


# Note: the boolean binary operators do not have reversed or in-place variants
def generate_params(
    func_family: Literal['elementwise', 'operator'],
    in_nargs: int,
    out_category: Literal['bool', 'promoted'],
) -> Iterator:
    if func_family == 'elementwise':
        funcs = [
            f for f in elementwise_functions.__all__
            if nargs(f) == in_nargs and dh.func_out_categories[f] == out_category
        ]
        if in_nargs == 1:
            for func in funcs:
                in_category = dh.func_in_categories[func]
                for in_dtype in dh.category_to_dtypes[in_category]:
                    yield pytest.param(func, in_dtype, id=f"{func}({in_dtype})")
        else:
            for func, ((d1, d2), d3) in product(funcs, dh.promotion_table.items()):
                if all(d in dh.category_to_dtypes[dh.func_in_categories[func]] for d in (d1, d2)):
                    if out_category == 'bool':
                        yield pytest.param(func, (d1, d2), id=f"{func}({d1}, {d2})")
                    else:
                        yield pytest.param(func, ((d1, d2), d3), id=f"{func}({d1}, {d2}) -> {d3}")
    else:
        if in_nargs == 1:
            for op, symbol in dh.unary_op_to_symbol.items():
                func = dh.op_to_func[op]
                if dh.func_out_categories[func] == out_category:
                    in_category = dh.func_in_categories[func]
                    for in_dtype in dh.category_to_dtypes[in_category]:
                        yield pytest.param(op, symbol, in_dtype, id=f"{op}({in_dtype})")
        else:
            for op, symbol in dh.binary_op_to_symbol.items():
                if op == "__matmul__":
                    continue
                func = dh.op_to_func[op]
                if dh.func_out_categories[func] == out_category:
                    in_category = dh.func_in_categories[func]
                    for ((d1, d2), d3) in dh.promotion_table.items():
                        if all(d in dh.category_to_dtypes[in_category] for d in (d1, d2)):
                            if out_category == 'bool':
                                yield pytest.param(op, symbol, (d1, d2), id=f"{op}({d1}, {d2})")
                            else:
                                if d1 == d3:
                                    yield pytest.param(op, symbol, ((d1, d2), d3), id=f"{op}({d1}, {d2}) -> {d3}")



# TODO: These functions should still do type promotion internally, but we do
# not test this here (it is tested in the corresponding tests in
# test_elementwise_functions.py). This can affect the resulting values if not
# done correctly. For example, greater_equal(array(1.0, dtype=float32),
# array(1.00000001, dtype=float64)) will be wrong if the float64 array is
# downcast to float32. See for instance
# https://github.com/numpy/numpy/issues/10322.
@pytest.mark.parametrize('func, dtypes', generate_params('elementwise', in_nargs=2, out_category='bool'))
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(two_shapes=hh.two_mutually_broadcastable_shapes, data=st.data())
def test_elementwise_two_args_return_bool(func, two_shapes, dtypes, data):
    assert nargs(func) == 2
    func = getattr(xp, func)

    dtype1, dtype2 = dtypes

    fillvalue1 = data.draw(hh.scalars(st.just(dtype1)))
    if func in ['bitwise_left_shift', 'bitwise_right_shift']:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)))


    for i in [func, dtype1, dtype2]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = ah.full(shape1, fillvalue1, dtype=dtype1)
    a2 = ah.full(shape2, fillvalue2, dtype=dtype2)
    res = func(a1, a2)

    assert res.dtype == xp.bool, f"{func}({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to bool (shapes={shape1, shape2})"

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func, dtypes', generate_params('elementwise', in_nargs=2, out_category='promoted'))
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(two_shapes=hh.two_mutually_broadcastable_shapes, data=st.data())
def test_elementwise_two_args_return_promoted(func,
                                                              two_shapes, dtypes,
                                                              data):
    assert nargs(func) == 2
    func = getattr(xp, func)

    (dtype1, dtype2), res_dtype = dtypes
    fillvalue1 = data.draw(hh.scalars(st.just(dtype1)))
    if func in ['bitwise_left_shift', 'bitwise_right_shift']:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)))


    for i in [func, dtype1, dtype2, res_dtype]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = ah.full(shape1, fillvalue1, dtype=dtype1)
    a2 = ah.full(shape2, fillvalue2, dtype=dtype2)
    res = func(a1, a2)

    assert res.dtype == res_dtype, f"{func}({dtype1}, {dtype2}) promoted to {res.dtype}, should have promoted to {res_dtype} (shapes={shape1, shape2})"

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func, dtype', generate_params('elementwise', in_nargs=1, out_category='bool'))
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=hh.shapes, data=st.data())
def test_elementwise_one_arg_return_bool(func, shape, dtype, data):
    assert nargs(func) == 1
    func = getattr(xp, func)

    fillvalue = data.draw(hh.scalars(st.just(dtype)))

    for i in [func, dtype]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    x = ah.full(shape, fillvalue, dtype=dtype)
    res = func(x)

    assert res.dtype == xp.bool, f"{func}({dtype}) returned to {res.dtype}, should have promoted to bool (shape={shape})"

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func,dtype', generate_params('elementwise', in_nargs=1, out_category='promoted'))
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=hh.shapes, data=st.data())
def test_elementwise_one_arg_return_promoted(func, shape,
                                                     dtype, data):
    assert nargs(func) == 1
    func = getattr(xp, func)

    fillvalue = data.draw(hh.scalars(st.just(dtype)))

    for i in [func, dtype]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    x = ah.full(shape, fillvalue, dtype=dtype)
    res = func(x)

    assert res.dtype == dtype, f"{func}({dtype}) returned to {res.dtype}, should have promoted to {dtype} (shape={shape})"


# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize(
    'unary_op_name, unary_op, dtype',
    generate_params('operator', in_nargs=1, out_category='promoted'),
)
# The spec explicitly requires type promotion to work for shape 0
# Unfortunately, data(), isn't compatible with @example, so this is commented
# out for now.
# @example(shape=(0,))
@given(shape=hh.shapes, data=st.data())
def test_operator_one_arg_return_promoted(unary_op_name, unary_op, shape, dtype, data):
    fillvalue = data.draw(hh.scalars(st.just(dtype)))

    if isinstance(dtype, xp._UndefinedStub):
        dtype._raise()

    a = ah.full(shape, fillvalue, dtype=dtype)

    get_locals = lambda: dict(a=a)

    if unary_op_name == '__abs__':
        # This is the built-in abs(), not the array module's abs()
        expression = 'abs(a)'
    else:
        expression = f'{unary_op} a'
    res = eval(expression, get_locals())

    assert res.dtype == dtype, f"{unary_op}({dtype}) returned to {res.dtype}, should have promoted to {dtype} (shape={shape})"

@pytest.mark.parametrize(
    'binary_op_name, binary_op, dtypes',
    generate_params('operator', in_nargs=2, out_category='bool')
)
@given(two_shapes=hh.two_mutually_broadcastable_shapes, data=st.data())
def test_operator_two_args_return_bool(binary_op_name, binary_op, dtypes, two_shapes, data):
    dtype1, dtype2 = dtypes
    fillvalue1 = data.draw(hh.scalars(st.just(dtype1)))
    fillvalue2 = data.draw(hh.scalars(st.just(dtype2)))

    for i in [dtype1, dtype2]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = ah.full(shape1, fillvalue1, dtype=dtype1)
    a2 = ah.full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)
    expression = f'a1 {binary_op} a2'
    res = eval(expression, get_locals())

    assert res.dtype == xp.bool, f"{dtype1} {binary_op} {dtype2} promoted to {res.dtype}, should have promoted to bool (shape={shape1, shape2})"

binary_operators_promoted = [binary_op_name for binary_op_name in sorted(set(dh.binary_op_to_symbol) - {'__matmul__'})
                             if dh.func_out_categories[dh.op_to_func[binary_op_name]] == 'promoted']
operator_two_args_promoted_parametrize_inputs = [(binary_op_name, dtypes)
                                       for binary_op_name in binary_operators_promoted
                                       for dtypes in dh.promotion_table.items()
                                       if all(d in dh.category_to_dtypes[dh.func_in_categories[dh.op_to_func[binary_op_name]]] for d in dtypes[0])
                                       ]
operator_two_args_promoted_parametrize_ids = [f"{n}-{d1}-{d2}" for n, ((d1, d2), _)
                                            in operator_two_args_promoted_parametrize_inputs]

@pytest.mark.parametrize('binary_op_name, binary_op, dtypes', generate_params('operator', in_nargs=2, out_category='promoted'))
@given(two_shapes=hh.two_mutually_broadcastable_shapes, data=st.data())
def test_operator_two_args_return_promoted(binary_op_name, binary_op, dtypes, two_shapes, data):
    (dtype1, dtype2), res_dtype = dtypes
    fillvalue1 = data.draw(hh.scalars(st.just(dtype1)))
    if binary_op_name in ['>>', '<<']:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)))


    for i in [dtype1, dtype2, res_dtype]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = ah.full(shape1, fillvalue1, dtype=dtype1)
    a2 = ah.full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)
    expression = f'a1 {binary_op} a2'
    res = eval(expression, get_locals())

    assert res.dtype == res_dtype, f"{dtype1} {binary_op} {dtype2} promoted to {res.dtype}, should have promoted to {res_dtype} (shape={shape1, shape2})"

operator_inplace_two_args_promoted_parametrize_inputs = [(binary_op, dtypes) for binary_op, dtypes in operator_two_args_promoted_parametrize_inputs
                                                        if dtypes[0][0] == dtypes[1]]
operator_inplace_two_args_promoted_parametrize_ids = ['-'.join((n[:2] + 'i' + n[2:], str(d1), str(d2))) for n, ((d1, d2), _)
                                            in operator_inplace_two_args_promoted_parametrize_inputs]

@pytest.mark.parametrize('binary_op_name, binary_op, dtypes', generate_params('operator', in_nargs=2, out_category='promoted'))
@given(two_shapes=hh.two_broadcastable_shapes(), data=st.data())
def test_operator_inplace_two_args_return_promoted(binary_op_name, binary_op, dtypes, two_shapes,
                                    data):
    (dtype1, dtype2), res_dtype = dtypes
    fillvalue1 = data.draw(hh.scalars(st.just(dtype1)))
    if binary_op_name in ['>>', '<<']:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)).filter(lambda x: x > 0))
    else:
        fillvalue2 = data.draw(hh.scalars(st.just(dtype2)))

    for i in [dtype1, dtype2, res_dtype]:
        if isinstance(i, xp._UndefinedStub):
            i._raise()

    shape1, shape2 = two_shapes
    a1 = ah.full(shape1, fillvalue1, dtype=dtype1)
    a2 = ah.full(shape2, fillvalue2, dtype=dtype2)

    get_locals = lambda: dict(a1=a1, a2=a2)

    res_locals = get_locals()
    expression = f'a1 {binary_op}= a2'
    exec(expression, res_locals)
    res = res_locals['a1']

    assert res.dtype == res_dtype, f"{dtype1} {binary_op}= {dtype2} promoted to {res.dtype}, should have promoted to {res_dtype} (shape={shape1, shape2})"

scalar_promotion_parametrize_inputs = [
    pytest.param(binary_op_name, dtype, scalar_type, id=f"{binary_op_name}-{dtype}-{scalar_type.__name__}")
    for binary_op_name in sorted(set(dh.binary_op_to_symbol) - {'__matmul__'})
    for dtype in dh.category_to_dtypes[dh.func_in_categories[dh.op_to_func[binary_op_name]]]
    for scalar_type in dh.dtypes_to_scalars[dtype]
]

@pytest.mark.parametrize('binary_op_name,dtype,scalar_type',
                         scalar_promotion_parametrize_inputs)
@given(shape=hh.shapes, python_scalars=st.data(), data=st.data())
def test_operator_scalar_arg_return_promoted(binary_op_name, dtype, scalar_type,
                                   shape, python_scalars, data):
    """
    See https://st.data-apis.github.io/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-hh.scalars
    """
    binary_op = dh.binary_op_to_symbol[binary_op_name]
    if binary_op == '@':
        pytest.skip("matmul (@) is not supported for hh.scalars")

    if dtype in dh.category_to_dtypes['integer']:
        s = python_scalars.draw(st.integers(*ah.dtype_ranges[dtype]))
    else:
        s = python_scalars.draw(st.from_type(scalar_type))
    scalar_as_array = ah.asarray(s, dtype=dtype)
    get_locals = lambda: dict(a=a, s=s, scalar_as_array=scalar_as_array)

    fillvalue = data.draw(hh.scalars(st.just(dtype)))
    a = ah.full(shape, fillvalue, dtype=dtype)

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
    ah.assert_exactly_equal(res, expected)

    scalar_array = f's {binary_op} a'
    scalar_array_expected = f'scalar_as_array {binary_op} a'
    res = eval(scalar_array, get_locals())
    expected = eval(scalar_array_expected, get_locals())
    ah.assert_exactly_equal(res, expected)

    # Test in-place operators
    if binary_op in ['==', '!=', '<', '>', '<=', '>=']:
        return
    array_scalar = f'a {binary_op}= s'
    array_scalar_expected = f'a {binary_op}= scalar_as_array'
    a = ah.full(shape, fillvalue, dtype=dtype)
    res_locals = get_locals()
    exec(array_scalar, get_locals())
    res = res_locals['a']
    a = ah.full(shape, fillvalue, dtype=dtype)
    expected_locals = get_locals()
    exec(array_scalar_expected, get_locals())
    expected = expected_locals['a']
    ah.assert_exactly_equal(res, expected)


if __name__ == '__main__':
    for (i, j), p in dh.promotion_table.items():
        print(f"({i}, {j}) -> {p}")
