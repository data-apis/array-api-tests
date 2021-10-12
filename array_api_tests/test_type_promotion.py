"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""

from itertools import product
from typing import Iterator, Literal

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps
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
            for func, op in dh.unary_op_to_symbol.items():
                if dh.op_out_categories[func] == out_category:
                    in_category = dh.op_in_categories[func]
                    for in_dtype in dh.category_to_dtypes[in_category]:
                        yield pytest.param(func, op, in_dtype, id=f"{func}({in_dtype})")
        else:
            for func, op in dh.binary_op_to_symbol.items():
                if func == "__matmul__":
                    continue
                if dh.op_out_categories[func] == out_category:
                    in_category = dh.op_in_categories[func]
                    for ((d1, d2), d3) in dh.promotion_table.items():
                        if all(d in dh.category_to_dtypes[in_category] for d in (d1, d2)):
                            if out_category == 'bool':
                                yield pytest.param(func, op, (d1, d2), id=f"{func}({d1}, {d2})")
                            else:
                                if d1 == d3:
                                    yield pytest.param(func, op, ((d1, d2), d3), id=f"{func}({d1}, {d2}) -> {d3}")


def generate_func_params() -> Iterator:
    for func_name in elementwise_functions.__all__:
        func = getattr(xp, func_name)
        in_category = dh.func_in_categories[func_name]
        out_category = dh.func_out_categories[func_name]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        ndtypes = nargs(func_name)
        if ndtypes == 1:
            for in_dtype in valid_in_dtypes:
                out_dtype = in_dtype if out_category == 'promoted' else xp.bool
                yield pytest.param(
                    func, (in_dtype,), out_dtype, id=f"{func_name}({in_dtype}) -> {out_dtype}"
                )
        elif ndtypes == 2:
            for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
                if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                    out_dtype = promoted_dtype if out_category == 'promoted' else xp.bool
                    yield pytest.param(
                        func, (in_dtype1, in_dtype2), out_dtype, id=f'{func_name}({in_dtype1}, {in_dtype2}) -> {out_dtype}'
                    )
        else:
            raise NotImplementedError()


@pytest.mark.parametrize('func, in_dtypes, out_dtype', generate_func_params())
@given(data=st.data())
def test_func_returns_array_with_correct_dtype(func, in_dtypes, out_dtype, data):
    arrays = []
    shapes = data.draw(hh.mutually_broadcastable_shapes(len(in_dtypes)), label='shapes')
    for i, (dtype, shape) in enumerate(zip(in_dtypes, shapes), 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label='x{i}')
        arrays.append(x)
    out = func(*arrays)
    assert out.dtype == out_dtype, f"{out.dtype=!s}, but should be {out_dtype}"


def generate_unary_op_params() -> Iterator:
    for op, symbol in dh.unary_op_to_symbol.items():
        if op == '__abs__':
            continue
        in_category = dh.op_in_categories[op]
        out_category = dh.op_out_categories[op]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        for in_dtype in valid_in_dtypes:
            out_dtype = in_dtype if out_category == 'promoted' else xp.bool
            yield pytest.param(symbol, in_dtype, out_dtype, id=f'{op}({in_dtype}) -> {out_dtype}')


@pytest.mark.parametrize('op_symbol, in_dtype, out_dtype', generate_unary_op_params())
@given(data=st.data())
def test_unary_operator_returns_array_with_correct_dtype(op_symbol, in_dtype, out_dtype, data):
    x = data.draw(xps.arrays(dtype=in_dtype, shape=hh.shapes), label='x')
    out = eval(f'{op_symbol}x', {"x": x})
    assert out.dtype == out_dtype, f"{out.dtype=!s}, but should be {out_dtype}"


def generate_abs_op_params() -> Iterator:
    in_category = dh.op_in_categories['__abs__']
    out_category = dh.op_out_categories['__abs__']
    valid_in_dtypes = dh.category_to_dtypes[in_category]
    for in_dtype in valid_in_dtypes:
        out_dtype = in_dtype if out_category == 'promoted' else xp.bool
        yield pytest.param(in_dtype, out_dtype, id=f'__abs__({in_dtype}) -> {out_dtype}')


@pytest.mark.parametrize('in_dtype, out_dtype', generate_abs_op_params())
@given(data=st.data())
def test_abs_operator_returns_array_with_correct_dtype(in_dtype, out_dtype, data):
    x = data.draw(xps.arrays(dtype=in_dtype, shape=hh.shapes), label='x')
    out = eval('abs(x)', {"x": x})
    assert out.dtype == out_dtype, f"{out.dtype=!s}, but should be {out_dtype}"


def generate_binary_op_params() -> Iterator:
    for op, symbol in dh.binary_op_to_symbol.items():
        if op == '__matmul__' or 'shift' in op:
            continue
        in_category = dh.op_in_categories[op]
        out_category = dh.op_out_categories[op]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = promoted_dtype if out_category == 'promoted' else xp.bool
                yield pytest.param(
                    symbol,
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=f'{op}({in_dtype1}, {in_dtype2}) -> {out_dtype}'
                )


@pytest.mark.parametrize('op_symbol, in_dtypes, out_dtype', generate_binary_op_params())
@given(shapes=hh.mutually_broadcastable_shapes(2), data=st.data())
def test_binary_operator_returns_array_with_correct_dtype(op_symbol, in_dtypes, out_dtype, shapes, data):
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    out = eval(f'x1 {op_symbol} x2', {"x1": x1, "x2": x2})
    assert out.dtype == out_dtype, f"{out.dtype=!s}, but should be {out_dtype}"


def generate_inplace_op_params() -> Iterator:
    for op, symbol in dh.binary_op_to_symbol.items():
        if op == '__matmul__' or 'shift' in op or '=' in symbol or '<' in symbol or '>' in symbol:
            continue
        in_category = dh.op_in_categories[op]
        out_category = dh.op_out_categories[op]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 == promoted_dtype and in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = promoted_dtype if out_category == 'promoted' else xp.bool
                yield pytest.param(
                    f'{symbol}=',
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=f'__i{op[2:]}({in_dtype1}, {in_dtype2}) -> {out_dtype}'
                )


@pytest.mark.parametrize('op_symbol, in_dtypes, out_dtype', generate_inplace_op_params())
@given(shapes=hh.mutually_broadcastable_shapes(2), data=st.data())
def test_inplace_operator_returns_array_with_correct_dtype(op_symbol, in_dtypes, out_dtype, shapes, data):
    assume(len(shapes[0]) >= len(shapes[1]))
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    locals_ = {"x1": x1, "x2": x2}
    exec(f'x1 {op_symbol} x2', locals_)
    x1 = locals_["x1"]
    assert x1.dtype == out_dtype, f"{x1.dtype=!s}, but should be {out_dtype}"


scalar_promotion_parametrize_inputs = [
    pytest.param(func, dtype, scalar_type, id=f"{func}-{dtype}-{scalar_type.__name__}")
    for func in sorted(set(dh.binary_op_to_symbol) - {'__matmul__'})
    for dtype in dh.category_to_dtypes[dh.op_in_categories[func]]
    for scalar_type in dh.dtypes_to_scalars[dtype]
]

@pytest.mark.parametrize('func,dtype,scalar_type',
                         scalar_promotion_parametrize_inputs)
@given(shape=hh.shapes, python_scalars=st.data(), data=st.data())
def test_operator_scalar_arg_return_promoted(func, dtype, scalar_type,
                                   shape, python_scalars, data):
    """
    See https://st.data-apis.github.io/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-hh.scalars
    """
    op = dh.binary_op_to_symbol[func]
    if op == '@':
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

    array_scalar = f'a {op} s'
    array_scalar_expected = f'a {op} scalar_as_array'
    res = eval(array_scalar, get_locals())
    expected = eval(array_scalar_expected, get_locals())
    ah.assert_exactly_equal(res, expected)

    scalar_array = f's {op} a'
    scalar_array_expected = f'scalar_as_array {op} a'
    res = eval(scalar_array, get_locals())
    expected = eval(scalar_array_expected, get_locals())
    ah.assert_exactly_equal(res, expected)

    # Test in-place operators
    if op in ['==', '!=', '<', '>', '<=', '>=']:
        return
    array_scalar = f'a {op}= s'
    array_scalar_expected = f'a {op}= scalar_as_array'
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
