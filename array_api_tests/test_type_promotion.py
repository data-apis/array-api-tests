"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""
from collections import defaultdict
from typing import Iterator, TypeVar, Tuple, Callable, Type, Union

import pytest
from hypothesis import assume, given, reject
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps
from .function_stubs import elementwise_functions
from .pytest_helpers import nargs


bitwise_shift_funcs = [
    'bitwise_left_shift',
    'bitwise_right_shift',
    '__lshift__',
    '__rshift__',
    '__ilshift__',
    '__irshift__',
]


DT = TypeVar('DT')


# We apply filters to xps.arrays() so we don't generate array elements that
# are erroneous or undefined for a function/operator.
filters = defaultdict(
    lambda: lambda _: True,
    {func: lambda x: ah.all(x > 0) for func in bitwise_shift_funcs},
)


def gen_func_params() -> Iterator[Tuple[Callable, Tuple[DT, ...], DT, Callable]]:
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
                    func,
                    (in_dtype,),
                    out_dtype,
                    filters[func_name],
                    id=f'{func_name}({in_dtype}) -> {out_dtype}',
                )
        elif ndtypes == 2:
            for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
                if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                    out_dtype = (
                        promoted_dtype if out_category == 'promoted' else xp.bool
                    )
                    yield pytest.param(
                        func,
                        (in_dtype1, in_dtype2),
                        out_dtype,
                        filters[func_name],
                        id=f'{func_name}({in_dtype1}, {in_dtype2}) -> {out_dtype}',
                    )
        else:
            raise NotImplementedError()


@pytest.mark.parametrize('func, in_dtypes, out_dtype, x_filter', gen_func_params())
@given(data=st.data())
def test_func_returns_array_with_correct_dtype(
    func, in_dtypes, out_dtype, x_filter, data
):
    if len(in_dtypes) == 1:
        x = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes).filter(x_filter), label='x'
        )
        arrays = [x]
    else:
        arrays = []
        shapes = data.draw(
            hh.mutually_broadcastable_shapes(len(in_dtypes)), label='shapes'
        )
        for i, (dtype, shape) in enumerate(zip(in_dtypes, shapes), 1):
            x = data.draw(
                xps.arrays(dtype=dtype, shape=shape).filter(x_filter), label=f'x{i}'
            )
            arrays.append(x)
    try:
        out = func(*arrays)
    except OverflowError:
        reject()
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


def gen_op_params() -> Iterator[Tuple[str, Tuple[DT, ...], DT, Callable]]:
    op_to_symbol = {**dh.unary_op_to_symbol, **dh.binary_op_to_symbol}
    for op, symbol in op_to_symbol.items():
        if op == '__matmul__':
            continue
        in_category = dh.op_in_categories[op]
        out_category = dh.op_out_categories[op]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        ndtypes = nargs(op)
        if ndtypes == 1:
            for in_dtype in valid_in_dtypes:
                out_dtype = in_dtype if out_category == 'promoted' else xp.bool
                yield pytest.param(
                    f'{symbol}x',
                    (in_dtype,),
                    out_dtype,
                    filters[op],
                    id=f'{op}({in_dtype}) -> {out_dtype}',
                )
        else:
            for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
                if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                    out_dtype = (
                        promoted_dtype if out_category == 'promoted' else xp.bool
                    )
                    yield pytest.param(
                        f'x1 {symbol} x2',
                        (in_dtype1, in_dtype2),
                        out_dtype,
                        filters[op],
                        id=f'{op}({in_dtype1}, {in_dtype2}) -> {out_dtype}',
                    )
    # We generate params for abs seperately as it does not have an associated symbol
    for in_dtype in dh.category_to_dtypes[dh.op_in_categories['__abs__']]:
        yield pytest.param(
            'abs(x)',
            (in_dtype,),
            in_dtype,
            filters['__abs__'],
            id=f'__abs__({in_dtype}) -> {in_dtype}',
        )


@pytest.mark.parametrize('expr, in_dtypes, out_dtype, x_filter', gen_op_params())
@given(data=st.data())
def test_operator_returns_array_with_correct_dtype(
    expr, in_dtypes, out_dtype, x_filter, data
):
    locals_ = {}
    if len(in_dtypes) == 1:
        locals_['x'] = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes).filter(x_filter), label='x'
        )
    else:
        shapes = data.draw(
            hh.mutually_broadcastable_shapes(len(in_dtypes)), label='shapes'
        )
        for i, (dtype, shape) in enumerate(zip(in_dtypes, shapes), 1):
            locals_[f'x{i}'] = data.draw(
                xps.arrays(dtype=dtype, shape=shape).filter(x_filter), label=f'x{i}'
            )
    try:
        out = eval(expr, locals_)
    except OverflowError:
        reject()
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


def gen_inplace_params() -> Iterator[Tuple[str, Tuple[DT, ...], DT, Callable]]:
    for op, symbol in dh.binary_op_to_symbol.items():
        if op == '__matmul__' or dh.op_out_categories[op] == 'bool':
            continue
        in_category = dh.op_in_categories[op]
        valid_in_dtypes = dh.category_to_dtypes[in_category]
        iop = f'__i{op[2:]}'
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if (
                in_dtype1 == promoted_dtype
                and in_dtype1 in valid_in_dtypes
                and in_dtype2 in valid_in_dtypes
            ):
                yield pytest.param(
                    f'x1 {symbol}= x2',
                    (in_dtype1, in_dtype2),
                    promoted_dtype,
                    filters[iop],
                    id=f'{iop}({in_dtype1}, {in_dtype2}) -> {promoted_dtype}',
                )


@pytest.mark.parametrize('expr, in_dtypes, out_dtype, x_filter', gen_inplace_params())
@given(shapes=hh.mutually_broadcastable_shapes(2), data=st.data())
def test_inplace_operator_returns_array_with_correct_dtype(
    expr, in_dtypes, out_dtype, x_filter, shapes, data
):
    assume(len(shapes[0]) >= len(shapes[1]))
    x1 = data.draw(
        xps.arrays(dtype=in_dtypes[0], shape=shapes[0]).filter(x_filter), label='x1'
    )
    x2 = data.draw(
        xps.arrays(dtype=in_dtypes[1], shape=shapes[1]).filter(x_filter), label='x2'
    )
    locals_ = {'x1': x1, 'x2': x2}
    try:
        exec(expr, locals_)
    except OverflowError:
        reject()
    x1 = locals_['x1']
    assert x1.dtype == out_dtype, f'{x1.dtype=!s}, but should be {out_dtype}'


finite_kw = {'allow_nan': False, 'allow_infinity': False}
ScalarType = Union[Type[bool], Type[int], Type[float]]


def gen_op_scalar_params() -> Iterator[Tuple[str, DT, ScalarType, DT, Callable]]:
    for op, symbol in dh.binary_op_to_symbol.items():
        if op == '__matmul__':
            continue
        in_category = dh.op_in_categories[op]
        out_category = dh.op_out_categories[op]
        for in_dtype in dh.category_to_dtypes[in_category]:
            out_dtype = in_dtype if out_category == 'promoted' else xp.bool
            for in_stype in dh.dtypes_to_scalars[in_dtype]:
                yield pytest.param(
                    f'x {symbol} s',
                    in_dtype,
                    in_stype,
                    out_dtype,
                    filters[op],
                    id=f'{op}({in_dtype}, {in_stype.__name__}) -> {out_dtype}',
                )


@pytest.mark.parametrize(
    'expr, in_dtype, in_stype, out_dtype, x_filter', gen_op_scalar_params()
)
@given(data=st.data())
def test_binary_operator_promotes_python_scalars(
    expr, in_dtype, in_stype, out_dtype, x_filter, data
):
    s = data.draw(xps.from_dtype(in_dtype, **finite_kw).map(in_stype), label='scalar')
    x = data.draw(
        xps.arrays(dtype=in_dtype, shape=hh.shapes).filter(x_filter), label='x'
    )
    try:
        out = eval(expr, {'x': x, 's': s})
    except OverflowError:
        reject()
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


def gen_inplace_scalar_params() -> Iterator[Tuple[str, DT, ScalarType, Callable]]:
    for op, symbol in dh.binary_op_to_symbol.items():
        if op == '__matmul__' or dh.op_out_categories[op] == 'bool':
            continue
        in_category = dh.op_in_categories[op]
        iop = f'__i{op[2:]}'
        for dtype in dh.category_to_dtypes[in_category]:
            for in_stype in dh.dtypes_to_scalars[dtype]:
                yield pytest.param(
                    f'x {symbol}= s',
                    dtype,
                    in_stype,
                    filters[iop],
                    id=f'{iop}({dtype}, {in_stype.__name__}) -> {dtype}',
                )


@pytest.mark.parametrize('expr, dtype, in_stype, x_filter', gen_inplace_scalar_params())
@given(data=st.data())
def test_inplace_operator_promotes_python_scalars(
    expr, dtype, in_stype, x_filter, data
):
    s = data.draw(xps.from_dtype(dtype, **finite_kw).map(in_stype), label='scalar')
    x = data.draw(xps.arrays(dtype=dtype, shape=hh.shapes).filter(x_filter), label='x')
    locals_ = {'x': x, 's': s}
    try:
        exec(expr, locals_)
    except OverflowError:
        reject()
    x = locals_['x']
    assert x.dtype == dtype, f'{x.dtype=!s}, but should be {dtype}'


if __name__ == '__main__':
    for (i, j), p in dh.promotion_table.items():
        print(f'({i}, {j}) -> {p}')
