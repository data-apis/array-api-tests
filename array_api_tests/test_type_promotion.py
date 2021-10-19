"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""
from collections import defaultdict
from typing import Tuple, Type, Union, List

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


DT = Type
ScalarType = Union[Type[bool], Type[int], Type[float]]


def multi_promotable_dtypes(
    allow_bool: bool = True,
) -> st.SearchStrategy[Tuple[DT, ...]]:
    strats = [
        st.lists(st.sampled_from(dh.all_int_dtypes), min_size=2).filter(
            lambda l: not (xp.uint64 in l and any(d in dh.int_dtypes for d in l))
        ),
        st.lists(st.sampled_from(dh.float_dtypes), min_size=2),
    ]
    if allow_bool:
        strats.append(st.lists(st.just(xp.bool), min_size=2))
    return st.one_of(strats).map(tuple)


@given(multi_promotable_dtypes())
def test_result_type(dtypes):
    out = xp.result_type(*dtypes)
    expected = dh.result_type(*dtypes)
    assert out == expected, f'{out=!s}, but should be {expected}'


@given(
    dtypes=multi_promotable_dtypes(allow_bool=False),
    kw=hh.kwargs(indexing=st.sampled_from(['xy', 'ij'])),
    data=st.data(),
)
def test_meshgrid(dtypes, kw, data):
    arrays = []
    shapes = data.draw(hh.mutually_broadcastable_shapes(len(dtypes)), label='shapes')
    for i, (dtype, shape) in enumerate(zip(dtypes, shapes), 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f'x{i}')
        arrays.append(x)
    out = xp.meshgrid(*arrays, **kw)
    expected = dh.result_type(*dtypes)
    for i in range(len(out)):
        assert (
            out[i].dtype == expected
        ), f'out[{i}]={out[i].dtype}, but should be {expected}'


@given(
    shape=hh.shapes(min_dims=1),
    dtypes=multi_promotable_dtypes(allow_bool=False),
    kw=hh.kwargs(axis=st.none() | st.just(0)),
    data=st.data(),
)
def test_concat(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f'x{i}')
        arrays.append(x)
    out = xp.concat(arrays, **kw)
    expected = dh.result_type(*dtypes)
    assert out.dtype == expected, f'{out.dtype=!s}, but should be {expected}'


@given(
    shape=hh.shapes(),
    dtypes=multi_promotable_dtypes(),
    kw=hh.kwargs(axis=st.just(0)),
    data=st.data(),
)
def test_stack(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f'x{i}')
        arrays.append(x)
    out = xp.stack(arrays, **kw)
    expected = dh.result_type(*dtypes)
    assert out.dtype == expected, f'{out.dtype=!s}, but should be {expected}'


bitwise_shift_funcs = [
    'bitwise_left_shift',
    'bitwise_right_shift',
    '__lshift__',
    '__rshift__',
    '__ilshift__',
    '__irshift__',
]


# We apply filters to xps.arrays() so we don't generate array elements that
# are erroneous or undefined for a function/operator.
filters = defaultdict(
    lambda: lambda _: True,
    {func: lambda x: ah.all(x > 0) for func in bitwise_shift_funcs},
)


def make_id(
    func_name: str, in_dtypes: Tuple[Union[DT, ScalarType], ...], out_dtype: DT
) -> str:
    f_in_dtypes = []
    for dtype in in_dtypes:
        try:
            f_in_dtypes.append(dh.dtype_to_name[dtype])
        except KeyError:
            # i.e. dtype is bool, int, or float
            f_in_dtypes.append(dtype.__name__)
    f_args = ', '.join(f_in_dtypes)
    f_out_dtype = dh.dtype_to_name[out_dtype]
    return f'{func_name}({f_args}) -> {f_out_dtype}'


func_params: List[Tuple[str, Tuple[DT, ...], DT]] = []
for func_name in elementwise_functions.__all__:
    valid_in_dtypes = dh.func_in_dtypes[func_name]
    ndtypes = nargs(func_name)
    if ndtypes == 1:
        for in_dtype in valid_in_dtypes:
            out_dtype = xp.bool if dh.func_returns_bool[func_name] else in_dtype
            p = pytest.param(
                func_name,
                (in_dtype,),
                out_dtype,
                id=make_id(func_name, (in_dtype,), out_dtype),
            )
            func_params.append(p)
    elif ndtypes == 2:
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = (
                    xp.bool if dh.func_returns_bool[func_name] else promoted_dtype
                )
                p = pytest.param(
                    func_name,
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=make_id(func_name, (in_dtype1, in_dtype2), out_dtype),
                )
                func_params.append(p)
    else:
        raise NotImplementedError()


@pytest.mark.parametrize('func_name, in_dtypes, out_dtype', func_params)
@given(data=st.data())
def test_func_promotion(func_name, in_dtypes, out_dtype, data):
    func = getattr(xp, func_name)
    x_filter = filters[func_name]
    if len(in_dtypes) == 1:
        x = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes()).filter(x_filter),
            label='x',
        )
        out = func(x)
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


promotion_params: List[Tuple[Tuple[DT, DT], DT]] = []
for (dtype1, dtype2), promoted_dtype in dh.promotion_table.items():
    p = pytest.param(
        (dtype1, dtype2),
        promoted_dtype,
        id=make_id('', (dtype1, dtype2), promoted_dtype),
    )
    promotion_params.append(p)


@pytest.mark.parametrize('in_dtypes, out_dtype', promotion_params)
@given(shapes=hh.mutually_broadcastable_shapes(3), data=st.data())
def test_where(in_dtypes, out_dtype, shapes, data):
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    cond = data.draw(xps.arrays(dtype=xp.bool, shape=shapes[2]), label='condition')
    out = xp.where(cond, x1, x2)
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


numeric_promotion_params = promotion_params[1:]


@pytest.mark.parametrize('in_dtypes, out_dtype', numeric_promotion_params)
@given(shapes=hh.mutually_broadcastable_shapes(2, min_dims=1), data=st.data())
def test_matmul(in_dtypes, out_dtype, shapes, data):
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    out = xp.matmul(x1, x2)
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


@pytest.mark.parametrize('in_dtypes, out_dtype', numeric_promotion_params)
@given(shapes=hh.mutually_broadcastable_shapes(2, min_dims=2), data=st.data())
def test_tensordot(in_dtypes, out_dtype, shapes, data):
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    out = xp.tensordot(x1, x2)
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


@pytest.mark.parametrize('in_dtypes, out_dtype', numeric_promotion_params)
@given(shapes=hh.mutually_broadcastable_shapes(2, min_dims=1), data=st.data())
def test_vecdot(in_dtypes, out_dtype, shapes, data):
    x1 = data.draw(xps.arrays(dtype=in_dtypes[0], shape=shapes[0]), label='x1')
    x2 = data.draw(xps.arrays(dtype=in_dtypes[1], shape=shapes[1]), label='x2')
    out = xp.vecdot(x1, x2)
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


op_params: List[Tuple[str, str, Tuple[DT, ...], DT]] = []
op_to_symbol = {**dh.unary_op_to_symbol, **dh.binary_op_to_symbol}
for op, symbol in op_to_symbol.items():
    if op == '__matmul__':
        continue
    valid_in_dtypes = dh.func_in_dtypes[op]
    ndtypes = nargs(op)
    if ndtypes == 1:
        for in_dtype in valid_in_dtypes:
            out_dtype = xp.bool if dh.func_returns_bool[op] else in_dtype
            p = pytest.param(
                op,
                f'{symbol}x',
                (in_dtype,),
                out_dtype,
                id=make_id(op, (in_dtype,), out_dtype),
            )
            op_params.append(p)
    else:
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = xp.bool if dh.func_returns_bool[op] else promoted_dtype
                p = pytest.param(
                    op,
                    f'x1 {symbol} x2',
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=make_id(op, (in_dtype1, in_dtype2), out_dtype),
                )
                op_params.append(p)
# We generate params for abs seperately as it does not have an associated symbol
for in_dtype in dh.func_in_dtypes['__abs__']:
    p = pytest.param(
        '__abs__',
        'abs(x)',
        (in_dtype,),
        in_dtype,
        id=make_id('__abs__', (in_dtype,), in_dtype),
    )
    op_params.append(p)


@pytest.mark.parametrize('op, expr, in_dtypes, out_dtype', op_params)
@given(data=st.data())
def test_op_promotion(op, expr, in_dtypes, out_dtype, data):
    x_filter = filters[op]
    if len(in_dtypes) == 1:
        x = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes()).filter(x_filter),
            label='x',
        )
        out = eval(expr, {'x': x})
    else:
        locals_ = {}
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


inplace_params: List[Tuple[str, str, Tuple[DT, ...], DT]] = []
for op, symbol in dh.inplace_op_to_symbol.items():
    if op == '__imatmul__':
        continue
    valid_in_dtypes = dh.func_in_dtypes[op]
    for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
        if (
            in_dtype1 == promoted_dtype
            and in_dtype1 in valid_in_dtypes
            and in_dtype2 in valid_in_dtypes
        ):
            p = pytest.param(
                op,
                f'x1 {symbol} x2',
                (in_dtype1, in_dtype2),
                promoted_dtype,
                id=make_id(op, (in_dtype1, in_dtype2), promoted_dtype),
            )
            inplace_params.append(p)


@pytest.mark.parametrize('op, expr, in_dtypes, out_dtype', inplace_params)
@given(shapes=hh.mutually_broadcastable_shapes(2), data=st.data())
def test_inplace_op_promotion(op, expr, in_dtypes, out_dtype, shapes, data):
    assume(len(shapes[0]) >= len(shapes[1]))
    x_filter = filters[op]
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


op_scalar_params: List[Tuple[str, str, DT, ScalarType, DT]] = []
for op, symbol in dh.binary_op_to_symbol.items():
    if op == '__matmul__':
        continue
    for in_dtype in dh.func_in_dtypes[op]:
        out_dtype = xp.bool if dh.func_returns_bool[op] else in_dtype
        for in_stype in dh.dtype_to_scalars[in_dtype]:
            p = pytest.param(
                op,
                f'x {symbol} s',
                in_dtype,
                in_stype,
                out_dtype,
                id=make_id(op, (in_dtype, in_stype), out_dtype),
            )
            op_scalar_params.append(p)


@pytest.mark.parametrize('op, expr, in_dtype, in_stype, out_dtype', op_scalar_params)
@given(data=st.data())
def test_op_scalar_promotion(op, expr, in_dtype, in_stype, out_dtype, data):
    x_filter = filters[op]
    kw = {k: in_stype is float for k in ('allow_nan', 'allow_infinity')}
    s = data.draw(xps.from_dtype(in_dtype, **kw).map(in_stype), label='scalar')
    x = data.draw(
        xps.arrays(dtype=in_dtype, shape=hh.shapes()).filter(x_filter), label='x'
    )
    try:
        out = eval(expr, {'x': x, 's': s})
    except OverflowError:
        reject()
    assert out.dtype == out_dtype, f'{out.dtype=!s}, but should be {out_dtype}'


inplace_scalar_params: List[Tuple[str, str, DT, ScalarType]] = []
for op, symbol in dh.inplace_op_to_symbol.items():
    if op == '__imatmul__':
        continue
    for dtype in dh.func_in_dtypes[op]:
        for in_stype in dh.dtype_to_scalars[dtype]:
            p = pytest.param(
                op,
                f'x {symbol} s',
                dtype,
                in_stype,
                id=make_id(op, (dtype, in_stype), dtype),
            )
            inplace_scalar_params.append(p)


@pytest.mark.parametrize('op, expr, dtype, in_stype', inplace_scalar_params)
@given(data=st.data())
def test_inplace_op_scalar_promotion(op, expr, dtype, in_stype, data):
    x_filter = filters[op]
    kw = {k: in_stype is float for k in ('allow_nan', 'allow_infinity')}
    s = data.draw(xps.from_dtype(dtype, **kw).map(in_stype), label='scalar')
    x = data.draw(
        xps.arrays(dtype=dtype, shape=hh.shapes()).filter(x_filter), label='x'
    )
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
