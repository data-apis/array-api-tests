import math
from itertools import product
from typing import Sequence, Union, get_args

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .typing import DataType, Param, Scalar, ScalarType, Shape


def reshape(
    flat_seq: Sequence[Scalar], shape: Shape
) -> Union[Scalar, Sequence[Scalar]]:
    """Reshape a flat sequence"""
    if len(shape) == 0:
        assert len(flat_seq) == 1  # sanity check
        return flat_seq[0]
    elif len(shape) == 1:
        return flat_seq
    size = len(flat_seq)
    n = math.prod(shape[1:])
    return [reshape(flat_seq[i * n : (i + 1) * n], shape[1:]) for i in range(size // n)]


@given(hh.shapes(min_side=1), st.data())  # TODO: test 0-sided arrays
def test_getitem(shape, data):
    size = math.prod(shape)
    dtype = data.draw(xps.scalar_dtypes(), label="dtype")
    obj = data.draw(
        st.lists(xps.from_dtype(dtype), min_size=size, max_size=size).map(
            lambda l: reshape(l, shape)
        ),
        label="obj",
    )
    x = xp.asarray(obj, dtype=dtype)
    note(f"{x=}")
    key = data.draw(xps.indices(shape=shape), label="key")

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)
    _key = tuple(key) if isinstance(key, tuple) else (key,)
    if Ellipsis in _key:
        start_a = _key.index(Ellipsis)
        stop_a = start_a + (len(shape) - (len(_key) - 1))
        slices = tuple(slice(None, None) for _ in range(start_a, stop_a))
        _key = _key[:start_a] + slices + _key[start_a + 1 :]
    axes_indices = []
    out_shape = []
    for a, i in enumerate(_key):
        if isinstance(i, int):
            axes_indices.append([i])
        else:
            side = shape[a]
            indices = range(side)[i]
            axes_indices.append(indices)
            out_shape.append(len(indices))
    out_shape = tuple(out_shape)
    ph.assert_shape("__getitem__", out.shape, out_shape)
    assume(all(len(indices) > 0 for indices in axes_indices))
    out_obj = []
    for idx in product(*axes_indices):
        val = obj
        for i in idx:
            val = val[i]
        out_obj.append(val)
    out_obj = reshape(out_obj, out_shape)
    expected = xp.asarray(out_obj, dtype=dtype)
    ph.assert_array("__getitem__", out, expected)


@given(hh.shapes(min_side=1), st.data())  # TODO: test 0-sided arrays
def test_setitem(shape, data):
    size = math.prod(shape)
    dtype = data.draw(xps.scalar_dtypes(), label="dtype")
    obj = data.draw(
        st.lists(xps.from_dtype(dtype), min_size=size, max_size=size).map(
            lambda l: reshape(l, shape)
        ),
        label="obj",
    )
    x = xp.asarray(obj, dtype=dtype)
    note(f"{x=}")
    key = data.draw(xps.indices(shape=shape, max_dims=0), label="key")
    value = data.draw(
        xps.from_dtype(dtype) | xps.arrays(dtype=dtype, shape=()), label="value"
    )

    res = xp.asarray(x, copy=True)
    res[key] = value

    ph.assert_dtype("__setitem__", x.dtype, res.dtype, repr_name="x.dtype")
    ph.assert_shape("__setitem__", res.shape, x.shape, repr_name="x.shape")
    if isinstance(value, get_args(Scalar)):
        msg = f"x[{key}]={res[key]!r}, but should be {value=} [__setitem__()]"
        if math.isnan(value):
            assert xp.isnan(res[key]), msg
        else:
            assert res[key] == value, msg
    else:
        ph.assert_0d_equals("__setitem__", "value", value, f"x[{key}]", res[key])


# TODO: make mask tests optional


@given(hh.shapes(), st.data())
def test_getitem_mask(shape, data):
    x = data.draw(xps.arrays(xps.scalar_dtypes(), shape=shape), label="x")
    mask_shapes = st.one_of(
        st.sampled_from([x.shape, ()]),
        st.lists(st.booleans(), min_size=x.ndim, max_size=x.ndim).map(
            lambda l: tuple(s if b else 0 for s, b in zip(x.shape, l))
        ),
        hh.shapes(),
    )
    key = data.draw(xps.arrays(dtype=xp.bool, shape=mask_shapes), label="key")

    if key.ndim > x.ndim or not all(
        ks in (xs, 0) for xs, ks in zip(x.shape, key.shape)
    ):
        with pytest.raises(IndexError):
            x[key]
        return

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)
    if key.ndim == 0:
        out_shape = (1,) if key else (0,)
        out_shape += x.shape
    else:
        size = int(xp.sum(xp.astype(key, xp.uint8)))
        out_shape = (size,) + x.shape[key.ndim :]
    ph.assert_shape("__getitem__", out.shape, out_shape)


@given(hh.shapes(min_side=1), st.data())
def test_setitem_mask(shape, data):
    x = data.draw(xps.arrays(xps.scalar_dtypes(), shape=shape), label="x")
    key = data.draw(xps.arrays(dtype=xp.bool, shape=shape), label="key")
    value = data.draw(xps.from_dtype(x.dtype), label="value")

    res = xp.asarray(x, copy=True)
    res[key] = value

    # TODO


def make_param(method_name: str, dtype: DataType, stype: ScalarType) -> Param:
    return pytest.param(
        method_name, dtype, stype, id=f"{method_name}({dh.dtype_to_name[dtype]})"
    )


@pytest.mark.parametrize(
    "method_name, dtype, stype",
    [make_param("__bool__", xp.bool, bool)]
    + [make_param("__int__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__index__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__float__", d, float) for d in dh.float_dtypes],
)
@given(data=st.data())
def test_duck_typing(method_name, dtype, stype, data):
    x = data.draw(xps.arrays(dtype, shape=()), label="x")
    method = getattr(x, method_name)
    out = method()
    assert isinstance(
        out, stype
    ), f"{method_name}({x})={out}, which is not a {stype.__name__} scalar"
