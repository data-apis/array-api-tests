import math
from typing import Optional, Union

from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .typing import Scalar, ScalarType, Shape


def axes(ndim: int) -> st.SearchStrategy[Optional[Union[int, Shape]]]:
    axes_strats = [st.none()]
    if ndim != 0:
        axes_strats.append(st.integers(-ndim, ndim - 1))
        axes_strats.append(xps.valid_tuple_axes(ndim))
    return st.one_of(axes_strats)


def assert_equals(
    func_name: str, type_: ScalarType, out: Scalar, expected: Scalar, /, **kw
):
    f_func = f"{func_name}({ph.fmt_kw(kw)})"
    if type_ is bool or type_ is int:
        msg = f"{out=}, should be {expected} [{f_func}]"
        assert out == expected, msg
    elif math.isnan(expected):
        msg = f"{out=}, should be {expected} [{f_func}]"
        assert math.isnan(out), msg
    else:
        msg = f"{out=}, should be roughly {expected} [{f_func}]"
        assert math.isclose(out, expected, rel_tol=0.05), msg


@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_min(x, data):
    kw = data.draw(hh.kwargs(axis=axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.min(x, **kw)

    ph.assert_dtype("min", x.dtype, out.dtype)

    f_func = f"min({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis", None) is None:
        keepdims = kw.get("keepdims", False)
        if keepdims:
            idx = tuple(1 for _ in x.shape)
            msg = f"{out.shape=}, should be reduced dimension {idx} [{f_func}]"
            assert out.shape == idx, msg
        else:
            ph.assert_shape("min", out.shape, (), **kw)

        # TODO: figure out NaN behaviour
        if dh.is_int_dtype(x.dtype) or not xp.any(xp.isnan(x)):
            _out = xp.reshape(out, ()) if keepdims else out
            scalar_type = dh.get_scalar_type(out.dtype)
            elements = []
            for idx in ah.ndindex(x.shape):
                s = scalar_type(x[idx])
                elements.append(s)
            min_ = scalar_type(_out)
            expected = min(elements)
            assert_equals("min", dh.get_scalar_type(out.dtype), min_, expected)


@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_max(x, data):
    kw = data.draw(hh.kwargs(axis=axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.max(x, **kw)

    ph.assert_dtype("max", x.dtype, out.dtype)

    f_func = f"max({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis", None) is None:
        keepdims = kw.get("keepdims", False)
        if keepdims:
            idx = tuple(1 for _ in x.shape)
            msg = f"{out.shape=}, should be reduced dimension {idx} [{f_func}]"
            assert out.shape == idx, msg
        else:
            ph.assert_shape("max", out.shape, (), **kw)

        # TODO: figure out NaN behaviour
        if dh.is_int_dtype(x.dtype) or not xp.any(xp.isnan(x)):
            _out = xp.reshape(out, ()) if keepdims else out
            scalar_type = dh.get_scalar_type(out.dtype)
            elements = []
            for idx in ah.ndindex(x.shape):
                s = scalar_type(x[idx])
                elements.append(s)
            max_ = scalar_type(_out)
            expected = max(elements)
            assert_equals("mean", dh.get_scalar_type(out.dtype), max_, expected)


@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_mean(x, data):
    kw = data.draw(hh.kwargs(axis=axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.mean(x, **kw)

    ph.assert_dtype("mean", x.dtype, out.dtype)

    f_func = f"mean({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis", None) is None:
        keepdims = kw.get("keepdims", False)
        if keepdims:
            idx = tuple(1 for _ in x.shape)
            msg = f"{out.shape=}, should be reduced dimension {idx} [{f_func}]"
            assert out.shape == idx, msg
        else:
            ph.assert_shape("max", out.shape, (), **kw)

        # TODO: figure out NaN behaviour
        if not xp.any(xp.isnan(x)):
            _out = xp.reshape(out, ()) if keepdims else out
            elements = []
            for idx in ah.ndindex(x.shape):
                s = float(x[idx])
                elements.append(s)
            mean = float(_out)
            expected = sum(elements) / len(elements)
            assert_equals("mean", float, mean, expected)


@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_prod(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=axes(x.ndim),
            dtype=st.none() | st.just(x.dtype),  # TODO: all valid dtypes
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    out = xp.prod(x, **kw)

    dtype = kw.get("dtype", None)
    if dtype is None:
        if dh.is_int_dtype(x.dtype):
            m, M = dh.dtype_ranges[x.dtype]
            d_m, d_M = dh.dtype_ranges[dh.default_int]
            if m < d_m or M > d_M:
                _dtype = x.dtype
            else:
                _dtype = dh.default_int
        else:
            if dh.dtype_nbits[x.dtype] > dh.dtype_nbits[dh.default_float]:
                _dtype = x.dtype
            else:
                _dtype = dh.default_float
    else:
        _dtype = dtype
    ph.assert_dtype("prod", x.dtype, out.dtype, _dtype)

    f_func = f"prod({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis", None) is None:
        keepdims = kw.get("keepdims", False)
        if keepdims:
            idx = tuple(1 for _ in x.shape)
            msg = f"{out.shape=}, should be reduced dimension {idx} [{f_func}]"
            assert out.shape == idx, msg
        else:
            ph.assert_shape("prod", out.shape, (), **kw)

        # TODO: figure out NaN behaviour
        if dh.is_int_dtype(x.dtype) or not xp.any(xp.isnan(x)):
            _out = xp.reshape(out, ()) if keepdims else out
            scalar_type = dh.get_scalar_type(out.dtype)
            elements = []
            for idx in ah.ndindex(x.shape):
                s = scalar_type(x[idx])
                elements.append(s)
            prod = scalar_type(_out)
            expected = math.prod(elements)
            if dh.is_int_dtype(out.dtype):
                m, M = dh.dtype_ranges[out.dtype]
                assume(m <= expected <= M)
            assert_equals("prod", dh.get_scalar_type(out.dtype), prod, expected)


@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_std(x, data):
    kw = data.draw(hh.kwargs(axis=axes(x.ndim), keepdims=st.booleans()), label="kw")

    xp.std(x, **kw)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_sum(x):
    xp.sum(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_var(x):
    xp.var(x)
    # TODO
