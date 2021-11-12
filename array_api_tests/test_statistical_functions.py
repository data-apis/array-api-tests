import math

from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps

RTOL = 0.05


@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_min(x, data):
    axis_strats = [st.none()]
    if x.shape != ():
        axis_strats.append(
            st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim)
        )
    kw = data.draw(
        hh.kwargs(axis=st.one_of(axis_strats), keepdims=st.booleans()), label="kw"
    )

    out = xp.min(x, **kw)

    ph.assert_dtype("min", x.dtype, out.dtype)

    f_func = f"min({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis") is None:
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
            msg = f"out={min_}, should be {expected} [{f_func}]"
            if math.isnan(min_):
                assert math.isnan(expected), msg
            else:
                assert min_ == expected, msg


@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_max(x, data):
    axis_strats = [st.none()]
    if x.shape != ():
        axis_strats.append(
            st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim)
        )
    kw = data.draw(
        hh.kwargs(axis=st.one_of(axis_strats), keepdims=st.booleans()), label="kw"
    )

    out = xp.max(x, **kw)

    ph.assert_dtype("max", x.dtype, out.dtype)

    f_func = f"max({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis") is None:
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
            msg = f"out={max_}, should be {expected} [{f_func}]"
            if math.isnan(max_):
                assert math.isnan(expected), msg
            else:
                assert max_ == expected, msg


@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_mean(x, data):
    axis_strats = [st.none()]
    if x.shape != ():
        axis_strats.append(
            st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim)
        )
    kw = data.draw(
        hh.kwargs(axis=st.one_of(axis_strats), keepdims=st.booleans()), label="kw"
    )

    out = xp.mean(x, **kw)

    ph.assert_dtype("mean", x.dtype, out.dtype)

    f_func = f"mean({ph.fmt_kw(kw)})"

    # TODO: support axis
    if kw.get("axis") is None:
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
            msg = f"out={mean}, should be roughly {expected} [{f_func}]"
            assert math.isclose(mean, expected, rel_tol=RTOL), msg


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_prod(x):
    xp.prod(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_std(x):
    xp.std(x)
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
