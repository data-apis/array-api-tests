import math
from itertools import product
from typing import Iterator, List, Optional, Tuple, Union

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.control import reject

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .typing import DataType, Scalar, ScalarType, Shape


def kwarg_dtypes(dtype: DataType) -> st.SearchStrategy[Optional[DataType]]:
    dtypes = [d2 for d1, d2 in dh.promotion_table if d1 == dtype]
    return st.none() | st.sampled_from(dtypes)


def normalise_axis(
    axis: Optional[Union[int, Tuple[int, ...]]], ndim: int
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    return axes


def axes_ndindex(shape: Shape, axes: Tuple[int, ...]) -> Iterator[List[Shape]]:
    """Generate indices that index all elements except in `axes` dimensions"""
    base_indices = []
    axes_indices = []
    for axis, side in enumerate(shape):
        if axis in axes:
            base_indices.append([None])
            axes_indices.append(range(side))
        else:
            base_indices.append(range(side))
            axes_indices.append([None])
    for base_idx in product(*base_indices):
        indices = []
        for idx in product(*axes_indices):
            idx = list(idx)
            for axis, side in enumerate(idx):
                if axis not in axes:
                    idx[axis] = base_idx[axis]
            idx = tuple(idx)
            indices.append(idx)
        yield list(indices)


def assert_keepdimable_shape(
    func_name: str,
    out_shape: Shape,
    in_shape: Shape,
    axes: Tuple[int, ...],
    keepdims: bool,
    /,
    **kw,
):
    if keepdims:
        shape = tuple(1 if axis in axes else side for axis, side in enumerate(in_shape))
    else:
        shape = tuple(side for axis, side in enumerate(in_shape) if axis not in axes)
    ph.assert_shape(func_name, out_shape, shape, **kw)


def assert_equals(
    func_name: str,
    type_: ScalarType,
    idx: Shape,
    out: Scalar,
    expected: Scalar,
    /,
    **kw,
):
    out_repr = "out" if idx == () else f"out[{idx}]"
    f_func = f"{func_name}({ph.fmt_kw(kw)})"
    if type_ is bool or type_ is int:
        msg = f"{out_repr}={out}, should be {expected} [{f_func}]"
        assert out == expected, msg
    elif math.isnan(expected):
        msg = f"{out_repr}={out}, should be {expected} [{f_func}]"
        assert math.isnan(out), msg
    else:
        msg = f"{out_repr}={out}, should be roughly {expected} [{f_func}]"
        assert math.isclose(out, expected, rel_tol=0.25, abs_tol=1), msg


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_max(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.max(x, **kw)

    ph.assert_dtype("max", x.dtype, out.dtype)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "max", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        max_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(elements)
        assert_equals("max", scalar_type, out_idx, max_, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_mean(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.mean(x, **kw)

    ph.assert_dtype("mean", x.dtype, out.dtype)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "mean", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        mean = float(out[out_idx])
        assume(not math.isinf(mean))  # mean may become inf due to internal overflows
        elements = []
        for idx in indices:
            s = float(x[idx])
            elements.append(s)
        expected = sum(elements) / len(elements)
        assert_equals("mean", float, out_idx, mean, expected)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_min(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.min(x, **kw)

    ph.assert_dtype("min", x.dtype, out.dtype)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "min", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        min_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(elements)
        assert_equals("min", scalar_type, out_idx, min_, expected)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_prod(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=hh.axes(x.ndim),
            dtype=kwarg_dtypes(x.dtype),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    try:
        out = xp.prod(x, **kw)
    except OverflowError:
        reject()

    dtype = kw.get("dtype", None)
    if dtype is None:
        if dh.is_int_dtype(x.dtype):
            if x.dtype in dh.uint_dtypes:
                default_dtype = dh.default_uint
            else:
                default_dtype = dh.default_int
            m, M = dh.dtype_ranges[x.dtype]
            d_m, d_M = dh.dtype_ranges[default_dtype]
            if m < d_m or M > d_M:
                _dtype = x.dtype
            else:
                _dtype = default_dtype
        else:
            if dh.dtype_nbits[x.dtype] > dh.dtype_nbits[dh.default_float]:
                _dtype = x.dtype
            else:
                _dtype = dh.default_float
    else:
        _dtype = dtype
    ph.assert_dtype("prod", x.dtype, out.dtype, _dtype)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "prod", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        prod = scalar_type(out[out_idx])
        assume(math.isfinite(prod))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = math.prod(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        assert_equals("prod", scalar_type, out_idx, prod, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: x.size >= 2),
    data=st.data(),
)
def test_std(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", keepdims, False),
        ),
        label="kw",
    )

    out = xp.std(x, **kw)

    ph.assert_dtype("std", x.dtype, out.dtype)
    assert_keepdimable_shape(
        "std", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    # We can't easily test the result(s) as standard deviation methods vary a lot


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_sum(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=hh.axes(x.ndim),
            dtype=kwarg_dtypes(x.dtype),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    try:
        out = xp.sum(x, **kw)
    except OverflowError:
        reject()

    dtype = kw.get("dtype", None)
    if dtype is None:
        if dh.is_int_dtype(x.dtype):
            if x.dtype in dh.uint_dtypes:
                default_dtype = dh.default_uint
            else:
                default_dtype = dh.default_int
            m, M = dh.dtype_ranges[x.dtype]
            d_m, d_M = dh.dtype_ranges[default_dtype]
            if m < d_m or M > d_M:
                _dtype = x.dtype
            else:
                _dtype = default_dtype
        else:
            if dh.dtype_nbits[x.dtype] > dh.dtype_nbits[dh.default_float]:
                _dtype = x.dtype
            else:
                _dtype = dh.default_float
    else:
        _dtype = dtype
    ph.assert_dtype("sum", x.dtype, out.dtype, _dtype)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "sum", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        sum_ = scalar_type(out[out_idx])
        assume(math.isfinite(sum_))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = sum(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        assert_equals("sum", scalar_type, out_idx, sum_, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: x.size >= 2),
    data=st.data(),
)
def test_var(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", keepdims, False),
        ),
        label="kw",
    )

    out = xp.var(x, **kw)

    ph.assert_dtype("var", x.dtype, out.dtype)
    assert_keepdimable_shape(
        "var", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    # We can't easily test the result(s) as variance methods vary a lot
