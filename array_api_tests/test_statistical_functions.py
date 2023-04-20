import cmath
import math
from typing import Optional

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from ._array_module import _UndefinedStub
from .typing import DataType

pytestmark = pytest.mark.ci


def kwarg_dtypes(dtype: DataType) -> st.SearchStrategy[Optional[DataType]]:
    dtypes = [d2 for d1, d2 in dh.promotion_table if d1 == dtype]
    if hh.FILTER_UNDEFINED_DTYPES:
        dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
    return st.none() | st.sampled_from(dtypes)


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
    keepdims = kw.get("keepdims", False)

    out = xp.max(x, **kw)

    ph.assert_dtype("max", in_dtype=x.dtype, out_dtype=out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "max", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        max_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(elements)
        ph.assert_scalar_equals("max", type_=scalar_type, idx=out_idx, out=max_, expected=expected)


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
    keepdims = kw.get("keepdims", False)

    out = xp.mean(x, **kw)

    ph.assert_dtype("mean", in_dtype=x.dtype, out_dtype=out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "mean", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    # Values testing mean is too finicky


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
    keepdims = kw.get("keepdims", False)

    out = xp.min(x, **kw)

    ph.assert_dtype("min", in_dtype=x.dtype, out_dtype=out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "min", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        min_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(elements)
        ph.assert_scalar_equals("min", type_=scalar_type, idx=out_idx, out=min_, expected=expected)


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
    keepdims = kw.get("keepdims", False)

    with hh.reject_overflow():
        out = xp.prod(x, **kw)

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
    if isinstance(_dtype, _UndefinedStub):
        # If a default uint cannot exist (i.e. in PyTorch which doesn't support
        # uint32 or uint64), we skip testing the output dtype.
        # See https://github.com/data-apis/array-api-tests/issues/106
        if _dtype in dh.uint_dtypes:
            assert dh.is_int_dtype(out.dtype)  # sanity check
    else:
        ph.assert_dtype("prod", in_dtype=x.dtype, out_dtype=out.dtype, expected=_dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "prod", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        prod = scalar_type(out[out_idx])
        assume(cmath.isfinite(prod))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = math.prod(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        ph.assert_scalar_equals("prod", type_=scalar_type, idx=out_idx, out=prod, expected=expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: math.prod(x.shape) >= 2),
    data=st.data(),
)
def test_std(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    _keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", _keepdims, False),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    out = xp.std(x, **kw)

    ph.assert_dtype("std", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_keepdimable_shape(
        "std", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
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
    keepdims = kw.get("keepdims", False)

    with hh.reject_overflow():
        out = xp.sum(x, **kw)

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
    if isinstance(_dtype, _UndefinedStub):
        # If a default uint cannot exist (i.e. in PyTorch which doesn't support
        # uint32 or uint64), we skip testing the output dtype.
        # See https://github.com/data-apis/array-api-tests/issues/160
        if _dtype in dh.uint_dtypes:
            assert dh.is_int_dtype(out.dtype)  # sanity check
    else:
        ph.assert_dtype("sum", in_dtype=x.dtype, out_dtype=out.dtype, expected=_dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "sum", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        sum_ = scalar_type(out[out_idx])
        assume(cmath.isfinite(sum_))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = sum(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        ph.assert_scalar_equals("sum", type_=scalar_type, idx=out_idx, out=sum_, expected=expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: math.prod(x.shape) >= 2),
    data=st.data(),
)
def test_var(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    _keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", _keepdims, False),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    out = xp.var(x, **kw)

    ph.assert_dtype("var", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_keepdimable_shape(
        "var", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    # We can't easily test the result(s) as variance methods vary a lot
