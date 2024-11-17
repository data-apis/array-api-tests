import cmath
import math
from typing import Optional

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from ndindex import iter_indices

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from ._array_module import _UndefinedStub
from .typing import DataType


@pytest.mark.min_version("2023.12")
@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.numeric_dtypes,
        shape=hh.shapes(min_dims=1)),
    data=st.data(),
)
def test_cumulative_sum(x, data):
    axes = st.integers(-x.ndim, x.ndim - 1)
    if x.ndim == 1:
        axes = axes | st.none()
    axis = data.draw(axes, label='axis')
    _axis, = sh.normalize_axis(axis, x.ndim)
    dtype = data.draw(kwarg_dtypes(x.dtype))
    include_initial = data.draw(st.booleans(), label="include_initial")

    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("dtype", dtype, None),
            ("include_initial", include_initial, False),
        ),
        label="kw",
    )

    out = xp.cumulative_sum(x, **kw)

    expected_shape = list(x.shape)
    if include_initial:
        expected_shape[_axis] += 1
    expected_shape = tuple(expected_shape)
    ph.assert_shape("cumulative_sum", out_shape=out.shape, expected=expected_shape)

    expected_dtype = dh.accumulation_result_dtype(x.dtype, dtype)
    if expected_dtype is None:
        # If a default uint cannot exist (i.e. in PyTorch which doesn't support
        # uint32 or uint64), we skip testing the output dtype.
        # See https://github.com/data-apis/array-api-tests/issues/106
        if x.dtype in dh.uint_dtypes:
            assert dh.is_int_dtype(out.dtype)  # sanity check
    else:
        ph.assert_dtype("cumulative_sum", in_dtype=x.dtype, out_dtype=out.dtype, expected=expected_dtype)

    scalar_type = dh.get_scalar_type(out.dtype)

    for x_idx, out_idx, in iter_indices(x.shape, expected_shape, skip_axes=_axis):
        x_arr = x[x_idx.raw]
        out_arr = out[out_idx.raw]

        if include_initial:
            ph.assert_scalar_equals("cumulative_sum", type_=scalar_type, idx=out_idx.raw, out=out_arr[0], expected=0)

        for n in range(x.shape[_axis]):
            start = 1 if include_initial else 0
            out_val = out_arr[n + start]
            assume(cmath.isfinite(out_val))
            elements = []
            for idx in range(n + 1):
                s = scalar_type(x_arr[idx])
                elements.append(s)
            expected = sum(elements)
            if dh.is_int_dtype(out.dtype):
                m, M = dh.dtype_ranges[out.dtype]
                assume(m <= expected <= M)
                ph.assert_scalar_equals("cumulative_sum", type_=scalar_type,
                                        idx=out_idx.raw, out=out_val,
                                        expected=expected)
            else:
                condition_number = _sum_condition_number(elements)
                assume(condition_number < 1e6)
                ph.assert_scalar_isclose("cumulative_sum", type_=scalar_type,
                                         idx=out_idx.raw, out=out_val,
                                         expected=expected)

def kwarg_dtypes(dtype: DataType) -> st.SearchStrategy[Optional[DataType]]:
    dtypes = [d2 for d1, d2 in dh.promotion_table if d1 == dtype]
    dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
    assert len(dtypes) > 0  # sanity check
    return st.none() | st.sampled_from(dtypes)


@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
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
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
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
    x=hh.arrays(
        dtype=hh.real_floating_dtypes,
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
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "mean", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    # Values testing mean is too finicky


@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
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
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
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


def _prod_condition_number(elements):
    # Relative condition number using the infinity norm
    abs_max = max([abs(i) for i in elements])
    abs_min = min([abs(i) for i in elements])

    if abs_min == 0:
        return float('inf')

    return abs_max / abs_min

@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.numeric_dtypes,
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
    expected_dtype = dh.accumulation_result_dtype(x.dtype, dtype)
    if expected_dtype is None:
        # If a default uint cannot exist (i.e. in PyTorch which doesn't support
        # uint32 or uint64), we skip testing the output dtype.
        # See https://github.com/data-apis/array-api-tests/issues/106
        if x.dtype in dh.uint_dtypes:
            assert dh.is_int_dtype(out.dtype)  # sanity check
    else:
        ph.assert_dtype("prod", in_dtype=x.dtype, out_dtype=out.dtype, expected=expected_dtype)
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
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
            ph.assert_scalar_equals("prod", type_=scalar_type, idx=out_idx,
                                    out=prod, expected=expected)
        else:
            condition_number = _prod_condition_number(elements)
            assume(condition_number < 1e15)
            ph.assert_scalar_isclose("prod", type_=scalar_type, idx=out_idx,
                                     out=prod, expected=expected)


@pytest.mark.skip(reason="flaky")  # TODO: fix!
@given(
    x=hh.arrays(
        dtype=hh.real_floating_dtypes,
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: math.prod(x.shape) >= 2),
    data=st.data(),
)
def test_std(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalize_axis(axis, x.ndim)
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


def _sum_condition_number(elements):
    sum_abs = sum([abs(i) for i in elements])
    abs_sum = abs(sum(elements))

    if abs_sum == 0:
        return float('inf')

    return sum_abs / abs_sum

# @pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.numeric_dtypes,
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
    expected_dtype = dh.accumulation_result_dtype(x.dtype, dtype)
    if expected_dtype is None:
        # If a default uint cannot exist (i.e. in PyTorch which doesn't support
        # uint32 or uint64), we skip testing the output dtype.
        # See https://github.com/data-apis/array-api-tests/issues/160
        if x.dtype in dh.uint_dtypes:
            assert dh.is_int_dtype(out.dtype)  # sanity check
    else:
        ph.assert_dtype("sum", in_dtype=x.dtype, out_dtype=out.dtype, expected=expected_dtype)
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
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
            ph.assert_scalar_equals("sum", type_=scalar_type, idx=out_idx,
                                    out=sum_, expected=expected)
        else:
            # Avoid value testing for ill conditioned summations. See
            # https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Accuracy and
            # https://en.wikipedia.org/wiki/Condition_number.
            condition_number = _sum_condition_number(elements)
            assume(condition_number < 1e6)
            ph.assert_scalar_isclose("sum", type_=scalar_type, idx=out_idx, out=sum_, expected=expected)


@pytest.mark.unvectorized
@pytest.mark.skip(reason="flaky")  # TODO: fix!
@given(
    x=hh.arrays(
        dtype=hh.real_floating_dtypes,
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: math.prod(x.shape) >= 2),
    data=st.data(),
)
def test_var(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalize_axis(axis, x.ndim)
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
