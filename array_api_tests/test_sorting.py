from hypothesis import given
from hypothesis import strategies as st
from hypothesis.control import assume

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .test_manipulation_functions import assert_equals as assert_equals_
from .test_searching_functions import assert_default_index
from .test_statistical_functions import assert_equals


# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argsort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.argsort(x, **kw)

    assert_default_index("sort", out.dtype)
    ph.assert_shape("sort", out.shape, x.shape, **kw)
    axis = kw.get("axis", -1)
    axes = sh.normalise_axis(axis, x.ndim)
    descending = kw.get("descending", False)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements = [scalar_type(x[idx]) for idx in indices]
        indices_order = sorted(range(len(indices)), key=elements.__getitem__)
        if descending:
            # sorted(..., reverse=descending) doesn't always work
            indices_order = reversed(indices_order)
        for idx, o in zip(indices, indices_order):
            assert_equals("argsort", int, idx, int(out[idx]), o)


# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_sort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.sort(x, **kw)

    ph.assert_dtype("sort", out.dtype, x.dtype)
    ph.assert_shape("sort", out.shape, x.shape, **kw)
    axis = kw.get("axis", -1)
    axes = sh.normalise_axis(axis, x.ndim)
    descending = kw.get("descending", False)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements = [scalar_type(x[idx]) for idx in indices]
        indices_order = sorted(
            range(len(indices)), key=elements.__getitem__, reverse=descending
        )
        x_indices = [indices[o] for o in indices_order]
        for out_idx, x_idx in zip(indices, x_indices):
            assert_equals_(
                "sort",
                f"x[{x_idx}]",
                x[x_idx],
                f"out[{out_idx}]",
                out[out_idx],
                **kw,
            )
