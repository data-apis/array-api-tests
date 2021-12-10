from hypothesis import given
from hypothesis import strategies as st
from hypothesis.control import assume

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .test_manipulation_functions import assert_equals, axis_ndindex


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_argsort(x):
    xp.argsort(x)
    # TODO


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
    _axis = axis if axis >= 0 else x.ndim + axis
    descending = kw.get("descending", False)
    scalar_type = dh.get_scalar_type(x.dtype)
    for idx in axis_ndindex(x.shape, _axis):
        f_idx = ", ".join(str(i) if isinstance(i, int) else ":" for i in idx)
        indexed_x = x[idx]
        indexed_out = out[idx]
        out_indices = list(ah.ndindex(indexed_x.shape))
        elements = [scalar_type(indexed_x[idx2]) for idx2 in out_indices]
        indices_order = sorted(
            range(len(out_indices)), key=elements.__getitem__, reverse=descending
        )
        x_indices = [out_indices[o] for o in indices_order]
        for out_idx, x_idx in zip(out_indices, x_indices):
            assert_equals(
                "sort",
                f"x[{f_idx}][{x_idx}]",
                indexed_x[x_idx],
                f"out[{f_idx}][{out_idx}]",
                indexed_out[out_idx],
                **kw,
            )
