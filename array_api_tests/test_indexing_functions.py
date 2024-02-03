import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps

pytestmark = pytest.mark.ci


@pytest.mark.min_version("2022.12")
@given(
    x=hh.arrays(xps.scalar_dtypes(), hh.shapes(min_dims=1, min_side=1)),
    data=st.data(),
)
def test_take(x, data):
    # TODO:
    # * negative axis
    # * negative indices
    # * different dtypes for indices
    axis = data.draw(st.integers(0, max(x.ndim - 1, 0)), label="axis")
    _indices = data.draw(
        st.lists(st.integers(0, x.shape[axis] - 1), min_size=1, unique=True),
        label="_indices",
    )
    indices = xp.asarray(_indices, dtype=dh.default_int)
    note(f"{indices=}")

    out = xp.take(x, indices, axis=axis)

    ph.assert_dtype("take", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape(
        "take",
        out_shape=out.shape,
        expected=x.shape[:axis] + (len(_indices),) + x.shape[axis + 1 :],
        kw=dict(
            x=x,
            indices=indices,
            axis=axis,
        ),
    )
    out_indices = sh.ndindex(out.shape)
    axis_indices = list(sh.axis_ndindex(x.shape, axis))
    for axis_idx in axis_indices:
        f_axis_idx = sh.fmt_idx("x", axis_idx)
        for i in _indices:
            f_take_idx = sh.fmt_idx(f_axis_idx, i)
            indexed_x = x[axis_idx][i, ...]
            for at_idx in sh.ndindex(indexed_x.shape):
                out_idx = next(out_indices)
                ph.assert_0d_equals(
                    "take",
                    x_repr=sh.fmt_idx(f_take_idx, at_idx),
                    x_val=indexed_x[at_idx],
                    out_repr=sh.fmt_idx("out", out_idx),
                    out_val=out[out_idx],
                )
    # sanity check
    with pytest.raises(StopIteration):
        next(out_indices)
