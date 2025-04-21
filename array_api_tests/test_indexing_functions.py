import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh


@pytest.mark.unvectorized
@pytest.mark.min_version("2022.12")
@given(
    x=hh.arrays(hh.all_dtypes, hh.shapes(min_dims=1, min_side=1)),
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



@pytest.mark.unvectorized
@pytest.mark.min_version("2024.12")
@given(
    x=hh.arrays(hh.all_dtypes, hh.shapes(min_dims=1, min_side=1)),
    data=st.data(),
)
def test_take_along_axis(x, data):
    # TODO
    # 2. negative indices
    # 3. different dtypes for indices
    axis = data.draw(st.integers(-x.ndim, max(x.ndim - 1, 0)), label="axis")
    len_axis = data.draw(st.integers(0, 2*x.shape[axis]), label="len_axis")

    n_axis = axis + x.ndim if axis < 0 else axis
    idx_shape = x.shape[:n_axis] + (len_axis,) + x.shape[n_axis+1:]
    indices = data.draw(
        hh.arrays(
            shape=idx_shape,
            dtype=dh.default_int,
            elements={"min_value": 0, "max_value": x.shape[axis]-1}
        ),
        label="indices"
    )
    note(f"{indices=}  {idx_shape=}")

    out = xp.take_along_axis(x, indices, axis=axis)

    ph.assert_dtype("take_along_axis", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape(
        "take_along_axis",
        out_shape=out.shape,
        expected=x.shape[:n_axis] + (len_axis,) + x.shape[n_axis+1:],
        kw=dict(
            x=x,
            indices=indices,
            axis=axis,
        ),
    )

    # value test: notation is from `np.take_along_axis` docstring
    Ni, Nk = x.shape[:n_axis], x.shape[n_axis+1:]
    for ii in sh.ndindex(Ni):
        for kk in sh.ndindex(Nk):
            a_1d = x[ii + (slice(None),) + kk]
            i_1d = indices[ii + (slice(None),) + kk]
            o_1d = out[ii + (slice(None),) + kk]
            for j in range(len_axis):
                assert o_1d[j] == a_1d[i_1d[j]], f'{ii=}, {kk=}, {j=}'
