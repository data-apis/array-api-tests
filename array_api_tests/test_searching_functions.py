import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps


pytestmark = pytest.mark.unvectorized

array_mod_bool = dh.get_array_module_bool()

@given(
    x=hh.arrays(
        dtype=xps.real_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmax(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    out = xp.argmax(x, **kw)

    ph.assert_default_index("argmax", out.dtype)
    axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "argmax", in_shape=x.shape, out_shape=out.shape, axes=axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
        max_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(range(len(elements)), key=elements.__getitem__)
        ph.assert_scalar_equals("argmax", type_=int, idx=out_idx, out=max_i,
                                expected=expected, kw=kw)


@given(
    x=hh.arrays(
        dtype=xps.real_dtypes(),
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmin(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    out = xp.argmin(x, **kw)

    ph.assert_default_index("argmin", out.dtype)
    axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "argmin", in_shape=x.shape, out_shape=out.shape, axes=axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
        min_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(range(len(elements)), key=elements.__getitem__)
        ph.assert_scalar_equals("argmin", type_=int, idx=out_idx, out=min_i, expected=expected)


@given(hh.arrays(dtype=xps.scalar_dtypes(), shape=()))
def test_nonzero_zerodim_error(x):
    with pytest.raises(Exception):
        xp.nonzero(x)


@pytest.mark.data_dependent_shapes
@given(hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_dims=1, min_side=1)))
def test_nonzero(x):
    out = xp.nonzero(x)
    assert len(out) == x.ndim, f"{len(out)=}, but should be {x.ndim=}"
    out_size = math.prod(out[0].shape)
    for i in range(len(out)):
        assert out[i].ndim == 1, f"out[{i}].ndim={x.ndim}, but should be 1"
        size_at = math.prod(out[i].shape)
        assert size_at == out_size, (
            f"prod(out[{i}].shape)={size_at}, "
            f"but should be prod(out[0].shape)={out_size}"
        )
        ph.assert_default_index("nonzero", out[i].dtype, repr_name=f"out[{i}].dtype")
    indices = []
    if x.dtype == array_mod_bool:
        for idx in sh.ndindex(x.shape):
            if x[idx]:
                indices.append(idx)
    else:
        for idx in sh.ndindex(x.shape):
            if x[idx] != 0:
                indices.append(idx)
    if x.ndim == 0:
        assert out_size == len(
            indices
        ), f"prod(out[0].shape)={out_size}, but should be {len(indices)}"
    else:
        for i in range(out_size):
            idx = tuple(int(x[i]) for x in out)
            f_idx = f"Extrapolated index (x[{i}] for x in out)={idx}"
            f_element = f"x[{idx}]={x[idx]}"
            assert idx in indices, f"{f_idx} results in {f_element}, a zero element"
            assert (
                idx == indices[i]
            ), f"{f_idx} is in the wrong position, should be {indices.index(idx)}"


@given(
    shapes=hh.mutually_broadcastable_shapes(3),
    dtypes=hh.mutually_promotable_dtypes(),
    data=st.data(),
)
def test_where(shapes, dtypes, data):
    cond = data.draw(hh.arrays(dtype=array_mod_bool, shape=shapes[0]), label="condition")
    x1 = data.draw(hh.arrays(dtype=dtypes[0], shape=shapes[1]), label="x1")
    x2 = data.draw(hh.arrays(dtype=dtypes[1], shape=shapes[2]), label="x2")

    out = xp.where(cond, x1, x2)

    shape = sh.broadcast_shapes(*shapes)
    ph.assert_shape("where", out_shape=out.shape, expected=shape)
    # TODO: generate indices without broadcasting arrays
    _cond = xp.broadcast_to(cond, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)
    for idx in sh.ndindex(shape):
        if _cond[idx]:
            ph.assert_0d_equals(
                "where",
                x_repr=f"_x1[{idx}]",
                x_val=_x1[idx],
                out_repr=f"out[{idx}]",
                out_val=out[idx]
            )
        else:
            ph.assert_0d_equals(
                "where",
                x_repr=f"_x2[{idx}]",
                x_val=_x2[idx],
                out_repr=f"out[{idx}]",
                out_val=out[idx]
            )
