import math

import pytest
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.control import assume

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps


pytestmark = pytest.mark.unvectorized


@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
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
        dtype=hh.real_dtypes,
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


@given(hh.arrays(dtype=hh.all_dtypes, shape=()))
def test_nonzero_zerodim_error(x):
    with pytest.raises(Exception):
        xp.nonzero(x)


@pytest.mark.data_dependent_shapes
@given(hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1, min_side=1)))
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
    if x.dtype == xp.bool:
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
    cond = data.draw(hh.arrays(dtype=xp.bool, shape=shapes[0]), label="condition")
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


@pytest.mark.min_version("2023.12")
@given(data=st.data())
def test_searchsorted(data):
    # TODO: test side="right"
    _x1 = data.draw(
        st.lists(xps.from_dtype(dh.default_float), min_size=1, unique=True),
        label="_x1",
    )
    x1 = xp.asarray(_x1, dtype=dh.default_float)
    if data.draw(st.booleans(), label="use sorter?"):
        sorter = data.draw(
            st.permutations(_x1).map(lambda o: xp.asarray(o, dtype=dh.default_float)),
            label="sorter",
        )
    else:
        sorter = None
        x1 = xp.sort(x1)
    note(f"{x1=}")
    x2 = data.draw(
        st.lists(st.sampled_from(_x1), unique=True, min_size=1).map(
            lambda o: xp.asarray(o, dtype=dh.default_float)
        ),
        label="x2",
    )

    out = xp.searchsorted(x1, x2, sorter=sorter)

    ph.assert_dtype(
        "searchsorted",
        in_dtype=[x1.dtype, x2.dtype],
        out_dtype=out.dtype,
        expected=xp.__array_namespace_info__().default_dtypes()["indexing"],
    )
    # TODO: shapes and values testing


@pytest.mark.unvectorized
# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data()
)
def test_top_k(x, data):

    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    axis = data.draw(
        st.integers(-x.ndim, x.ndim - 1), label='axis')
    largest = data.draw(st.booleans(), label='largest')
    if axis is None:
        k = data.draw(st.integers(1, math.prod(x.shape)))
    else:
        k = data.draw(st.integers(1, x.shape[axis]))

    kw = dict(
        x=x,
        k=k,
        axis=axis,
        largest=largest,
    )

    (out_values, out_indices) = xp.top_k(x, k, axis, largest=largest)
    if axis is None:
        x = xp.reshape(x, (-1,))
        axis = 0

    ph.assert_dtype("top_k", in_dtype=x.dtype, out_dtype=out_values.dtype)
    ph.assert_dtype(
        "top_k",
        in_dtype=x.dtype,
        out_dtype=out_indices.dtype,
        expected=dh.default_int
    )
    axes, = sh.normalise_axis(axis, x.ndim)
    for arr in [out_values, out_indices]:
        ph.assert_shape(
            "top_k",
            out_shape=arr.shape,
            expected=x.shape[:axes] + (k,) + x.shape[axes + 1:],
            kw=kw
        )

    scalar_type = dh.get_scalar_type(x.dtype)

    for indices in sh.axes_ndindex(x.shape, (axes,)):

        # Test if the values indexed by out_indices corresponds to
        # the correct top_k values.
        elements = [scalar_type(x[idx]) for idx in indices]
        size = len(elements)
        correct_order = sorted(
            range(size),
            key=elements.__getitem__,
            reverse=largest
        )
        correct_order = correct_order[:k]
        test_order = [out_indices[idx] for idx in indices[:k]]
        # Sort because top_k does not necessarily return the values in
        # sorted order.
        test_sorted_order = sorted(
            test_order,
            key=elements.__getitem__,
            reverse=largest
        )

        for y_o, x_o in zip(correct_order, test_sorted_order):
            y_idx = indices[y_o]
            x_idx = indices[x_o]
            ph.assert_0d_equals(
                "top_k",
                x_repr=f"x[{x_idx}]",
                x_val=x[x_idx],
                out_repr=f"x[{y_idx}]",
                out_val=x[y_idx],
                kw=kw,
            )

        # Test if the values indexed by out_indices corresponds to out_values.
        for y_o, x_idx in zip(test_order, indices[:k]):
            y_idx = indices[y_o]
            ph.assert_0d_equals(
                "top_k",
                x_repr=f"out_values[{x_idx}]",
                x_val=scalar_type(out_values[x_idx]),
                out_repr=f"x[{y_idx}]",
                out_val=x[y_idx],
                kw=kw
            )
