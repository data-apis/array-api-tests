import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_all(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")
    keepdims = kw.get("keepdims", False)

    out = xp.all(x, **kw)

    ph.assert_dtype("all", in_dtype=x.dtype, out_dtype=out.dtype, expected=xp.bool)
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "all", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        result = bool(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = all(elements)
        ph.assert_scalar_equals("all", type_=scalar_type, idx=out_idx,
                                out=result, expected=expected, kw=kw)


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes()),
    data=st.data(),
)
def test_any(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")
    keepdims = kw.get("keepdims", False)

    out = xp.any(x, **kw)

    ph.assert_dtype("any", in_dtype=x.dtype, out_dtype=out.dtype, expected=xp.bool)
    _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "any", in_shape=x.shape, out_shape=out.shape, axes=_axes, keepdims=keepdims, kw=kw,
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        result = bool(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = any(elements)
        ph.assert_scalar_equals("any", type_=scalar_type, idx=out_idx,
                                out=result, expected=expected, kw=kw)


@pytest.mark.unvectorized
@pytest.mark.min_version("2024.12")
@pytest.mark.has_setup_funcs
@given(
    x=hh.arrays(hh.numeric_dtypes, hh.shapes(min_dims=1, min_side=1)),
    data=st.data(),
)
def test_diff(x, data):
    axis = data.draw(
        st.integers(-x.ndim, max(x.ndim - 1, 0)) | st.none(),
        label="axis"
    )
    if axis is None:
        axis_kw = {"axis": -1}
        n_axis = x.ndim - 1
    else:
        axis_kw = {"axis": axis}
        n_axis = axis + x.ndim if axis < 0 else axis

    n = data.draw(st.integers(1, min(x.shape[n_axis], 3)))

    out = xp.diff(x, **axis_kw, n=n)

    expected_shape = list(x.shape)
    expected_shape[n_axis] -= n

    assert out.shape == tuple(expected_shape)

    # value test
    if n == 1:
        for idx in sh.ndindex(out.shape):
            l = list(idx)
            l[n_axis] += 1
            assert out[idx] == x[tuple(l)] - x[idx], f"diff failed with {idx = }"


@pytest.mark.min_version("2024.12")
@pytest.mark.unvectorized
@given(
    x=hh.arrays(hh.numeric_dtypes, hh.shapes(min_dims=1, min_side=1)),
    data=st.data(),
)
def test_diff_append_prepend(x, data):
    axis = data.draw(
        st.integers(-x.ndim, max(x.ndim - 1, 0)) | st.none(),
        label="axis"
    )
    if axis is None:
        axis_kw = {"axis": -1}
        n_axis = x.ndim - 1
    else:
        axis_kw = {"axis": axis}
        n_axis = axis + x.ndim if axis < 0 else axis

    n = data.draw(st.integers(1, min(x.shape[n_axis], 3)))

    append_shape = list(x.shape)
    append_axis_len = data.draw(st.integers(1, 2*append_shape[n_axis]), label="append_axis")
    append_shape[n_axis] = append_axis_len
    append = data.draw(hh.arrays(dtype=x.dtype, shape=tuple(append_shape)), label="append")

    prepend_shape = list(x.shape)
    prepend_axis_len = data.draw(st.integers(1, 2*prepend_shape[n_axis]), label="prepend_axis")
    prepend_shape[n_axis] = prepend_axis_len
    prepend = data.draw(hh.arrays(dtype=x.dtype, shape=tuple(prepend_shape)), label="prepend")

    out = xp.diff(x, **axis_kw, n=n, append=append, prepend=prepend)

    in_1 = xp.concat((prepend, x, append), **axis_kw)
    out_1 = xp.diff(in_1, **axis_kw, n=n)

    assert out.shape == out_1.shape
    for idx in sh.ndindex(out.shape):
        assert out[idx] == out_1[idx], f"{idx = }"

