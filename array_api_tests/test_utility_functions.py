import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_all(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")
    keepdims = kw.get("keepdims", False)

    out = xp.all(x, **kw)

    ph.assert_dtype("all", in_dtype=x.dtype, out_dtype=out.dtype, expected=xp.bool)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
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
    x=hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    data=st.data(),
)
def test_any(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")
    keepdims = kw.get("keepdims", False)

    out = xp.any(x, **kw)

    ph.assert_dtype("any", in_dtype=x.dtype, out_dtype=out.dtype, expected=xp.bool)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
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
