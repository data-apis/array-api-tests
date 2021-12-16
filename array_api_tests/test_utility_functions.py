from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .test_statistical_functions import (
    assert_equals,
    assert_keepdimable_shape,
    axes,
    axes_ndindex,
    normalise_axis,
)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_all(x, data):
    kw = data.draw(hh.kwargs(axis=axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.all(x, **kw)

    ph.assert_dtype("all", x.dtype, out.dtype, xp.bool)
    _axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "all", out.shape, x.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, _axes), ah.ndindex(out.shape)):
        result = bool(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = all(elements)
        assert_equals("all", scalar_type, out_idx, result, expected)


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_any(x):
    xp.any(x)
    # TODO
