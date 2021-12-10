from hypothesis import given
from hypothesis import strategies as st

from array_api_tests.test_statistical_functions import (
    assert_equals,
    assert_keepdimable_shape,
    axes_ndindex,
    normalise_axis,
)
from array_api_tests.typing import DataType

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps


def assert_default_index(func_name: str, dtype: DataType):
    f_dtype = dh.dtype_to_name[dtype]
    msg = (
        f"out.dtype={f_dtype}, should be the default index dtype, "
        f"which is either int32 or int64 [{func_name}()]"
    )
    assert dtype in (xp.int32, xp.int64), msg


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
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

    out = xp.argmax(x, **kw)

    assert_default_index("argmax", out.dtype)
    axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "argmax", out.shape, x.shape, axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, axes), ah.ndindex(out.shape)):
        max_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(range(len(elements)), key=elements.__getitem__)
        assert_equals("argmax", int, out_idx, max_i, expected)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
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

    out = xp.argmin(x, **kw)

    assert_default_index("argmin", out.dtype)
    axes = normalise_axis(kw.get("axis", None), x.ndim)
    assert_keepdimable_shape(
        "argmin", out.shape, x.shape, axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices, out_idx in zip(axes_ndindex(x.shape, axes), ah.ndindex(out.shape)):
        min_i = int(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(range(len(elements)), key=elements.__getitem__)
        assert_equals("argmin", int, out_idx, min_i, expected)


# TODO: generate kwargs, skip if opted out
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_nonzero(x):
    xp.nonzero(x)
    # TODO


# TODO: skip if opted out
@given(
    shapes=hh.mutually_broadcastable_shapes(3),
    dtypes=hh.mutually_promotable_dtypes(),
    data=st.data(),
)
def test_where(shapes, dtypes, data):
    cond = data.draw(xps.arrays(dtype=xp.bool, shape=shapes[0]), label="condition")
    x1 = data.draw(xps.arrays(dtype=dtypes[0], shape=shapes[1]), label="x1")
    x2 = data.draw(xps.arrays(dtype=dtypes[1], shape=shapes[2]), label="x2")
    xp.where(cond, x1, x2)
    # TODO
