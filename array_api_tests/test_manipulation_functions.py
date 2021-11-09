import math

from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps


@given(
    shape=hh.shapes(min_dims=1),
    dtypes=hh.mutually_promotable_dtypes(None, dtypes=dh.numeric_dtypes),
    kw=hh.kwargs(axis=st.just(0) | st.none()),  # TODO: test with axis >= 1
    data=st.data(),
)
def test_concat(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    out = xp.concat(arrays, **kw)
    ph.assert_dtype("concat", dtypes, out.dtype)
    shapes = tuple(x.shape for x in arrays)
    if kw.get("axis", 0) == 0:
        pass  # TODO: assert expected shape
    elif kw["axis"] is None:
        size = sum(math.prod(s) for s in shapes)
        ph.assert_result_shape("concat", shapes, out.shape, (size,), **kw)
    # TODO: assert out elements match input arrays


@given(
    shape=hh.shapes(),
    dtypes=hh.mutually_promotable_dtypes(None),
    data=st.data(),
)
def test_stack(shape, dtypes, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    out = xp.stack(arrays)
    ph.assert_dtype("stack", dtypes, out.dtype)
    # TODO
