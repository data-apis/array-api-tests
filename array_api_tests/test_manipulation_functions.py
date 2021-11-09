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
    data=st.data(),
)
def test_concat(shape, dtypes, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    out = xp.concat(arrays)
    ph.assert_dtype("concat", dtypes, out.dtype)
    # TODO


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
