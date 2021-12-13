import math

from hypothesis import assume, given

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_all(x):
    xp.unique_all(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_counts(x):
    xp.unique_counts(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_inverse(x):
    xp.unique_inverse(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_unique_values(x):
    out = xp.unique_values(x)
    ph.assert_dtype("unique_values", x.dtype, out.dtype)
    scalar_type = dh.get_scalar_type(x.dtype)
    distinct = set(scalar_type(x[idx]) for idx in ah.ndindex(x.shape))
    vals_idx = {}
    nans = 0
    for idx in ah.ndindex(out.shape):
        val = scalar_type(out[idx])
        if math.isnan(val):
            nans += 1
        else:
            assert val in distinct, f"out[{idx}]={val}, but {val} not in input array"
            assert (
                val not in vals_idx.keys()
            ), f"out[{idx}]={val}, but {val} is also in out[{vals_idx[val]}]"
            vals_idx[val] = idx
    if dh.is_float_dtype(out.dtype):
        assume(x.size <= 128)  # may not be representable
        expected_nans = xp.sum(xp.astype(xp.isnan(x), xp.uint8))
        assert nans == expected_nans, f"{nans} NaNs in out, expected {expected_nans}"
