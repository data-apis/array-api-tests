import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers

from ..array_helpers import exactly_equal, notequal, int_to_dtype, promote_dtypes
from ..hypothesis_helpers import integer_dtypes
from ..dtype_helpers import dtype_nbits, dtype_signed
from .. import _array_module as xp

# TODO: These meta-tests currently only work with NumPy

def test_exactly_equal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([True, False, True, False, True, False, True, False])
    assert xp.all(xp.equal(exactly_equal(a, b), res))

def test_notequal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([False, True, False, False, False, True, False, True])
    assert xp.all(xp.equal(notequal(a, b), res))

@given(integers(), integer_dtypes)
def test_int_to_dtype(x, dtype):
    n = dtype_nbits(dtype)
    signed = dtype_signed(dtype)
    try:
        d = xp.asarray(x, dtype=dtype)
    except OverflowError:
        assume(False)
    assert int_to_dtype(x, n, signed) == d

@pytest.mark.parametrize(
    "dtype1, dtype2, result",
    [
        (xp.uint8, xp.uint8, xp.uint8),
        (xp.uint8, xp.int8, xp.int16),
        (xp.int8, xp.int8, xp.int8),
    ]
)
def test_promote_dtypes(dtype1, dtype2, result):
    assert promote_dtypes(dtype1, dtype2) == result


@pytest.mark.parametrize("dtype1, dtype2", [(xp.uint8, xp.float32)])
def test_promote_dtypes_incompatible_dtypes_fail(dtype1, dtype2):
    with pytest.raises(ValueError):
        promote_dtypes(dtype1, dtype2)
