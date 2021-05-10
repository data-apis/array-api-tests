from ..array_helpers import exactly_equal, notequal, int_to_dtype
from ..hypothesis_helpers import integer_dtypes
from ..test_type_promotion import dtype_nbits, dtype_signed
from .._array_module import asarray, nan, equal, all

from hypothesis import given, assume
from hypothesis.strategies import integers

# TODO: These meta-tests currently only work with NumPy

def test_exactly_equal():
    a = asarray([0, 0., -0., -0., nan, nan, 1, 1])
    b = asarray([0, -1, -0.,  0., nan,      1, 1, 2])

    res = asarray([True, False, True, False, True, False, True, False])
    assert all(equal(exactly_equal(a, b), res))

def test_notequal():
    a = asarray([0, 0., -0., -0., nan, nan, 1, 1])
    b = asarray([0, -1, -0.,  0., nan,      1, 1, 2])

    res = asarray([False, True, False, False, False, True, False, True])
    assert all(equal(notequal(a, b), res))

@given(integers(), integer_dtypes)
def test_int_to_dtype(x, dtype):
    n = dtype_nbits(dtype)
    signed = dtype_signed(dtype)
    try:
        d = asarray(x, dtype=dtype)
    except OverflowError:
        assume(False)
    assert int_to_dtype(x, n, signed) == d
