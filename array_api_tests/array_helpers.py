from _array_module import (isnan, all, equal, logical_not, isfinite, greater,
                           less, zeros, ones, full, bool, int8, int16, int32,
                           int64, uint8, uint16, uint32, uint64, float32,
                           float64, nan, inf, pi, remainder)

def zero(dtype):
    """
    Returns a scalar 0 of the given dtype.

    This should be used in place of the literal "0" in the test suite, as the
    spec does not require any behavior with Python literals (and in
    particular, it does not specify how the integer 0 and the float 0.0 work
    with type promotion).

    To get -0, use -zero(dtype) (note that -0 is only defined for floating
    point dtypes).
    """
    return zeros((), dtype=dtype)

def one(dtype):
    """
    Returns a scalar 1 of the given dtype.

    This should be used in place of the literal "1" in the test suite, as the
    spec does not require any behavior with Python literals (and in
    particular, it does not specify how the integer 1 and the float 1.0 work
    with type promotion).

    To get -1, use -one(dtype).
    """
    return ones((), dtype=dtype)

def NaN(dtype):
    """
    Returns a scalar nan of the given dtype.

    Note that this is only defined for floating point dtypes.
    """
    if dtype not in [float32, float64]:
        raise RuntimeError(f"Unexpected dtype {dtype} in nan().")
    return full((), nan, dtype=dtype)

def infinity(dtype):
    """
    Returns a scalar positive infinity of the given dtype.

    Note that this is only defined for floating point dtypes.

    To get negative infinity, use -infinity(dtype).

    """
    if dtype not in [float32, float64]:
        raise RuntimeError(f"Unexpected dtype {dtype} in infinity().")
    return full((), inf, dtype=dtype)

def π(dtype):
    """
    Returns a scalar π.

    Note that this function is only defined for floating point dtype.

    To get rational multiples of π, use, e.g., 3*π(dtype)/2.

    """
    if dtype not in [float32, float64]:
        raise RuntimeError(f"Unexpected dtype {dtype} in infinity().")
    return full((), pi, dtype=dtype)

def equal_nan(x, y):
    """
    Same as equal(x, y) except if y is nan, it is True where x is nan.
    """
    if all(isnan(y)):
        return isnan(x)
    return equal(x, y)

def assert_equal(x, y):
    """
    Test that the arrays x and y are equal.
    """
    assert x.shape == y.shape, "The input arrays do not have the same shapes"

    assert x.dtype == y.dtype, "The input arrays do not have the same dtype"

    xnans = isnan(x)
    ynans = isnan(y)

    assert all(equal(xnans, ynans)), "The input arrays have NaN values in different locations."

    notxnan = logical_not(xnans)
    notynan = logical_not(ynans)

    assert all(equal(x[notxnan], y[notynan])), "The input arrays have different values"

def assert_finite(x):
    """
    Test that the array x is finite
    """
    assert all(isfinite(x)), "The input array is not finite"

def assert_positive(x):
    assert all(greater(x, zero(x.dtype))), "The input array is not positive"

def assert_negative(x):
    assert all(less(x, zero(x.dtype))), "The input array is not negative"

def isintegral(x):
    """
    Returns a mask the shape of x where the values are integral

    x is integral if its dtype is an integer dtype, or if it is a floating
    point value that can be exactly represented as an integer.
    """
    if x.dtype in [int8, int16, int32, int64, uint8, uint16, uint32, uint64]:
        return full(x.shape, True, dtype=bool)
    elif x.dtype in [float32, float64]:
        return equal(remainder(x, one(x.dtype)), zero(x.dtype))
    else:
        return full(x.shape, False, dtype=bool)

def assert_integral(x):
    """
    Check that x has only integer values
    """
    assert all(isintegral(x)), "The input array has nonintegral values"
