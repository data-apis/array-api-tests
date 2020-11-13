from ._array_module import (isnan, all, equal, not_equal, logical_and,
                            logical_or, isfinite, greater, less, zeros, ones,
                            full, bool, int8, int16, int32, int64, uint8,
                            uint16, uint32, uint64, float32, float64, nan,
                            inf, pi, remainder, divide, isinf)

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

def isnegzero(x):
    """
    Returns a mask where x is -0.
    """
    # TODO: If copysign or signbit are added to the spec, use those instead.
    dtype = x.dtype
    return equal(divide(one(dtype), x), -infinity(dtype))

def isposzero(x):
    """
    Returns a mask where x is +0 (but not -0).
    """
    # TODO: If copysign or signbit are added to the spec, use those instead.
    dtype = x.dtype
    return equal(divide(one(dtype), x), infinity(dtype))

def exactly_equal(x, y):
    """
    Same as equal(x, y) except it gives True where both values are nan, and
    distinguishes +0 and -0.

    This function implicitly assumes x and y have the same shape and dtype.
    """
    if x.dtype in [float32, float64]:
        xnegzero = isnegzero(x)
        ynegzero = isnegzero(y)

        xposzero = isposzero(x)
        yposzero = isposzero(y)

        xnan = isnan(x)
        ynan = isnan(y)

        # (x == y OR x == y == NaN) AND xnegzero == ynegzero AND xposzero == y poszero
        return logical_and(logical_and(
            logical_or(equal(x, y), logical_and(xnan, ynan)),
            equal(xnegzero, ynegzero)),
            equal(xposzero, yposzero))

    return equal(x, y)

def assert_exactly_equal(x, y):
    """
    Test that the arrays x and y are exactly equal.

    If x and y do not have the same shape and dtype, they are not considered
    equal.

    """
    assert x.shape == y.shape, "The input arrays do not have the same shapes"

    assert x.dtype == y.dtype, "The input arrays do not have the same dtype"

    assert all(exactly_equal(x, y)), "The input arrays have different values"

def assert_finite(x):
    """
    Test that the array x is finite
    """
    assert all(isfinite(x)), "The input array is not finite"

def nonzero(x):
    not_equal(x, zero(x.dtype))

def assert_nonzero(x):
    assert all(nonzero(x)), "The input array is not nonzero"

def ispositive(x):
    return greater(x, zero(x.dtype))

def assert_positive(x):
    assert all(ispositive(x)), "The input array is not positive"

def isnegative(x):
    return less(x, zero(x.dtype))

def assert_negative(x):
    assert all(isnegative(x)), "The input array is not negative"

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

def isodd(x):
    return logical_and(isintegral(x), equal(remainder(x, 2*one(x.dtype)), one(x.dtype)))

def assert_isinf(x):
    """
    Check that x is an infinity
    """
    assert all(isinf(x)), "The input array is not infinite"

def same_sign(x, y):
    """
    Check if x and y have the "same sign"

    x and y have the same sign if they are both nonnegative or both negative.
    For the purposes of this function 0 and 1 have the same sign and -0 and -1
    have the same sign. The value of this function is False if either x or y
    is nan, as signed nans are not required by the spec.
    """
    logical_or(
        logical_and(
            logical_or(greater(x, 0), isposzero(x)),
            logical_or(greater(y, 0), isposzero(y))),
        logical_and(
            logical_or(less(x, 0), isnegzero(x)),
            logical_or(less(y, 0), isnegzero(y))))

def assert_same_sign(x, y):
    assert all(same_sign(x, y)), "The input arrays do not have the same sign"
