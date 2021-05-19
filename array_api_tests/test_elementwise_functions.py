"""
Tests for elementwise functions

https://data-apis.github.io/array-api/latest/API_specification/elementwise_functions.html

This tests behavior that is explicitly mentioned in the spec. Note that the
spec does not make any accuracy requirements for functions, so this does not
test that. Tests for the special cases are generated and tested separately in
special_cases/

Note: Due to current limitations in Hypothesis, the tests below only test
arrays of shape (). In the future, the tests should be updated to test
arrays of any shape, using masking patterns (similar to the tests in special_cases/

"""

from hypothesis import given, assume
from hypothesis.strategies import composite, just

from .hypothesis_helpers import (integer_dtype_objects,
                                 floating_dtype_objects,
                                 numeric_dtype_objects,
                                 integer_or_boolean_dtype_objects,
                                 boolean_dtype_objects, floating_dtypes,
                                 numeric_dtypes, integer_or_boolean_dtypes,
                                 boolean_dtypes, mutually_promotable_dtypes,
                                 array_scalars)
from .array_helpers import (assert_exactly_equal, negative,
                            positive_mathematical_sign,
                            negative_mathematical_sign, logical_not,
                            logical_or, logical_and, inrange, π, one, zero,
                            infinity, full, isnegzero, isnegative, any as
                            array_any, int_to_dtype, bool as bool_dtype)

from . import _array_module

# integer_scalars = array_scalars(integer_dtypes)
floating_scalars = array_scalars(floating_dtypes)
numeric_scalars = array_scalars(numeric_dtypes)
integer_or_boolean_scalars = array_scalars(integer_or_boolean_dtypes)
boolean_scalars = array_scalars(boolean_dtypes)

two_integer_dtypes = mutually_promotable_dtypes(integer_dtype_objects)
two_floating_dtypes = mutually_promotable_dtypes(floating_dtype_objects)
two_numeric_dtypes = mutually_promotable_dtypes(numeric_dtype_objects)
two_integer_or_boolean_dtypes = mutually_promotable_dtypes(integer_or_boolean_dtype_objects)
two_boolean_dtypes = mutually_promotable_dtypes(boolean_dtype_objects)
two_any_dtypes = mutually_promotable_dtypes()

@composite
def two_array_scalars(draw, dtype1, dtype2):
    # two_dtypes should be a strategy that returns two dtypes (like
    # mutually_promotable_dtypes())
    return draw(array_scalars(just(dtype1))), draw(array_scalars(just(dtype2)))

def sanity_check(x1, x2):
    from .test_type_promotion import promotion_table, dtype_mapping
    t1 = [i for i in dtype_mapping if dtype_mapping[i] == x1.dtype][0]
    t2 = [i for i in dtype_mapping if dtype_mapping[i] == x2.dtype][0]

    if (t1, t2) not in promotion_table:
        raise RuntimeError("Error in test generation (probably a bug in the test suite")

@given(numeric_scalars)
def test_abs(x):
    a = _array_module.abs(x)
    assert _array_module.all(logical_not(negative_mathematical_sign(a))), "abs(x) did not have positive sign"
    less_zero = negative_mathematical_sign(x)
    negx = negative(x)
    # abs(x) = -x for x < 0
    assert_exactly_equal(a[less_zero], negx[less_zero])
    # abs(x) = x for x >= 0
    assert_exactly_equal(a[logical_not(less_zero)], x[logical_not(less_zero)])

@given(floating_scalars)
def test_acos(x):
    a = _array_module.acos(x)
    ONE = one(x.shape, x.dtype)
    # Here (and elsewhere), should technically be a.dtype, but this is the
    # same as x.dtype, as tested by the type_promotion tests.
    PI = π(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, -ONE, ONE)
    codomain = inrange(a, ZERO, PI)
    # acos maps [-1, 1] to [0, pi]. Values outside this domain are mapped to
    # nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_acosh(x):
    a = _array_module.acosh(x)
    ONE = one(x.shape, x.dtype)
    INFINITY = infinity(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, ONE, INFINITY)
    codomain = inrange(a, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_add(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.add(x1, x2)
    b = _array_module.add(x2, x1)
    # add is commutative
    assert_exactly_equal(a, b)
    # TODO: Test that add is actually addition

@given(floating_scalars)
def test_asin(x):
    a = _array_module.asin(x)
    ONE = one(x.shape, x.dtype)
    PI = π(x.shape, x.dtype)
    domain = inrange(x, -ONE, ONE)
    codomain = inrange(a, -PI/2, PI/2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_asinh(x):
    a = _array_module.asinh(x)
    INFINITY = infinity(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY)
    codomain = inrange(a, -INFINITY, INFINITY)
    # asinh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_atan(x):
    a = _array_module.atan(x)
    INFINITY = infinity(x.shape, x.dtype)
    PI = π(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY)
    codomain = inrange(a, -PI/2, PI/2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_atan2(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.atan2(x1, x2)
    INFINITY1 = infinity(x1.shape, x1.dtype)
    INFINITY2 = infinity(x2.shape, x2.dtype)
    PI = π(a.shape, a.dtype)
    domainx1 = inrange(x1, -INFINITY1, INFINITY1)
    domainx2 = inrange(x2, -INFINITY2, INFINITY2)
    # codomain = inrange(a, -PI, PI, 1e-5)
    codomain = inrange(a, -PI, PI)
    # atan2 maps [-inf, inf] x [-inf, inf] to [-pi, pi]. Values outside
    # this domain are mapped to nan, which is already tested in the special
    # cases.
    assert_exactly_equal(logical_and(domainx1, domainx2), codomain)
    # From the spec:
    #
    # The mathematical signs of `x1_i` and `x2_i` determine the quadrant of
    # each element-wise result. The quadrant (i.e., branch) is chosen such
    # that each element-wise result is the signed angle in radians between the
    # ray ending at the origin and passing through the point `(1,0)` and the
    # ray ending at the origin and passing through the point `(x2_i, x1_i)`.

    # This is equivalent to atan2(x1, x2) has the same sign as x1 when x2 is
    # finite.
    posx1 = positive_mathematical_sign(x1)
    negx1 = negative_mathematical_sign(x1)
    posx2 = positive_mathematical_sign(x2)
    negx2 = negative_mathematical_sign(x2)
    posa = positive_mathematical_sign(a)
    nega = negative_mathematical_sign(a)
    assert_exactly_equal(logical_or(logical_and(posx1, posx2),
                                    logical_and(posx1, negx2)), posa)
    assert_exactly_equal(logical_or(logical_and(negx1, posx2),
                                    logical_and(negx1, negx2)), nega)

@given(floating_scalars)
def test_atanh(x):
    a = _array_module.atanh(x)
    ONE = one(x.shape, x.dtype)
    INFINITY = infinity(x.shape, x.dtype)
    domain = inrange(x, -ONE, ONE)
    codomain = inrange(a, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_and(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_and(x1, x2)
    # Compare against the Python & operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_and needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    res = int(a)
    if a.dtype == bool_dtype:
        assert (x and y) == res
    else:
        ans = int_to_dtype(x & y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_left_shift(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    negative_x2 = isnegative(x2)
    if array_any(negative_x2):
        assume(False)
    a = _array_module.bitwise_left_shift(x1, x2)
    # Compare against the Python << operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_left_shift needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    if y >= dtype_nbits(a.dtype):
        # Avoid shifting very large y in Python ints
        ans = 0
    else:
        ans = x << y
    ans = int_to_dtype(ans, dtype_nbits(a.dtype), dtype_signed(a.dtype))
    res = int(a)
    assert ans == res

@given(integer_or_boolean_scalars)
def test_bitwise_invert(x):
    from .test_type_promotion import dtype_nbits, dtype_signed
    a = _array_module.bitwise_invert(x)
    # Compare against the Python ~ operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x.shape == ()):
        raise RuntimeError("Error: test_bitwise_invert needs to be updated for nonscalar array inputs")
    x = int(x)
    res = int(a)
    if a.dtype == bool_dtype:
        assert (not x) == res
    else:
        ans = int_to_dtype(~x, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_or(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_or(x1, x2)
    # Compare against the Python | operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_or needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    res = int(a)
    if a.dtype == bool_dtype:
        assert (x or y) == res
    else:
        ans = int_to_dtype(x | y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_right_shift(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    negative_x2 = isnegative(x2)
    if array_any(negative_x2):
        assume(False)
    a = _array_module.bitwise_right_shift(x1, x2)
    # Compare against the Python >> operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_right_shift needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    ans = int_to_dtype(x >> y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
    res = int(a)
    assert ans == res

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_xor(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_xor(x1, x2)
    # Compare against the Python ^ operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_xor needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    res = int(a)
    if a.dtype == bool_dtype:
        assert (x ^ y) == res
    else:
        ans = int_to_dtype(x ^ y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(numeric_scalars)
def test_ceil(x):
    a = _array_module.ceil(x)

@given(floating_scalars)
def test_cos(x):
    a = _array_module.cos(x)

@given(floating_scalars)
def test_cosh(x):
    a = _array_module.cosh(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.divide(x1, x2)

@given(two_any_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.equal(x1, x2)

@given(floating_scalars)
def test_exp(x):
    a = _array_module.exp(x)

@given(floating_scalars)
def test_expm1(x):
    a = _array_module.expm1(x)

@given(numeric_scalars)
def test_floor(x):
    a = _array_module.floor(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_floor_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.floor_divide(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater_equal(x1, x2)

@given(numeric_scalars)
def test_isfinite(x):
    a = _array_module.isfinite(x)

@given(numeric_scalars)
def test_isinf(x):
    a = _array_module.isinf(x)

@given(numeric_scalars)
def test_isnan(x):
    a = _array_module.isnan(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less_equal(x1, x2)

@given(floating_scalars)
def test_log(x):
    a = _array_module.log(x)

@given(floating_scalars)
def test_log1p(x):
    a = _array_module.log1p(x)

@given(floating_scalars)
def test_log2(x):
    a = _array_module.log2(x)

@given(floating_scalars)
def test_log10(x):
    a = _array_module.log10(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logaddexp(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logaddexp(x1, x2)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_and(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_and(x1, x2)

@given(boolean_scalars)
def test_logical_not(x):
    a = _array_module.logical_not(x)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_or(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_or(x1, x2)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_xor(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_xor(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_multiply(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.multiply(x1, x2)

@given(numeric_scalars)
def test_negative(x):
    a = _array_module.negative(x)

@given(two_any_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_not_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.not_equal(x1, x2)

@given(numeric_scalars)
def test_positive(x):
    a = _array_module.positive(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_pow(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.pow(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_remainder(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.remainder(x1, x2)

@given(numeric_scalars)
def test_round(x):
    a = _array_module.round(x)

@given(numeric_scalars)
def test_sign(x):
    a = _array_module.sign(x)

@given(floating_scalars)
def test_sin(x):
    a = _array_module.sin(x)

@given(floating_scalars)
def test_sinh(x):
    a = _array_module.sinh(x)

@given(numeric_scalars)
def test_square(x):
    a = _array_module.square(x)

@given(floating_scalars)
def test_sqrt(x):
    a = _array_module.sqrt(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_subtract(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.subtract(x1, x2)

@given(floating_scalars)
def test_tan(x):
    a = _array_module.tan(x)

@given(floating_scalars)
def test_tanh(x):
    a = _array_module.tanh(x)

@given(numeric_scalars)
def test_trunc(x):
    a = _array_module.trunc(x)
