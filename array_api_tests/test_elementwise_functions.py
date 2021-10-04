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

import math

from .hypothesis_helpers import (integer_dtype_objects,
                                 floating_dtype_objects,
                                 numeric_dtype_objects,
                                 integer_or_boolean_dtype_objects,
                                 boolean_dtype_objects, floating_dtypes,
                                 numeric_dtypes, integer_or_boolean_dtypes,
                                 boolean_dtypes, mutually_promotable_dtypes,
                                 array_scalars, two_mutual_arrays, xps, shapes)
from .array_helpers import (assert_exactly_equal, negative,
                            positive_mathematical_sign,
                            negative_mathematical_sign, logical_not,
                            logical_or, logical_and, inrange, π, one, zero,
                            infinity, isnegative, all as array_all, any as
                            array_any, int_to_dtype, bool as bool_dtype,
                            assert_integral, less_equal, isintegral, isfinite,
                            ndindex, promote_dtypes, is_integer_dtype,
                            is_float_dtype, not_equal, float64, asarray,
                            dtype_ranges, full, true, false, assert_same_sign,
                            isnan, equal, less)
# We might as well use this implementation rather than requiring
# mod.broadcast_shapes(). See test_equal() and others.
from .test_broadcasting import broadcast_shapes

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
    try:
        promote_dtypes(x1.dtype, x2.dtype)
    except ValueError:
        raise RuntimeError("Error in test generation (probably a bug in the test suite")

@given(numeric_scalars)
def test_abs(x):
    if is_integer_dtype(x.dtype):
        minval = dtype_ranges[x.dtype][0]
        if minval < 0:
            # abs of the smallest representable negative integer is not defined
            mask = not_equal(x, full(x.shape, minval, dtype=x.dtype))
            x = x[mask]
    a = _array_module.abs(x)
    assert array_all(logical_not(negative_mathematical_sign(a))), "abs(x) did not have positive sign"
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

    if a.dtype == bool_dtype:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x and y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
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
    if a.dtype == bool_dtype:
        x = bool(x)
        res = bool(a)
        assert (not x) == res
    else:
        x = int(x)
        res = int(a)
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
    if a.dtype == bool_dtype:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x or y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
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
    if a.dtype == bool_dtype:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x ^ y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
        ans = int_to_dtype(x ^ y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(numeric_scalars)
def test_ceil(x):
    # This test is almost identical to test_floor()
    a = _array_module.ceil(x)
    finite = isfinite(x)
    assert_integral(a[finite])
    assert array_all(less_equal(x[finite], a[finite]))
    assert array_all(less_equal(a[finite] - x[finite], one(x[finite].shape, x.dtype)))
    integers = isintegral(x)
    assert_exactly_equal(a[integers], x[integers])

@given(floating_scalars)
def test_cos(x):
    a = _array_module.cos(x)
    ONE = one(x.shape, x.dtype)
    INFINITY = infinity(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY, open=True)
    codomain = inrange(a, -ONE, ONE)
    # cos maps (-inf, inf) to [-1, 1]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_cosh(x):
    a = _array_module.cosh(x)
    INFINITY = infinity(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY)
    codomain = inrange(a, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    _array_module.divide(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@given(two_mutual_arrays())
def test_equal(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = _array_module.equal(x1, x2)
    # NOTE: assert_exactly_equal() itself uses equal(), so we must be careful
    # not to use it here. Otherwise, the test would be circular and
    # meaningless. Instead, we implement this by iterating every element of
    # the arrays and comparing them. The logic here is also used for the tests
    # for the other elementwise functions that accept any input dtype but
    # always return bool (greater(), greater_equal(), less(), less_equal(),
    # and not_equal()).

    # First we broadcast the arrays so that they can be indexed uniformly.
    # TODO: it should be possible to skip this step if we instead generate
    # indices to x1 and x2 that correspond to the broadcasted shapes. This
    # would avoid the dependence in this test on broadcast_to().
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    # Second, manually promote the dtypes. This is important. If the internal
    # type promotion in equal() is wrong, it will not be directly visible in
    # the output type, but it can lead to wrong answers. For example,
    # equal(array(1.0, dtype=float32), array(1.00000001, dtype=float64)) will
    # be wrong if the float64 is downcast to float32. # be wrong if the
    # float64 is downcast to float32. See the comment on
    # test_elementwise_function_two_arg_bool_type_promotion() in
    # test_type_promotion.py. The type promotion for equal() is not *really*
    # tested in that file, because doing so requires doing the consistency
    # check we do here rather than just checking the result dtype.
    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) == scalar_func(x2idx))

@given(floating_scalars)
def test_exp(x):
    a = _array_module.exp(x)
    INFINITY = infinity(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY)
    codomain = inrange(a, ZERO, INFINITY)
    # exp maps [-inf, inf] to [0, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_expm1(x):
    a = _array_module.expm1(x)
    INFINITY = infinity(x.shape, x.dtype)
    NEGONE = -one(x.shape, x.dtype)
    domain = inrange(x, -INFINITY, INFINITY)
    codomain = inrange(a, NEGONE, INFINITY)
    # expm1 maps [-inf, inf] to [1, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(numeric_scalars)
def test_floor(x):
    # This test is almost identical to test_ceil
    a = _array_module.floor(x)
    finite = isfinite(x)
    assert_integral(a[finite])
    assert array_all(less_equal(a[finite], x[finite]))
    assert array_all(less_equal(x[finite] - a[finite], one(x[finite].shape, x.dtype)))
    integers = isintegral(x)
    assert_exactly_equal(a[integers], x[integers])

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_floor_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    if is_integer_dtype(x1.dtype):
        # The spec does not specify the behavior for division by 0 for integer
        # dtypes. A library may choose to raise an exception in this case, so
        # we avoid passing it in entirely.
        nonzero = not_equal(x2, zero(x2.shape, x2.dtype))
        div = _array_module.divide(
            asarray(x1[nonzero], dtype=float64),
            asarray(x2[nonzero], dtype=float64))
        a = _array_module.floor_divide(x1[nonzero], x2[nonzero])
    else:
        div = _array_module.divide(x1, x2)
        a = _array_module.floor_divide(x1, x2)

    # TODO: The spec doesn't clearly specify the behavior of floor_divide on
    # infinities. See https://github.com/data-apis/array-api/issues/199.
    finite = isfinite(div)
    assert_integral(a[finite])

    # TODO: Test the exact output for floor_divide.

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) > scalar_func(x2idx))

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) >= scalar_func(x2idx))

@given(numeric_scalars)
def test_isfinite(x):
    a = _array_module.isfinite(x)
    TRUE = true(x.shape)
    if is_integer_dtype(x.dtype):
        assert_exactly_equal(a, TRUE)
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = logical_or(_array_module.isinf(x), _array_module.isnan(x))
    assert_exactly_equal(a, logical_not(inf))

    # Test the exact value by comparing to the math version
    if is_float_dtype(x.dtype):
        for idx in ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isfinite(s)

@given(numeric_scalars)
def test_isinf(x):
    a = _array_module.isinf(x)
    FALSE = false(x.shape)
    if is_integer_dtype(x.dtype):
        assert_exactly_equal(a, FALSE)
    finite_or_nan = logical_or(_array_module.isfinite(x), _array_module.isnan(x))
    assert_exactly_equal(a, logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if is_float_dtype(x.dtype):
        for idx in ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isinf(s)

@given(numeric_scalars)
def test_isnan(x):
    a = _array_module.isnan(x)
    FALSE = false(x.shape)
    if is_integer_dtype(x.dtype):
        assert_exactly_equal(a, FALSE)
    finite_or_inf = logical_or(_array_module.isfinite(x), _array_module.isinf(x))
    assert_exactly_equal(a, logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if is_float_dtype(x.dtype):
        for idx in ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isnan(s)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) < scalar_func(x2idx))

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) <= scalar_func(x2idx))

@given(floating_scalars)
def test_log(x):
    a = _array_module.log(x)
    INFINITY = infinity(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, ZERO, INFINITY)
    codomain = inrange(a, -INFINITY, INFINITY)
    # log maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_log1p(x):
    a = _array_module.log1p(x)
    INFINITY = infinity(x.shape, x.dtype)
    NEGONE = -one(x.shape, x.dtype)
    codomain = inrange(x, NEGONE, INFINITY)
    domain = inrange(a, -INFINITY, INFINITY)
    # log1p maps [1, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_log2(x):
    a = _array_module.log2(x)
    INFINITY = infinity(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, ZERO, INFINITY)
    codomain = inrange(a, -INFINITY, INFINITY)
    # log2 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(floating_scalars)
def test_log10(x):
    a = _array_module.log10(x)
    INFINITY = infinity(x.shape, x.dtype)
    ZERO = zero(x.shape, x.dtype)
    domain = inrange(x, ZERO, INFINITY)
    codomain = inrange(a, -INFINITY, INFINITY)
    # log10 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    assert_exactly_equal(domain, codomain)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logaddexp(args):
    x1, x2 = args
    sanity_check(x1, x2)
    _array_module.logaddexp(x1, x2)
    # The spec doesn't require any behavior for this function. We could test
    # that this is indeed an approximation of log(exp(x1) + exp(x2)), but we
    # don't have tests for this sort of thing for any functions yet.

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_and(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_and(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    for idx in ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) and bool(_x2[idx]))

@given(boolean_scalars)
def test_logical_not(x):
    a = _array_module.logical_not(x)

    for idx in ndindex(x.shape):
        assert a[idx] == (not bool(x[idx]))

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_or(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_or(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    for idx in ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) or bool(_x2[idx]))

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_xor(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_xor(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    for idx in ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) ^ bool(_x2[idx]))

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_multiply(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.multiply(x1, x2)

    b = _array_module.multiply(x2, x1)
    # multiply is commutative
    assert_exactly_equal(a, b)

@given(numeric_scalars)
def test_negative(x):
    a = _array_module.negative(x)

    # Negation is an involution
    b = _array_module.negative(a)
    assert_exactly_equal(x, b)

    mask = isfinite(x)
    if is_integer_dtype(x.dtype):
        minval = dtype_ranges[x.dtype][0]
        if minval < 0:
            # negative of the smallest representable negative integer is not defined
            mask = not_equal(x, full(x.shape, minval, dtype=x.dtype))
            x = x[mask]

    # Additive inverse
    y = _array_module.add(x[mask], a[mask])
    ZERO = zero(x[mask].shape, x.dtype)
    assert_exactly_equal(y, ZERO)

@given(two_any_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_not_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.not_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = _array_module.broadcast_to(x1, shape)
    _x2 = _array_module.broadcast_to(x2, shape)

    promoted_dtype = promote_dtypes(x1.dtype, x2.dtype)
    _x1 = _array_module.asarray(_x1, dtype=promoted_dtype)
    _x2 = _array_module.asarray(_x2, dtype=promoted_dtype)

    if is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) != scalar_func(x2idx))

@given(numeric_scalars)
def test_positive(x):
    a = _array_module.positive(x)
    # Positive does nothing
    assert_exactly_equal(a, x)

@given(two_mutual_arrays(floating_dtype_objects))
def test_pow(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    _array_module.pow(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.

@given(two_mutual_arrays(numeric_dtype_objects))
def test_remainder(x1_and_x2):
    x1, x2 = x1_and_x2
    assume(len(x1.shape) <= len(x2.shape)) # TODO: rework same sign testing below to remove this
    sanity_check(x1, x2)
    out = _array_module.remainder(x1, x2)

    # out and x2 should have the same sign.
    # assert_same_sign returns False for nans
    not_nan = logical_not(logical_or(isnan(out), isnan(x2)))
    assert_same_sign(out[not_nan], x2[not_nan])

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=shapes))
def test_round(x):
    a = _array_module.round(x)

    # Test that the result is integral
    finite = isfinite(x)
    assert_integral(a[finite])

    # round(x) should be the nearest integer to x. The case where there is a
    # tie (round to even) is already handled by the special cases tests.

    # This is the same strategy used in the mask in the
    # test_round_special_cases_one_arg_two_integers_equally_close special
    # cases test.
    floor = _array_module.floor(x)
    ceil = _array_module.ceil(x)
    over = _array_module.subtract(x, floor)
    under = _array_module.subtract(ceil, x)
    round_down = less(over, under)
    round_up = less(under, over)
    assert_exactly_equal(a[round_down], floor[round_down])
    assert_exactly_equal(a[round_up], ceil[round_up])

@given(numeric_scalars)
def test_sign(x):
    # a = _array_module.sign(x)
    pass

@given(floating_scalars)
def test_sin(x):
    # a = _array_module.sin(x)
    pass

@given(floating_scalars)
def test_sinh(x):
    # a = _array_module.sinh(x)
    pass

@given(numeric_scalars)
def test_square(x):
    # a = _array_module.square(x)
    pass

@given(floating_scalars)
def test_sqrt(x):
    # a = _array_module.sqrt(x)
    pass

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_subtract(args):
    x1, x2 = args
    sanity_check(x1, x2)
    # a = _array_module.subtract(x1, x2)

@given(floating_scalars)
def test_tan(x):
    # a = _array_module.tan(x)
    pass

@given(floating_scalars)
def test_tanh(x):
    # a = _array_module.tanh(x)
    pass

@given(xps.arrays(dtype=numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    out = _array_module.trunc(x)
    assert out.dtype == x.dtype, f"{x.dtype=!s} but {out.dtype=!s}"
    assert out.shape == x.shape, f"{x.shape} but {out.shape}"
    if x.dtype in integer_dtype_objects:
        assert array_all(equal(x, out)), f"{x=!s} but {out=!s}"
    else:
        finite_mask = _array_module.isfinite(out)
        for idx in ndindex(out.shape):
            if finite_mask[idx]:
                assert float(out[idx]).is_integer(), f"x at {idx=} is {x[idx]}, but out at idx is {out[idx]}"
