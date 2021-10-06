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

import math

from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import hypothesis_helpers as hh
from . import xps
# We might as well use this implementation rather than requiring
# mod.broadcast_shapes(). See test_equal() and others.
from .test_broadcasting import broadcast_shapes

# integer_scalars = hh.array_scalars(integer_dtypes)
floating_scalars = hh.array_scalars(hh.floating_dtypes)
numeric_scalars = hh.array_scalars(hh.numeric_dtypes)
integer_or_boolean_scalars = hh.array_scalars(hh.integer_or_boolean_dtypes)
boolean_scalars = hh.array_scalars(hh.boolean_dtypes)

two_integer_dtypes = hh.mutually_promotable_dtypes(hh.integer_dtype_objects)
two_floating_dtypes = hh.mutually_promotable_dtypes(hh.floating_dtype_objects)
two_numeric_dtypes = hh.mutually_promotable_dtypes(hh.numeric_dtype_objects)
two_integer_or_boolean_dtypes = hh.mutually_promotable_dtypes(hh.integer_or_boolean_dtype_objects)
two_boolean_dtypes = hh.mutually_promotable_dtypes(hh.boolean_dtype_objects)
two_any_dtypes = hh.mutually_promotable_dtypes()

@st.composite
def two_array_scalars(draw, dtype1, dtype2):
    # two_dtypes should be a strategy that returns two dtypes (like
    # hh.mutually_promotable_dtypes())
    return draw(hh.array_scalars(st.just(dtype1))), draw(hh.array_scalars(st.just(dtype2)))

def sanity_check(x1, x2):
    try:
        ah.promote_dtypes(x1.dtype, x2.dtype)
    except ValueError:
        raise RuntimeError("Error in test generation (probably a bug in the test suite")

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_abs(x):
    if ah.is_integer_dtype(x.dtype):
        minval = ah.dtype_ranges[x.dtype][0]
        if minval < 0:
            # abs of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))
            x = x[mask]
    a = xp.abs(x)
    assert ah.all(ah.logical_not(ah.negative_mathematical_sign(a))), "abs(x) did not have positive sign"
    less_zero = ah.negative_mathematical_sign(x)
    negx = ah.negative(x)
    # abs(x) = -x for x < 0
    ah.assert_exactly_equal(a[less_zero], negx[less_zero])
    # abs(x) = x for x >= 0
    ah.assert_exactly_equal(a[ah.logical_not(less_zero)], x[ah.logical_not(less_zero)])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_acos(x):
    a = xp.acos(x)
    ONE = ah.one(x.shape, x.dtype)
    # Here (and elsewhere), should technically be a.dtype, but this is the
    # same as x.dtype, as tested by the type_promotion tests.
    PI = ah.π(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(a, ZERO, PI)
    # acos maps [-1, 1] to [0, pi]. Values outside this domain are mapped to
    # nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_acosh(x):
    a = xp.acosh(x)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ONE, INFINITY)
    codomain = ah.inrange(a, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_add(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.add(x1, x2)

    b = xp.add(x2, x1)
    # add is commutative
    ah.assert_exactly_equal(a, b)
    # TODO: Test that add is actually addition

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_asin(x):
    a = xp.asin(x)
    ONE = ah.one(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(a, -PI/2, PI/2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_asinh(x):
    a = xp.asinh(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # asinh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_atan(x):
    a = xp.atan(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -PI/2, PI/2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(hh.two_mutual_arrays(hh.floating_dtype_objects))
def test_atan2(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.atan2(x1, x2)
    INFINITY1 = ah.infinity(x1.shape, x1.dtype)
    INFINITY2 = ah.infinity(x2.shape, x2.dtype)
    PI = ah.π(a.shape, a.dtype)
    domainx1 = ah.inrange(x1, -INFINITY1, INFINITY1)
    domainx2 = ah.inrange(x2, -INFINITY2, INFINITY2)
    # codomain = ah.inrange(a, -PI, PI, 1e-5)
    codomain = ah.inrange(a, -PI, PI)
    # atan2 maps [-inf, inf] x [-inf, inf] to [-pi, pi]. Values outside
    # this domain are mapped to nan, which is already tested in the special
    # cases.
    ah.assert_exactly_equal(ah.logical_and(domainx1, domainx2), codomain)
    # From the spec:
    #
    # The mathematical signs of `x1_i` and `x2_i` determine the quadrant of
    # each element-wise result. The quadrant (i.e., branch) is chosen such
    # that each element-wise result is the signed angle in radians between the
    # ray ending at the origin and passing through the point `(1,0)` and the
    # ray ending at the origin and passing through the point `(x2_i, x1_i)`.

    # This is equivalent to atan2(x1, x2) has the same sign as x1 when x2 is
    # finite.
    posx1 = ah.positive_mathematical_sign(x1)
    negx1 = ah.negative_mathematical_sign(x1)
    posx2 = ah.positive_mathematical_sign(x2)
    negx2 = ah.negative_mathematical_sign(x2)
    posa = ah.positive_mathematical_sign(a)
    nega = ah.negative_mathematical_sign(a)
    ah.assert_exactly_equal(ah.logical_or(ah.logical_and(posx1, posx2),
                                    ah.logical_and(posx1, negx2)), posa)
    ah.assert_exactly_equal(ah.logical_or(ah.logical_and(negx1, posx2),
                                    ah.logical_and(negx1, negx2)), nega)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_atanh(x):
    a = xp.atanh(x)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_and(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = xp.bitwise_and(x1, x2)
    # Compare against the Python & operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_and needs to be updated for nonscalar array inputs")

    if a.dtype == xp.bool:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x and y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
        ans = ah.int_to_dtype(x & y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_left_shift(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    negative_x2 = ah.isnegative(x2)
    assume(not xp.any(negative_x2))
    a = xp.bitwise_left_shift(x1, x2)
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
    ans = ah.int_to_dtype(ans, dtype_nbits(a.dtype), dtype_signed(a.dtype))
    res = int(a)
    assert ans == res

@given(integer_or_boolean_scalars)
def test_bitwise_invert(x):
    from .test_type_promotion import dtype_nbits, dtype_signed
    a = xp.bitwise_invert(x)
    # Compare against the Python ~ operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x.shape == ()):
        raise RuntimeError("Error: test_bitwise_invert needs to be updated for nonscalar array inputs")
    if a.dtype == xp.bool:
        x = bool(x)
        res = bool(a)
        assert (not x) == res
    else:
        x = int(x)
        res = int(a)
        ans = ah.int_to_dtype(~x, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_or(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = xp.bitwise_or(x1, x2)
    # Compare against the Python | operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_or needs to be updated for nonscalar array inputs")
    if a.dtype == xp.bool:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x or y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
        ans = ah.int_to_dtype(x | y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_right_shift(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    negative_x2 = ah.isnegative(x2)
    assume(not xp.any(negative_x2))
    a = xp.bitwise_right_shift(x1, x2)
    # Compare against the Python >> operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_right_shift needs to be updated for nonscalar array inputs")
    x = int(x1)
    y = int(x2)
    ans = ah.int_to_dtype(x >> y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
    res = int(a)
    assert ans == res

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_xor(args):
    from .test_type_promotion import dtype_nbits, dtype_signed
    x1, x2 = args
    sanity_check(x1, x2)
    a = xp.bitwise_xor(x1, x2)
    # Compare against the Python ^ operator.
    # TODO: Generalize this properly for inputs that are arrays.
    if not (x1.shape == x2.shape == ()):
        raise RuntimeError("Error: test_bitwise_xor needs to be updated for nonscalar array inputs")
    if a.dtype == xp.bool:
        x = bool(x1)
        y = bool(x2)
        res = bool(a)
        assert (x ^ y) == res
    else:
        x = int(x1)
        y = int(x2)
        res = int(a)
        ans = ah.int_to_dtype(x ^ y, dtype_nbits(a.dtype), dtype_signed(a.dtype))
        assert ans == res

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_ceil(x):
    # This test is almost identical to test_floor()
    a = xp.ceil(x)
    finite = ah.isfinite(x)
    ah.assert_integral(a[finite])
    assert ah.all(ah.less_equal(x[finite], a[finite]))
    assert ah.all(ah.less_equal(a[finite] - x[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(a[integers], x[integers])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_cos(x):
    a = xp.cos(x)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY, open=True)
    codomain = ah.inrange(a, -ONE, ONE)
    # cos maps (-inf, inf) to [-1, 1]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_cosh(x):
    a = xp.cosh(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(hh.two_mutual_arrays(hh.floating_dtype_objects))
def test_divide(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    xp.divide(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@given(hh.two_mutual_arrays())
def test_equal(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = ah.equal(x1, x2)
    # NOTE: ah.assert_exactly_equal() itself uses ah.equal(), so we must be careful
    # not to use it here. Otherwise, the test would be circular and
    # meaningless. Instead, we implement this by iterating every element of
    # the arrays and comparing them. The logic here is also used for the tests
    # for the other elementwise functions that accept any input dtype but
    # always return bool (greater(), greater_equal(), less(), less_equal(),
    # and not_equal()).

    # First we broadcast the arrays so that they can be indexed uniformly.
    # TODO: it should be possible to skip this step if we instead generate
    # indices to x1 and x2 that correspond to the broadcasted hh.shapes. This
    # would avoid the dependence in this test on broadcast_to().
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Second, manually promote the dtypes. This is important. If the internal
    # type promotion in ah.equal() is wrong, it will not be directly visible in
    # the output type, but it can lead to wrong answers. For example,
    # ah.equal(array(1.0, dtype=xp.float32), array(1.00000001, dtype=xp.float64)) will
    # be wrong if the float64 is downcast to float32. # be wrong if the
    # xp.float64 is downcast to float32. See the comment on
    # test_elementwise_function_two_arg_bool_type_promotion() in
    # test_type_promotion.py. The type promotion for ah.equal() is not *really*
    # tested in that file, because doing so requires doing the consistency
    # check we do here rather than st.just checking the result dtype.
    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) == scalar_func(x2idx))

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_exp(x):
    a = xp.exp(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, ZERO, INFINITY)
    # exp maps [-inf, inf] to [0, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_expm1(x):
    a = xp.expm1(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, NEGONE, INFINITY)
    # expm1 maps [-inf, inf] to [1, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_floor(x):
    # This test is almost identical to test_ceil
    a = xp.floor(x)
    finite = ah.isfinite(x)
    ah.assert_integral(a[finite])
    assert ah.all(ah.less_equal(a[finite], x[finite]))
    assert ah.all(ah.less_equal(x[finite] - a[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(a[integers], x[integers])

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_floor_divide(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    if ah.is_integer_dtype(x1.dtype):
        # The spec does not specify the behavior for division by 0 for integer
        # dtypes. A library may choose to raise an exception in this case, so
        # we avoid passing it in entirely.
        assume(not xp.any(x1 == 0) and not xp.any(x2 == 0))
        div = xp.divide(
            ah.asarray(x1, dtype=xp.float64),
            ah.asarray(x2, dtype=xp.float64),
        )
    else:
        div = xp.divide(x1, x2)

    out = xp.floor_divide(x1, x2)

    # TODO: The spec doesn't clearly specify the behavior of floor_divide on
    # infinities. See https://github.com/data-apis/array-api/issues/199.
    finite = ah.isfinite(div)
    ah.assert_integral(out[finite])

    # TODO: Test the exact output for floor_divide.

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_greater(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.greater(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) > scalar_func(x2idx))

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_greater_equal(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.greater_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) >= scalar_func(x2idx))

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_isfinite(x):
    a = ah.isfinite(x)
    TRUE = ah.true(x.shape)
    if ah.is_integer_dtype(x.dtype):
        ah.assert_exactly_equal(a, TRUE)
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = ah.logical_or(xp.isinf(x), ah.isnan(x))
    ah.assert_exactly_equal(a, ah.logical_not(inf))

    # Test the exact value by comparing to the math version
    if ah.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isfinite(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_isinf(x):
    a = xp.isinf(x)
    FALSE = ah.false(x.shape)
    if ah.is_integer_dtype(x.dtype):
        ah.assert_exactly_equal(a, FALSE)
    finite_or_nan = ah.logical_or(ah.isfinite(x), ah.isnan(x))
    ah.assert_exactly_equal(a, ah.logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if ah.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isinf(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_isnan(x):
    a = ah.isnan(x)
    FALSE = ah.false(x.shape)
    if ah.is_integer_dtype(x.dtype):
        ah.assert_exactly_equal(a, FALSE)
    finite_or_inf = ah.logical_or(ah.isfinite(x), xp.isinf(x))
    ah.assert_exactly_equal(a, ah.logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if ah.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isnan(s)

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_less(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = ah.less(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) < scalar_func(x2idx))

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_less_equal(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = ah.less_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) <= scalar_func(x2idx))

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_log(x):
    a = xp.log(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_log1p(x):
    a = xp.log1p(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    codomain = ah.inrange(x, NEGONE, INFINITY)
    domain = ah.inrange(a, -INFINITY, INFINITY)
    # log1p maps [1, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_log2(x):
    a = xp.log2(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log2 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes))
def test_log10(x):
    a = xp.log10(x)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log10 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(hh.two_mutual_arrays(hh.floating_dtype_objects))
def test_logaddexp(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    xp.logaddexp(x1, x2)
    # The spec doesn't require any behavior for this function. We could test
    # that this is indeed an approximation of log(exp(x1) + exp(x2)), but we
    # don't have tests for this sort of thing for any functions yet.

@given(hh.two_mutual_arrays([xp.bool]))
def test_logical_and(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = ah.logical_and(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) and bool(_x2[idx]))

@given(xps.arrays(dtype=xp.bool, shape=hh.shapes))
def test_logical_not(x):
    a = ah.logical_not(x)

    for idx in ah.ndindex(x.shape):
        assert a[idx] == (not bool(x[idx]))

@given(hh.two_mutual_arrays([xp.bool]))
def test_logical_or(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = ah.logical_or(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) or bool(_x2[idx]))

@given(hh.two_mutual_arrays([xp.bool]))
def test_logical_xor(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.logical_xor(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) ^ bool(_x2[idx]))

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_multiply(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.multiply(x1, x2)

    b = xp.multiply(x2, x1)
    # multiply is commutative
    ah.assert_exactly_equal(a, b)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_negative(x):
    out = ah.negative(x)

    # Negation is an involution
    ah.assert_exactly_equal(x, ah.negative(out))

    mask = ah.isfinite(x)
    if ah.is_integer_dtype(x.dtype):
        minval = ah.dtype_ranges[x.dtype][0]
        if minval < 0:
            # negative of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))

    # Additive inverse
    y = xp.add(x[mask], out[mask])
    ZERO = ah.zero(x[mask].shape, x.dtype)
    ah.assert_exactly_equal(y, ZERO)


@given(hh.two_mutual_arrays())
def test_not_equal(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    a = xp.not_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = ah.promote_dtypes(x1.dtype, x2.dtype)
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if ah.is_integer_dtype(promoted_dtype):
        scalar_func = int
    elif ah.is_float_dtype(promoted_dtype):
        scalar_func = float
    else:
        scalar_func = bool
    for idx in ah.ndindex(shape):
        aidx = a[idx]
        x1idx = _x1[idx]
        x2idx = _x2[idx]
        # Sanity check
        assert aidx.shape == x1idx.shape == x2idx.shape
        assert bool(aidx) == (scalar_func(x1idx) != scalar_func(x2idx))


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_positive(x):
    out = xp.positive(x)
    # Positive does nothing
    ah.assert_exactly_equal(out, x)

@given(hh.two_mutual_arrays(hh.floating_dtype_objects))
def test_pow(x1_and_x2):
    x1, x2 = x1_and_x2
    sanity_check(x1, x2)
    xp.pow(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.

@given(hh.two_mutual_arrays(hh.numeric_dtype_objects))
def test_remainder(x1_and_x2):
    x1, x2 = x1_and_x2
    assume(len(x1.shape) <= len(x2.shape)) # TODO: rework same sign testing below to remove this
    sanity_check(x1, x2)
    out = xp.remainder(x1, x2)

    # out and x2 should have the same sign.
    # ah.assert_same_sign returns False for nans
    not_nan = ah.logical_not(ah.logical_or(ah.isnan(out), ah.isnan(x2)))
    ah.assert_same_sign(out[not_nan], x2[not_nan])

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes))
def test_round(x):
    a = xp.round(x)

    # Test that the result is integral
    finite = ah.isfinite(x)
    ah.assert_integral(a[finite])

    # round(x) should be the nearest integer to x. The case where there is a
    # tie (round to even) is already handled by the special cases tests.

    # This is the same strategy used in the mask in the
    # test_round_special_cases_one_arg_two_integers_equally_close special
    # cases test.
    floor = xp.floor(x)
    ceil = xp.ceil(x)
    over = xp.subtract(x, floor)
    under = xp.subtract(ceil, x)
    round_down = ah.less(over, under)
    round_up = ah.less(under, over)
    ah.assert_exactly_equal(a[round_down], floor[round_down])
    ah.assert_exactly_equal(a[round_up], ceil[round_up])

@given(numeric_scalars)
def test_sign(x):
    # a = xp.sign(x)
    pass

@given(floating_scalars)
def test_sin(x):
    # a = xp.sin(x)
    pass

@given(floating_scalars)
def test_sinh(x):
    # a = xp.sinh(x)
    pass

@given(numeric_scalars)
def test_square(x):
    # a = xp.square(x)
    pass

@given(floating_scalars)
def test_sqrt(x):
    # a = xp.sqrt(x)
    pass

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_subtract(args):
    x1, x2 = args
    sanity_check(x1, x2)
    # a = xp.subtract(x1, x2)

@given(floating_scalars)
def test_tan(x):
    # a = xp.tan(x)
    pass

@given(floating_scalars)
def test_tanh(x):
    # a = xp.tanh(x)
    pass

@given(xps.arrays(dtype=hh.numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    out = xp.trunc(x)
    assert out.dtype == x.dtype, f"{x.dtype=!s} but {out.dtype=!s}"
    assert out.shape == x.shape, f"{x.shape} but {out.shape}"
    if x.dtype in hh.integer_dtype_objects:
        assert ah.all(ah.equal(x, out)), f"{x=!s} but {out=!s}"
    else:
        finite_mask = ah.isfinite(out)
        for idx in ah.ndindex(out.shape):
            if finite_mask[idx]:
                assert float(out[idx]).is_integer(), f"x at {idx=} is {x[idx]}, but out at idx is {out[idx]}"
