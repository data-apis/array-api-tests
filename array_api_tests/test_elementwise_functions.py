"""
Tests for elementwise functions

https://data-apis.github.io/array-api/latest/API_specification/elementwise_functions.html

This tests behavior that is explicitly mentioned in the spec. Note that the
spec does not make any accuracy requirements for functions, so this does not
test that. Tests for the special cases are generated and tested separately in
special_cases/
"""

import math

from hypothesis import assume, given

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
# We might as well use this implementation rather than requiring
# mod.broadcast_shapes(). See test_equal() and others.
from .test_broadcasting import broadcast_shapes


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_abs(x):
    if dh.is_int_dtype(x.dtype):
        minval = dh.dtype_ranges[x.dtype][0]
        if minval < 0:
            # abs of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))
            x = x[mask]
    a = xp.abs(x)
    ph.assert_shape("abs", a.shape, x.shape)
    assert ah.all(ah.logical_not(ah.negative_mathematical_sign(a))), "abs(x) did not have positive sign"
    less_zero = ah.negative_mathematical_sign(x)
    negx = ah.negative(x)
    # abs(x) = -x for x < 0
    ah.assert_exactly_equal(a[less_zero], negx[less_zero])
    # abs(x) = x for x >= 0
    ah.assert_exactly_equal(a[ah.logical_not(less_zero)], x[ah.logical_not(less_zero)])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acos(x):
    a = xp.acos(x)
    ph.assert_shape("acos", a.shape, x.shape)
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

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acosh(x):
    a = xp.acosh(x)
    ph.assert_shape("acosh", a.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ONE, INFINITY)
    codomain = ah.inrange(a, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_add(x1, x2):
    a = xp.add(x1, x2)

    b = xp.add(x2, x1)
    # add is commutative
    ah.assert_exactly_equal(a, b)
    # TODO: Test that add is actually addition

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asin(x):
    a = xp.asin(x)
    ph.assert_shape("asin", a.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(a, -PI/2, PI/2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asinh(x):
    a = xp.asinh(x)
    ph.assert_shape("asinh", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # asinh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atan(x):
    a = xp.atan(x)
    ph.assert_shape("atan", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -PI/2, PI/2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_atan2(x1, x2):
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
    # each element-wise res. The quadrant (i.e., branch) is chosen such
    # that each element-wise res is the signed angle in radians between the
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

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atanh(x):
    a = xp.atanh(x)
    ph.assert_shape("atanh", a.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_and(x1, x2):
    out = xp.bitwise_and(x1, x2)

    # TODO: generate indices without broadcasting arrays (see test_equal comment)
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("bitwise_and", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Compare against the Python & operator.
    if out.dtype == xp.bool:
        for idx in ah.ndindex(out.shape):
            val1 = bool(_x1[idx])
            val2 = bool(_x2[idx])
            res = bool(out[idx])
            assert (val1 and val2) == res
    else:
        for idx in ah.ndindex(out.shape):
            val1 = int(_x1[idx])
            val2 = int(_x2[idx])
            res = int(out[idx])
            vals_and = val1 & val2
            vals_and = ah.int_to_dtype(vals_and, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
            assert vals_and == res

@given(*hh.two_mutual_arrays(dh.all_int_dtypes))
def test_bitwise_left_shift(x1, x2):
    assume(not ah.any(ah.isnegative(x2)))
    out = xp.bitwise_left_shift(x1, x2)

    # TODO: generate indices without broadcasting arrays (see test_equal comment)
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("bitwise_left_shift", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Compare against the Python << operator.
    for idx in ah.ndindex(out.shape):
        val1 = int(_x1[idx])
        val2 = int(_x2[idx])
        res = int(out[idx])
        # We avoid shifting very large ints
        vals_shift = val1 << val2 if val2 < dh.dtype_nbits[out.dtype] else 0
        vals_shift = ah.int_to_dtype(vals_shift, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
        assert vals_shift == res

@given(xps.arrays(dtype=hh.integer_or_boolean_dtypes, shape=hh.shapes()))
def test_bitwise_invert(x):
    out = xp.bitwise_invert(x)
    ph.assert_shape("bitwise_invert", out.shape, x.shape)
    # Compare against the Python ~ operator.
    if out.dtype == xp.bool:
        for idx in ah.ndindex(out.shape):
            val = bool(x[idx])
            res = bool(out[idx])
            assert (not val) == res
    else:
        for idx in ah.ndindex(out.shape):
            val = int(x[idx])
            res = int(out[idx])
            val_invert = ~val
            val_invert = ah.int_to_dtype(val_invert, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
            assert val_invert == res

@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_or(x1, x2):
    out = xp.bitwise_or(x1, x2)

    # TODO: generate indices without broadcasting arrays (see test_equal comment)
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("bitwise_or", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Compare against the Python | operator.
    if out.dtype == xp.bool:
        for idx in ah.ndindex(out.shape):
            val1 = bool(_x1[idx])
            val2 = bool(_x2[idx])
            res = bool(out[idx])
            assert (val1 or val2) == res
    else:
        for idx in ah.ndindex(out.shape):
            val1 = int(_x1[idx])
            val2 = int(_x2[idx])
            res = int(out[idx])
            vals_or = val1 | val2
            vals_or = ah.int_to_dtype(vals_or, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
            assert vals_or == res

@given(*hh.two_mutual_arrays(dh.all_int_dtypes))
def test_bitwise_right_shift(x1, x2):
    assume(not ah.any(ah.isnegative(x2)))
    out = xp.bitwise_right_shift(x1, x2)

    # TODO: generate indices without broadcasting arrays (see test_equal comment)
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("bitwise_right_shift", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Compare against the Python >> operator.
    for idx in ah.ndindex(out.shape):
        val1 = int(_x1[idx])
        val2 = int(_x2[idx])
        res = int(out[idx])
        vals_shift = val1 >> val2
        vals_shift = ah.int_to_dtype(vals_shift, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
        assert vals_shift == res

@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_xor(x1, x2):
    out = xp.bitwise_xor(x1, x2)

    # TODO: generate indices without broadcasting arrays (see test_equal comment)
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("bitwise_xor", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    # Compare against the Python ^ operator.
    if out.dtype == xp.bool:
        for idx in ah.ndindex(out.shape):
            val1 = bool(_x1[idx])
            val2 = bool(_x2[idx])
            res = bool(out[idx])
            assert (val1 ^ val2) == res
    else:
        for idx in ah.ndindex(out.shape):
            val1 = int(_x1[idx])
            val2 = int(_x2[idx])
            res = int(out[idx])
            vals_xor = val1 ^ val2
            vals_xor = ah.int_to_dtype(vals_xor, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype])
            assert vals_xor == res

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_ceil(x):
    # This test is almost identical to test_floor()
    a = xp.ceil(x)
    ph.assert_shape("ceil", a.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(a[finite])
    assert ah.all(ah.less_equal(x[finite], a[finite]))
    assert ah.all(ah.less_equal(a[finite] - x[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(a[integers], x[integers])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cos(x):
    a = xp.cos(x)
    ph.assert_shape("cos", a.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY, open=True)
    codomain = ah.inrange(a, -ONE, ONE)
    # cos maps (-inf, inf) to [-1, 1]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cosh(x):
    a = xp.cosh(x)
    ph.assert_shape("cosh", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_divide(x1, x2):
    xp.divide(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@given(*hh.two_mutual_arrays())
def test_equal(x1, x2):
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
    # indices to x1 and x2 that correspond to the broadcasted shapes. This
    # would avoid the dependence in this test on broadcast_to().
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("equal", a.shape, shape)
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

    # check we do here rather than just checking the result dtype.
    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_exp(x):
    a = xp.exp(x)
    ph.assert_shape("exp", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, ZERO, INFINITY)
    # exp maps [-inf, inf] to [0, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_expm1(x):
    a = xp.expm1(x)
    ph.assert_shape("expm1", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(a, NEGONE, INFINITY)
    # expm1 maps [-inf, inf] to [1, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_floor(x):
    # This test is almost identical to test_ceil
    a = xp.floor(x)
    ph.assert_shape("floor", a.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(a[finite])
    assert ah.all(ah.less_equal(a[finite], x[finite]))
    assert ah.all(ah.less_equal(x[finite] - a[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(a[integers], x[integers])

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_floor_divide(x1, x2):
    if dh.is_int_dtype(x1.dtype):
        # The spec does not specify the behavior for division by 0 for integer
        # dtypes. A library may choose to raise an exception in this case, so
        # we avoid passing it in entirely.
        assume(not ah.any(x1 == 0) and not ah.any(x2 == 0))
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

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_greater(x1, x2):
    a = xp.greater(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("greater", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_greater_equal(x1, x2):
    a = xp.greater_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("greater_equal", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isfinite(x):
    a = ah.isfinite(x)
    ph.assert_shape("isfinite", a.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(a, ah.true(x.shape))
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = ah.logical_or(xp.isinf(x), ah.isnan(x))
    ah.assert_exactly_equal(a, ah.logical_not(inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isfinite(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isinf(x):
    a = xp.isinf(x)

    ph.assert_shape("isinf", a.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(a, ah.false(x.shape))
    finite_or_nan = ah.logical_or(ah.isfinite(x), ah.isnan(x))
    ah.assert_exactly_equal(a, ah.logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isinf(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isnan(x):
    a = ah.isnan(x)

    ph.assert_shape("isnan", a.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(a, ah.false(x.shape))
    finite_or_inf = ah.logical_or(ah.isfinite(x), xp.isinf(x))
    ah.assert_exactly_equal(a, ah.logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(a[idx]) == math.isnan(s)

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_less(x1, x2):
    a = ah.less(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("less", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_less_equal(x1, x2):
    a = ah.less_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("less_equal", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log(x):
    a = xp.log(x)

    ph.assert_shape("log", a.shape, x.shape)

    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log1p(x):
    a = xp.log1p(x)
    ph.assert_shape("log1p", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    codomain = ah.inrange(x, NEGONE, INFINITY)
    domain = ah.inrange(a, -INFINITY, INFINITY)
    # log1p maps [1, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log2(x):
    a = xp.log2(x)
    ph.assert_shape("log2", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log2 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log10(x):
    a = xp.log10(x)
    ph.assert_shape("log10", a.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(a, -INFINITY, INFINITY)
    # log10 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_logaddexp(x1, x2):
    xp.logaddexp(x1, x2)
    # The spec doesn't require any behavior for this function. We could test
    # that this is indeed an approximation of log(exp(x1) + exp(x2)), but we
    # don't have tests for this sort of thing for any functions yet.

@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_and(x1, x2):
    a = ah.logical_and(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_and", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) and bool(_x2[idx]))

@given(xps.arrays(dtype=xp.bool, shape=hh.shapes()))
def test_logical_not(x):
    a = ah.logical_not(x)
    ph.assert_shape("logical_not", a.shape, x.shape)
    for idx in ah.ndindex(x.shape):
        assert a[idx] == (not bool(x[idx]))

@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_or(x1, x2):
    a = ah.logical_or(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_or", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) or bool(_x2[idx]))

@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_xor(x1, x2):
    a = xp.logical_xor(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_xor", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert a[idx] == (bool(_x1[idx]) ^ bool(_x2[idx]))

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_multiply(x1, x2):
    a = xp.multiply(x1, x2)

    b = xp.multiply(x2, x1)
    # multiply is commutative
    ah.assert_exactly_equal(a, b)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_negative(x):
    out = ah.negative(x)

    ph.assert_shape("negative", out.shape, x.shape)

    # Negation is an involution
    ah.assert_exactly_equal(x, ah.negative(out))

    mask = ah.isfinite(x)
    if dh.is_int_dtype(x.dtype):
        minval = dh.dtype_ranges[x.dtype][0]
        if minval < 0:
            # negative of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))

    # Additive inverse
    y = xp.add(x[mask], out[mask])
    ZERO = ah.zero(x[mask].shape, x.dtype)
    ah.assert_exactly_equal(y, ZERO)


@given(*hh.two_mutual_arrays())
def test_not_equal(x1, x2):
    a = xp.not_equal(x1, x2)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("not_equal", a.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    promoted_dtype = dh.promotion_table[x1.dtype, x2.dtype]
    _x1 = ah.asarray(_x1, dtype=promoted_dtype)
    _x2 = ah.asarray(_x2, dtype=promoted_dtype)

    if dh.is_int_dtype(promoted_dtype):
        scalar_func = int
    elif dh.is_float_dtype(promoted_dtype):
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


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_positive(x):
    out = xp.positive(x)
    ph.assert_shape("positive", out.shape, x.shape)
    # Positive does nothing
    ah.assert_exactly_equal(out, x)

@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_pow(x1, x2):
    xp.pow(x1, x2)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_remainder(x1, x2):
    assume(len(x1.shape) <= len(x2.shape)) # TODO: rework same sign testing below to remove this
    out = xp.remainder(x1, x2)

    # out and x2 should have the same sign.
    # ah.assert_same_sign returns False for nans
    not_nan = ah.logical_not(ah.logical_or(ah.isnan(out), ah.isnan(x2)))
    ah.assert_same_sign(out[not_nan], x2[not_nan])

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_round(x):
    a = xp.round(x)

    ph.assert_shape("round", a.shape, x.shape)

    # Test that the res is integral
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

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_sign(x):
    out = xp.sign(x)
    ph.assert_shape("sign", out.shape, x.shape)
    # TODO

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sin(x):
    out = xp.sin(x)
    ph.assert_shape("sin", out.shape, x.shape)
    # TODO

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sinh(x):
    out = xp.sinh(x)
    ph.assert_shape("sinh", out.shape, x.shape)
    # TODO

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_square(x):
    out = xp.square(x)
    ph.assert_shape("square", out.shape, x.shape)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sqrt(x):
    out = xp.sqrt(x)
    ph.assert_shape("sqrt", out.shape, x.shape)

@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_subtract(x1, x2):
    # out = xp.subtract(x1, x2)
    pass

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tan(x):
    out = xp.tan(x)
    ph.assert_shape("tan", out.shape, x.shape)
    # TODO

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tanh(x):
    out = xp.tanh(x)
    ph.assert_shape("tanh", out.shape, x.shape)
    # TODO

@given(xps.arrays(dtype=hh.numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    out = xp.trunc(x)
    ph.assert_shape("bitwise_trunc", out.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, x)
    else:
        finite = ah.isfinite(x)
        ah.assert_integral(out[finite])
