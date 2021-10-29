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
import pytest

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
# We might as well use this implementation rather than requiring
# mod.broadcast_shapes(). See test_equal() and others.
from .test_broadcasting import broadcast_shapes


@pytest.mark.parametrize("expr", ["xp.abs(x)", "abs(x)"])
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_abs(expr, x):
    if x.dtype in dh.int_dtypes:
        # abs of the smallest representable negative integer is not defined
        m, _ = dh.dtype_ranges[x.dtype]
        mask = xp.not_equal(x, ah.full(x.shape, m, dtype=x.dtype))
        x = x[mask]
    out = eval(expr)
    ph.assert_shape("abs", out.shape, x.shape)
    assert ah.all(ah.logical_not(ah.negative_mathematical_sign(out))), "abs(x) did not have positive sign"
    less_zero = ah.negative_mathematical_sign(x)
    negx = ah.negative(x)
    # abs(x) = -x for x < 0
    ah.assert_exactly_equal(out[less_zero], negx[less_zero])
    # abs(x) = x for x >= 0
    ah.assert_exactly_equal(out[ah.logical_not(less_zero)], x[ah.logical_not(less_zero)])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acos(x):
    out = xp.acos(x)
    ph.assert_shape("acos", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    # Here (and elsewhere), should technically be out.dtype, but this is the
    # same as x.dtype, as tested by the type_promotion tests.
    PI = ah.π(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(out, ZERO, PI)
    # acos maps [-1, 1] to [0, pi]. Values outside this domain are mapped to
    # nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acosh(x):
    out = xp.acosh(x)
    ph.assert_shape("acosh", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ONE, INFINITY)
    codomain = ah.inrange(out, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@pytest.mark.parametrize("expr", ["xp.add(x1, x2)", "x1 + x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_add(expr, x1, x2):
    out = eval(expr)
    # add is commutative
    expected = xp.add(x2, x1)
    ah.assert_exactly_equal(out, expected)
    # TODO: Test that add is actually addition

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asin(x):
    out = xp.asin(x)
    ph.assert_shape("asin", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(out, -PI/2, PI/2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asinh(x):
    out = xp.asinh(x)
    ph.assert_shape("asinh", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # asinh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atan(x):
    out = xp.atan(x)
    ph.assert_shape("atan", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, -PI/2, PI/2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_atan2(x1, x2):
    out = xp.atan2(x1, x2)
    INFINITY1 = ah.infinity(x1.shape, x1.dtype)
    INFINITY2 = ah.infinity(x2.shape, x2.dtype)
    PI = ah.π(out.shape, out.dtype)
    domainx1 = ah.inrange(x1, -INFINITY1, INFINITY1)
    domainx2 = ah.inrange(x2, -INFINITY2, INFINITY2)
    # codomain = ah.inrange(out, -PI, PI, 1e-5)
    codomain = ah.inrange(out, -PI, PI)
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
    pos_x1 = ah.positive_mathematical_sign(x1)
    neg_x1 = ah.negative_mathematical_sign(x1)
    pos_x2 = ah.positive_mathematical_sign(x2)
    neg_x2 = ah.negative_mathematical_sign(x2)
    pos_out = ah.positive_mathematical_sign(out)
    neg_out = ah.negative_mathematical_sign(out)
    ah.assert_exactly_equal(ah.logical_or(ah.logical_and(pos_x1, pos_x2),
                                          ah.logical_and(pos_x1, neg_x2)), pos_out)
    ah.assert_exactly_equal(ah.logical_or(ah.logical_and(neg_x1, pos_x2),
                                    ah.logical_and(neg_x1, neg_x2)), neg_out)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atanh(x):
    out = xp.atanh(x)
    ph.assert_shape("atanh", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@pytest.mark.parametrize("expr", ["xp.bitwise_and(x1, x2)", "x1 & x2"])
@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_and(expr, x1, x2):
    out = eval(expr)

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

@pytest.mark.parametrize("expr", ["xp.bitwise_left_shift(x1, x2)", "x1 << x2"])
@given(*hh.two_mutual_arrays(dh.all_int_dtypes))
def test_bitwise_left_shift(expr, x1, x2):
    assume(not ah.any(ah.isnegative(x2)))
    out = eval(expr)

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

@pytest.mark.parametrize("expr", ["xp.bitwise_invert(x)", "~x"])
@given(xps.arrays(dtype=hh.integer_or_boolean_dtypes, shape=hh.shapes()))
def test_bitwise_invert(expr, x):
    out = eval(expr)
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

@pytest.mark.parametrize("expr", ["xp.bitwise_or(x1, x2)", "x1 | x2"])
@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_or(expr, x1, x2):
    out = eval(expr)

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

@pytest.mark.parametrize("expr", ["xp.bitwise_right_shift(x1, x2)", "x1 >> x2"])
@given(*hh.two_mutual_arrays(dh.all_int_dtypes))
def test_bitwise_right_shift(expr, x1, x2):
    assume(not ah.any(ah.isnegative(x2)))
    out = eval(expr)

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

@pytest.mark.parametrize("expr", ["xp.bitwise_xor(x1, x2)", "x1 ^ x2"])
@given(*hh.two_mutual_arrays(dh.bool_and_all_int_dtypes))
def test_bitwise_xor(expr, x1, x2):
    out = eval(expr)

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
    out = xp.ceil(x)
    ph.assert_shape("ceil", out.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])
    assert ah.all(ah.less_equal(x[finite], out[finite]))
    assert ah.all(ah.less_equal(out[finite] - x[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(out[integers], x[integers])

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cos(x):
    out = xp.cos(x)
    ph.assert_shape("cos", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY, open=True)
    codomain = ah.inrange(out, -ONE, ONE)
    # cos maps (-inf, inf) to [-1, 1]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cosh(x):
    out = xp.cosh(x)
    ph.assert_shape("cosh", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@pytest.mark.parametrize("expr", ["xp.divide(x1, x2)", "x1 / x2"])
@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_divide(expr, x1, x2):
    eval(expr)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@pytest.mark.parametrize("expr", ["xp.equal(x1, x2)", "x1 == x2"])
@given(*hh.two_mutual_arrays())
def test_equal(expr, x1, x2):
    out = eval(expr)
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
    ph.assert_shape("equal", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        # Sanity check
        assert out_idx.shape == x1_idx.shape == x2_idx.shape
        assert bool(out_idx) == (scalar_func(x1_idx) == scalar_func(x2_idx))

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_exp(x):
    out = xp.exp(x)
    ph.assert_shape("exp", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, ZERO, INFINITY)
    # exp maps [-inf, inf] to [0, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_expm1(x):
    out = xp.expm1(x)
    ph.assert_shape("expm1", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, NEGONE, INFINITY)
    # expm1 maps [-inf, inf] to [1, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_floor(x):
    # This test is almost identical to test_ceil
    out = xp.floor(x)
    ph.assert_shape("floor", out.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])
    assert ah.all(ah.less_equal(out[finite], x[finite]))
    assert ah.all(ah.less_equal(x[finite] - out[finite], ah.one(x[finite].shape, x.dtype)))
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(out[integers], x[integers])

@pytest.mark.parametrize("expr", ["xp.floor_divide(x1, x2)", "x1 // x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_floor_divide(expr, x1, x2):
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

    out = eval(expr)

    # TODO: The spec doesn't clearly specify the behavior of floor_divide on
    # infinities. See https://github.com/data-apis/array-api/issues/199.
    finite = ah.isfinite(div)
    ah.assert_integral(out[finite])

    # TODO: Test the exact output for floor_divide.

@pytest.mark.parametrize("expr", ["xp.greater(x1, x2)", "x1 > x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_greater(expr, x1, x2):
    out = eval(expr)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("greater", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
        assert bool(out_idx) == (scalar_func(x1_idx) > scalar_func(x2_idx))

@pytest.mark.parametrize("expr", ["xp.greater_equal(x1, x2)", "x1 >= x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_greater_equal(expr, x1, x2):
    out = eval(expr)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("greater_equal", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
        assert bool(out_idx) == (scalar_func(x1_idx) >= scalar_func(x2_idx))

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isfinite(x):
    out = ah.isfinite(x)
    ph.assert_shape("isfinite", out.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.true(x.shape))
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = ah.logical_or(xp.isinf(x), ah.isnan(x))
    ah.assert_exactly_equal(out, ah.logical_not(inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isfinite(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isinf(x):
    out = xp.isinf(x)

    ph.assert_shape("isinf", out.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.false(x.shape))
    finite_or_nan = ah.logical_or(ah.isfinite(x), ah.isnan(x))
    ah.assert_exactly_equal(out, ah.logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isinf(s)

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isnan(x):
    out = ah.isnan(x)

    ph.assert_shape("isnan", out.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.false(x.shape))
    finite_or_inf = ah.logical_or(ah.isfinite(x), xp.isinf(x))
    ah.assert_exactly_equal(out, ah.logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isnan(s)

@pytest.mark.parametrize("expr", ["xp.less(x1, x2)", "x1 < x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_less(expr, x1, x2):
    out = eval(expr)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("less", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        # Sanity check
        assert out_idx.shape == x1_idx.shape == x2_idx.shape
        assert bool(out_idx) == (scalar_func(x1_idx) < scalar_func(x2_idx))

@pytest.mark.parametrize("expr", ["xp.less_equal(x1, x2)", "x1 <= x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_less_equal(expr, x1, x2):
    out = eval(expr)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("less_equal", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
        assert bool(out_idx) == (scalar_func(x1_idx) <= scalar_func(x2_idx))

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log(x):
    out = xp.log(x)

    ph.assert_shape("log", out.shape, x.shape)

    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # log maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log1p(x):
    out = xp.log1p(x)
    ph.assert_shape("log1p", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    codomain = ah.inrange(x, NEGONE, INFINITY)
    domain = ah.inrange(out, -INFINITY, INFINITY)
    # log1p maps [1, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log2(x):
    out = xp.log2(x)
    ph.assert_shape("log2", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # log2 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)

@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log10(x):
    out = xp.log10(x)
    ph.assert_shape("log10", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
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
    out = ah.logical_and(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_and", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert out[idx] == (bool(_x1[idx]) and bool(_x2[idx]))

@given(xps.arrays(dtype=xp.bool, shape=hh.shapes()))
def test_logical_not(x):
    out = ah.logical_not(x)
    ph.assert_shape("logical_not", out.shape, x.shape)
    for idx in ah.ndindex(x.shape):
        assert out[idx] == (not bool(x[idx]))

@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_or(x1, x2):
    out = ah.logical_or(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_or", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert out[idx] == (bool(_x1[idx]) or bool(_x2[idx]))

@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_xor(x1, x2):
    out = xp.logical_xor(x1, x2)

    # See the comments in test_equal
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_xor", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)

    for idx in ah.ndindex(shape):
        assert out[idx] == (bool(_x1[idx]) ^ bool(_x2[idx]))

@pytest.mark.parametrize("expr", ["xp.multiply(x1, x2)", "x1 * x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_multiply(expr, x1, x2):
    out = eval(expr)

    expected = xp.multiply(x2, x1)
    # multiply is commutative
    ah.assert_exactly_equal(out, expected)

@pytest.mark.parametrize("expr", ["xp.negative(x)", "-x"])
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_negative(expr, x):
    out = eval(expr)

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


@pytest.mark.parametrize("expr", ["xp.not_equal(x1, x2)", "x1 != x2"])
@given(*hh.two_mutual_arrays())
def test_not_equal(expr, x1, x2):
    out = eval(expr)

    # See the comments in test_equal() for a description of how this test
    # works.
    shape = broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("not_equal", out.shape, shape)
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
        out_idx = out[idx]
        x1_idx = _x1[idx]
        x2_idx = _x2[idx]
        assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
        assert bool(out_idx) == (scalar_func(x1_idx) != scalar_func(x2_idx))


@pytest.mark.parametrize("expr", ["xp.positive(x)", "+x"])
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_positive(expr, x):
    out = eval(expr)
    ph.assert_shape("positive", out.shape, x.shape)
    # Positive does nothing
    ah.assert_exactly_equal(out, x)

@pytest.mark.parametrize("expr", ["xp.pow(x1, x2)", "x1 ** x2"])
@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_pow(expr, x1, x2):
    eval(expr)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.

@pytest.mark.parametrize("expr", ["xp.remainder(x1, x2)", "x1 % x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_remainder(expr, x1, x2):
    # TODO: rework same sign testing below to remove this
    assume(len(x1.shape) <= len(x2.shape))
    out = eval(expr)

    # out and x2 should have the same sign.
    # ah.assert_same_sign returns False for nans
    not_nan = ah.logical_not(ah.logical_or(ah.isnan(out), ah.isnan(x2)))
    ah.assert_same_sign(out[not_nan], x2[not_nan])

@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_round(x):
    out = xp.round(x)

    ph.assert_shape("round", out.shape, x.shape)

    # Test that the res is integral
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])

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
    ah.assert_exactly_equal(out[round_down], floor[round_down])
    ah.assert_exactly_equal(out[round_up], ceil[round_up])

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

@pytest.mark.parametrize("expr", ["xp.subtract(x1, x2)", "x1 - x2"])
@given(*hh.two_mutual_arrays(dh.numeric_dtypes))
def test_subtract(expr, x1, x2):
    eval(expr)
    # TODO

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
