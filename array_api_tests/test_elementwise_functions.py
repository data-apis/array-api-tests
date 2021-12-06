"""
Tests for elementwise functions

https://data-apis.github.io/array-api/latest/API_specification/elementwise_functions.html

This tests behavior that is explicitly mentioned in the spec. Note that the
spec does not make any accuracy requirements for functions, so this does not
test that. Tests for the special cases are generated and tested separately in
special_cases/
"""

import math
from enum import Enum, auto
from typing import Callable, List, Sequence, Union

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.control import reject

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .algos import broadcast_shapes
from .typing import Array, DataType, Param, Scalar

# When appropiate, this module tests operators alongside their respective
# elementwise methods. We do this by parametrizing a generalised test method
# with every relevant method and operator.
#
# Notable arguments in the parameter:
# - The function object, which for operator test cases is a wrapper that allows
#   test logic to be generalised.
# - The argument strategies, which can be used to draw arguments for the test
#   case. They may require additional filtering for certain test cases.
# - right_is_scalar (binary parameters), which denotes if the right argument is
#   a scalar in a test case. This can be used to appropiately adjust draw
#   filtering and test logic.


func_to_op = {v: k for k, v in dh.op_to_func.items()}
all_op_to_symbol = {**dh.binary_op_to_symbol, **dh.inplace_op_to_symbol}
finite_kw = {"allow_nan": False, "allow_infinity": False}

unary_argnames = ("func_name", "func", "strat")
UnaryParam = Param[str, Callable[[Array], Array], st.SearchStrategy[Array]]


def make_unary_params(
    elwise_func_name: str, dtypes: Sequence[DataType]
) -> List[UnaryParam]:
    strat = xps.arrays(dtype=st.sampled_from(dtypes), shape=hh.shapes())
    func = getattr(xp, elwise_func_name)
    op_name = func_to_op[elwise_func_name]
    op = lambda x: getattr(x, op_name)()
    return [
        pytest.param(elwise_func_name, func, strat, id=elwise_func_name),
        pytest.param(op_name, op, strat, id=op_name),
    ]


binary_argnames = (
    "func_name",
    "func",
    "left_sym",
    "left_strat",
    "right_sym",
    "right_strat",
    "right_is_scalar",
    "res_name",
)
BinaryParam = Param[
    str,
    Callable[[Array, Union[Scalar, Array]], Array],
    str,
    st.SearchStrategy[Array],
    str,
    st.SearchStrategy[Union[Scalar, Array]],
    bool,
]


class FuncType(Enum):
    FUNC = auto()
    OP = auto()
    IOP = auto()


def make_binary_params(
    elwise_func_name: str, dtypes: Sequence[DataType]
) -> List[BinaryParam]:
    dtypes_strat = st.sampled_from(dtypes)

    def make_param(
        func_name: str, func_type: FuncType, right_is_scalar: bool
    ) -> BinaryParam:
        if right_is_scalar:
            left_sym = "x"
            right_sym = "s"
        else:
            left_sym = "x1"
            right_sym = "x2"

        shared_dtypes = st.shared(dtypes_strat)
        if right_is_scalar:
            left_strat = xps.arrays(dtype=shared_dtypes, shape=hh.shapes())
            right_strat = shared_dtypes.flatmap(
                lambda d: xps.from_dtype(d, **finite_kw)
            )
        else:
            if func_type is FuncType.IOP:
                shared_shapes = st.shared(hh.shapes())
                left_strat = xps.arrays(dtype=shared_dtypes, shape=shared_shapes)
                right_strat = xps.arrays(dtype=shared_dtypes, shape=shared_shapes)
            else:
                left_strat, right_strat = hh.two_mutual_arrays(dtypes)

        if func_type is FuncType.FUNC:
            func = getattr(xp, func_name)
        else:
            op_sym = all_op_to_symbol[func_name]
            expr = f"{left_sym} {op_sym} {right_sym}"
            if func_type is FuncType.OP:

                def func(l: Array, r: Union[Scalar, Array]) -> Array:
                    locals_ = {}
                    locals_[left_sym] = l
                    locals_[right_sym] = r
                    return eval(expr, locals_)

            else:

                def func(l: Array, r: Union[Scalar, Array]) -> Array:
                    locals_ = {}
                    locals_[left_sym] = ah.asarray(
                        l, copy=True
                    )  # prevents left mutating
                    locals_[right_sym] = r
                    exec(expr, locals_)
                    return locals_[left_sym]

            func.__name__ = func_name  # for repr

        if func_type is FuncType.IOP:
            res_name = left_sym
        else:
            res_name = "out"

        return pytest.param(
            func_name,
            func,
            left_sym,
            left_strat,
            right_sym,
            right_strat,
            right_is_scalar,
            res_name,
            id=f"{func_name}({left_sym}, {right_sym})",
        )

    op_name = func_to_op[elwise_func_name]
    params = [
        make_param(elwise_func_name, FuncType.FUNC, False),
        make_param(op_name, FuncType.OP, False),
        make_param(op_name, FuncType.OP, True),
    ]
    iop_name = f"__i{op_name[2:]}"
    if iop_name in dh.inplace_op_to_symbol.keys():
        params.append(make_param(iop_name, FuncType.IOP, False))
        params.append(make_param(iop_name, FuncType.IOP, True))

    return params


@pytest.mark.parametrize(unary_argnames, make_unary_params("abs", dh.numeric_dtypes))
@given(data=st.data())
def test_abs(func_name, func, strat, data):
    x = data.draw(strat, label="x")
    if x.dtype in dh.int_dtypes:
        # abs of the smallest representable negative integer is not defined
        mask = xp.not_equal(
            x, ah.full(x.shape, dh.dtype_ranges[x.dtype].min, dtype=x.dtype)
        )
        x = x[mask]
    out = func(x)
    ph.assert_shape(func_name, out.shape, x.shape)
    assert ah.all(
        ah.logical_not(ah.negative_mathematical_sign(out))
    ), f"out elements not all positively signed [{func_name}()]\n{out=}"
    less_zero = ah.negative_mathematical_sign(x)
    negx = ah.negative(x)
    # abs(x) = -x for x < 0
    ah.assert_exactly_equal(out[less_zero], negx[less_zero])
    # abs(x) = x for x >= 0
    ah.assert_exactly_equal(
        out[ah.logical_not(less_zero)], x[ah.logical_not(less_zero)]
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acos(x):
    res = xp.acos(x)
    ph.assert_shape("acos", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    # Here (and elsewhere), should technically be res.dtype, but this is the
    # same as x.dtype, as tested by the type_promotion tests.
    PI = ah.π(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(res, ZERO, PI)
    # acos maps [-1, 1] to [0, pi]. Values outside this domain are mapped to
    # nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acosh(x):
    res = xp.acosh(x)
    ph.assert_shape("acosh", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ONE, INFINITY)
    codomain = ah.inrange(res, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize(binary_argnames, make_binary_params("add", dh.numeric_dtypes))
@given(data=st.data())
def test_add(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    try:
        res = func(left, right)
    except OverflowError:
        reject()

    if not right_is_scalar:
        # add is commutative
        expected = func(right, left)
        ah.assert_exactly_equal(res, expected)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asin(x):
    res = xp.asin(x)
    ph.assert_shape("asin", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(res, -PI / 2, PI / 2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asinh(x):
    res = xp.asinh(x)
    ph.assert_shape("asinh", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
    # asinh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atan(x):
    res = xp.atan(x)
    ph.assert_shape("atan", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(res, -PI / 2, PI / 2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_atan2(x1, x2):
    res = xp.atan2(x1, x2)
    INFINITY1 = ah.infinity(x1.shape, x1.dtype)
    INFINITY2 = ah.infinity(x2.shape, x2.dtype)
    PI = ah.π(res.shape, res.dtype)
    domainx1 = ah.inrange(x1, -INFINITY1, INFINITY1)
    domainx2 = ah.inrange(x2, -INFINITY2, INFINITY2)
    # codomain = ah.inrange(res, -PI, PI, 1e-5)
    codomain = ah.inrange(res, -PI, PI)
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
    pos_out = ah.positive_mathematical_sign(res)
    neg_out = ah.negative_mathematical_sign(res)
    ah.assert_exactly_equal(
        ah.logical_or(ah.logical_and(pos_x1, pos_x2), ah.logical_and(pos_x1, neg_x2)),
        pos_out,
    )
    ah.assert_exactly_equal(
        ah.logical_or(ah.logical_and(neg_x1, pos_x2), ah.logical_and(neg_x1, neg_x2)),
        neg_out,
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atanh(x):
    res = xp.atanh(x)
    ph.assert_shape("atanh", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("bitwise_and", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_and(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    res = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, res.shape, shape, repr_name=f"{res_name}.shape")
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Compare against the Python & operator.
        if res.dtype == xp.bool:
            for idx in ah.ndindex(res.shape):
                s_left = bool(_left[idx])
                s_right = bool(_right[idx])
                s_res = bool(res[idx])
                assert (s_left and s_right) == s_res
        else:
            for idx in ah.ndindex(res.shape):
                s_left = int(_left[idx])
                s_right = int(_right[idx])
                s_res = int(res[idx])
                s_and = ah.int_to_dtype(
                    s_left & s_right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
                assert s_and == s_res


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("bitwise_left_shift", dh.all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_left_shift(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)
    if right_is_scalar:
        assume(right >= 0)
    else:
        assume(not ah.any(ah.isnegative(right)))

    res = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, res.shape, shape, repr_name=f"{res_name}.shape")
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Compare against the Python << operator.
        for idx in ah.ndindex(res.shape):
            s_left = int(_left[idx])
            s_right = int(_right[idx])
            s_res = int(res[idx])
            s_shift = ah.int_to_dtype(
                # We avoid shifting very large ints
                s_left << s_right if s_right < dh.dtype_nbits[res.dtype] else 0,
                dh.dtype_nbits[res.dtype],
                dh.dtype_signed[res.dtype],
            )
            assert s_shift == s_res


@pytest.mark.parametrize(
    unary_argnames, make_unary_params("bitwise_invert", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_invert(func_name, func, strat, data):
    x = data.draw(strat, label="x")

    out = func(x)

    ph.assert_shape(func_name, out.shape, x.shape)
    # Compare against the Python ~ operator.
    if out.dtype == xp.bool:
        for idx in ah.ndindex(out.shape):
            s_x = bool(x[idx])
            s_out = bool(out[idx])
            assert (not s_x) == s_out
    else:
        for idx in ah.ndindex(out.shape):
            s_x = int(x[idx])
            s_out = int(out[idx])
            s_invert = ah.int_to_dtype(
                ~s_x, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype]
            )
            assert s_invert == s_out


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("bitwise_or", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_or(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    res = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, res.shape, shape, repr_name=f"{res_name}.shape")
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Compare against the Python | operator.
        if res.dtype == xp.bool:
            for idx in ah.ndindex(res.shape):
                s_left = bool(_left[idx])
                s_right = bool(_right[idx])
                s_res = bool(res[idx])
                assert (s_left or s_right) == s_res
        else:
            for idx in ah.ndindex(res.shape):
                s_left = int(_left[idx])
                s_right = int(_right[idx])
                s_res = int(res[idx])
                s_or = ah.int_to_dtype(
                    s_left | s_right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
                assert s_or == s_res


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("bitwise_right_shift", dh.all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_right_shift(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)
    if right_is_scalar:
        assume(right >= 0)
    else:
        assume(not ah.any(ah.isnegative(right)))

    res = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(
            "bitwise_right_shift", res.shape, shape, repr_name=f"{res_name}.shape"
        )
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Compare against the Python >> operator.
        for idx in ah.ndindex(res.shape):
            s_left = int(_left[idx])
            s_right = int(_right[idx])
            s_res = int(res[idx])
            s_shift = ah.int_to_dtype(
                s_left >> s_right, dh.dtype_nbits[res.dtype], dh.dtype_signed[res.dtype]
            )
            assert s_shift == s_res


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("bitwise_xor", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_xor(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    res = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, res.shape, shape, repr_name=f"{res_name}.shape")
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Compare against the Python ^ operator.
        if res.dtype == xp.bool:
            for idx in ah.ndindex(res.shape):
                s_left = bool(_left[idx])
                s_right = bool(_right[idx])
                s_res = bool(res[idx])
                assert (s_left ^ s_right) == s_res
        else:
            for idx in ah.ndindex(res.shape):
                s_left = int(_left[idx])
                s_right = int(_right[idx])
                s_res = int(res[idx])
                s_xor = ah.int_to_dtype(
                    s_left ^ s_right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
                assert s_xor == s_res


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_ceil(x):
    # This test is almost identical to test_floor()
    res = xp.ceil(x)
    ph.assert_shape("ceil", res.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(res[finite])
    assert ah.all(ah.less_equal(x[finite], res[finite]))
    assert ah.all(
        ah.less_equal(res[finite] - x[finite], ah.one(x[finite].shape, x.dtype))
    )
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(res[integers], x[integers])


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cos(x):
    res = xp.cos(x)
    ph.assert_shape("cos", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY, open=True)
    codomain = ah.inrange(res, -ONE, ONE)
    # cos maps (-inf, inf) to [-1, 1]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cosh(x):
    res = xp.cosh(x)
    ph.assert_shape("cosh", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize(binary_argnames, make_binary_params("divide", dh.float_dtypes))
@given(data=st.data())
def test_divide(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    func(left, right)

    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@pytest.mark.parametrize(binary_argnames, make_binary_params("equal", dh.all_dtypes))
@given(data=st.data())
def test_equal(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    # NOTE: ah.assert_exactly_equal() itself uses ah.equal(), so we must be careful
    # not to use it here. Otherwise, the test would be circular and
    # meaningless. Instead, we implement this by iterating every element of
    # the arrays and comparing them. The logic here is also used for the tests
    # for the other elementwise functions that accept any input dtype but
    # always return bool (greater(), greater_equal(), less(), less_equal(),
    # and not_equal()).

    if not right_is_scalar:
        # First we broadcast the arrays so that they can be indexed uniformly.
        # TODO: it should be possible to skip this step if we instead generate
        # indices to x1 and x2 that correspond to the broadcasted shapes. This
        # would avoid the dependence in this test on broadcast_to().
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        # Second, manually promote the dtypes. This is important. If the internal
        # type promotion in ah.equal() is wrong, it will not be directly visible in
        # the output type, but it can lead to wrong answers. For example,
        # ah.equal(array(1.0, dtype=xp.float32), array(1.00000001, dtype=xp.float64)) will
        # be wrong if the float64 is downcast to float32. # be wrong if the
        # xp.float64 is downcast to float32. See the comment on
        # test_elementwise_function_two_arg_bool_type_promotion() in
        # test_type_promotion.py. The type promotion for ah.equal() is not *really*
        # tested in that file, because doing so requires doing the consistency

        # check we do here rather than just checking the res dtype.
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            out_idx = out[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) == scalar_type(x2_idx))


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_exp(x):
    res = xp.exp(x)
    ph.assert_shape("exp", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(res, ZERO, INFINITY)
    # exp maps [-inf, inf] to [0, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_expm1(x):
    res = xp.expm1(x)
    ph.assert_shape("expm1", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(res, NEGONE, INFINITY)
    # expm1 maps [-inf, inf] to [1, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_floor(x):
    # This test is almost identical to test_ceil
    res = xp.floor(x)
    ph.assert_shape("floor", res.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(res[finite])
    assert ah.all(ah.less_equal(res[finite], x[finite]))
    assert ah.all(
        ah.less_equal(x[finite] - res[finite], ah.one(x[finite].shape, x.dtype))
    )
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(res[integers], x[integers])


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("floor_divide", dh.numeric_dtypes)
)
@given(data=st.data())
def test_floor_divide(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat.filter(lambda x: not ah.any(x == 0)), label=left_sym)
    right = data.draw(right_strat, label=right_sym)
    if right_is_scalar:
        assume(right != 0)
    else:
        assume(not ah.any(right == 0))

    res = func(left, right)

    if not right_is_scalar:
        if dh.is_int_dtype(left.dtype):
            # The spec does not specify the behavior for division by 0 for integer
            # dtypes. A library may choose to raise an exception in this case, so
            # we avoid passing it in entirely.
            div = xp.divide(
                ah.asarray(left, dtype=xp.float64),
                ah.asarray(right, dtype=xp.float64),
            )
        else:
            div = xp.divide(left, right)

        # TODO: The spec doesn't clearly specify the behavior of floor_divide on
        # infinities. See https://github.com/data-apis/array-api/issues/199.
        finite = ah.isfinite(div)
        ah.assert_integral(res[finite])

    # TODO: Test the exact output for floor_divide.


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("greater", dh.numeric_dtypes)
)
@given(data=st.data())
def test_greater(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)
        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            out_idx = out[idx]
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) > scalar_type(x2_idx))


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("greater_equal", dh.numeric_dtypes)
)
@given(data=st.data())
def test_greater_equal(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)

        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            out_idx = out[idx]
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) >= scalar_type(x2_idx))


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isfinite(x):
    res = ah.isfinite(x)
    ph.assert_shape("isfinite", res.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(res, ah.true(x.shape))
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = ah.logical_or(xp.isinf(x), ah.isnan(x))
    ah.assert_exactly_equal(res, ah.logical_not(inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(res[idx]) == math.isfinite(s)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isinf(x):
    res = xp.isinf(x)

    ph.assert_shape("isinf", res.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(res, ah.false(x.shape))
    finite_or_nan = ah.logical_or(ah.isfinite(x), ah.isnan(x))
    ah.assert_exactly_equal(res, ah.logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(res[idx]) == math.isinf(s)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isnan(x):
    res = ah.isnan(x)

    ph.assert_shape("isnan", res.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(res, ah.false(x.shape))
    finite_or_inf = ah.logical_or(ah.isfinite(x), xp.isinf(x))
    ah.assert_exactly_equal(res, ah.logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in ah.ndindex(x.shape):
            s = float(x[idx])
            assert bool(res[idx]) == math.isnan(s)


@pytest.mark.parametrize(binary_argnames, make_binary_params("less", dh.numeric_dtypes))
@given(data=st.data())
def test_less(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)

        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            out_idx = out[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) < scalar_type(x2_idx))


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("less_equal", dh.numeric_dtypes)
)
@given(data=st.data())
def test_less_equal(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)

        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            out_idx = out[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) <= scalar_type(x2_idx))


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log(x):
    res = xp.log(x)

    ph.assert_shape("log", res.shape, x.shape)

    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
    # log maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log1p(x):
    res = xp.log1p(x)
    ph.assert_shape("log1p", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    NEGONE = -ah.one(x.shape, x.dtype)
    codomain = ah.inrange(x, NEGONE, INFINITY)
    domain = ah.inrange(res, -INFINITY, INFINITY)
    # log1p maps [1, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log2(x):
    res = xp.log2(x)
    ph.assert_shape("log2", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
    # log2 maps [0, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log10(x):
    res = xp.log10(x)
    ph.assert_shape("log10", res.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ZERO, INFINITY)
    codomain = ah.inrange(res, -INFINITY, INFINITY)
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


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("multiply", dh.numeric_dtypes)
)
@given(data=st.data())
def test_multiply(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    res = func(left, right)

    if not right_is_scalar:
        # multiply is commutative
        expected = func(right, left)
        ah.assert_exactly_equal(res, expected)


@pytest.mark.parametrize(
    unary_argnames, make_unary_params("negative", dh.numeric_dtypes)
)
@given(data=st.data())
def test_negative(func_name, func, strat, data):
    x = data.draw(strat, label="x")

    out = func(x)

    ph.assert_shape(func_name, out.shape, x.shape)

    # Negation is an involution
    ah.assert_exactly_equal(x, func(out))

    mask = ah.isfinite(x)
    if dh.is_int_dtype(x.dtype):
        minval = dh.dtype_ranges[x.dtype][0]
        if minval < 0:
            # negative of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))

    # Additive inverse
    y = xp.add(x[mask], out[mask])
    ah.assert_exactly_equal(y, ah.zero(x[mask].shape, x.dtype))


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("not_equal", dh.all_dtypes)
)
@given(data=st.data())
def test_not_equal(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    out = func(left, right)

    if not right_is_scalar:
        # TODO: generate indices without broadcasting arrays (see test_equal comment)

        shape = broadcast_shapes(left.shape, right.shape)
        ph.assert_shape(func_name, out.shape, shape)
        _left = xp.broadcast_to(left, shape)
        _right = xp.broadcast_to(right, shape)

        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = ah.asarray(_left, dtype=promoted_dtype)
        _right = ah.asarray(_right, dtype=promoted_dtype)

        scalar_type = dh.get_scalar_type(promoted_dtype)
        for idx in ah.ndindex(shape):
            out_idx = out[idx]
            x1_idx = _left[idx]
            x2_idx = _right[idx]
            assert out_idx.shape == x1_idx.shape == x2_idx.shape  # sanity check
            assert bool(out_idx) == (scalar_type(x1_idx) != scalar_type(x2_idx))


@pytest.mark.parametrize(
    unary_argnames, make_unary_params("positive", dh.numeric_dtypes)
)
@given(data=st.data())
def test_positive(func_name, func, strat, data):
    x = data.draw(strat, label="x")

    out = func(x)

    ph.assert_shape(func_name, out.shape, x.shape)
    # Positive does nothing
    ah.assert_exactly_equal(out, x)


@pytest.mark.parametrize(binary_argnames, make_binary_params("pow", dh.float_dtypes))
@given(data=st.data())
def test_pow(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    try:
        func(left, right)
    except OverflowError:
        reject()

    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("remainder", dh.numeric_dtypes)
)
@given(data=st.data())
def test_remainder(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)
    # TODO: rework same sign testing below to remove this
    if not right_is_scalar:
        assume(len(left.shape) <= len(right.shape))

    res = func(left, right)

    if not right_is_scalar:
        # res and x2 should have the same sign.
        # ah.assert_same_sign returns False for nans
        not_nan = ah.logical_not(ah.logical_or(ah.isnan(res), ah.isnan(left)))
        ah.assert_same_sign(res[not_nan], right[not_nan])


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_round(x):
    res = xp.round(x)

    ph.assert_shape("round", res.shape, x.shape)

    # Test that the res is integral
    finite = ah.isfinite(x)
    ah.assert_integral(res[finite])

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
    ah.assert_exactly_equal(res[round_down], floor[round_down])
    ah.assert_exactly_equal(res[round_up], ceil[round_up])


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_sign(x):
    res = xp.sign(x)
    ph.assert_shape("sign", res.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sin(x):
    res = xp.sin(x)
    ph.assert_shape("sin", res.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sinh(x):
    res = xp.sinh(x)
    ph.assert_shape("sinh", res.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_square(x):
    res = xp.square(x)
    ph.assert_shape("square", res.shape, x.shape)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sqrt(x):
    res = xp.sqrt(x)
    ph.assert_shape("sqrt", res.shape, x.shape)


@pytest.mark.parametrize(
    binary_argnames, make_binary_params("subtract", dh.numeric_dtypes)
)
@given(data=st.data())
def test_subtract(
    func_name,
    func,
    left_sym,
    left_strat,
    right_sym,
    right_strat,
    right_is_scalar,
    res_name,
    data,
):
    left = data.draw(left_strat, label=left_sym)
    right = data.draw(right_strat, label=right_sym)

    try:
        func(left, right)
    except OverflowError:
        reject()

    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tan(x):
    res = xp.tan(x)
    ph.assert_shape("tan", res.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tanh(x):
    res = xp.tanh(x)
    ph.assert_shape("tanh", res.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=hh.numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    res = xp.trunc(x)
    ph.assert_shape("bitwise_trunc", res.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(res, x)
    else:
        finite = ah.isfinite(x)
        ah.assert_integral(res[finite])
