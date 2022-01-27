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
from typing import Callable, List, NamedTuple, Optional, Union

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.control import reject

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import Array, DataType, Param, Scalar, Shape

pytestmark = pytest.mark.ci


def all_integer_dtypes() -> st.SearchStrategy[DataType]:
    return xps.unsigned_integer_dtypes() | xps.integer_dtypes()


def boolean_and_all_integer_dtypes() -> st.SearchStrategy[DataType]:
    return xps.boolean_dtypes() | all_integer_dtypes()


def isclose(n1: Union[int, float], n2: Union[int, float]):
    if not (math.isfinite(n1) and math.isfinite(n2)):
        raise ValueError(f"{n1=} and {n1=}, but input must be finite")
    return math.isclose(n1, n2, rel_tol=0.25, abs_tol=1)


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


class UnaryParamContext(NamedTuple):
    func_name: str
    func: Callable[[Array], Array]
    strat: st.SearchStrategy[Array]

    @property
    def id(self) -> str:
        return f"{self.func_name}"

    def __repr__(self):
        return f"UnaryParamContext(<{self.id}>)"


def make_unary_params(
    elwise_func_name: str, dtypes_strat: st.SearchStrategy[DataType]
) -> List[Param[UnaryParamContext]]:
    strat = xps.arrays(dtype=dtypes_strat, shape=hh.shapes())
    func_ctx = UnaryParamContext(
        func_name=elwise_func_name, func=getattr(xp, elwise_func_name), strat=strat
    )
    op_name = func_to_op[elwise_func_name]
    op_ctx = UnaryParamContext(
        func_name=op_name, func=lambda x: getattr(x, op_name)(), strat=strat
    )
    return [pytest.param(func_ctx, id=func_ctx.id), pytest.param(op_ctx, id=op_ctx.id)]


class FuncType(Enum):
    FUNC = auto()
    OP = auto()
    IOP = auto()


shapes_kw = {"min_side": 1}


class BinaryParamContext(NamedTuple):
    func_name: str
    func: Callable[[Array, Union[Scalar, Array]], Array]
    left_sym: str
    left_strat: st.SearchStrategy[Array]
    right_sym: str
    right_strat: st.SearchStrategy[Union[Scalar, Array]]
    right_is_scalar: bool
    res_name: str

    @property
    def id(self) -> str:
        return f"{self.func_name}({self.left_sym}, {self.right_sym})"

    def __repr__(self):
        return f"BinaryParamContext(<{self.id}>)"


def make_binary_params(
    elwise_func_name: str, dtypes_strat: st.SearchStrategy[DataType]
) -> List[Param[BinaryParamContext]]:
    def make_param(
        func_name: str, func_type: FuncType, right_is_scalar: bool
    ) -> Param[BinaryParamContext]:
        if right_is_scalar:
            left_sym = "x"
            right_sym = "s"
        else:
            left_sym = "x1"
            right_sym = "x2"

        shared_dtypes = st.shared(dtypes_strat)
        if right_is_scalar:
            left_strat = xps.arrays(dtype=shared_dtypes, shape=hh.shapes(**shapes_kw))
            right_strat = shared_dtypes.flatmap(
                lambda d: xps.from_dtype(d, **finite_kw)
            )
        else:
            if func_type is FuncType.IOP:
                shared_shapes = st.shared(hh.shapes(**shapes_kw))
                left_strat = xps.arrays(dtype=shared_dtypes, shape=shared_shapes)
                right_strat = xps.arrays(dtype=shared_dtypes, shape=shared_shapes)
            else:
                mutual_shapes = st.shared(
                    hh.mutually_broadcastable_shapes(2, **shapes_kw)
                )
                left_strat = xps.arrays(
                    dtype=shared_dtypes, shape=mutual_shapes.map(lambda pair: pair[0])
                )
                right_strat = xps.arrays(
                    dtype=shared_dtypes, shape=mutual_shapes.map(lambda pair: pair[1])
                )

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
                    locals_[left_sym] = ah.asarray(l, copy=True)  # prevents mutating l
                    locals_[right_sym] = r
                    exec(expr, locals_)
                    return locals_[left_sym]

            func.__name__ = func_name  # for repr

        if func_type is FuncType.IOP:
            res_name = left_sym
        else:
            res_name = "out"

        ctx = BinaryParamContext(
            func_name,
            func,
            left_sym,
            left_strat,
            right_sym,
            right_strat,
            right_is_scalar,
            res_name,
        )
        return pytest.param(ctx, id=ctx.id)

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


def assert_binary_param_dtype(
    ctx: BinaryParamContext,
    left: Array,
    right: Union[Array, Scalar],
    res: Array,
    expected: Optional[DataType] = None,
):
    if ctx.right_is_scalar:
        in_dtypes = left.dtype
    else:
        in_dtypes = (left.dtype, right.dtype)  # type: ignore
    ph.assert_dtype(
        ctx.func_name, in_dtypes, res.dtype, expected, repr_name=f"{ctx.res_name}.dtype"
    )


def assert_binary_param_shape(
    ctx: BinaryParamContext,
    left: Array,
    right: Union[Array, Scalar],
    res: Array,
    expected: Optional[Shape] = None,
):
    if ctx.right_is_scalar:
        in_shapes = (left.shape,)
    else:
        in_shapes = (left.shape, right.shape)  # type: ignore
    ph.assert_result_shape(
        ctx.func_name, in_shapes, res.shape, expected, repr_name=f"{ctx.res_name}.shape"
    )


@pytest.mark.parametrize("ctx", make_unary_params("abs", xps.numeric_dtypes()))
@given(data=st.data())
def test_abs(ctx, data):
    x = data.draw(ctx.strat, label="x")
    if x.dtype in dh.int_dtypes:
        # abs of the smallest representable negative integer is not defined
        mask = xp.not_equal(
            x, ah.full(x.shape, dh.dtype_ranges[x.dtype].min, dtype=x.dtype)
        )
        x = x[mask]
    out = ctx.func(x)
    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    assert ah.all(
        ah.logical_not(ah.negative_mathematical_sign(out))
    ), f"out elements not all positively signed [{ctx.func_name}()]\n{out=}"
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
    ph.assert_dtype("acos", x.dtype, res.dtype)
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
    ph.assert_dtype("acosh", x.dtype, res.dtype)
    ph.assert_shape("acosh", res.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    ZERO = ah.zero(x.shape, x.dtype)
    domain = ah.inrange(x, ONE, INFINITY)
    codomain = ah.inrange(res, ZERO, INFINITY)
    # acosh maps [-1, inf] to [0, inf]. Values outside this domain are mapped
    # to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize("ctx,", make_binary_params("add", xps.numeric_dtypes()))
@given(data=st.data())
def test_add(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if not ctx.right_is_scalar:
        # add is commutative
        expected = ctx.func(right, left)
        ah.assert_exactly_equal(res, expected)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asin(x):
    out = xp.asin(x)
    ph.assert_dtype("asin", x.dtype, out.dtype)
    ph.assert_shape("asin", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(out, -PI / 2, PI / 2)
    # asin maps [-1, 1] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asinh(x):
    out = xp.asinh(x)
    ph.assert_dtype("asinh", x.dtype, out.dtype)
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
    ph.assert_dtype("atan", x.dtype, out.dtype)
    ph.assert_shape("atan", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    PI = ah.π(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, -PI / 2, PI / 2)
    # atan maps [-inf, inf] to [-pi/2, pi/2]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_atan2(x1, x2):
    out = xp.atan2(x1, x2)
    ph.assert_dtype("atan2", (x1.dtype, x2.dtype), out.dtype)
    ph.assert_result_shape("atan2", (x1.shape, x2.shape), out.shape)
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
    # each element-wise out. The quadrant (i.e., branch) is chosen such
    # that each element-wise out is the signed angle in radians between the
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
    out = xp.atanh(x)
    ph.assert_dtype("atanh", x.dtype, out.dtype)
    ph.assert_shape("atanh", out.shape, x.shape)
    ONE = ah.one(x.shape, x.dtype)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -ONE, ONE)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # atanh maps [-1, 1] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_and", boolean_and_all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_and(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    scalar_type = dh.get_scalar_type(res.dtype)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            scalar_l = scalar_type(left[idx])
            if res.dtype == xp.bool:
                expected = scalar_l and right
            else:
                # for mypy
                assert isinstance(scalar_l, int)
                assert isinstance(right, int)
                expected = ah.int_to_dtype(
                    scalar_l & right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            scalar_o = scalar_type(res[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} & {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            scalar_l = scalar_type(left[l_idx])
            scalar_r = scalar_type(right[r_idx])
            if res.dtype == xp.bool:
                expected = scalar_l and scalar_r
            else:
                # for mypy
                assert isinstance(scalar_l, int)
                assert isinstance(scalar_r, int)
                expected = ah.int_to_dtype(
                    scalar_l & scalar_r,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            scalar_o = scalar_type(res[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} & {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_left_shift", all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_left_shift(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right >= 0)
    else:
        assume(not ah.any(ah.isnegative(right)))

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            scalar_l = int(left[idx])
            expected = ah.int_to_dtype(
                # We avoid shifting very large ints
                scalar_l << right if right < dh.dtype_nbits[res.dtype] else 0,
                dh.dtype_nbits[res.dtype],
                dh.dtype_signed[res.dtype],
            )
            scalar_o = int(res[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} << {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            scalar_l = int(left[l_idx])
            scalar_r = int(right[r_idx])
            expected = ah.int_to_dtype(
                # We avoid shifting very large ints
                scalar_l << scalar_r if scalar_r < dh.dtype_nbits[res.dtype] else 0,
                dh.dtype_nbits[res.dtype],
                dh.dtype_signed[res.dtype],
            )
            scalar_o = int(res[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} << {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize(
    "ctx",
    make_unary_params("bitwise_invert", boolean_and_all_integer_dtypes()),
)
@given(data=st.data())
def test_bitwise_invert(ctx, data):
    x = data.draw(ctx.strat, label="x")

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    for idx in sh.ndindex(out.shape):
        if out.dtype == xp.bool:
            scalar_x = bool(x[idx])
            scalar_o = bool(out[idx])
            expected = not scalar_x
        else:
            scalar_x = int(x[idx])
            scalar_o = int(out[idx])
            expected = ah.int_to_dtype(
                ~scalar_x, dh.dtype_nbits[out.dtype], dh.dtype_signed[out.dtype]
            )
        f_x = sh.fmt_idx("x", idx)
        f_o = sh.fmt_idx("out", idx)
        assert scalar_o == expected, (
            f"{f_o}={scalar_o}, but should be ~{f_x}={scalar_x} "
            f"[{ctx.func_name}()]\n{f_x}={scalar_x}"
        )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_or", boolean_and_all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_or(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            if res.dtype == xp.bool:
                scalar_l = bool(left[idx])
                scalar_o = bool(res[idx])
                expected = scalar_l or right
            else:
                scalar_l = int(left[idx])
                scalar_o = int(res[idx])
                expected = ah.int_to_dtype(
                    scalar_l | right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} | {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            if res.dtype == xp.bool:
                scalar_l = bool(left[l_idx])
                scalar_r = bool(right[r_idx])
                scalar_o = bool(res[o_idx])
                expected = scalar_l or scalar_r
            else:
                scalar_l = int(left[l_idx])
                scalar_r = int(right[r_idx])
                scalar_o = int(res[o_idx])
                expected = ah.int_to_dtype(
                    scalar_l | scalar_r,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} | {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_right_shift", all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_right_shift(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right >= 0)
    else:
        assume(not ah.any(ah.isnegative(right)))

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            scalar_l = int(left[idx])
            expected = ah.int_to_dtype(
                scalar_l >> right,
                dh.dtype_nbits[res.dtype],
                dh.dtype_signed[res.dtype],
            )
            scalar_o = int(res[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} >> {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            scalar_l = int(left[l_idx])
            scalar_r = int(right[r_idx])
            expected = ah.int_to_dtype(
                scalar_l >> scalar_r,
                dh.dtype_nbits[res.dtype],
                dh.dtype_signed[res.dtype],
            )
            scalar_o = int(res[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} >> {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_xor", boolean_and_all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_xor(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            if res.dtype == xp.bool:
                scalar_l = bool(left[idx])
                scalar_o = bool(res[idx])
                expected = scalar_l ^ right
            else:
                scalar_l = int(left[idx])
                scalar_o = int(res[idx])
                expected = ah.int_to_dtype(
                    scalar_l ^ right,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} ^ {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            if res.dtype == xp.bool:
                scalar_l = bool(left[l_idx])
                scalar_r = bool(right[r_idx])
                scalar_o = bool(res[o_idx])
                expected = scalar_l ^ scalar_r
            else:
                scalar_l = int(left[l_idx])
                scalar_r = int(right[r_idx])
                scalar_o = int(res[o_idx])
                expected = ah.int_to_dtype(
                    scalar_l ^ scalar_r,
                    dh.dtype_nbits[res.dtype],
                    dh.dtype_signed[res.dtype],
                )
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} ^ {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_ceil(x):
    # This test is almost identical to test_floor()
    out = xp.ceil(x)
    ph.assert_dtype("ceil", x.dtype, out.dtype)
    ph.assert_shape("ceil", out.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])
    assert ah.all(ah.less_equal(x[finite], out[finite]))
    assert ah.all(
        ah.less_equal(out[finite] - x[finite], ah.one(x[finite].shape, x.dtype))
    )
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(out[integers], x[integers])


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cos(x):
    out = xp.cos(x)
    ph.assert_dtype("cos", x.dtype, out.dtype)
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
    ph.assert_dtype("cosh", x.dtype, out.dtype)
    ph.assert_shape("cosh", out.shape, x.shape)
    INFINITY = ah.infinity(x.shape, x.dtype)
    domain = ah.inrange(x, -INFINITY, INFINITY)
    codomain = ah.inrange(out, -INFINITY, INFINITY)
    # cosh maps [-inf, inf] to [-inf, inf]. Values outside this domain are
    # mapped to nan, which is already tested in the special cases.
    ah.assert_exactly_equal(domain, codomain)


@pytest.mark.parametrize("ctx", make_binary_params("divide", xps.floating_dtypes()))
@given(data=st.data())
def test_divide(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of division that strictly hold for floating-point numbers. We
    # could test that this does implement IEEE 754 division, but we don't yet
    # have those sorts in general for this module.


@pytest.mark.parametrize("ctx", make_binary_params("equal", xps.scalar_dtypes()))
@given(data=st.data())
def test_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l == right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} == {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # We manually promote the dtypes as incorrect internal type promotion
        # could lead to false positives. For example
        #
        #     >>> xp.equal(
        #     ...     xp.asarray(1.0, dtype=xp.float32),
        #     ...     xp.asarray(1.00000001, dtype=xp.float64),
        #     ... )
        #
        # would erroneously be True if float64 downcasted to float32.
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l == scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} == {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_exp(x):
    out = xp.exp(x)
    ph.assert_dtype("exp", x.dtype, out.dtype)
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
    ph.assert_dtype("expm1", x.dtype, out.dtype)
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
    ph.assert_dtype("floor", x.dtype, out.dtype)
    ph.assert_shape("floor", out.shape, x.shape)
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])
    assert ah.all(ah.less_equal(out[finite], x[finite]))
    assert ah.all(
        ah.less_equal(x[finite] - out[finite], ah.one(x[finite].shape, x.dtype))
    )
    integers = ah.isintegral(x)
    ah.assert_exactly_equal(out[integers], x[integers])


@pytest.mark.parametrize(
    "ctx", make_binary_params("floor_divide", xps.numeric_dtypes())
)
@given(data=st.data())
def test_floor_divide(ctx, data):
    left = data.draw(
        ctx.left_strat.filter(lambda x: not ah.any(x == 0)), label=ctx.left_sym
    )
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right != 0)
    else:
        assume(not ah.any(right == 0))

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    scalar_type = dh.get_scalar_type(res.dtype)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l // right
            scalar_o = scalar_type(res[idx])
            if not all(math.isfinite(n) for n in [scalar_l, right, scalar_o, expected]):
                continue
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly ({f_l} // {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            scalar_l = scalar_type(left[l_idx])
            scalar_r = scalar_type(right[r_idx])
            expected = scalar_l // scalar_r
            scalar_o = scalar_type(res[o_idx])
            if not all(
                math.isfinite(n) for n in [scalar_l, scalar_r, scalar_o, expected]
            ):
                continue
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly ({f_l} // {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize("ctx", make_binary_params("greater", xps.numeric_dtypes()))
@given(data=st.data())
def test_greater(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l > right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} > {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l > scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} > {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize(
    "ctx", make_binary_params("greater_equal", xps.numeric_dtypes())
)
@given(data=st.data())
def test_greater_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l >= right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} >= {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l >= scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} >= {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isfinite(x):
    out = ah.isfinite(x)
    ph.assert_dtype("isfinite", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isfinite", out.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.true(x.shape))
    # Test that isfinite, isinf, and isnan are self-consistent.
    inf = ah.logical_or(xp.isinf(x), ah.isnan(x))
    ah.assert_exactly_equal(out, ah.logical_not(inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in sh.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isfinite(s)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isinf(x):
    out = xp.isinf(x)

    ph.assert_dtype("isfinite", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isinf", out.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.false(x.shape))
    finite_or_nan = ah.logical_or(ah.isfinite(x), ah.isnan(x))
    ah.assert_exactly_equal(out, ah.logical_not(finite_or_nan))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in sh.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isinf(s)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isnan(x):
    out = ah.isnan(x)

    ph.assert_dtype("isnan", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isnan", out.shape, x.shape)

    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, ah.false(x.shape))
    finite_or_inf = ah.logical_or(ah.isfinite(x), xp.isinf(x))
    ah.assert_exactly_equal(out, ah.logical_not(finite_or_inf))

    # Test the exact value by comparing to the math version
    if dh.is_float_dtype(x.dtype):
        for idx in sh.ndindex(x.shape):
            s = float(x[idx])
            assert bool(out[idx]) == math.isnan(s)


@pytest.mark.parametrize("ctx", make_binary_params("less", xps.numeric_dtypes()))
@given(data=st.data())
def test_less(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l < right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} < {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l < scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} < {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize("ctx", make_binary_params("less_equal", xps.numeric_dtypes()))
@given(data=st.data())
def test_less_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l <= right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} <= {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l <= scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} <= {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log(x):
    out = xp.log(x)

    ph.assert_dtype("log", x.dtype, out.dtype)
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
    ph.assert_dtype("log1p", x.dtype, out.dtype)
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
    ph.assert_dtype("log2", x.dtype, out.dtype)
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
    ph.assert_dtype("log10", x.dtype, out.dtype)
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
    out = xp.logaddexp(x1, x2)
    ph.assert_dtype("logaddexp", (x1.dtype, x2.dtype), out.dtype)
    # The spec doesn't require any behavior for this function. We could test
    # that this is indeed an approximation of log(exp(x1) + exp(x2)), but we
    # don't have tests for this sort of thing for any functions yet.


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_and(x1, x2):
    out = ah.logical_and(x1, x2)
    ph.assert_dtype("logical_and", (x1.dtype, x2.dtype), out.dtype)
    ph.assert_result_shape("logical_and", (x1.shape, x2.shape), out.shape)
    for l_idx, r_idx, o_idx in sh.iter_indices(x1.shape, x2.shape, out.shape):
        scalar_l = bool(x1[l_idx])
        scalar_r = bool(x2[r_idx])
        expected = scalar_l and scalar_r
        scalar_o = bool(out[o_idx])
        f_l = sh.fmt_idx("x1", l_idx)
        f_r = sh.fmt_idx("x2", r_idx)
        f_o = sh.fmt_idx("out", o_idx)
        assert scalar_o == expected, (
            f"{f_o}={scalar_o}, but should be ({f_l} and {f_r})={expected} "
            f"[logical_and()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
        )


@given(xps.arrays(dtype=xp.bool, shape=hh.shapes()))
def test_logical_not(x):
    out = ah.logical_not(x)
    ph.assert_dtype("logical_not", x.dtype, out.dtype)
    ph.assert_shape("logical_not", out.shape, x.shape)
    for idx in sh.ndindex(x.shape):
        assert out[idx] == (not bool(x[idx]))


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_or(x1, x2):
    out = ah.logical_or(x1, x2)
    ph.assert_dtype("logical_or", (x1.dtype, x2.dtype), out.dtype)
    # See the comments in test_equal
    shape = sh.broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_or", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)
    for idx in sh.ndindex(shape):
        assert out[idx] == (bool(_x1[idx]) or bool(_x2[idx]))


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_xor(x1, x2):
    out = xp.logical_xor(x1, x2)
    ph.assert_dtype("logical_xor", (x1.dtype, x2.dtype), out.dtype)
    # See the comments in test_equal
    shape = sh.broadcast_shapes(x1.shape, x2.shape)
    ph.assert_shape("logical_xor", out.shape, shape)
    _x1 = xp.broadcast_to(x1, shape)
    _x2 = xp.broadcast_to(x2, shape)
    for idx in sh.ndindex(shape):
        assert out[idx] == (bool(_x1[idx]) ^ bool(_x2[idx]))


@pytest.mark.parametrize("ctx", make_binary_params("multiply", xps.numeric_dtypes()))
@given(data=st.data())
def test_multiply(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    if not ctx.right_is_scalar:
        # multiply is commutative
        expected = ctx.func(right, left)
        ah.assert_exactly_equal(res, expected)


@pytest.mark.parametrize("ctx", make_unary_params("negative", xps.numeric_dtypes()))
@given(data=st.data())
def test_negative(ctx, data):
    x = data.draw(ctx.strat, label="x")

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)

    # Negation is an involution
    ah.assert_exactly_equal(x, ctx.func(out))

    mask = ah.isfinite(x)
    if dh.is_int_dtype(x.dtype):
        minval = dh.dtype_ranges[x.dtype][0]
        if minval < 0:
            # negative of the smallest representable negative integer is not defined
            mask = xp.not_equal(x, ah.full(x.shape, minval, dtype=x.dtype))

    # Additive inverse
    y = xp.add(x[mask], out[mask])
    ah.assert_exactly_equal(y, ah.zero(x[mask].shape, x.dtype))


@pytest.mark.parametrize("ctx", make_binary_params("not_equal", xps.scalar_dtypes()))
@given(data=st.data())
def test_not_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, out, xp.bool)
    assert_binary_param_shape(ctx, left, right, out)
    if ctx.right_is_scalar:
        scalar_type = dh.get_scalar_type(left.dtype)
        for idx in sh.ndindex(left.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l != right
            scalar_o = bool(out[idx])
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} != {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        _left = xp.astype(left, promoted_dtype)
        _right = xp.astype(right, promoted_dtype)
        scalar_type = dh.get_scalar_type(promoted_dtype)
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, out.shape):
            scalar_l = scalar_type(_left[l_idx])
            scalar_r = scalar_type(_right[r_idx])
            expected = scalar_l != scalar_r
            scalar_o = bool(out[o_idx])
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be ({f_l} != {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@pytest.mark.parametrize("ctx", make_unary_params("positive", xps.numeric_dtypes()))
@given(data=st.data())
def test_positive(ctx, data):
    x = data.draw(ctx.strat, label="x")

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    # Positive does nothing
    ah.assert_exactly_equal(out, x)


@pytest.mark.parametrize("ctx", make_binary_params("pow", xps.numeric_dtypes()))
@given(data=st.data())
def test_pow(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        if isinstance(right, int):
            assume(right >= 0)
    else:
        if dh.is_int_dtype(right.dtype):
            assume(xp.all(right >= 0))

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    # There isn't much we can test here. The spec doesn't require any behavior
    # beyond the special cases, and indeed, there aren't many mathematical
    # properties of exponentiation that strictly hold for floating-point
    # numbers. We could test that this does implement IEEE 754 pow, but we
    # don't yet have those sorts in general for this module.


@pytest.mark.parametrize("ctx", make_binary_params("remainder", xps.numeric_dtypes()))
@given(data=st.data())
def test_remainder(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right != 0)
    else:
        assume(not ah.any(right == 0))

    res = ctx.func(left, right)

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    scalar_type = dh.get_scalar_type(res.dtype)
    if ctx.right_is_scalar:
        for idx in sh.ndindex(res.shape):
            scalar_l = scalar_type(left[idx])
            expected = scalar_l % right
            scalar_o = scalar_type(res[idx])
            if not all(math.isfinite(n) for n in [scalar_l, right, scalar_o, expected]):
                continue
            f_l = sh.fmt_idx(ctx.left_sym, idx)
            f_o = sh.fmt_idx(ctx.res_name, idx)
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly ({f_l} % {right})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}"
            )
    else:
        for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
            scalar_l = scalar_type(left[l_idx])
            scalar_r = scalar_type(right[r_idx])
            expected = scalar_l % scalar_r
            scalar_o = scalar_type(res[o_idx])
            if not all(
                math.isfinite(n) for n in [scalar_l, scalar_r, scalar_o, expected]
            ):
                continue
            f_l = sh.fmt_idx(ctx.left_sym, l_idx)
            f_r = sh.fmt_idx(ctx.right_sym, r_idx)
            f_o = sh.fmt_idx(ctx.res_name, o_idx)
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly ({f_l} % {f_r})={expected} "
                f"[{ctx.func_name}()]\n{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_round(x):
    out = xp.round(x)

    ph.assert_dtype("round", x.dtype, out.dtype)

    ph.assert_shape("round", out.shape, x.shape)

    # Test that the out is integral
    finite = ah.isfinite(x)
    ah.assert_integral(out[finite])

    # round(x) should be the neaoutt integer to x. The case where there is a
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
    ph.assert_dtype("sign", x.dtype, out.dtype)
    ph.assert_shape("sign", out.shape, x.shape)
    scalar_type = dh.get_scalar_type(x.dtype)
    for idx in sh.ndindex(x.shape):
        scalar_x = scalar_type(x[idx])
        f_x = sh.fmt_idx("x", idx)
        if math.isnan(scalar_x):
            continue
        if scalar_x == 0:
            expected = 0
            expr = f"{f_x}=0"
        else:
            expected = 1 if scalar_x > 0 else -1
            expr = f"({f_x} / |{f_x}|)={expected}"
        scalar_o = scalar_type(out[idx])
        f_o = sh.fmt_idx("out", idx)
        assert scalar_o == expected, (
            f"{f_o}={scalar_o}, but should be {expr} [sign()]\n{f_x}={scalar_x}"
        )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sin(x):
    out = xp.sin(x)
    ph.assert_dtype("sin", x.dtype, out.dtype)
    ph.assert_shape("sin", out.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sinh(x):
    out = xp.sinh(x)
    ph.assert_dtype("sinh", x.dtype, out.dtype)
    ph.assert_shape("sinh", out.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_square(x):
    out = xp.square(x)
    ph.assert_dtype("square", x.dtype, out.dtype)
    ph.assert_shape("square", out.shape, x.shape)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sqrt(x):
    out = xp.sqrt(x)
    ph.assert_dtype("sqrt", x.dtype, out.dtype)
    ph.assert_shape("sqrt", out.shape, x.shape)


@pytest.mark.parametrize("ctx", make_binary_params("subtract", xps.numeric_dtypes()))
@given(data=st.data())
def test_subtract(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    assert_binary_param_dtype(ctx, left, right, res)
    assert_binary_param_shape(ctx, left, right, res)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tan(x):
    out = xp.tan(x)
    ph.assert_dtype("tan", x.dtype, out.dtype)
    ph.assert_shape("tan", out.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tanh(x):
    out = xp.tanh(x)
    ph.assert_dtype("tanh", x.dtype, out.dtype)
    ph.assert_shape("tanh", out.shape, x.shape)
    # TODO


@given(xps.arrays(dtype=hh.numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    out = xp.trunc(x)
    ph.assert_dtype("trunc", x.dtype, out.dtype)
    ph.assert_shape("trunc", out.shape, x.shape)
    if dh.is_int_dtype(x.dtype):
        ah.assert_exactly_equal(out, x)
    else:
        finite = ah.isfinite(x)
        ah.assert_integral(out[finite])
