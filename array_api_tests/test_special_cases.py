# We use __future__ for forward reference type hints - this will work for even py3.8.0
# See https://stackoverflow.com/a/33533514/5193926
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from warnings import warn

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from array_api_tests.typing import Array, DataType

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from ._array_module import mod as xp
from .stubs import category_to_funcs

pytestmark = pytest.mark.ci

UnaryCheck = Callable[[float], bool]
BinaryCheck = Callable[[float, float], bool]


def make_strict_eq(v: float) -> UnaryCheck:
    if math.isnan(v):
        return math.isnan
    if v == 0:
        if ph.is_pos_zero(v):
            return ph.is_pos_zero
        else:
            return ph.is_neg_zero

    def strict_eq(i: float) -> bool:
        return i == v

    return strict_eq


def make_neq(v: float) -> UnaryCheck:
    eq = make_strict_eq(v)

    def neq(i: float) -> bool:
        return not eq(i)

    return neq


def make_rough_eq(v: float) -> UnaryCheck:
    assert math.isfinite(v)  # sanity check

    def rough_eq(i: float) -> bool:
        return math.isclose(i, v, abs_tol=0.01)

    return rough_eq


def make_gt(v: float) -> UnaryCheck:
    assert not math.isnan(v)  # sanity check

    def gt(i: float) -> bool:
        return i > v

    return gt


def make_lt(v: float) -> UnaryCheck:
    assert not math.isnan(v)  # sanity check

    def lt(i: float) -> bool:
        return i < v

    return lt


def make_or(cond1: UnaryCheck, cond2: UnaryCheck) -> UnaryCheck:
    def or_(i: float) -> bool:
        return cond1(i) or cond2(i)

    return or_


def make_and(cond1: UnaryCheck, cond2: UnaryCheck) -> UnaryCheck:
    def and_(i: float) -> bool:
        return cond1(i) or cond2(i)

    return and_


def make_not_cond(cond: UnaryCheck) -> UnaryCheck:
    def not_cond(i: float) -> bool:
        return not cond(i)

    return not_cond


def absify_cond(cond: UnaryCheck) -> UnaryCheck:
    def abs_cond(i: float) -> bool:
        return cond(abs(i))

    return abs_cond


repr_to_value = {
    "NaN": float("nan"),
    "infinity": float("inf"),
    "0": 0.0,
    "1": 1.0,
}
r_value = re.compile(r"([+-]?)(.+)")
r_pi = re.compile(r"(\d?)π(?:/(\d))?")


@dataclass
class ValueParseError(ValueError):
    value: str


def parse_value(value_str: str) -> float:
    m = r_value.match(value_str)
    if m is None:
        raise ValueParseError(value_str)
    if pi_m := r_pi.match(m.group(2)):
        value = math.pi
        if numerator := pi_m.group(1):
            value *= int(numerator)
        if denominator := pi_m.group(2):
            value /= int(denominator)
    else:
        value = repr_to_value[m.group(2)]
    if sign := m.group(1):
        if sign == "-":
            value *= -1
    return value


r_code = re.compile(r"``([^\s]+)``")
r_approx_value = re.compile(
    rf"an implementation-dependent approximation to {r_code.pattern}"
)


def parse_inline_code(inline_code: str) -> float:
    if m := r_code.match(inline_code):
        return parse_value(m.group(1))
    else:
        raise ValueParseError(inline_code)


r_not = re.compile("not (.+)")
r_equal_to = re.compile(f"equal to {r_code.pattern}")
r_array_element = re.compile(r"``([+-]?)x([12])_i``")
r_either_code = re.compile(f"either {r_code.pattern} or {r_code.pattern}")
r_gt = re.compile(f"greater than {r_code.pattern}")
r_lt = re.compile(f"less than {r_code.pattern}")


class FromDtypeFunc(Protocol):
    def __call__(self, dtype: DataType, **kw) -> st.SearchStrategy[float]:
        ...


@dataclass
class BoundFromDtype(FromDtypeFunc):
    kwargs: Dict[str, Any] = field(default_factory=dict)
    filter_: Optional[Callable[[Array], bool]] = None
    base_func: Optional[FromDtypeFunc] = None

    def __add__(self, other: BoundFromDtype) -> BoundFromDtype:
        for k in self.kwargs.keys():
            if k in other.kwargs.keys():
                assert self.kwargs[k] == other.kwargs[k]  # sanity check
        kwargs = {**self.kwargs, **other.kwargs}

        if self.filter_ is not None and other.filter_ is not None:
            filter_ = lambda i: self.filter_(i) and other.filter_(i)
        else:
            if self.filter_ is not None:
                filter_ = self.filter_
            elif other.filter_ is not None:
                filter_ = other.filter_
            else:
                filter_ = None

        # sanity check
        assert not (self.base_func is not None and other.base_func is not None)
        if self.base_func is not None:
            base_func = self.base_func
        elif other.base_func is not None:
            base_func = other.base_func
        else:
            base_func = None

        return BoundFromDtype(kwargs, filter_, base_func)

    def __call__(self, dtype: DataType, **kw) -> st.SearchStrategy[float]:
        assert len(kw) == 0  # sanity check
        from_dtype = self.base_func or xps.from_dtype
        strat = from_dtype(dtype, **self.kwargs)
        if self.filter_ is not None:
            strat = strat.filter(self.filter_)
        return strat


def wrap_strat_as_from_dtype(strat: st.SearchStrategy[float]) -> FromDtypeFunc:
    def from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
        assert kw == {}  # sanity check
        return strat

    return from_dtype


def parse_cond(cond_str: str) -> Tuple[UnaryCheck, str, FromDtypeFunc]:
    if m := r_not.match(cond_str):
        cond_str = m.group(1)
        not_cond = True
    else:
        not_cond = False

    kwargs = {}
    filter_ = None
    from_dtype = None  # type: ignore
    strat = None
    if m := r_code.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_strict_eq(value)
        expr_template = "{} == " + m.group(1)
        if not not_cond:
            strat = st.just(value)
    elif m := r_equal_to.match(cond_str):
        value = parse_value(m.group(1))
        assert not math.isnan(value)  # sanity check
        cond = lambda i: i == value
        expr_template = "{} == " + m.group(1)
    elif m := r_gt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_gt(value)
        expr_template = "{} > " + m.group(1)
        if not not_cond:
            kwargs = {"min_value": value, "exclude_min": True}
    elif m := r_lt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_lt(value)
        expr_template = "{} < " + m.group(1)
        if not not_cond:
            kwargs = {"max_value": value, "exclude_max": True}
    elif m := r_either_code.match(cond_str):
        v1 = parse_value(m.group(1))
        v2 = parse_value(m.group(2))
        cond = make_or(make_strict_eq(v1), make_strict_eq(v2))
        expr_template = "{} == " + m.group(1) + " or {} == " + m.group(2)
        if not not_cond:
            strat = st.sampled_from([v1, v2])
    elif cond_str in ["finite", "a finite number"]:
        cond = math.isfinite
        expr_template = "isfinite({})"
        if not not_cond:
            kwargs = {"allow_nan": False, "allow_infinity": False}
    elif cond_str in "a positive (i.e., greater than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i > 0
        expr_template = "isfinite({}) and {} > 0"
        if not not_cond:
            kwargs = {
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": 0,
                "exclude_min": True,
            }
    elif cond_str == "a negative (i.e., less than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i < 0
        expr_template = "isfinite({}) and {} < 0"
        if not not_cond:
            kwargs = {
                "allow_nan": False,
                "allow_infinity": False,
                "max_value": 0,
                "exclude_max": True,
            }
    elif cond_str == "positive":
        cond = lambda i: math.copysign(1, i) == 1
        expr_template = "copysign(1, {}) == 1"
        if not not_cond:
            # We assume (positive) zero is special cased seperately
            kwargs = {"min_value": 0, "exclude_min": True}
    elif cond_str == "negative":
        cond = lambda i: math.copysign(1, i) == -1
        expr_template = "copysign(1, {}) == -1"
        if not not_cond:
            # We assume (negative) zero is special cased seperately
            kwargs = {"max_value": 0, "exclude_max": True}
    elif "nonzero finite" in cond_str:
        cond = lambda i: math.isfinite(i) and i != 0
        expr_template = "isfinite({}) and {} != 0"
        if not not_cond:
            kwargs = {"allow_nan": False, "allow_infinity": False}
            filter_ = lambda n: n != 0
    elif cond_str == "an integer value":
        cond = lambda i: i.is_integer()
        expr_template = "{}.is_integer()"
        if not not_cond:
            from_dtype = integers_from_dtype  # type: ignore
    elif cond_str == "an odd integer value":
        cond = lambda i: i.is_integer() and i % 2 == 1
        expr_template = "{}.is_integer() and {} % 2 == 1"
        if not not_cond:

            def from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
                return integers_from_dtype(dtype, **kw).filter(lambda n: n % 2 == 1)

    else:
        raise ValueParseError(cond_str)

    if strat is not None:
        # sanity checks
        assert not not_cond
        assert kwargs == {}
        assert filter_ is None
        assert from_dtype is None
        return cond, expr_template, wrap_strat_as_from_dtype(strat)

    if not_cond:
        expr_template = f"not {expr_template}"
        cond = make_not_cond(cond)
        kwargs = {}
        filter_ = cond
    assert kwargs is not None
    return cond, expr_template, BoundFromDtype(kwargs, filter_, from_dtype)


def parse_result(result_str: str) -> Tuple[UnaryCheck, str]:
    if m := r_code.match(result_str):
        value = parse_value(m.group(1))
        check_result = make_strict_eq(value)  # type: ignore
        expr = m.group(1)
    elif m := r_approx_value.match(result_str):
        value = parse_value(m.group(1))
        check_result = make_rough_eq(value)  # type: ignore
        repr_ = m.group(1).replace("π", "pi")  # for pytest param names
        expr = f"roughly {repr_}"
    elif "positive" in result_str:

        def check_result(result: float) -> bool:
            if math.isnan(result):
                # The sign of NaN is out-of-scope
                return True
            return math.copysign(1, result) == 1

        expr = "+"
    elif "negative" in result_str:

        def check_result(result: float) -> bool:
            if math.isnan(result):
                # The sign of NaN is out-of-scope
                return True
            return math.copysign(1, result) == -1

        expr = "-"
    else:
        raise ValueParseError(result_str)

    return check_result, expr


class Case(Protocol):
    cond_expr: str
    result_expr: str

    def cond(self, *args) -> bool:
        ...

    def check_result(self, *args) -> bool:
        ...

    def __str__(self) -> str:
        return f"{self.cond_expr} -> {self.result_expr}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{self}>)"


class UnaryCond(Protocol):
    def __call__(self, i: float) -> bool:
        ...


class UnaryResultCheck(Protocol):
    def __call__(self, i: float, result: float) -> bool:
        ...


@dataclass(repr=False)
class UnaryCase(Case):
    cond_expr: str
    result_expr: str
    cond_from_dtype: FromDtypeFunc
    cond: UnaryCheck
    check_result: UnaryResultCheck

    @classmethod
    def from_strings(cls, cond_str: str, result_str: str):
        cond, cond_expr_template, cond_from_dtype = parse_cond(cond_str)
        cond_expr = cond_expr_template.replace("{}", "x_i")
        _check_result, result_expr = parse_result(result_str)

        def check_result(i: float, result: float) -> bool:
            return _check_result(result)

        return cls(
            cond_expr=cond_expr,
            cond=cond,
            cond_from_dtype=cond_from_dtype,
            result_expr=result_expr,
            check_result=check_result,
        )


r_unary_case = re.compile("If ``x_i`` is (.+), the result is (.+)")
r_even_int_round_case = re.compile(
    "If two integers are equally close to ``x_i``, "
    "the result is the even integer closest to ``x_i``"
)


def trailing_halves_from_dtype(dtype: DataType):
    m, M = dh.dtype_ranges[dtype]
    return st.integers(math.ceil(m) // 2, math.floor(M) // 2).map(lambda n: n * 0.5)


even_int_round_case = UnaryCase(
    cond_expr="i % 0.5 == 0",
    cond=lambda i: i % 0.5 == 0,
    cond_from_dtype=trailing_halves_from_dtype,
    result_expr="Decimal(i).to_integral_exact(ROUND_HALF_EVEN)",
    check_result=lambda i, result: (
        result == float(Decimal(i).to_integral_exact(ROUND_HALF_EVEN))
    ),
)


def parse_unary_docstring(docstring: str) -> List[UnaryCase]:
    match = r_special_cases.search(docstring)
    if match is None:
        return []
    lines = match.group(1).split("\n")[:-1]
    cases = []
    for line in lines:
        if m := r_case.match(line):
            case = m.group(1)
        else:
            warn(f"line not machine-readable: '{line}'")
            continue
        if m := r_unary_case.search(case):
            try:
                case = UnaryCase.from_strings(*m.groups())
            except ValueParseError as e:
                warn(f"not machine-readable: '{e.value}'")
                continue
            cases.append(case)
        elif m := r_even_int_round_case.search(case):
            cases.append(even_int_round_case)
        else:
            if not r_remaining_case.search(case):
                warn(f"case not machine-readable: '{case}'")
    return cases


class BinaryCond(Protocol):
    def __call__(self, i1: float, i2: float) -> bool:
        ...


class BinaryResultCheck(Protocol):
    def __call__(self, i1: float, i2: float, result: float) -> bool:
        ...


@dataclass(repr=False)
class BinaryCase(Case):
    cond_expr: str
    result_expr: str
    x1_cond_from_dtype: FromDtypeFunc
    x2_cond_from_dtype: FromDtypeFunc
    cond: BinaryCond
    check_result: BinaryResultCheck


r_special_cases = re.compile(
    r"\*\*Special [Cc]ases\*\*(?:\n.*)+"
    r"For floating-point operands,\n+"
    r"((?:\s*-\s*.*\n)+)"
)
r_case = re.compile(r"\s+-\s*(.*)\.\n?")
r_binary_case = re.compile("If (.+), the result (.+)")
r_remaining_case = re.compile("In the remaining cases.+")
r_cond_sep = re.compile(r"(?<!``x1_i``),? and |(?<!i\.e\.), ")
r_cond = re.compile("(.+) (?:is|have) (.+)")
r_input_is_array_element = re.compile(
    f"{r_array_element.pattern} is {r_array_element.pattern}"
)
r_both_inputs_are_value = re.compile("are both (.+)")
r_element = re.compile("x([12])_i")
r_input = re.compile(rf"``{r_element.pattern}``")
r_abs_input = re.compile(rf"``abs\({r_element.pattern}\)``")
r_and_input = re.compile(f"{r_input.pattern} and {r_input.pattern}")
r_or_input = re.compile(f"either {r_input.pattern} or {r_input.pattern}")
r_result = re.compile(r"(?:is|has a) (.+)")


class BinaryCondArg(Enum):
    FIRST = auto()
    SECOND = auto()
    BOTH = auto()
    EITHER = auto()

    @classmethod
    def from_x_no(cls, string):
        if string == "1":
            return cls.FIRST
        elif string == "2":
            return cls.SECOND
        else:
            raise ValueError(f"{string=} not '1' or '2'")


def noop(n: float) -> float:
    return n


def make_binary_cond(
    cond_arg: BinaryCondArg,
    unary_cond: UnaryCheck,
    *,
    input_wrapper: Optional[Callable[[float], float]] = None,
) -> BinaryCond:
    if input_wrapper is None:
        input_wrapper = noop

    if cond_arg == BinaryCondArg.FIRST:

        def partial_cond(i1: float, i2: float) -> bool:
            return unary_cond(input_wrapper(i1))

    elif cond_arg == BinaryCondArg.SECOND:

        def partial_cond(i1: float, i2: float) -> bool:
            return unary_cond(input_wrapper(i2))

    elif cond_arg == BinaryCondArg.BOTH:

        def partial_cond(i1: float, i2: float) -> bool:
            return unary_cond(input_wrapper(i1)) and unary_cond(input_wrapper(i2))

    else:

        def partial_cond(i1: float, i2: float) -> bool:
            return unary_cond(input_wrapper(i1)) or unary_cond(input_wrapper(i2))

    return partial_cond


def make_eq_input_check_result(
    eq_to: BinaryCondArg, *, eq_neg: bool = False
) -> BinaryResultCheck:
    if eq_neg:
        input_wrapper = lambda i: -i
    else:
        input_wrapper = noop

    if eq_to == BinaryCondArg.FIRST:

        def check_result(i1: float, i2: float, result: float) -> bool:
            eq = make_strict_eq(input_wrapper(i1))
            return eq(result)

    elif eq_to == BinaryCondArg.SECOND:

        def check_result(i1: float, i2: float, result: float) -> bool:
            eq = make_strict_eq(input_wrapper(i2))
            return eq(result)

    else:
        raise ValueError(f"{eq_to=} must be FIRST or SECOND")


def integers_from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
    for k in kw.keys():
        # sanity check
        assert k in ["min_value", "max_value", "exclude_min", "exclude_max"]
    m, M = dh.dtype_ranges[dtype]
    if "min_value" in kw.keys():
        m = kw["min_value"]
        if "exclude_min" in kw.keys():
            m += 1
    if "max_value" in kw.keys():
        M = kw["max_value"]
        if "exclude_max" in kw.keys():
            M -= 1
    return st.integers(math.ceil(m), math.floor(M)).map(float)


def parse_binary_case(case_str: str) -> BinaryCase:
    case_m = r_binary_case.match(case_str)
    assert case_m is not None  # sanity check
    cond_strs = r_cond_sep.split(case_m.group(1))

    partial_conds = []
    partial_exprs = []
    x1_cond_from_dtypes = []
    x2_cond_from_dtypes = []
    for cond_str in cond_strs:
        if m := r_input_is_array_element.match(cond_str):
            in_sign, in_no, other_sign, other_no = m.groups()
            assert in_sign == "" and other_no != in_no  # sanity check
            partial_expr = f"{in_sign}x{in_no}_i == {other_sign}x{other_no}_i"
            input_wrapper = lambda i: -i if other_sign == "-" else noop
            shared_from_dtype = lambda d, **kw: st.shared(
                xps.from_dtype(d, **kw), key=cond_str
            )

            if other_no == "1":

                def partial_cond(i1: float, i2: float) -> bool:
                    eq = make_strict_eq(input_wrapper(i1))
                    return eq(i2)

                _x2_cond_from_dtype = shared_from_dtype  # type: ignore

                def _x1_cond_from_dtype(dtype, **kw) -> st.SearchStrategy[float]:
                    return shared_from_dtype(dtype, **kw).map(input_wrapper)

            elif other_no == "2":

                def partial_cond(i1: float, i2: float) -> bool:
                    eq = make_strict_eq(input_wrapper(i2))
                    return eq(i1)

                _x1_cond_from_dtype = shared_from_dtype  # type: ignore

                def _x2_cond_from_dtype(dtype, **kw) -> st.SearchStrategy[float]:
                    return shared_from_dtype(dtype, **kw).map(input_wrapper)

            else:
                raise ValueParseError(cond_str)

            x1_cond_from_dtypes.append(BoundFromDtype(base_func=_x1_cond_from_dtype))
            x2_cond_from_dtypes.append(BoundFromDtype(base_func=_x2_cond_from_dtype))

        elif m := r_both_inputs_are_value.match(cond_str):
            unary_cond, expr_template, cond_from_dtype = parse_cond(m.group(1))
            left_expr = expr_template.replace("{}", "x1_i")
            right_expr = expr_template.replace("{}", "x2_i")
            partial_expr = f"({left_expr}) and ({right_expr})"
            partial_cond = make_binary_cond(  # type: ignore
                BinaryCondArg.BOTH, unary_cond
            )
            x1_cond_from_dtypes.append(cond_from_dtype)
            x2_cond_from_dtypes.append(cond_from_dtype)
        else:
            cond_m = r_cond.match(cond_str)
            if cond_m is None:
                raise ValueParseError(cond_str)
            input_str, value_str = cond_m.groups()

            if value_str == "the same mathematical sign":
                partial_expr = "copysign(1, x1_i) == copysign(1, x2_i)"

                def partial_cond(i1: float, i2: float) -> bool:
                    return math.copysign(1, i1) == math.copysign(1, i2)

            elif value_str == "different mathematical signs":
                partial_expr = "copysign(1, x1_i) != copysign(1, x2_i)"

                def partial_cond(i1: float, i2: float) -> bool:
                    return math.copysign(1, i1) != math.copysign(1, i2)

            else:
                unary_cond, expr_template, cond_from_dtype = parse_cond(value_str)
                # Do not define partial_cond via the def keyword, as one
                # partial_cond definition can mess up previous definitions
                # in the partial_conds list. This is a hard-limitation of
                # using local functions with the same name and that use the same
                # outer variables (i.e. unary_cond).
                input_wrapper = None
                if m := r_input.match(input_str):
                    x_no = m.group(1)
                    partial_expr = expr_template.replace("{}", f"x{x_no}_i")
                    cond_arg = BinaryCondArg.from_x_no(x_no)
                elif m := r_abs_input.match(input_str):
                    x_no = m.group(1)
                    partial_expr = expr_template.replace("{}", f"abs(x{x_no}_i)")
                    cond_arg = BinaryCondArg.from_x_no(x_no)
                    input_wrapper = abs
                elif r_and_input.match(input_str):
                    left_expr = expr_template.replace("{}", "x1_i")
                    right_expr = expr_template.replace("{}", "x2_i")
                    partial_expr = f"({left_expr}) and ({right_expr})"
                    cond_arg = BinaryCondArg.BOTH
                elif r_or_input.match(input_str):
                    left_expr = expr_template.replace("{}", "x1_i")
                    right_expr = expr_template.replace("{}", "x2_i")
                    partial_expr = f"{left_expr} or {right_expr}"
                    if len(cond_strs) != 1:
                        partial_expr = f"({partial_expr})"
                    cond_arg = BinaryCondArg.EITHER
                else:
                    raise ValueParseError(input_str)
                partial_cond = make_binary_cond(  # type: ignore
                    cond_arg, unary_cond, input_wrapper=input_wrapper
                )
                if cond_arg == BinaryCondArg.FIRST:
                    x1_cond_from_dtypes.append(cond_from_dtype)
                elif cond_arg == BinaryCondArg.SECOND:
                    x2_cond_from_dtypes.append(cond_from_dtype)
                elif cond_arg == BinaryCondArg.BOTH:
                    x1_cond_from_dtypes.append(cond_from_dtype)
                    x2_cond_from_dtypes.append(cond_from_dtype)
                else:
                    # For "either x1_i or x2_i is <condition>" cases, we want to
                    # test three scenarios:
                    #
                    # 1. x1_i is <condition>
                    # 2. x2_i is <condition>
                    # 3. x1_i AND x2_i is <condition>
                    #
                    # This is achieved by a shared base strategy that picks one
                    # of these scenarios to determine whether each array will
                    # use either cond_from_dtype() (i.e. meet the condition), or
                    # simply xps.from_dtype() (i.e. be any value).

                    use_x1_or_x2_strat = st.shared(
                        st.sampled_from([(True, False), (False, True), (True, True)])
                    )

                    def _x1_cond_from_dtype(dtype) -> st.SearchStrategy[float]:
                        return use_x1_or_x2_strat.flatmap(
                            lambda t: cond_from_dtype(dtype)
                            if t[0]
                            else xps.from_dtype(dtype)
                        )

                    def _x2_cond_from_dtype(dtype) -> st.SearchStrategy[float]:
                        return use_x1_or_x2_strat.flatmap(
                            lambda t: cond_from_dtype(dtype)
                            if t[1]
                            else xps.from_dtype(dtype)
                        )

                    x1_cond_from_dtypes.append(_x1_cond_from_dtype)
                    x2_cond_from_dtypes.append(_x2_cond_from_dtype)

        partial_conds.append(partial_cond)
        partial_exprs.append(partial_expr)

    result_m = r_result.match(case_m.group(2))
    if result_m is None:
        raise ValueParseError(case_m.group(2))
    result_str = result_m.group(1)
    if m := r_array_element.match(result_str):
        sign, x_no = m.groups()
        result_expr = f"{sign}x{x_no}_i"
        check_result = make_eq_input_check_result(  # type: ignore
            BinaryCondArg.from_x_no(x_no), eq_neg=sign == "-"
        )
    else:
        _check_result, result_expr = parse_result(result_m.group(1))

        def check_result(i1: float, i2: float, result: float) -> bool:
            return _check_result(result)

    cond_expr = " and ".join(partial_exprs)

    def cond(i1: float, i2: float) -> bool:
        return all(pc(i1, i2) for pc in partial_conds)

    if len(x1_cond_from_dtypes) == 0:
        x1_cond_from_dtype = xps.from_dtype
    elif len(x1_cond_from_dtypes) == 1:
        x1_cond_from_dtype = x1_cond_from_dtypes[0]
    else:
        if not all(isinstance(fd, BoundFromDtype) for fd in x1_cond_from_dtypes):
            raise ValueParseError(case_str)
        x1_cond_from_dtype = sum(x1_cond_from_dtypes, start=BoundFromDtype())
    if len(x2_cond_from_dtypes) == 0:
        x2_cond_from_dtype = xps.from_dtype
    elif len(x2_cond_from_dtypes) == 1:
        x2_cond_from_dtype = x2_cond_from_dtypes[0]
    else:
        if not all(isinstance(fd, BoundFromDtype) for fd in x2_cond_from_dtypes):
            raise ValueParseError(case_str)
        x2_cond_from_dtype = sum(x2_cond_from_dtypes, start=BoundFromDtype())

    return BinaryCase(
        cond_expr=cond_expr,
        cond=cond,
        x1_cond_from_dtype=x1_cond_from_dtype,
        x2_cond_from_dtype=x2_cond_from_dtype,
        result_expr=result_expr,
        check_result=check_result,
    )


r_redundant_case = re.compile("result.+determined by the rule already stated above")


def parse_binary_docstring(docstring: str) -> List[BinaryCase]:
    match = r_special_cases.search(docstring)
    if match is None:
        return []
    lines = match.group(1).split("\n")[:-1]
    cases = []
    for line in lines:
        if m := r_case.match(line):
            case_str = m.group(1)
        else:
            warn(f"line not machine-readable: '{line}'")
            continue
        if r_redundant_case.search(case_str):
            continue
        if m := r_binary_case.match(case_str):
            try:
                case = parse_binary_case(case_str)
                cases.append(case)
            except ValueParseError as e:
                warn(f"not machine-readable: '{e.value}'")
        else:
            if not r_remaining_case.match(case_str):
                warn(f"case not machine-readable: '{case_str}'")
    return cases


unary_params = []
binary_params = []
for stub in category_to_funcs["elementwise"]:
    if stub.__doc__ is None:
        warn(f"{stub.__name__}() stub has no docstring")
        continue
    marks = []
    try:
        func = getattr(xp, stub.__name__)
    except AttributeError:
        marks.append(
            pytest.mark.skip(reason=f"{stub.__name__} not found in array module")
        )
        func = None
    sig = inspect.signature(stub)
    param_names = list(sig.parameters.keys())
    if len(sig.parameters) == 0:
        warn(f"{func=} has no parameters")
        continue
    if param_names[0] == "x":
        if cases := parse_unary_docstring(stub.__doc__):
            for case in cases:
                id_ = f"{stub.__name__}({case.cond_expr}) -> {case.result_expr}"
                p = pytest.param(stub.__name__, func, case, id=id_)
                unary_params.append(p)
        continue
    if len(sig.parameters) == 1:
        warn(f"{func=} has one parameter '{param_names[0]}' which is not named 'x'")
        continue
    if param_names[0] == "x1" and param_names[1] == "x2":
        if cases := parse_binary_docstring(stub.__doc__):
            for case in cases:
                id_ = f"{stub.__name__}({case.cond_expr}) -> {case.result_expr}"
                p = pytest.param(stub.__name__, func, case, id=id_)
                binary_params.append(p)
        continue
    else:
        warn(
            f"{func=} starts with two parameters '{param_names[0]}' and "
            f"'{param_names[1]}', which are not named 'x1' and 'x2'"
        )


# good_example is a flag that tells us whether Hypothesis generated an array
# with at least on element that is special-cased. We reject the example when
# its False - Hypothesis will complain if we reject too many examples, thus
# indicating we should modify the array strategy being used.


@pytest.mark.parametrize("func_name, func, case", unary_params)
@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_unary(func_name, func, case, x, data):
    set_idx = data.draw(
        xps.indices(x.shape, max_dims=0, allow_ellipsis=False), label="set idx"
    )
    set_value = data.draw(case.cond_from_dtype(x.dtype), label="set value")
    x[set_idx] = set_value
    note(f"{x=}")

    res = func(x)

    good_example = False
    for idx in sh.ndindex(res.shape):
        in_ = float(x[idx])
        if case.cond(in_):
            good_example = True
            out = float(res[idx])
            f_in = f"{sh.fmt_idx('x', idx)}={in_}"
            f_out = f"{sh.fmt_idx('out', idx)}={out}"
            assert case.check_result(in_, out), (
                f"{f_out} not good [{func_name}()]\n" f"{case}\n" f"{f_in}"
            )
            break
    assume(good_example)


x1_strat, x2_strat = hh.two_mutual_arrays(
    dtypes=dh.float_dtypes,
    two_shapes=hh.mutually_broadcastable_shapes(2, min_side=1),
)


@pytest.mark.parametrize("func_name, func, case", binary_params)
@given(x1=x1_strat, x2=x2_strat, data=st.data())
def test_binary(func_name, func, case, x1, x2, data):
    result_shape = sh.broadcast_shapes(x1.shape, x2.shape)
    all_indices = list(sh.iter_indices(x1.shape, x2.shape, result_shape))

    indices_strat = st.shared(st.sampled_from(all_indices))
    set_x1_idx = data.draw(indices_strat.map(lambda t: t[0]), label="set x1 idx")
    set_x1_value = data.draw(case.x1_cond_from_dtype(x1.dtype), label="set x1 value")
    x1[set_x1_idx] = set_x1_value
    note(f"{x1=}")
    set_x2_idx = data.draw(indices_strat.map(lambda t: t[1]), label="set x2 idx")
    set_x2_value = data.draw(case.x2_cond_from_dtype(x2.dtype), label="set x2 value")
    x2[set_x2_idx] = set_x2_value
    note(f"{x2=}")

    res = func(x1, x2)
    # sanity check
    ph.assert_result_shape(func_name, [x1.shape, x2.shape], res.shape, result_shape)

    good_example = False
    for l_idx, r_idx, o_idx in all_indices:
        l = float(x1[l_idx])
        r = float(x2[r_idx])
        if case.cond(l, r):
            good_example = True
            o = float(res[o_idx])
            f_left = f"{sh.fmt_idx('x1', l_idx)}={l}"
            f_right = f"{sh.fmt_idx('x2', r_idx)}={r}"
            f_out = f"{sh.fmt_idx('out', o_idx)}={o}"
            assert case.check_result(l, r, o), (
                f"{f_out} not good [{func_name}()]\n" f"{case}\n" f"{f_left}, {f_right}"
            )
            break
    assume(good_example)
