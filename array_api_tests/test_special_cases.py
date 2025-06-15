"""
Tests for special cases.

Most test cases for special casing are built on runtime via the parametrized
tests test_unary/test_binary/test_iop. Most of this file consists of utility
classes and functions, all bought together to create the test cases (pytest
params), to finally be run through generalised test logic.

TODO: test integer arrays for relevant special cases
"""
# We use __future__ for forward reference type hints - this will work for even py3.8.0
# See https://stackoverflow.com/a/33533514/5193926
from __future__ import annotations

import inspect
import math
import operator
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Literal
from warnings import warn, filterwarnings, catch_warnings

import pytest
from hypothesis import given, note, settings, assume
from hypothesis import strategies as st
from hypothesis.errors import NonInteractiveExampleWarning

from array_api_tests.typing import Array, DataType

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xp, xps
from .stubs import category_to_funcs

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


def make_strict_neq(v: float) -> UnaryCheck:
    strict_eq = make_strict_eq(v)

    def strict_neq(i: float) -> bool:
        return not strict_eq(i)

    return strict_neq


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
    "False": 0.0,
    "True": 1.0,
}
r_value = re.compile(r"([+-]?)(.+)")
r_pi = re.compile(r"(\d?)π(?:/(\d))?")


@dataclass
class ParseError(ValueError):
    value: str


def parse_value(value_str: str) -> float:
    """
    Parses a value string to return a float, e.g.

        >>> parse_value('1')
        1.
        >>> parse_value('-infinity')
        -float('inf')
        >>> parse_value('3π/4')
        2.356194490192345

    """
    m = r_value.match(value_str)
    if m is None:
        raise ParseError(value_str)
    if pi_m := r_pi.match(m.group(2)):
        value = math.pi
        if numerator := pi_m.group(1):
            value *= int(numerator)
        if denominator := pi_m.group(2):
            value /= int(denominator)
    else:
        try:
            value = repr_to_value[m.group(2)]
        except KeyError as e:
            raise ParseError(value_str) from e
    if sign := m.group(1):
        if sign == "-":
            value *= -1
    return value


r_code = re.compile(r"``([^\s]+)``")
r_approx_value = re.compile(
    rf"an implementation-dependent approximation to {r_code.pattern}"
)
r_not = re.compile("not (.+)")
r_equal_to = re.compile(f"equal to {r_code.pattern}")
r_array_element = re.compile(r"``([+-]?)x([12])_i``")
r_either_code = re.compile(f"either {r_code.pattern} or {r_code.pattern}")
r_gt = re.compile(f"greater than {r_code.pattern}")
r_lt = re.compile(f"less than {r_code.pattern}")


class FromDtypeFunc(Protocol):
    """
    Type hint for functions that return an elements strategy for arrays of the
    given dtype, e.g. xps.from_dtype().
    """

    def __call__(self, dtype: DataType, **kw) -> st.SearchStrategy[float]:
        ...


@dataclass
class BoundFromDtype(FromDtypeFunc):
    """
    A xps.from_dtype()-like callable with bounded kwargs, filters and base function.

    We can bound:

     1. Keyword arguments that xps.from_dtype() can use, e.g.

        >>> from_dtype = BoundFromDtype(kwargs={'min_value': 0, 'allow_infinity': False})
        >>> strategy = from_dtype(xp.float64)

        is equivalent to

        >>> strategy = xps.from_dtype(xp.float64, min_value=0, allow_infinity=False)

        i.e. a strategy that generates finite floats above 0

     2. Functions that filter the elements strategy that xps.from_dtype() returns, e.g.

        >>> from_dtype = BoundFromDtype(filter=lambda i: i != 0)
        >>> strategy = from_dtype(xp.float64)

        is equivalent to

        >>> strategy = xps.from_dtype(xp.float64).filter(lambda i: i != 0)

        i.e. a strategy that generates any float except +0 and -0

     3. The underlying function that returns an elements strategy from a dtype, e.g.

        >>> from_dtype = BoundFromDtype(
        ...     from_dtype=lambda d: st.integers(
        ...         math.ceil(xp.finfo(d).min), math.floor(xp.finfo(d).max)
        ...     )
        ... )
        >>> strategy = from_dtype(xp.float64)

        is equivalent to

        >>> strategy = st.integers(
        ...     math.ceil(xp.finfo(xp.float64).min), math.floor(xp.finfo(xp.float64).max)
        ... )

        i.e. a strategy that generates integers (within the dtype's range)

    This is useful to avoid translating special case conditions into either a
    dict, filter or "base func", and instead allows us to generalise these three
    components into a callable equivalent of xps.from_dtype().

    Additionally, BoundFromDtype instances can be added together. This allows us
    to keep parsing each condition individually - so we don't need to duplicate
    complicated parsing code - as ultimately we can represent (and subsequently
    test for) special cases which have more than one condition per array, e.g.

        "If x1_i is greater than 0 and x1_i is not 42, ..."

        could be translated as

        >>> gt_0_from_dtype = BoundFromDtype(kwargs={'min_value': 0})
        >>> not_42_from_dtype = BoundFromDtype(filter=lambda i: i != 42)
        >>> gt_0_from_dtype + not_42_from_dtype
        BoundFromDtype(kwargs={'min_value': 0}, filter=<lambda>(i))

    """

    kwargs: Dict[str, Any] = field(default_factory=dict)
    filter_: Optional[Callable[[Array], bool]] = None
    base_func: Optional[FromDtypeFunc] = None

    def __call__(self, dtype: DataType, **kw) -> st.SearchStrategy[float]:
        assert len(kw) == 0  # sanity check
        from_dtype = self.base_func or hh.from_dtype
        strat = from_dtype(dtype, **self.kwargs)
        if self.filter_ is not None:
            strat = strat.filter(self.filter_)
        return strat

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


def wrap_strat_as_from_dtype(strat: st.SearchStrategy[float]) -> FromDtypeFunc:
    """
    Wraps an elements strategy as a xps.from_dtype()-like function
    """

    def from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
        assert len(kw) == 0  # sanity check
        return strat

    return from_dtype


def parse_cond(cond_str: str) -> Tuple[UnaryCheck, str, BoundFromDtype]:
    """
    Parses a Sphinx-formatted condition string to return:

     1. A function which takes an input and returns True if it meets the
        condition, otherwise False.
     2. A string template for expressing the condition.
     3. A xps.from_dtype()-like function which returns a strategy that generates
        elements that meet the condition.

    e.g.

        >>> cond, expr_template, from_dtype = parse_cond('greater than ``0``')
        >>> cond(42)
        True
        >>> cond(-123)
        False
        >>> expr_template.replace('{}', 'x_i')
        'x_i > 0'
        >>> strategy = from_dtype(xp.float64)
        >>> for _ in range(5):
        ...     print(strategy.example())
        1.
        0.1
        1.7976931348623155e+179
        inf
        124.978

    """
    # We first identify whether the condition starts with "not". If so, we note
    # this but parse the condition as if it was not negated.
    if m := r_not.match(cond_str):
        cond_str = m.group(1)
        not_cond = True
    else:
        not_cond = False

    # We parse the condition to identify the condition function, expression
    # template, and xps.from_dtype()-like condition strategy.
    kwargs = {}
    filter_ = None
    from_dtype = None  # type: ignore
    if m := r_code.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_strict_eq(value)
        expr_template = "{} is " + m.group(1)
        from_dtype = wrap_strat_as_from_dtype(st.just(value))
    elif m := r_either_code.match(cond_str):
        v1 = parse_value(m.group(1))
        v2 = parse_value(m.group(2))
        cond = make_or(make_strict_eq(v1), make_strict_eq(v2))
        expr_template = "({} is " + m.group(1) + " or {} == " + m.group(2) + ")"
        from_dtype = wrap_strat_as_from_dtype(st.sampled_from([v1, v2]))
    elif m := r_equal_to.match(cond_str):
        value = parse_value(m.group(1))
        if math.isnan(value):
            raise ParseError(cond_str)
        cond = lambda i: i == value
        expr_template = "{} == " + m.group(1)
    elif m := r_gt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_gt(value)
        expr_template = "{} > " + m.group(1)
        kwargs = {"min_value": value, "exclude_min": True}
    elif m := r_lt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_lt(value)
        expr_template = "{} < " + m.group(1)
        kwargs = {"max_value": value, "exclude_max": True}
    elif cond_str in ["finite", "a finite number"]:
        cond = math.isfinite
        expr_template = "isfinite({})"
        kwargs = {"allow_nan": False, "allow_infinity": False}
    elif cond_str in "a positive (i.e., greater than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i > 0
        expr_template = "isfinite({}) and {} > 0"
        kwargs = {
            "allow_nan": False,
            "allow_infinity": False,
            "min_value": 0,
            "exclude_min": True,
        }
    elif cond_str == "a negative (i.e., less than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i < 0
        expr_template = "isfinite({}) and {} < 0"
        kwargs = {
            "allow_nan": False,
            "allow_infinity": False,
            "max_value": 0,
            "exclude_max": True,
        }
    elif cond_str == "positive":
        cond = lambda i: math.copysign(1, i) == 1
        expr_template = "copysign(1, {}) == 1"
        # We assume (positive) zero is special cased seperately
        kwargs = {"min_value": 0, "exclude_min": True}
    elif cond_str == "negative":
        cond = lambda i: math.copysign(1, i) == -1
        expr_template = "copysign(1, {}) == -1"
        # We assume (negative) zero is special cased seperately
        kwargs = {"max_value": 0, "exclude_max": True}
    elif "nonzero finite" in cond_str:
        cond = lambda i: math.isfinite(i) and i != 0
        expr_template = "isfinite({}) and {} != 0"
        kwargs = {"allow_nan": False, "allow_infinity": False}
        filter_ = lambda n: n != 0
    elif cond_str == "an integer value":
        cond = lambda i: i.is_integer()
        expr_template = "{}.is_integer()"
        from_dtype = integers_from_dtype  # type: ignore
    elif cond_str == "an odd integer value":
        cond = lambda i: i.is_integer() and i % 2 == 1
        expr_template = "{}.is_integer() and {} % 2 == 1"
        if not_cond:
            expr_template = f"({expr_template})"

        def from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
            return integers_from_dtype(dtype, **kw).filter(lambda n: n % 2 == 1)

    else:
        raise ParseError(cond_str)

    if not_cond:
        # We handle negated conitions by simply negating the condition function
        # and using it as a filter for xps.from_dtype() (or an equivalent).
        cond = make_not_cond(cond)
        expr_template = f"not {expr_template}"
        filter_ = cond
        return cond, expr_template, BoundFromDtype(filter_=filter_)
    else:
        return cond, expr_template, BoundFromDtype(kwargs, filter_, from_dtype)


def parse_result(result_str: str) -> Tuple[UnaryCheck, str]:
    """
    Parses a Sphinx-formatted result string to return:

     1. A function which takes an input and returns True if it is the expected
        result (or meets the condition of the expected result), otherwise False.
     2. A string that expresses the result.

    e.g.

        >>> check_result, expr = parse_result('``42``')
        >>> check_result(7)
        False
        >>> check_result(42)
        True
        >>> expr
        '42'

    """
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

        expr = "positive sign"
    elif "negative" in result_str:

        def check_result(result: float) -> bool:
            if math.isnan(result):
                # The sign of NaN is out-of-scope
                return True
            return math.copysign(1, result) == -1

        expr = "negative sign"
    else:
        raise ParseError(result_str)

    return check_result, expr


class Case(Protocol):
    cond_expr: str
    result_expr: str
    raw_case: Optional[str]

    def cond(self, *args) -> bool:
        ...

    def check_result(self, *args) -> bool:
        ...

    def __str__(self) -> str:
        return f"{self.cond_expr} -> {self.result_expr}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{self}>)"


r_case_block = re.compile(
    r"\*\*Special [Cc]ases\*\*\n+((?:(.*\n)+))\n+\s*"
    r"(?:.+\n--+)?(?:\.\. versionchanged.*)?"
)
r_case = re.compile(r"\s+-\s*(.*)\.")


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
    raw_case: Optional[str] = field(default=None)


r_unary_case = re.compile("If ``x_i`` is (.+), the result is (.+)")
r_already_int_case = re.compile(
    "If ``x_i`` is already integer-valued, the result is ``x_i``"
)
r_even_round_halves_case = re.compile(
    "If two integers are equally close to ``x_i``, "
    "the result is the even integer closest to ``x_i``"
)
r_nan_signbit = re.compile(
    "If ``x_i`` is ``NaN`` and the sign bit of ``x_i`` is ``(.+)``, "
    "the result is ``(.+)``"
)


def integers_from_dtype(dtype: DataType, **kw) -> st.SearchStrategy[float]:
    """
    Returns a strategy that generates float-casted integers within the bounds of dtype.
    """
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


def trailing_halves_from_dtype(dtype: DataType) -> st.SearchStrategy[float]:
    """
    Returns a strategy that generates floats that end with .5 and are within the
    bounds of dtype.
    """
    # We bound our base integers strategy to a range of values which should be
    # able to represent a decimal 5 when .5 is added or subtracted.
    if dtype == xp.float32:
        abs_max = 10**4
    else:
        abs_max = 10**16
    return st.sampled_from([0.5, -0.5]).flatmap(
        lambda half: st.integers(-abs_max, abs_max).map(lambda n: n + half)
    )


already_int_case = UnaryCase(
    cond_expr="x_i.is_integer()",
    cond=lambda i: i.is_integer(),
    cond_from_dtype=integers_from_dtype,
    result_expr="x_i",
    check_result=lambda i, result: i == result,
)
even_round_halves_case = UnaryCase(
    cond_expr="modf(i)[0] == 0.5",
    cond=lambda i: math.modf(i)[0] == 0.5,
    cond_from_dtype=trailing_halves_from_dtype,
    result_expr="Decimal(i).to_integral_exact(ROUND_HALF_EVEN)",
    check_result=lambda i, result: (
        result == float(Decimal(i).to_integral_exact(ROUND_HALF_EVEN))
    ),
)


def make_nan_signbit_case(signbit: Literal[0, 1], expected: bool) -> UnaryCase:
    if signbit:
        sign = -1
        nan_expr = "-NaN"
        float_arg = "-nan"
    else:
        sign = 1
        nan_expr = "+NaN"
        float_arg = "nan"

    return UnaryCase(
        cond_expr=f"x_i is {nan_expr}",
        cond=lambda i: math.isnan(i) and math.copysign(1, i) == sign,
        cond_from_dtype=lambda _: st.just(float(float_arg)),
        result_expr=str(expected),
        check_result=lambda _, result: result == float(expected),
    )


def make_unary_check_result(check_just_result: UnaryCheck) -> UnaryResultCheck:
    def check_result(i: float, result: float) -> bool:
        return check_just_result(result)

    return check_result


def parse_unary_case_block(case_block: str, func_name: str) -> List[UnaryCase]:
    """
    Parses a Sphinx-formatted docstring of a unary function to return a list of
    codified unary cases, e.g.

        >>> def sqrt(x):
        ...     '''
        ...     Calculates the square root
        ...
        ...     **Special Cases**
        ...
        ...     For floating-point operands,
        ...
        ...     - If ``x_i`` is less than ``0``, the result is ``NaN``.
        ...     - If ``x_i`` is ``NaN``, the result is ``NaN``.
        ...     - If ``x_i`` is ``+0``, the result is ``+0``.
        ...     - If ``x_i`` is ``-0``, the result is ``-0``.
        ...     - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        ...
        ...     Parameters
        ...     ----------
        ...     x: array
        ...         input array
        ...
        ...     Returns
        ...     -------
        ...     out: array
        ...         an array containing the square root of each element in ``x``
        ...     '''
        ...
        >>> case_block = r_case_block.search(sqrt.__doc__).group(1)
        >>> unary_cases = parse_unary_case_block(case_block, 'sqrt')
        >>> for case in unary_cases:
        ...     print(repr(case))
        UnaryCase(<x_i < 0 -> NaN>)
        UnaryCase(<x_i == NaN -> NaN>)
        UnaryCase(<x_i == +0 -> +0>)
        UnaryCase(<x_i == -0 -> -0>)
        UnaryCase(<x_i == +infinity -> +infinity>)
        >>> lt_0_case = unary_cases[0]
        >>> lt_0_case.cond(-123)
        True
        >>> lt_0_case.check_result(-123, float('nan'))
        True

    """
    cases = []
    for case_m in r_case.finditer(case_block):
        case_str = case_m.group(1)
        if r_already_int_case.search(case_str):
            cases.append(already_int_case)
        elif r_even_round_halves_case.search(case_str):
            cases.append(even_round_halves_case)
        elif m := r_nan_signbit.search(case_str):
            signbit = parse_value(m.group(1))
            expected = bool(parse_value(m.group(2)))
            cases.append(make_nan_signbit_case(signbit, expected))
        elif m := r_unary_case.search(case_str):
            try:
                cond, cond_expr_template, cond_from_dtype = parse_cond(m.group(1))
                _check_result, result_expr = parse_result(m.group(2))
            except ParseError as e:
                warn(f"case for {func_name} not machine-readable: '{e.value}'")
                continue
            cond_expr = cond_expr_template.replace("{}", "x_i")
            # Do not define check_result in this function's body - see
            # parse_binary_case comment.
            check_result = make_unary_check_result(_check_result)
            case = UnaryCase(
                cond_expr=cond_expr,
                cond=cond,
                cond_from_dtype=cond_from_dtype,
                result_expr=result_expr,
                check_result=check_result,
                raw_case=case_str,
            )
            cases.append(case)
        else:
            if not r_remaining_case.search(case_str):
                warn(f"case for {func_name} not machine-readable: '{case_str}'")
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
    raw_case: Optional[str] = field(default=None)


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
    """
    Wraps a unary condition as a binary condition, e.g.

        >>> unary_cond = lambda i: i == 42
        >>> binary_cond_first = make_binary_cond(BinaryCondArg.FIRST, unary_cond)
        >>> binary_cond_first(42, 0)
        True
        >>> binary_cond_second = make_binary_cond(BinaryCondArg.SECOND, unary_cond)
        >>> binary_cond_second(42, 0)
        False
        >>> binary_cond_second(0, 42)
        True
        >>> binary_cond_both = make_binary_cond(BinaryCondArg.BOTH, unary_cond)
        >>> binary_cond_both(42, 0)
        False
        >>> binary_cond_both(42, 42)
        True
        >>> binary_cond_either = make_binary_cond(BinaryCondArg.EITHER, unary_cond)
        >>> binary_cond_either(0, 0)
        False
        >>> binary_cond_either(42, 0)
        True
        >>> binary_cond_either(0, 42)
        True
        >>> binary_cond_either(42, 42)
        True

    """
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
    """
    Returns a result checker for cases where the result equals an array element

        >>> check_result_first = make_eq_input_check_result(BinaryCondArg.FIRST)
        >>> check_result(42, 0, 42)
        True
        >>> check_result_second = make_eq_input_check_result(BinaryCondArg.SECOND)
        >>> check_result(42, 0, 42)
        False
        >>> check_result(0, 42, 42)
        True
        >>> check_result_neg_first = make_eq_input_check_result(BinaryCondArg.FIRST, eq_neg=True)
        >>> check_result_neg_first(42, 0, 42)
        False
        >>> check_result_neg_first(42, 0, -42)
        True

    """
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

    return check_result


def make_binary_check_result(check_just_result: UnaryCheck) -> BinaryResultCheck:
    def check_result(i1: float, i2: float, result: float) -> bool:
        return check_just_result(result)

    return check_result


def parse_binary_case(case_str: str) -> BinaryCase:
    """
    Parses a Sphinx-formatted binary case string to return codified binary cases, e.g.

        >>> case_str = (
        ...     "If ``x1_i`` is greater than ``0``, ``x1_i`` is a finite number, "
        ...     "and ``x2_i`` is ``+infinity``, the result is ``NaN``."
        ... )
        >>> case = parse_binary_case(case_str)
        >>> case
        BinaryCase(<x1_i > 0 and isfinite(x1_i) and x2_i == +infinity -> NaN>)
        >>> case.cond(42, float('inf'))
        True
        >>> case.check_result(42, float('inf'), float('nan'))
        True

    """
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
            if in_sign != "" or other_no == in_no:
                raise ParseError(cond_str)
            partial_expr = f"{in_sign}x{in_no}_i == {other_sign}x{other_no}_i"

            # For these scenarios, we want to make sure both array elements
            # generate respective to one another by using a shared strategy.
            shared_from_dtype = lambda d, **kw: st.shared(
                xps.from_dtype(d, **kw), key=cond_str
            )
            input_wrapper = lambda i: -i if other_sign == "-" else noop
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
                raise ParseError(cond_str)

            x1_cond_from_dtypes.append(BoundFromDtype(base_func=_x1_cond_from_dtype))
            x2_cond_from_dtypes.append(BoundFromDtype(base_func=_x2_cond_from_dtype))

        elif m := r_both_inputs_are_value.match(cond_str):
            unary_cond, expr_template, cond_from_dtype = parse_cond(m.group(1))
            left_expr = expr_template.replace("{}", "x1_i")
            right_expr = expr_template.replace("{}", "x2_i")
            partial_expr = f"{left_expr} and {right_expr}"
            partial_cond = make_binary_cond(  # type: ignore
                BinaryCondArg.BOTH, unary_cond
            )
            x1_cond_from_dtypes.append(cond_from_dtype)
            x2_cond_from_dtypes.append(cond_from_dtype)
        else:
            cond_m = r_cond.match(cond_str)
            if cond_m is None:
                raise ParseError(cond_str)
            input_str, value_str = cond_m.groups()

            if value_str == "the same mathematical sign":
                partial_expr = "copysign(1, x1_i) == copysign(1, x2_i)"

                def partial_cond(i1: float, i2: float) -> bool:
                    return math.copysign(1, i1) == math.copysign(1, i2)

                x1_cond_from_dtypes.append(BoundFromDtype(kwargs={"min_value": 1}))
                x2_cond_from_dtypes.append(BoundFromDtype(kwargs={"min_value": 1}))

            elif value_str == "different mathematical signs":
                partial_expr = "copysign(1, x1_i) != copysign(1, x2_i)"

                def partial_cond(i1: float, i2: float) -> bool:
                    return math.copysign(1, i1) != math.copysign(1, i2)

                x1_cond_from_dtypes.append(BoundFromDtype(kwargs={"min_value": 1}))
                x2_cond_from_dtypes.append(BoundFromDtype(kwargs={"max_value": -1}))

            else:
                unary_cond, expr_template, cond_from_dtype = parse_cond(value_str)
                # Do not define partial_cond via the def keyword or lambda
                # expressions, as one partial_cond definition can mess up
                # previous definitions in the partial_conds list. This is a
                # hard-limitation of using local functions with the same name
                # and that use the same outer variables (i.e. unary_cond). Use
                # def in a called function avoids this problem.
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
                    partial_expr = f"{left_expr} and {right_expr}"
                    cond_arg = BinaryCondArg.BOTH
                elif r_or_input.match(input_str):
                    left_expr = expr_template.replace("{}", "x1_i")
                    right_expr = expr_template.replace("{}", "x2_i")
                    partial_expr = f"{left_expr} or {right_expr}"
                    if len(cond_strs) != 1:
                        partial_expr = f"({partial_expr})"
                    cond_arg = BinaryCondArg.EITHER
                else:
                    raise ParseError(input_str)
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

                    def _x1_cond_from_dtype(dtype, **kw) -> st.SearchStrategy[float]:
                        assert len(kw) == 0  # sanity check
                        return use_x1_or_x2_strat.flatmap(
                            lambda t: cond_from_dtype(dtype)
                            if t[0]
                            else hh.from_dtype(dtype)
                        )

                    def _x2_cond_from_dtype(dtype, **kw) -> st.SearchStrategy[float]:
                        assert len(kw) == 0  # sanity check
                        return use_x1_or_x2_strat.flatmap(
                            lambda t: cond_from_dtype(dtype)
                            if t[1]
                            else hh.from_dtype(dtype)
                        )

                    x1_cond_from_dtypes.append(
                        BoundFromDtype(base_func=_x1_cond_from_dtype)
                    )
                    x2_cond_from_dtypes.append(
                        BoundFromDtype(base_func=_x2_cond_from_dtype)
                    )

        partial_conds.append(partial_cond)
        partial_exprs.append(partial_expr)

    result_m = r_result.match(case_m.group(2))
    if result_m is None:
        raise ParseError(case_m.group(2))
    result_str = result_m.group(1)
    # Like with partial_cond, do not define check_result in this function's body.
    if m := r_array_element.match(result_str):
        sign, x_no = m.groups()
        result_expr = f"{sign}x{x_no}_i"
        check_result = make_eq_input_check_result(  # type: ignore
            BinaryCondArg.from_x_no(x_no), eq_neg=sign == "-"
        )
    else:
        _check_result, result_expr = parse_result(result_m.group(1))
        check_result = make_binary_check_result(_check_result)

    cond_expr = " and ".join(partial_exprs)

    def cond(i1: float, i2: float) -> bool:
        return all(pc(i1, i2) for pc in partial_conds)

    x1_cond_from_dtype = sum(x1_cond_from_dtypes, start=BoundFromDtype())
    x2_cond_from_dtype = sum(x2_cond_from_dtypes, start=BoundFromDtype())

    return BinaryCase(
        cond_expr=cond_expr,
        cond=cond,
        x1_cond_from_dtype=x1_cond_from_dtype,
        x2_cond_from_dtype=x2_cond_from_dtype,
        result_expr=result_expr,
        check_result=check_result,
        raw_case=case_str,
    )


r_redundant_case = re.compile("result.+determined by the rule already stated above")


def parse_binary_case_block(case_block: str, func_name: str) -> List[BinaryCase]:
    """
    Parses a Sphinx-formatted docstring of a binary function to return a list of
    codified binary cases, e.g.

        >>> def logaddexp(x1, x2):
        ...     '''
        ...     Calculates the logarithm of the sum of exponentiations
        ...
        ...     **Special Cases**
        ...
        ...     For floating-point operands,
        ...
        ...     - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        ...     - If ``x1_i`` is ``+infinity`` and ``x2_i`` is not ``NaN``, the result is ``+infinity``.
        ...     - If ``x1_i`` is not ``NaN`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        ...
        ...     Parameters
        ...     ----------
        ...     x1: array
        ...         first input array
        ...     x2: array
        ...         second input array
        ...
        ...     Returns
        ...     -------
        ...     out: array
        ...         an array containing the results
        ...     '''
        ...
        >>> case_block = r_case_block.search(logaddexp.__doc__).group(1)
        >>> binary_cases = parse_binary_case_block(case_block, 'logaddexp')
        >>> for case in binary_cases:
        ...     print(repr(case))
        BinaryCase(<x1_i == NaN or x2_i == NaN -> NaN>)
        BinaryCase(<x1_i == +infinity and not x2_i == NaN -> +infinity>)
        BinaryCase(<not x1_i == NaN and x2_i == +infinity -> +infinity>)

    """
    cases = []
    for case_m in r_case.finditer(case_block):
        case_str = case_m.group(1)
        if r_redundant_case.search(case_str):
            continue
        if r_binary_case.match(case_str):
            try:
                case = parse_binary_case(case_str)
                cases.append(case)
            except ParseError as e:
                warn(f"case for {func_name} not machine-readable: '{e.value}'")
        else:
            if not r_remaining_case.match(case_str):
                warn(f"case for {func_name} not machine-readable: '{case_str}'")
    return cases


unary_params = []
binary_params = []
iop_params = []
func_to_op: Dict[str, str] = {v: k for k, v in dh.op_to_func.items()}
for stub in category_to_funcs["elementwise"]:
    func_name = stub.__name__
    if stub.__doc__ is None:
        warn(f"{func_name}() stub has no docstring")
        continue
    if m := r_case_block.search(stub.__doc__):
        case_block = m.group(1)
    else:
        continue
    marks = []
    try:
        func = getattr(xp, func_name)
    except AttributeError:
        marks.append(
            pytest.mark.skip(reason=f"{func_name} not found in array module")
        )
        func = None
    sig = inspect.signature(stub)
    param_names = list(sig.parameters.keys())
    if len(sig.parameters) == 0:
        warn(f"{func=} has no parameters")
        continue
    if param_names[0] == "x":
        if cases := parse_unary_case_block(case_block, func_name):
            name_to_func = {func_name: func}
            if func_name in func_to_op.keys():
                op_name = func_to_op[func_name]
                op = getattr(operator, op_name)
                name_to_func[op_name] = op
            for func_name, func in name_to_func.items():
                for case in cases:
                    id_ = f"{func_name}({case.cond_expr}) -> {case.result_expr}"
                    p = pytest.param(func_name, func, case, id=id_)
                    unary_params.append(p)
        else:
            warn(f"Special cases found for {func_name} but none were parsed")
        continue
    if len(sig.parameters) == 1:
        warn(f"{func=} has one parameter '{param_names[0]}' which is not named 'x'")
        continue
    if param_names[0] == "x1" and param_names[1] == "x2":
        if cases := parse_binary_case_block(case_block, func_name):
            name_to_func = {func_name: func}
            if func_name in func_to_op.keys():
                op_name = func_to_op[func_name]
                op = getattr(operator, op_name)
                name_to_func[op_name] = op
                # We collect inplace operator test cases seperately
                if "equal" in func_name:
                    continue
                iop_name = "__i" + op_name[2:]
                iop = getattr(operator, iop_name)
                for case in cases:
                    id_ = f"{iop_name}({case.cond_expr}) -> {case.result_expr}"
                    p = pytest.param(iop_name, iop, case, id=id_)
                    iop_params.append(p)
            for func_name, func in name_to_func.items():
                for case in cases:
                    id_ = f"{func_name}({case.cond_expr}) -> {case.result_expr}"
                    p = pytest.param(func_name, func, case, id=id_)
                    binary_params.append(p)
        else:
            warn(f"Special cases found for {func_name} but none were parsed")
        continue
    else:
        warn(
            f"{func=} starts with two parameters '{param_names[0]}' and "
            f"'{param_names[1]}', which are not named 'x1' and 'x2'"
        )


# test_{unary/binary/iop} naively generate arrays, i.e. arrays that might not
# meet the condition that is being test. We then forcibly make the array meet
# the condition by picking a random index to insert an acceptable element.
#
# good_example is a flag that tells us whether Hypothesis generated an array
# with at least on element that is special-cased. We reject the example when
# its False - Hypothesis will complain if we reject too many examples, thus
# indicating we've done something wrong.

# sanity checks
assert len(unary_params) != 0
assert len(binary_params) != 0
assert len(iop_params) != 0


@pytest.mark.parametrize("func_name, func, case", unary_params)
def test_unary(func_name, func, case):
    with catch_warnings():
        # XXX: We are using example here to generate one example draw, but
        # hypothesis issues a warning from this. We should consider either
        # drawing multiple examples like a normal test, or just hard-coding a
        # single example test case without using hypothesis.
        filterwarnings('ignore', category=NonInteractiveExampleWarning)
        in_value = case.cond_from_dtype(xp.float64).example()
    x = xp.asarray(in_value, dtype=xp.float64)
    out = func(x)
    out_value = float(out)
    assert case.check_result(in_value, out_value), (
        f"out={out_value}, but should be {case.result_expr} [{func_name}()]\n"
    )


@pytest.mark.parametrize("func_name, func, case", binary_params)
@settings(max_examples=1)
@given(data=st.data())
def test_binary(func_name, func, case, data):
    # We don't use example() like in test_unary because the same internal shared
    # strategies used in both x1's and x2's don't "sync" with example() draws.
    x1_value = data.draw(case.x1_cond_from_dtype(xp.float64), label="x1_value")
    x2_value = data.draw(case.x2_cond_from_dtype(xp.float64), label="x2_value")
    x1 = xp.asarray(x1_value, dtype=xp.float64)
    x2 = xp.asarray(x2_value, dtype=xp.float64)

    out = func(x1, x2)
    out_value = float(out)

    assert case.check_result(x1_value, x2_value, out_value), (
        f"out={out_value}, but should be {case.result_expr} [{func_name}()]\n"
        f"condition: {case}\n"
        f"x1={x1_value}, x2={x2_value}"
    )



@pytest.mark.parametrize("iop_name, iop, case", iop_params)
@settings(max_examples=1)
@given(data=st.data())
def test_iop(iop_name, iop, case, data):
    # See test_binary comment
    x1_value = data.draw(case.x1_cond_from_dtype(xp.float64), label="x1_value")
    x2_value = data.draw(case.x2_cond_from_dtype(xp.float64), label="x2_value")
    x1 = xp.asarray(x1_value, dtype=xp.float64)
    x2 = xp.asarray(x2_value, dtype=xp.float64)

    res = iop(x1, x2)
    res_value = float(res)

    assert case.check_result(x1_value, x2_value, res_value), (
        f"x1={res}, but should be {case.result_expr} [{func_name}()]\n"
        f"condition: {case}\n"
        f"x1={x1_value}, x2={x2_value}"
    )


@pytest.mark.parametrize(
    "func_name, expected",
    [
        ("mean", float("nan")),
        ("prod", 1),
        ("std", float("nan")),
        ("sum", 0),
        ("var", float("nan")),
    ],
    ids=["mean", "prod", "std", "sum", "var"],
)
def test_empty_arrays(func_name, expected):  # TODO: parse docstrings to get expected
    func = getattr(xp, func_name)
    out = func(xp.asarray([], dtype=dh.default_float))
    ph.assert_shape(func_name, out_shape=out.shape, expected=())  # sanity check
    msg = f"{out=!r}, but should be {expected}"
    if math.isnan(expected):
        assert xp.isnan(out), msg
    else:
        assert out == expected, msg


@pytest.mark.parametrize(
    "func_name", [f.__name__ for f in category_to_funcs["statistical"]
                  if f.__name__ not in ['cumulative_sum', 'cumulative_prod']]
)
@given(
    x=hh.arrays(dtype=hh.real_floating_dtypes, shape=hh.shapes(min_side=1)),
    data=st.data(),
)
def test_nan_propagation(func_name, x, data):
    func = getattr(xp, func_name)
    nan_positions = data.draw(
        hh.arrays(dtype=hh.bool_dtype, shape=x.shape), label="nan_positions"
    )
    assume(xp.any(nan_positions))
    x = xp.where(nan_positions, xp.asarray(float("nan")), x)
    note(f"{x=}")

    out = func(x)

    ph.assert_shape(func_name, out_shape=out.shape, expected=())  # sanity check
    assert xp.isnan(out), f"{out=!r}, but should be NaN"
