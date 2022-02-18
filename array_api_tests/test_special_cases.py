import inspect
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Pattern,
    Protocol,
    Tuple,
    Union,
)
from warnings import warn

import pytest
from hypothesis import assume, given

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


def make_eq(v: float) -> UnaryCheck:
    if math.isnan(v):
        return math.isnan
    if v == 0:
        if ph.is_pos_zero(v):
            return ph.is_pos_zero
        else:
            return ph.is_neg_zero

    def eq(i: float) -> bool:
        return i == v

    return eq


def make_neq(v: float) -> UnaryCheck:
    eq = make_eq(v)

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


def notify_cond(cond: UnaryCheck) -> UnaryCheck:
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


r_special_cases = re.compile(
    r"\*\*Special [Cc]ases\*\*(?:\n.*)+"
    r"For floating-point operands,\n+"
    r"((?:\s*-\s*.*\n)+)"
)
r_case = re.compile(r"\s+-\s*(.*)\.\n?")
r_remaining_case = re.compile("In the remaining cases.+")


def parse_cond(cond_str: str) -> Tuple[UnaryCheck, str]:
    if m := r_not.match(cond_str):
        cond_str = m.group(1)
        notify = True
    else:
        notify = False

    if m := r_code.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_eq(value)
        expr_template = "{} == " + str(value)
    elif m := r_gt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_gt(value)
        expr_template = "{} > " + str(value)
    elif m := r_lt.match(cond_str):
        value = parse_value(m.group(1))
        cond = make_lt(value)
        expr_template = "{} < " + str(value)
    elif m := r_either_code.match(cond_str):
        v1 = parse_value(m.group(1))
        v2 = parse_value(m.group(2))
        cond = make_or(make_eq(v1), make_eq(v2))
        expr_template = "{} == " + str(v1) + " or {} == " + str(v2)
    elif cond_str in ["finite", "a finite number"]:
        cond = math.isfinite
        expr_template = "isfinite({})"
    elif cond_str in "a positive (i.e., greater than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i > 0
        expr_template = "isfinite({}) and {} > 0"
    elif cond_str == "a negative (i.e., less than ``0``) finite number":
        cond = lambda i: math.isfinite(i) and i < 0
        expr_template = "isfinite({}) and {} < 0"
    elif cond_str == "positive":
        cond = lambda i: math.copysign(1, i) == 1
        expr_template = "copysign(1, {}) == 1"
    elif cond_str == "negative":
        cond = lambda i: math.copysign(1, i) == -1
        expr_template = "copysign(1, {}) == -1"
    elif "nonzero finite" in cond_str:
        cond = lambda i: math.isfinite(i) and i != 0
        expr_template = "isfinite({}) and {} != 0"
    elif cond_str == "an integer value":
        cond = lambda i: i.is_integer()
        expr_template = "{}.is_integer()"
    elif cond_str == "an odd integer value":
        cond = lambda i: i.is_integer() and i % 2 == 1
        expr_template = "{}.is_integer() and {} % 2 == 1"
    else:
        raise ValueParseError(cond_str)

    if notify:
        cond = notify_cond(cond)
        expr_template = f"not ({expr_template})"

    return cond, expr_template


def parse_result(result_str: str) -> Tuple[UnaryCheck, str]:
    if m := r_code.match(result_str):
        value = parse_value(m.group(1))
        check_result = make_eq(value)
        expr = str(value)
    elif m := r_approx_value.match(result_str):
        value = parse_value(m.group(1))
        check_result = make_rough_eq(value)
        expr = f"~{value}"
    elif result_str == "positive":

        def check_result(result: float) -> bool:
            if math.isnan(result):
                # The sign of NaN is out-of-scope
                return True
            return math.copysign(1, result) == 1

        expr = "+"
    elif result_str == "negative":

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
    expr: str

    def cond(self, *args) -> bool:
        ...

    def check_result(self, *args) -> bool:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(<{self.expr}>)"

    def __str__(self):
        return self.expr


class UnaryCond(Protocol):
    def __call__(self, i: float) -> bool:
        ...


class UnaryResultCheck(Protocol):
    def __call__(self, i: float, result: float) -> bool:
        ...


@dataclass(repr=False)
class UnaryCase(Case):
    expr: str
    cond: UnaryCheck
    check_result: UnaryResultCheck

    @classmethod
    def from_strings(cls, cond_str: str, result_str: str):
        cond, cond_expr_template = parse_cond(cond_str)
        check_result, check_result_expr = parse_result(result_str)
        cond_expr = cond_expr_template.replace("{}", "xᵢ")
        expr = f"{cond_expr} -> {check_result_expr}"
        return cls(expr, cond, lambda i, result: check_result(result))


r_unary_case = re.compile("If ``x_i`` is (.+), the result is (.+)")
r_even_int_round_case = re.compile(
    "If two integers are equally close to ``x_i``, "
    "the result is the even integer closest to ``x_i``"
)
even_int_round_case = UnaryCase(
    expr="i % 0.5 == 0 -> Decimal(i).to_integral_exact(ROUND_HALF_EVEN)",
    cond=lambda i: i % 0.5 == 0,
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


@dataclass
class BinaryCond:
    cond: BinaryCheck
    expr: str

    def __call__(self, i1: float, i2: float) -> bool:
        return self.cond(i1, i2)


class BinaryCondFactory(Protocol):
    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        ...


r_not = re.compile("not (?:equal to )?(.+)")
r_array_element = re.compile(r"``([+-]?)x([12])_i``")
r_either_code = re.compile(f"either {r_code.pattern} or {r_code.pattern}")
r_gt = re.compile(f"greater than {r_code.pattern}")
r_lt = re.compile(f"less than {r_code.pattern}")


@dataclass
class ValueCondFactory(BinaryCondFactory):
    input_: Union[Literal["i1"], Literal["i2"], Literal["either"], Literal["both"]]
    re_groups_i: int
    abs_: bool = False

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        group = groups[self.re_groups_i]

        if m := r_array_element.match(group):
            assert not self.abs_  # sanity check
            sign = m.group(1)
            if sign == "-":
                signer = lambda i: -i
            else:
                signer = lambda i: i

            if self.input_ == "i1":
                expr = f"x1ᵢ == {sign}x2ᵢ"

                def cond(i1: float, i2: float) -> bool:
                    _cond = make_eq(signer(i2))
                    return _cond(i1)

            else:
                assert self.input_ == "i2"  # sanity check
                expr = f"x2ᵢ == {sign}x1ᵢ"

                def cond(i1: float, i2: float) -> bool:
                    _cond = make_eq(signer(i1))
                    return _cond(i2)

            return BinaryCond(cond, expr)

        _cond, expr_template = parse_cond(group)

        if self.abs_:
            _cond = absify_cond(_cond)

        f_i1 = "x1ᵢ"
        f_i2 = "x2ᵢ"
        if self.abs_:
            f_i1 = f"abs({f_i1})"
            f_i2 = f"abs({f_i2})"

        if self.input_ == "i1":
            expr = expr_template.replace("{}", f_i1)

            def cond(i1: float, i2: float) -> bool:
                return _cond(i1)

        elif self.input_ == "i2":
            expr = expr_template.replace("{}", f_i2)

            def cond(i1: float, i2: float) -> bool:
                return _cond(i2)

        elif self.input_ == "either":
            expr = f"({expr_template.replace('{}', f_i1)}) or ({expr_template.replace('{}', f_i2)})"

            def cond(i1: float, i2: float) -> bool:
                return _cond(i1) or _cond(i2)

        else:
            assert self.input_ == "both"  # sanity check
            expr = f"({expr_template.replace('{}', f_i1)}) and ({expr_template.replace('{}', f_i2)})"

            def cond(i1: float, i2: float) -> bool:
                return _cond(i1) and _cond(i2)

        return BinaryCond(cond, expr)


class AndCondFactory(BinaryCondFactory):
    def __init__(self, *cond_factories: BinaryCondFactory):
        self.cond_factories = cond_factories

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        conds = [cond_factory(groups) for cond_factory in self.cond_factories]
        expr = " and ".join(f"({cond.expr})" for cond in conds)

        def cond(i1: float, i2: float) -> bool:
            return all(cond(i1, i2) for cond in conds)

        return BinaryCond(cond, expr)


@dataclass
class SignCondFactory(BinaryCondFactory):
    re_groups_i: int

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        group = groups[self.re_groups_i]
        if group == "the same mathematical sign":

            def cond(i1: float, i2: float) -> bool:
                return math.copysign(1, i1) == math.copysign(1, i2)

            expr = "copysign(1, x1ᵢ) == copysign(1, x2ᵢ)"
        elif group == "different mathematical signs":

            def cond(i1: float, i2: float) -> bool:
                return math.copysign(1, i1) != math.copysign(1, i2)

            expr = "copysign(1, x1ᵢ) != copysign(1, x2ᵢ)"
        else:
            raise ValueParseError(group)
        return BinaryCond(cond, expr)


class BinaryResultCheck(NamedTuple):
    check_result: Callable[[float, float, float], bool]
    expr: str

    def __call__(self, i1: float, i2: float, result: float) -> bool:
        return self.check_result(i1, i2, result)

    def __repr__(self):
        return self.expr


class BinaryResultCheckFactory(Protocol):
    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        ...


@dataclass
class ResultCheckFactory(BinaryResultCheckFactory):
    re_groups_i: int

    def __call__(self, groups: Tuple[str, ...]) -> BinaryResultCheck:
        group = groups[self.re_groups_i]

        if m := r_array_element.match(group):
            sign, input_ = m.groups()
            if sign == "-":
                signer = lambda i: -i
            else:
                signer = lambda i: i

            if input_ == "1":
                expr = f"{sign}x1ᵢ"

                def check_result(i1: float, i2: float, result: float) -> bool:
                    _check_result = make_eq(signer(i1))
                    return _check_result(result)

            else:
                expr = f"{sign}x2ᵢ"

                def check_result(i1: float, i2: float, result: float) -> bool:
                    _check_result = make_eq(signer(i2))
                    return _check_result(result)

            return BinaryResultCheck(check_result, expr)

        _check_result, expr = parse_result(group)

        def check_result(i1: float, i2: float, result: float) -> bool:
            return _check_result(result)

        return BinaryResultCheck(check_result, expr)


@dataclass(repr=False)
class BinaryCase(Case):
    expr: str
    cond: BinaryCond
    check_result: BinaryResultCheck


class BinaryCaseFactory(NamedTuple):
    cond_factory: BinaryCondFactory
    check_result_factory: ResultCheckFactory

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCase:
        cond = self.cond_factory(groups)
        check_result = self.check_result_factory(groups)
        expr = f"{cond.expr} -> {check_result.expr}"
        return BinaryCase(expr, cond, check_result)


r_result_sign = re.compile("([a-z]+) mathematical sign")

binary_pattern_to_case_factory: Dict[Pattern, BinaryCaseFactory] = {
    re.compile(
        "If ``x1_i`` is (.+), ``x1_i`` is (.+), ``x2_i`` is (.+), "
        "and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(
            ValueCondFactory("i1", 0),
            ValueCondFactory("i1", 1),
            ValueCondFactory("i2", 2),
            ValueCondFactory("i2", 3),
        ),
        ResultCheckFactory(4),
    ),
    re.compile(
        "If ``x1_i`` is (.+), ``x1_i`` is (.+), "
        "and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(
            ValueCondFactory("i1", 0),
            ValueCondFactory("i1", 1),
            ValueCondFactory("i2", 2),
        ),
        ResultCheckFactory(3),
    ),
    re.compile(
        "If ``x1_i`` is (.+), ``x2_i`` is (.+), "
        "and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(
            ValueCondFactory("i1", 0),
            ValueCondFactory("i2", 1),
            ValueCondFactory("i2", 2),
        ),
        ResultCheckFactory(3),
    ),
    # This pattern must come after the above to avoid false matches
    re.compile(
        "If ``x1_i`` is (.+) and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(ValueCondFactory("i1", 0), ValueCondFactory("i2", 1)),
        ResultCheckFactory(2),
    ),
    re.compile(
        r"If ``abs\(x1_i\)`` is (.+) and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(ValueCondFactory("i1", 0, abs_=True), ValueCondFactory("i2", 1)),
        ResultCheckFactory(2),
    ),
    re.compile(
        "If either ``x1_i`` or ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(ValueCondFactory("either", 0), ResultCheckFactory(1)),
    re.compile(
        "If ``x1_i`` and ``x2_i`` have (.+signs?), "
        f"the result has a {r_result_sign.pattern}"
    ): BinaryCaseFactory(SignCondFactory(0), ResultCheckFactory(1)),
    re.compile(
        "If ``x1_i`` and ``x2_i`` have (.+signs?) and are both (.+), "
        f"the result has a {r_result_sign.pattern}"
    ): BinaryCaseFactory(
        AndCondFactory(SignCondFactory(0), ValueCondFactory("both", 1)),
        ResultCheckFactory(2),
    ),
    re.compile(
        "If ``x1_i`` and ``x2_i`` have (.+signs?), the result has a "
        rf"{r_result_sign.pattern} , unless the result is (.+)\. If the result "
        r"is ``NaN``, the \"sign\" of ``NaN`` is implementation-defined\."
    ): BinaryCaseFactory(SignCondFactory(0), ResultCheckFactory(1)),
    re.compile(
        "If ``x2_i`` is (.+), the result is (.+), even if ``x1_i`` is .+"
    ): BinaryCaseFactory(ValueCondFactory("i2", 0), ResultCheckFactory(1)),
}


r_redundant_case = re.compile("result.+determined by the rule already stated above")


def parse_binary_docstring(docstring: str) -> List[BinaryCase]:
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
        if r_redundant_case.search(case):
            continue
        for pattern, make_case in binary_pattern_to_case_factory.items():
            if m := pattern.search(case):
                try:
                    case = make_case(m.groups())
                except ValueParseError as e:
                    warn(f"not machine-readable: '{e.value}'")
                    break
                cases.append(case)
                break
        else:
            if not r_remaining_case.search(case):
                warn(f"case not machine-readable: '{case}'")
    return cases


# Here be the tests
# ------------------------------------------------------------------------------


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
            p = pytest.param(stub.__name__, func, cases, id=stub.__name__)
            unary_params.append(p)
        continue
    if len(sig.parameters) == 1:
        warn(f"{func=} has one parameter '{param_names[0]}' which is not named 'x'")
        continue
    if param_names[0] == "x1" and param_names[1] == "x2":
        if cases := parse_binary_docstring(stub.__doc__):
            p = pytest.param(stub.__name__, func, cases, id=stub.__name__)
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


@pytest.mark.parametrize("func_name, func, cases", unary_params)
@given(x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_unary(func_name, func, cases, x):
    res = func(x)
    good_example = False
    for idx in sh.ndindex(res.shape):
        in_ = float(x[idx])
        for case in cases:
            if case.cond(in_):
                good_example = True
                out = float(res[idx])
                f_in = f"{sh.fmt_idx('x', idx)}={in_}"
                f_out = f"{sh.fmt_idx('out', idx)}={out}"
                assert case.check_result(
                    in_, out
                ), f"{f_out} not good [{func_name}()]\n{f_in}"
                break
    assume(good_example)


@pytest.mark.parametrize("func_name, func, cases", binary_params)
@given(
    *hh.two_mutual_arrays(
        dtypes=dh.float_dtypes,
        two_shapes=hh.mutually_broadcastable_shapes(2, min_side=1),
    )
)
def test_binary(func_name, func, cases, x1, x2):
    res = func(x1, x2)
    good_example = False
    for l_idx, r_idx, o_idx in sh.iter_indices(x1.shape, x2.shape, res.shape):
        l = float(x1[l_idx])
        r = float(x2[r_idx])
        for case in cases:
            if case.cond(l, r):
                good_example = True
                o = float(res[o_idx])
                f_left = f"{sh.fmt_idx('x1', l_idx)}={l}"
                f_right = f"{sh.fmt_idx('x2', r_idx)}={r}"
                f_out = f"{sh.fmt_idx('out', o_idx)}={o}"
                assert case.check_result(
                    l, r, o
                ), f"{f_out} not good [{func_name}()]\n{f_left}, {f_right}"
                break
    assume(good_example)
