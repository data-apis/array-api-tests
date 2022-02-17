import inspect
import math
import re
from dataclasses import dataclass
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
from hypothesis import HealthCheck, assume, given, settings

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from ._array_module import mod as xp
from .stubs import category_to_funcs

pytestmark = pytest.mark.ci

# Condition factories
# ------------------------------------------------------------------------------


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


def make_bin_and_factory(
    make_cond1: Callable[[float], UnaryCheck], make_cond2: Callable[[float], UnaryCheck]
) -> Callable[[float, float], BinaryCheck]:
    def make_bin_and(v1: float, v2: float) -> BinaryCheck:
        cond1 = make_cond1(v1)
        cond2 = make_cond2(v2)

        def bin_and(i1: float, i2: float) -> bool:
            return cond1(i1) and cond2(i2)

        return bin_and

    return make_bin_and


def make_bin_or_factory(
    make_cond: Callable[[float], UnaryCheck]
) -> Callable[[float], BinaryCheck]:
    def make_bin_or(v: float) -> BinaryCheck:
        cond = make_cond(v)

        def bin_or(i1: float, i2: float) -> bool:
            return cond(i1) or cond(i2)

        return bin_or

    return make_bin_or


def absify_cond_factory(
    make_cond: Callable[[float], UnaryCheck]
) -> Callable[[float], UnaryCheck]:
    def make_abs_cond(v: float) -> UnaryCheck:
        cond = make_cond(v)

        def abs_cond(i: float) -> bool:
            i = abs(i)
            return cond(i)

        return abs_cond

    return make_abs_cond


def make_bin_multi_and_factory(
    make_conds1: List[Callable[[float], UnaryCheck]],
    make_conds2: List[Callable[[float], UnaryCheck]],
) -> Callable:
    def make_bin_multi_and(*values: float) -> BinaryCheck:
        assert len(values) == len(make_conds1) + len(make_conds2)
        conds1 = [make_cond(v) for make_cond, v in zip(make_conds1, values)]
        conds2 = [make_cond(v) for make_cond, v in zip(make_conds2, values[::-1])]

        def bin_multi_and(i1: float, i2: float) -> bool:
            return all(cond(i1) for cond in conds1) and all(cond(i2) for cond in conds2)

        return bin_multi_and

    return make_bin_multi_and


def same_sign(i1: float, i2: float) -> bool:
    return math.copysign(1, i1) == math.copysign(1, i2)


def diff_sign(i1: float, i2: float) -> bool:
    return not same_sign(i1, i2)


# Parse utils
# ------------------------------------------------------------------------------


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


def parse_value(s_value: str) -> float:
    assert not s_value.startswith("``") and not s_value.endswith("``")  # sanity check
    m = r_value.match(s_value)
    if m is None:
        raise ValueParseError(s_value)
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


class Result(NamedTuple):
    value: float
    repr_: str
    strict_check: bool


def parse_result(s_result: str) -> Result:
    match = None
    if m := r_code.match(s_result):
        match = m
        strict_check = True
    elif m := r_approx_value.match(s_result):
        match = m
        strict_check = False
    else:
        raise ValueParseError(s_result)
    value = parse_value(match.group(1))
    repr_ = match.group(1)
    return Result(value, repr_, strict_check)


r_special_cases = re.compile(
    r"\*\*Special [Cc]ases\*\*\n+\s*"
    r"For floating-point operands,\n+"
    r"((?:\s*-\s*.*\n)+)"
)
r_case = re.compile(r"\s+-\s*(.*)\.\n?")
r_remaining_case = re.compile("In the remaining cases.+")


unary_pattern_to_condition_factory: Dict[Pattern, Callable] = {
    re.compile("If ``x_i`` is greater than (.+), the result is (.+)"): make_gt,
    re.compile("If ``x_i`` is less than (.+), the result is (.+)"): make_lt,
    re.compile("If ``x_i`` is either (.+) or (.+), the result is (.+)"): (
        lambda v1, v2: make_or(make_eq(v1), make_eq(v2))
    ),
    # This pattern must come after the previous patterns to avoid unwanted matches
    re.compile("If ``x_i`` is (.+), the result is (.+)"): make_eq,
    re.compile(
        "If two integers are equally close to ``x_i``, the result is (.+)"
    ): lambda: (lambda i: (abs(i) - math.floor(abs(i))) == 0.5),
}


def parse_unary_docstring(docstring: str) -> Dict[Callable, Result]:
    match = r_special_cases.search(docstring)
    if match is None:
        return {}
    cases = match.group(1).split("\n")[:-1]
    cases = {}
    for line in cases:
        if m := r_case.match(line):
            case = m.group(1)
        else:
            warn(f"line not machine-readable: '{line}'")
            continue
        for pattern, make_cond in unary_pattern_to_condition_factory.items():
            if m := pattern.search(case):
                *s_values, s_result = m.groups()
                try:
                    values = [parse_inline_code(v) for v in s_values]
                except ValueParseError as e:
                    warn(f"value not machine-readable: '{e.value}'")
                    break
                cond = make_cond(*values)
                try:
                    result = parse_result(s_result)
                except ValueParseError as e:
                    warn(f"result not machine-readable: '{e.value}'")

                    break
                cases[cond] = result
                break
        else:
            if not r_remaining_case.search(case):
                warn(f"case not machine-readable: '{case}'")
    return cases


class BinaryCond(NamedTuple):
    cond: BinaryCheck
    repr_: str

    def __call__(self, i1: float, i2: float) -> bool:
        return self.cond(i1, i2)

    def __repr__(self):
        return self.repr_


class BinaryCondFactory(Protocol):
    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        ...


r_not = re.compile("not (?:equal to )?(.+)")
r_array_element = re.compile(r"``([+-]?)x([12])_i``")
r_either_code = re.compile(f"either {r_code.pattern} or {r_code.pattern}")
r_gt = re.compile(f"greater than {r_code.pattern}")
r_lt = re.compile(f"less than {r_code.pattern}")

x1_i = "x1ᵢ"
x2_i = "x2ᵢ"


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
                repr_ = f"{x1_i} == {sign}{x2_i}"

                def cond(i1: float, i2: float) -> bool:
                    _cond = make_eq(signer(i2))
                    return _cond(i1)

            else:
                assert self.input_ == "i2"  # sanity check
                repr_ = f"{x2_i} == {sign}{x1_i}"

                def cond(i1: float, i2: float) -> bool:
                    _cond = make_eq(signer(i1))
                    return _cond(i2)

            return BinaryCond(cond, repr_)

        if m := r_not.match(group):
            group = m.group(1)
            notify = True
        else:
            notify = False

        if m := r_code.match(group):
            value = parse_value(m.group(1))
            _cond = make_eq(value)
            repr_template = "{} == " + str(value)
        elif m := r_gt.match(group):
            value = parse_value(m.group(1))
            _cond = make_gt(value)
            repr_template = "{} > " + str(value)
        elif m := r_lt.match(group):
            value = parse_value(m.group(1))
            _cond = make_lt(value)
            repr_template = "{} < " + str(value)
        elif m := r_either_code.match(group):
            v1 = parse_value(m.group(1))
            v2 = parse_value(m.group(2))
            _cond = make_or(make_eq(v1), make_eq(v2))
            repr_template = "{} == " + str(v1) + " or {} == " + str(v2)
        elif group in ["finite", "a finite number"]:
            _cond = math.isfinite
            repr_template = "isfinite({})"
        elif group in "a positive (i.e., greater than ``0``) finite number":
            _cond = lambda i: math.isfinite(i) and i > 0
            repr_template = "isfinite({}) and {} > 0"
        elif group == "a negative (i.e., less than ``0``) finite number":
            _cond = lambda i: math.isfinite(i) and i < 0
            repr_template = "isfinite({}) and {} < 0"
        elif group == "positive":
            _cond = lambda i: math.copysign(1, i) == 1
            repr_template = "copysign(1, {}) == 1"
        elif group == "negative":
            _cond = lambda i: math.copysign(1, i) == -1
            repr_template = "copysign(1, {}) == -1"
        elif "nonzero finite" in group:
            _cond = lambda i: math.isfinite(i) and i != 0
            repr_template = "copysign(1, {}) == -1"
        elif group == "an integer value":
            _cond = lambda i: i.is_integer()
            repr_template = "{}.is_integer()"
        elif group == "an odd integer value":
            _cond = lambda i: i.is_integer() and i % 2 == 1
            repr_template = "{}.is_integer() and {} % 2 == 1"
        else:
            raise ValueParseError(group)

        if notify:
            final_cond = lambda i: not _cond(i)
        else:
            final_cond = _cond

        f_i1 = x1_i
        f_i2 = x2_i
        if self.abs_:
            f_i1 = f"abs{f_i1}"
            f_i2 = f"abs{f_i2}"

        if self.input_ == "i1":
            repr_ = repr_template.replace("{}", f_i1)

            def cond(i1: float, i2: float) -> bool:
                return final_cond(i1)

        elif self.input_ == "i2":
            repr_ = repr_template.replace("{}", f_i2)

            def cond(i1: float, i2: float) -> bool:
                return final_cond(i2)

        elif self.input_ == "either":
            repr_ = f"({repr_template.replace('{}', f_i1)}) or ({repr_template.replace('{}', f_i2)})"

            def cond(i1: float, i2: float) -> bool:
                return final_cond(i1) or final_cond(i2)

        else:
            assert self.input_ == "both"  # sanity check
            repr_ = f"({repr_template.replace('{}', f_i1)}) and ({repr_template.replace('{}', f_i2)})"

            def cond(i1: float, i2: float) -> bool:
                return final_cond(i1) and final_cond(i2)

        if notify:
            repr_ = f"not ({repr_})"

        return BinaryCond(cond, repr_)


class AndCondFactory(BinaryCondFactory):
    def __init__(self, *cond_factories: BinaryCondFactory):
        self.cond_factories = cond_factories

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCond:
        conds = [cond_factory(groups) for cond_factory in self.cond_factories]
        repr_ = " and ".join(f"({cond!r})" for cond in conds)

        def cond(i1: float, i2: float) -> bool:
            return all(cond(i1, i2) for cond in conds)

        return BinaryCond(cond, repr_)


@dataclass
class SignCondFactory(BinaryCondFactory):
    re_groups_i: int

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCheck:
        group = groups[self.re_groups_i]
        if group == "the same mathematical sign":
            return same_sign
        elif group == "different mathematical signs":
            return diff_sign
        else:
            raise ValueParseError(group)


class BinaryResultCheck(NamedTuple):
    check_result: Callable[[float, float, float], bool]
    repr_: str

    def __call__(self, i1: float, i2: float, result: float) -> bool:
        return self.check_result(i1, i2, result)

    def __repr__(self):
        return self.repr_


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
                repr_ = f"{sign}{x1_i}"

                def check_result(i1: float, i2: float, result: float) -> bool:
                    _check_result = make_eq(signer(i1))
                    return _check_result(result)

            else:
                repr_ = f"{sign}{x2_i}"

                def check_result(i1: float, i2: float, result: float) -> bool:
                    _check_result = make_eq(signer(i2))
                    return _check_result(result)

            return BinaryResultCheck(check_result, repr_)

        if m := r_code.match(group):
            value = parse_value(m.group(1))
            _check_result = make_eq(value)
            repr_ = str(value)
        elif m := r_approx_value.match(group):
            value = parse_value(m.group(1))
            _check_result = make_rough_eq(value)
            repr_ = f"~{value}"
        else:
            raise ValueParseError(group)

        def check_result(i1: float, i2: float, result: float) -> bool:
            return _check_result(result)

        return BinaryResultCheck(check_result, repr_)


class ResultSignCheckFactory(ResultCheckFactory):
    def __call__(self, groups: Tuple[str, ...]) -> BinaryResultCheck:
        group = groups[self.re_groups_i]
        if group == "positive":

            def cond(i1: float, i2: float, result: float) -> bool:
                if math.isnan(result):
                    return True
                return result > 0 or ph.is_pos_zero(result)

        elif group == "negative":

            def cond(i1: float, i2: float, result: float) -> bool:
                if math.isnan(result):
                    return True
                return result < 0 or ph.is_neg_zero(result)

        else:
            raise ValueParseError(group)

        return cond


class BinaryCase(NamedTuple):
    cond: BinaryCond
    check_result: BinaryResultCheck

    def __repr__(self):
        return f"BinaryCase(<{self.cond} -> {self.check_result}>)"


class BinaryCaseFactory(NamedTuple):
    cond_factory: BinaryCondFactory
    check_result_factory: ResultCheckFactory

    def __call__(self, groups: Tuple[str, ...]) -> BinaryCase:
        cond = self.cond_factory(groups)
        check_result = self.check_result_factory(groups)
        return BinaryCase(cond, check_result)


r_result_sign = re.compile("([a-z]+) mathematical sign")

binary_pattern_to_case_factory: Dict[Pattern, BinaryCaseFactory] = {
    re.compile(
        "If ``x1_i`` is (.+) and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(ValueCondFactory("i1", 0), ValueCondFactory("i2", 1)),
        ResultCheckFactory(2),
    ),
    re.compile(
        "If ``x1_i`` is (.+), ``x1_i`` (.+), "
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
        "If ``x1_i`` is (.+), ``x2_i`` (.+), "
        "and ``x2_i`` is (.+), the result is (.+)"
    ): BinaryCaseFactory(
        AndCondFactory(
            ValueCondFactory("i1", 0),
            ValueCondFactory("i2", 1),
            ValueCondFactory("i2", 2),
        ),
        ResultCheckFactory(3),
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
    ): BinaryCaseFactory(SignCondFactory(0), ResultSignCheckFactory(1)),
    re.compile(
        "If ``x1_i`` and ``x2_i`` have (.+signs?) and are both (.+), "
        f"the result has a {r_result_sign.pattern}"
    ): BinaryCaseFactory(
        AndCondFactory(SignCondFactory(0), ValueCondFactory("both", 1)),
        ResultSignCheckFactory(2),
    ),
    re.compile(
        "If ``x1_i`` and ``x2_i`` have (.+signs?), the result has a "
        rf"{r_result_sign.pattern} , unless the result is (.+)\. If the result "
        r"is ``NaN``, the \"sign\" of ``NaN`` is implementation-defined\."
    ): BinaryCaseFactory(SignCondFactory(0), ResultSignCheckFactory(1)),
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
        # if cases := parse_unary_docstring(stub.__doc__):
        #     p = pytest.param(stub.__name__, func, cases, id=stub.__name__)
        #     unary_params.append(p)
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
        for cond, result in cases.items():
            if cond(in_):
                good_example = True
                out = float(res[idx])
                f_in = f"{sh.fmt_idx('x', idx)}={in_}"
                f_out = f"{sh.fmt_idx('out', idx)}={out}"
                if result.strict_check:
                    msg = (
                        f"{f_out}, but should be {result.repr_} [{func_name}()]\n"
                        f"{f_in}"
                    )
                    if math.isnan(result.value):
                        assert math.isnan(out), msg
                    else:
                        assert out == result.value, msg
                else:
                    assert math.isfinite(result.value)  # sanity check
                    assert math.isclose(out, result.value, abs_tol=0.1), (
                        f"{f_out}, but should be roughly {result.repr_}={result.value} "
                        f"[{func_name}()]\n"
                        f"{f_in}"
                    )
                break
    assume(good_example)


@pytest.mark.parametrize("func_name, func, cases", binary_params)
@given(
    *hh.two_mutual_arrays(
        dtypes=dh.float_dtypes,
        two_shapes=hh.mutually_broadcastable_shapes(2, min_side=1),
    )
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])  # TODO: remove
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
                assert case.check_result(l, r, o), (
                    f"{f_out} not good [{func_name}()]\n" f"{f_left}, {f_right}"
                )
                break
    assume(good_example)
