import inspect
import math
import re
from typing import Callable, Dict, NamedTuple, Pattern
from warnings import warn

import pytest
from attr import dataclass
from hypothesis import HealthCheck, assume, given, settings

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from ._array_module import mod as xp
from .stubs import category_to_funcs


def make_eq(v: float) -> Callable[[float], bool]:
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


def make_rough_eq(v: float) -> Callable[[float], bool]:
    assert math.isfinite(v)  # sanity check

    def rough_eq(i: float) -> bool:
        return math.isclose(i, v, abs_tol=0.01)

    return rough_eq


def make_gt(v: float):
    assert not math.isnan(v)  # sanity check

    def gt(i: float):
        return i > v

    return gt


def make_lt(v: float):
    assert not math.isnan(v)  # sanity check

    def lt(i: float):
        return i < v

    return lt


def make_or(cond1: Callable, cond2: Callable):
    def or_(i: float):
        return cond1(i) or cond2(i)

    return or_


repr_to_value = {
    "NaN": float("nan"),
    "infinity": float("infinity"),
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


r_inline_code = re.compile(r"``([^\s]+)``")
r_approx_value = re.compile(
    rf"an implementation-dependent approximation to {r_inline_code.pattern}"
)


def parse_inline_code(inline_code: str) -> float:
    if m := r_inline_code.match(inline_code):
        return parse_value(m.group(1))
    else:
        raise ValueParseError(inline_code)


class Result(NamedTuple):
    value: float
    repr_: str
    strict_check: bool


def parse_result(s_result: str) -> Result:
    match = None
    if m := r_inline_code.match(s_result):
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
    condition_to_result = {}
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
                condition_to_result[cond] = result
                break
        else:
            if not r_remaining_case.search(case):
                warn(f"case not machine-readable: '{case}'")
    return condition_to_result


binary_pattern_to_condition_factory: Dict[Pattern, Callable] = {
    re.compile(
        "If ``x1_i`` is (.+) and ``x2_i`` is (.+), the result is (.+)"
    ): lambda v1, v2: lambda i1, i2: make_eq(v1)(i1)
    and make_eq(v2)(i2),
}


def parse_binary_docstring(docstring: str) -> Dict[Callable, Result]:
    match = r_special_cases.search(docstring)
    if match is None:
        return {}
    cases = match.group(1).split("\n")[:-1]
    condition_to_result = {}
    for line in cases:
        if m := r_case.match(line):
            case = m.group(1)
        else:
            warn(f"line not machine-readable: '{line}'")
            continue
        for pattern, make_cond in binary_pattern_to_condition_factory.items():
            if m := pattern.search(case):
                *s_values, s_result = m.groups()
                try:
                    values = [parse_inline_code(v) for v in s_values]
                except ValueParseError as e:
                    warn(f"value not machine-readable: '{e.value}'")
                    break
                cond = make_cond(*values)
                if (
                    "atan2" in docstring
                    and ph.is_pos_zero(values[0])
                    and ph.is_neg_zero(values[1])
                ):
                    breakpoint()
                try:
                    result = parse_result(s_result)
                except ValueParseError as e:
                    warn(f"result not machine-readable: '{e.value}'")
                    break
                condition_to_result[cond] = result
                break
        else:
            if not r_remaining_case.search(case):
                warn(f"case not machine-readable: '{case}'")
    return condition_to_result


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
        if condition_to_result := parse_unary_docstring(stub.__doc__):
            p = pytest.param(stub.__name__, func, condition_to_result, id=stub.__name__)
            unary_params.append(p)
        continue
    if len(sig.parameters) == 1:
        warn(f"{func=} has one parameter '{param_names[0]}' which is not named 'x'")
        continue
    if param_names[0] == "x1" and param_names[1] == "x2":
        if condition_to_result := parse_binary_docstring(stub.__doc__):
            p = pytest.param(stub.__name__, func, condition_to_result, id=stub.__name__)
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


@pytest.mark.parametrize("func_name, func, condition_to_result", unary_params)
@given(x=xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_unary(func_name, func, condition_to_result, x):
    res = func(x)
    good_example = False
    for idx in sh.ndindex(res.shape):
        in_ = float(x[idx])
        for cond, result in condition_to_result.items():
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


@pytest.mark.parametrize("func_name, func, condition_to_result", binary_params)
@given(
    *hh.two_mutual_arrays(
        dtypes=dh.float_dtypes,
        two_shapes=hh.mutually_broadcastable_shapes(2, min_side=1),
    )
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])  # TODO: remove
def test_binary(func_name, func, condition_to_result, x1, x2):
    res = func(x1, x2)
    good_example = False
    for l_idx, r_idx, o_idx in sh.iter_indices(x1.shape, x2.shape, res.shape):
        l = float(x1[l_idx])
        r = float(x2[r_idx])
        for cond, result in condition_to_result.items():
            if cond(l, r):
                good_example = True
                out = float(res[o_idx])
                f_left = f"{sh.fmt_idx('x1', l_idx)}={l}"
                f_right = f"{sh.fmt_idx('x2', r_idx)}={r}"
                f_out = f"{sh.fmt_idx('out', o_idx)}={out}"
                if result.strict_check:
                    msg = (
                        f"{f_out}, but should be {result.repr_} [{func_name}()]\n"
                        f"{f_left}, {f_right}"
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
                        f"{f_left}, {f_right}"
                    )
                break
    assume(good_example)
