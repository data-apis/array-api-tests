import inspect
import math
import re
from typing import Callable, Dict, NamedTuple, Pattern
from warnings import warn

import pytest
from attr import dataclass
from hypothesis import assume, given

from . import hypothesis_helpers as hh
from . import shape_helpers as sh
from . import xps
from ._array_module import mod as xp
from .stubs import category_to_funcs

repr_to_value = {
    "NaN": float("nan"),
    "+infinity": float("infinity"),
    "infinity": float("infinity"),
    "-infinity": float("-infinity"),
    "+0": 0.0,
    "0": 0.0,
    "-0": -0.0,
    "+1": 1.0,
    "1": 1.0,
    "-1": -1.0,
    "+π/2": math.pi / 2,
    "π/2": math.pi / 2,
    "-π/2": -math.pi / 2,
}


def make_eq(v: float) -> Callable[[float], bool]:
    if math.isnan(v):
        return math.isnan

    def eq(i: float) -> bool:
        return i == v

    return eq


def make_rough_eq(v: float) -> Callable[[float], bool]:
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


r_value = re.compile(r"``([^\s]+)``")
r_approx_value = re.compile(
    rf"an implementation-dependent approximation to {r_value.pattern}"
)


@dataclass
class ValueParseError(ValueError):
    value: str


def parse_value(value: str) -> float:
    if m := r_value.match(value):
        return repr_to_value[m.group(1)]
    raise ValueParseError(value)


class Result(NamedTuple):
    value: float
    repr_: str
    strict_check: bool


def parse_result(result: str) -> Result:
    if m := r_value.match(result):
        repr_ = m.group(1)
        strict_check = True
    elif m := r_approx_value.match(result):
        repr_ = m.group(1)
        strict_check = False
    else:
        raise ValueParseError(result)
    value = repr_to_value[repr_]
    return Result(value, repr_, strict_check)


r_special_cases = re.compile(
    r"\*\*Special [Cc]ases\*\*\n\n\s*"
    r"For floating-point operands,\n\n"
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
                    values = [parse_value(v) for v in s_values]
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


unary_params = []
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
        pass  # TODO
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
def test_unary_special_cases(func_name, func, condition_to_result, x):
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
