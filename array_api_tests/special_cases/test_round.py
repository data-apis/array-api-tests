"""
Special cases tests for round.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (assert_exactly_equal, ceil, equal, floor, greater, isintegral,
                             subtract, where, zero)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import round

from hypothesis import given


@given(numeric_arrays)
def test_round_special_cases_one_arg_equal(arg1):
    """
    Special case test for `round(x)`:

        -   If `x_i` is already integer-valued, the result is `x_i`.

    """
    res = round(arg1)
    mask = isintegral(arg1)
    assert_exactly_equal(res[mask], arg1)


@given(numeric_arrays)
def test_round_special_cases_one_arg_two_integers_equally_close(arg1):
    """
    Special case test for `round(x)`:

        -   If two integers are equally close to `x_i`, the result is whichever integer is farthest from `0`.

    """
    res = round(arg1)
    mask = equal(subtract(arg1, floor(arg1)), subtract(ceil(arg1), arg1))
    assert_exactly_equal(res[mask], where(greater(abs(subtract(zero(arg1.dtype), floor(arg1))), abs(subtract(ceil(arg1), zero(arg1.dtype)))), floor(arg1), ceil(arg1)))
