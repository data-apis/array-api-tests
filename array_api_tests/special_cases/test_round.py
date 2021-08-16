"""
Special cases tests for round.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (assert_exactly_equal, assert_iseven, assert_positive, ceil, equal,
                             floor, isintegral, logical_and, not_equal, one, subtract)
from ..hypothesis_helpers import floating_arrays, broadcastable_floating_array_pairs
from .._array_module import round

from hypothesis import given


@given(floating_arrays)
def test_round_special_cases_one_arg_equal(arg1):
    """
    Special case test for `round(x, /)`:

        -   If `x_i` is already integer-valued, the result is `x_i`.

    """
    res = round(arg1)
    mask = isintegral(arg1)
    assert_exactly_equal(res[mask], (arg1)[mask])


@given(floating_arrays)
def test_round_special_cases_one_arg_two_integers_equally_close(arg1):
    """
    Special case test for `round(x, /)`:

        -   If two integers are equally close to `x_i`, the result is the even integer closest to `x_i`.

    """
    res = round(arg1)
    mask = logical_and(not_equal(floor(arg1), ceil(arg1)), equal(subtract(arg1, floor(arg1)), subtract(ceil(arg1), arg1)))
    assert_iseven(res[mask])
    assert_positive(subtract(one(arg1[mask].shape, arg1[mask].dtype), abs(subtract(arg1[mask], res[mask]))))
