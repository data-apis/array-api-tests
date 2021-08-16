"""
Special cases tests for logaddexp.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import NaN, assert_exactly_equal, exactly_equal, infinity, logical_or
from ..hypothesis_helpers import floating_arrays, broadcastable_floating_array_pairs
from .._array_module import logaddexp

from hypothesis import given


@given(broadcastable_floating_array_pairs())
def test_logaddexp_special_cases_two_args_either_1(pair):
    """
    Special case test for `logaddexp(x1, x2)`:

        - If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    arg1, arg2 = pair
    res = logaddexp(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_logaddexp_special_cases_two_args_either_2(pair):
    """
    Special case test for `logaddexp(x1, x2)`:

        - If either `x1_i` or `x2_i` is `+infinity`, the result is `+infinity`.

    """
    arg1, arg2 = pair
    res = logaddexp(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])
