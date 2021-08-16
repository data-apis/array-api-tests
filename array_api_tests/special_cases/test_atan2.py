"""
Special cases tests for atan2.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, exactly_equal, greater, infinity, isfinite,
                             less, logical_and, logical_or, zero, π)
from ..hypothesis_helpers import floating_arrays, broadcastable_floating_array_pairs
from .._array_module import atan2

from hypothesis import given


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_either(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_greater__equal_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `+0`, the result is an implementation-dependent approximation to `+π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_greater__equal_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `+π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__greater_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is greater than `0`, the result is `+0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__greater_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is greater than `0`, the result is `-0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `+0`, the result is `+0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `+π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_3(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `+0`, the result is `-0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_4(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `-π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_5(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is finite, the result is an implementation-dependent approximation to `+π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_6(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is finite, the result is an implementation-dependent approximation to `-π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_7(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `+infinity`, the result is an implementation-dependent approximation to `+π/4`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/4)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_8(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `+3π/4`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+3*π(arg1.shape, arg1.dtype)/4)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_9(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `+infinity`, the result is an implementation-dependent approximation to `-π/4`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/4)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__equal_10(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `-3π/4`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-3*π(arg1.shape, arg1.dtype)/4)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__less_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is less than `0`, the result is an implementation-dependent approximation to `+π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_equal__less_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is less than `0`, the result is an implementation-dependent approximation to `-π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_less__equal_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `+0`, the result is an implementation-dependent approximation to `-π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_less__equal_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `-π/2`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_greater_equal__equal_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0`, `x1_i` is a finite number, and `x2_i` is `+infinity`, the result is `+0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_greater_equal__equal_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0`, `x1_i` is a finite number, and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `+π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_less_equal__equal_1(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0`, `x1_i` is a finite number, and `x2_i` is `+infinity`, the result is `-0`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(broadcastable_floating_array_pairs())
def test_atan2_special_cases_two_args_less_equal__equal_2(pair):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0`, `x1_i` is a finite number, and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `-π`.

    """
    arg1, arg2 = pair
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])
