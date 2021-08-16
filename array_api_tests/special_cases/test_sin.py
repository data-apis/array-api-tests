"""
Special cases tests for sin.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import NaN, assert_exactly_equal, exactly_equal, infinity, logical_or, zero
from ..hypothesis_helpers import floating_arrays, broadcastable_floating_array_pairs
from .._array_module import sin

from hypothesis import given


@given(floating_arrays)
def test_sin_special_cases_one_arg_equal_1(arg1):
    """
    Special case test for `sin(x, /)`:

        -   If `x_i` is `NaN`, the result is `NaN`.

    """
    res = sin(arg1)
    mask = exactly_equal(arg1, NaN(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(floating_arrays)
def test_sin_special_cases_one_arg_equal_2(arg1):
    """
    Special case test for `sin(x, /)`:

        -   If `x_i` is `+0`, the result is `+0`.

    """
    res = sin(arg1)
    mask = exactly_equal(arg1, zero(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(floating_arrays)
def test_sin_special_cases_one_arg_equal_3(arg1):
    """
    Special case test for `sin(x, /)`:

        -   If `x_i` is `-0`, the result is `-0`.

    """
    res = sin(arg1)
    mask = exactly_equal(arg1, -zero(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(floating_arrays)
def test_sin_special_cases_one_arg_either(arg1):
    """
    Special case test for `sin(x, /)`:

        -   If `x_i` is either `+infinity` or `-infinity`, the result is `NaN`.

    """
    res = sin(arg1)
    mask = logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])
