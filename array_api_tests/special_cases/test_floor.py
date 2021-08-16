"""
Special cases tests for floor.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import assert_exactly_equal, isintegral
from ..hypothesis_helpers import floating_arrays, broadcastable_floating_array_pairs
from .._array_module import floor

from hypothesis import given


@given(floating_arrays)
def test_floor_special_cases_one_arg_equal(arg1):
    """
    Special case test for `floor(x, /)`:

        -   If `x_i` is already integer-valued, the result is `x_i`.

    """
    res = floor(arg1)
    mask = isintegral(arg1)
    assert_exactly_equal(res[mask], (arg1)[mask])
