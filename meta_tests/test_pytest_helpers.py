from pytest import raises

from array_api_tests import _array_module as xp
from array_api_tests import pytest_helpers as ph


def test_assert_dtype():
    ph.assert_dtype("promoted_func", in_dtype=[xp.uint8, xp.int8], out_dtype=xp.int16)
    with raises(AssertionError):
        ph.assert_dtype("bad_func", in_dtype=[xp.uint8, xp.int8], out_dtype=xp.float32)
    ph.assert_dtype("bool_func", in_dtype=[xp.uint8, xp.int8], out_dtype=xp.bool, expected=xp.bool)
    ph.assert_dtype("single_promoted_func", in_dtype=[xp.uint8], out_dtype=xp.uint8)
    ph.assert_dtype("single_bool_func", in_dtype=[xp.uint8], out_dtype=xp.bool, expected=xp.bool)


def test_assert_array_elements():
    ph.assert_array_elements("int zeros", out=xp.asarray(0), expected=xp.asarray(0))
    ph.assert_array_elements("pos zeros", out=xp.asarray(0.0), expected=xp.asarray(0.0))
    with raises(AssertionError):
        ph.assert_array_elements("mixed sign zeros", out=xp.asarray(0.0), expected=xp.asarray(-0.0))
    with raises(AssertionError):
        ph.assert_array_elements("mixed sign zeros", out=xp.asarray(-0.0), expected=xp.asarray(0.0))

    ph.assert_array_elements("nans", out=xp.asarray(float("nan")), expected=xp.asarray(float("nan")))
    with raises(AssertionError):
        ph.assert_array_elements("nan and zero", out=xp.asarray(float("nan")), expected=xp.asarray(0.0))
