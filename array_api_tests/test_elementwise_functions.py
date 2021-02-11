"""
Tests for elementwise functions

https://data-apis.github.io/array-api/latest/API_specification/elementwise_functions.html

This tests behavior that is explicitly mentioned in the spec. Note that the
spec does not make any accuracy requirements for functions, so this does not
test that. Tests for the special cases are generated and tested separately in
special_cases/

Note: Due to current limitations in Hypothesis, the tests below only test
arrays of shape (1,). In the future, the tests should be updated to test
arrays of any shape, using masking patterns (similar to the tests in special_cases/

"""

from hypothesis import given
from hypothesis.strategies import composite, just

from .hypothesis_helpers import (integer_dtype_objects,
                                 floating_dtype_objects,
                                 numeric_dtype_objects,
                                 integer_or_boolean_dtype_objects,
                                 boolean_dtype_objects, floating_dtypes,
                                 numeric_dtypes, integer_or_boolean_dtypes,
                                 boolean_dtypes, mutually_promotable_dtypes,
                                 array_scalars)

from . import _array_module

# integer_scalars = array_scalars(integer_dtypes)
floating_scalars = array_scalars(floating_dtypes)
numeric_scalars = array_scalars(numeric_dtypes)
integer_or_boolean_scalars = array_scalars(integer_or_boolean_dtypes)
boolean_scalars = array_scalars(boolean_dtypes)

two_integer_dtypes = mutually_promotable_dtypes(integer_dtype_objects)
two_floating_dtypes = mutually_promotable_dtypes(floating_dtype_objects)
two_numeric_dtypes = mutually_promotable_dtypes(numeric_dtype_objects)
two_integer_or_boolean_dtypes = mutually_promotable_dtypes(integer_or_boolean_dtype_objects)
two_boolean_dtypes = mutually_promotable_dtypes(boolean_dtype_objects)
two_any_dtypes = mutually_promotable_dtypes()

@composite
def two_array_scalars(draw, dtype1, dtype2):
    # two_dtypes should be a strategy that returns two dtypes (like
    # mutually_promotable_dtypes())
    return draw(array_scalars(just(dtype1))), draw(array_scalars(just(dtype2)))

def sanity_check(x1, x2):
    from .test_type_promotion import promotion_table, dtype_mapping
    t1 = [i for i in dtype_mapping if dtype_mapping[i] == x1.dtype][0]
    t2 = [i for i in dtype_mapping if dtype_mapping[i] == x2.dtype][0]

    if (t1, t2) not in promotion_table:
        raise RuntimeError("Error in test generation (probably a bug in the test suite")

input_types = {
    'abs': 'numeric',
    'acos': 'floating',
    'acosh': 'floating',
    'add': 'numeric',
    'asin': 'floating',
    'asinh': 'floating',
    'atan': 'floating',
    'atan2': 'floating',
    'atanh': 'floating',
    'bitwise_and': 'integer_or_boolean',
    'bitwise_invert': 'integer_or_boolean',
    'bitwise_left_shift': 'integer',
    'bitwise_or': 'integer_or_boolean',
    'bitwise_right_shift': 'integer',
    'bitwise_xor': 'integer_or_boolean',
    'ceil': 'numeric',
    'cos': 'floating',
    'cosh': 'floating',
    'divide': 'floating',
    'equal': 'any',
    'exp': 'floating',
    'expm1': 'floating',
    'floor': 'numeric',
    'floor_divide': 'numeric',
    'greater': 'numeric',
    'greater_equal': 'numeric',
    'isfinite': 'numeric',
    'isinf': 'numeric',
    'isnan': 'numeric',
    'less': 'numeric',
    'less_equal': 'numeric',
    'log': 'floating',
    'log10': 'floating',
    'log1p': 'floating',
    'log2': 'floating',
    'logical_and': 'boolean',
    'logical_not': 'boolean',
    'logical_or': 'boolean',
    'logical_xor': 'boolean',
    'multiply': 'numeric',
    'negative': 'numeric',
    'not_equal': 'any',
    'positive': 'numeric',
    'pow': 'floating',
    'remainder': 'numeric',
    'round': 'numeric',
    'sign': 'numeric',
    'sin': 'floating',
    'sinh': 'floating',
    'sqrt': 'floating',
    'square': 'numeric',
    'subtract': 'numeric',
    'tan': 'floating',
    'tanh': 'floating',
    'trunc': 'numeric',
}
@given(numeric_scalars)
def test_abs(x):
    a = _array_module.abs(x)

@given(floating_scalars)
def test_acos(x):
    a = _array_module.acos(x)

@given(floating_scalars)
def test_acosh(x):
    a = _array_module.acosh(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_add(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.add(x1, x2)

@given(floating_scalars)
def test_asin(x):
    a = _array_module.asin(x)

@given(floating_scalars)
def test_asinh(x):
    a = _array_module.asinh(x)

@given(floating_scalars)
def test_atan(x):
    a = _array_module.atan(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_atan2(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.atan2(x1, x2)

@given(floating_scalars)
def test_atanh(x):
    a = _array_module.atanh(x)

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_and(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_and(x1, x2)

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_left_shift(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_left_shift(x1, x2)

@given(integer_or_boolean_scalars)
def test_bitwise_invert(x):
    a = _array_module.bitwise_invert(x)

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_or(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_or(x1, x2)

@given(two_integer_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_right_shift(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_right_shift(x1, x2)

@given(two_integer_or_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_bitwise_xor(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.bitwise_xor(x1, x2)

@given(numeric_scalars)
def test_ceil(x):
    a = _array_module.ceil(x)

@given(floating_scalars)
def test_cos(x):
    a = _array_module.cos(x)

@given(floating_scalars)
def test_cosh(x):
    a = _array_module.cosh(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.divide(x1, x2)

@given(two_any_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.equal(x1, x2)

@given(floating_scalars)
def test_exp(x):
    a = _array_module.exp(x)

@given(floating_scalars)
def test_expm1(x):
    a = _array_module.expm1(x)

@given(numeric_scalars)
def test_floor(x):
    a = _array_module.floor(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_floor_divide(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.floor_divide(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_greater_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.greater_equal(x1, x2)

@given(numeric_scalars)
def test_isfinite(x):
    a = _array_module.isfinite(x)

@given(numeric_scalars)
def test_isinf(x):
    a = _array_module.isinf(x)

@given(numeric_scalars)
def test_isnan(x):
    a = _array_module.isnan(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_less_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.less_equal(x1, x2)

@given(floating_scalars)
def test_log(x):
    a = _array_module.log(x)

@given(floating_scalars)
def test_log1p(x):
    a = _array_module.log1p(x)

@given(floating_scalars)
def test_log2(x):
    a = _array_module.log2(x)

@given(floating_scalars)
def test_log10(x):
    a = _array_module.log10(x)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_and(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_and(x1, x2)

@given(boolean_scalars)
def test_logical_not(x):
    a = _array_module.logical_not(x)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_or(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_or(x1, x2)

@given(two_boolean_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_logical_xor(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.logical_xor(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_multiply(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.multiply(x1, x2)

@given(numeric_scalars)
def test_negative(x):
    a = _array_module.negative(x)

@given(two_any_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_not_equal(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.not_equal(x1, x2)

@given(numeric_scalars)
def test_positive(x):
    a = _array_module.positive(x)

@given(two_floating_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_pow(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.pow(x1, x2)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_remainder(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.remainder(x1, x2)

@given(numeric_scalars)
def test_round(x):
    a = _array_module.round(x)

@given(numeric_scalars)
def test_sign(x):
    a = _array_module.sign(x)

@given(floating_scalars)
def test_sin(x):
    a = _array_module.sin(x)

@given(floating_scalars)
def test_sinh(x):
    a = _array_module.sinh(x)

@given(numeric_scalars)
def test_square(x):
    a = _array_module.square(x)

@given(floating_scalars)
def test_sqrt(x):
    a = _array_module.sqrt(x)

@given(two_numeric_dtypes.flatmap(lambda i: two_array_scalars(*i)))
def test_subtract(args):
    x1, x2 = args
    sanity_check(x1, x2)
    a = _array_module.subtract(x1, x2)

@given(floating_scalars)
def test_tan(x):
    a = _array_module.tan(x)

@given(floating_scalars)
def test_tanh(x):
    a = _array_module.tanh(x)

@given(numeric_scalars)
def test_trunc(x):
    a = _array_module.trunc(x)
