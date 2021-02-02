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
from hypothesis.strategies import shared, composite, just

from .hypothesis_helpers import (integer_dtypes, floating_dtypes,
                                 numeric_dtypes, integer_or_boolean_dtypes,
                                 boolean_dtypes, mutually_promotable_dtypes,
                                 scalars)

x_integer_dtypes = shared(integer_dtypes, key='x')
x_floating_dtypes = shared(floating_dtypes, key='x')
x_numeric_dtypes = shared(numeric_dtypes, key='x')
x_integer_or_boolean_dtypes = shared(integer_or_boolean_dtypes, key='x')
x_boolean_dtypes = shared(boolean_dtypes, key='x')

x_integer_scalars = scalars(x_integer_dtypes)
x_floating_scalars = scalars(x_floating_dtypes)
x_numeric_scalars = scalars(x_numeric_dtypes)
x_integer_or_boolean_scalars = scalars(x_integer_or_boolean_dtypes)
x_boolean_scalars = scalars(x_boolean_dtypes)

x1_integer_dtypes = shared(integer_dtypes, key='x1')
x1_floating_dtypes = shared(floating_dtypes, key='x1')
x1_numeric_dtypes = shared(numeric_dtypes, key='x1')
x1_integer_or_boolean_dtypes = shared(integer_or_boolean_dtypes, key='x1')
x1_boolean_dtypes = shared(boolean_dtypes, key='x1')

x1_integer_scalars = scalars(x1_integer_dtypes)
x1_floating_scalars = scalars(x1_floating_dtypes)
x1_numeric_scalars = scalars(x1_numeric_dtypes)
x1_integer_or_boolean_scalars = scalars(x1_integer_or_boolean_dtypes)
x1_boolean_scalars = scalars(x1_boolean_dtypes)

x2_integer_dtypes = shared(integer_dtypes, key='x2')
x2_floating_dtypes = shared(floating_dtypes, key='x2')
x2_numeric_dtypes = shared(numeric_dtypes, key='x2')
x2_integer_or_boolean_dtypes = shared(integer_or_boolean_dtypes, key='x2')
x2_boolean_dtypes = shared(boolean_dtypes, key='x2')

x2_integer_scalars = scalars(x2_integer_dtypes)
x2_floating_scalars = scalars(x2_floating_dtypes)
x2_numeric_scalars = scalars(x2_numeric_dtypes)
x2_integer_or_boolean_scalars = scalars(x2_integer_or_boolean_dtypes)
x2_boolean_scalars = scalars(x2_boolean_dtypes)

any_dtypes = shared(mutually_promotable_dtypes())

@composite
def two_scalars(draw, two_dtypes):
    # two_dtypes should be a strategy that returns two dtypes (like
    # mutually_promotable_dtypes())
    dtype1, dtype2 = draw(two_dtypes)
    return draw(scalars(just(dtype1))), draw(scalars(just(dtype2)))

@given(x_numeric_scalars, x_numeric_dtypes)
def test_abs(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_acos(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_acosh(x, x_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_add(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_asin(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_asinh(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_atan(x, x_dtype):
    pass

@given(x1_floating_scalars, x1_floating_dtypes, x2_floating_scalars, x2_floating_dtypes)
def test_atan2(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_atanh(x, x_dtype):
    pass

@given(x1_integer_or_boolean_scalars, x1_integer_or_boolean_dtypes, x2_integer_or_boolean_scalars, x2_integer_or_boolean_dtypes)
def test_bitwise_and(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_integer_scalars, x1_integer_dtypes, x2_integer_scalars, x2_integer_dtypes)
def test_bitwise_left_shift(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_integer_or_boolean_scalars, x_integer_or_boolean_dtypes)
def test_bitwise_invert(x, x_dtype):
    pass

@given(x1_integer_or_boolean_scalars, x1_integer_or_boolean_dtypes, x2_integer_or_boolean_scalars, x2_integer_or_boolean_dtypes)
def test_bitwise_or(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_integer_scalars, x1_integer_dtypes, x2_integer_scalars, x2_integer_dtypes)
def test_bitwise_right_shift(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_integer_or_boolean_scalars, x1_integer_or_boolean_dtypes, x2_integer_or_boolean_scalars, x2_integer_or_boolean_dtypes)
def test_bitwise_xor(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_ceil(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_cos(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_cosh(x, x_dtype):
    pass

@given(x1_floating_scalars, x1_floating_dtypes, x2_floating_scalars, x2_floating_dtypes)
def test_divide(x1, x1_dtype, x2, x2_dtype):
    pass

@given(two_scalars(any_dtypes), any_dtypes)
def test_equal(args, dtypes):
    x1, x2 = args
    x1_dtype, x2_dtype = dtypes

@given(x_floating_scalars, x_floating_dtypes)
def test_exp(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_expm1(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_floor(x, x_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_floor_divide(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_greater(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_greater_equal(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_isfinite(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_isinf(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_isnan(x, x_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_less(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_less_equal(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_log(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_log1p(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_log2(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_log10(x, x_dtype):
    pass

@given(x1_boolean_scalars, x1_boolean_dtypes, x2_boolean_scalars, x2_boolean_dtypes)
def test_logical_and(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_boolean_scalars, x_boolean_dtypes)
def test_logical_not(x, x_dtype):
    pass

@given(x1_boolean_scalars, x1_boolean_dtypes, x2_boolean_scalars, x2_boolean_dtypes)
def test_logical_or(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_boolean_scalars, x1_boolean_dtypes, x2_boolean_scalars, x2_boolean_dtypes)
def test_logical_xor(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_multiply(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_negative(x, x_dtype):
    pass

@given(two_scalars(any_dtypes), any_dtypes)
def test_not_equal(args, dtypes):
    x1, x2 = args
    x1_dtype, x2_dtype = dtypes

@given(x_numeric_scalars, x_numeric_dtypes)
def test_positive(x, x_dtype):
    pass

@given(x1_floating_scalars, x1_floating_dtypes, x2_floating_scalars, x2_floating_dtypes)
def test_pow(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_remainder(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_round(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_sign(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_sin(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_sinh(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_square(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_sqrt(x, x_dtype):
    pass

@given(x1_numeric_scalars, x1_numeric_dtypes, x2_numeric_scalars, x2_numeric_dtypes)
def test_subtract(x1, x1_dtype, x2, x2_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_tan(x, x_dtype):
    pass

@given(x_floating_scalars, x_floating_dtypes)
def test_tanh(x, x_dtype):
    pass

@given(x_numeric_scalars, x_numeric_dtypes)
def test_trunc(x, x_dtype):
    pass


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
