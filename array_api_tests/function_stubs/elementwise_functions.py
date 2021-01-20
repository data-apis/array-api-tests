"""
Function stubs for elementwise functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/elementwise_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

from ._types import array

def abs(x: array):
    pass

def acos(x: array):
    pass

def acosh(x: array):
    pass

def add(x1: array, x2: array):
    pass

def asin(x: array):
    pass

def asinh(x: array):
    pass

def atan(x: array):
    pass

def atan2(x1: array, x2: array):
    pass

def atanh(x: array):
    pass

def bitwise_and(x1: array, x2: array):
    pass

def bitwise_left_shift(x1: array, x2: array):
    pass

def bitwise_invert(x: array):
    pass

def bitwise_or(x1: array, x2: array):
    pass

def bitwise_right_shift(x1: array, x2: array):
    pass

def bitwise_xor(x1: array, x2: array):
    pass

def ceil(x: array):
    pass

def cos(x: array):
    pass

def cosh(x: array):
    pass

def divide(x1: array, x2: array):
    pass

def equal(x1: array, x2: array):
    pass

def exp(x: array):
    pass

def expm1(x: array):
    pass

def floor(x: array):
    pass

def floor_divide(x1: array, x2: array):
    pass

def greater(x1: array, x2: array):
    pass

def greater_equal(x1: array, x2: array):
    pass

def isfinite(x: array):
    pass

def isinf(x: array):
    pass

def isnan(x: array):
    pass

def less(x1: array, x2: array):
    pass

def less_equal(x1: array, x2: array):
    pass

def log(x: array):
    pass

def log1p(x: array):
    pass

def log2(x: array):
    pass

def log10(x: array):
    pass

def logical_and(x1: array, x2: array):
    pass

def logical_not(x: array):
    pass

def logical_or(x1: array, x2: array):
    pass

def logical_xor(x1: array, x2: array):
    pass

def multiply(x1: array, x2: array):
    pass

def negative(x: array):
    pass

def not_equal(x1: array, x2: array):
    pass

def positive(x: array):
    pass

def pow(x1: array, x2: array):
    pass

def remainder(x1: array, x2: array):
    pass

def round(x: array):
    pass

def sign(x: array):
    pass

def sin(x: array):
    pass

def sinh(x: array):
    pass

def square(x: array):
    pass

def sqrt(x: array):
    pass

def subtract(x1: array, x2: array):
    pass

def tan(x: array):
    pass

def tanh(x: array):
    pass

def trunc(x: array):
    pass

__all__ = ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor', 'floor_divide', 'greater', 'greater_equal', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'multiply', 'negative', 'not_equal', 'positive', 'pow', 'remainder', 'round', 'sign', 'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh', 'trunc']
