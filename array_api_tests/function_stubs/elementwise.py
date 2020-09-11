"""
Function stubs for elementwise functions

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/elementwise_functions.md

Note, all non-keyword arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

def abs(x, *, out=None):
    pass

def acos(x, *, out=None):
    pass

def acosh(x, *, out=None):
    pass

def add(x1, x2, *, out=None):
    pass

def asin(x, *, out=None):
    pass

def asinh(x, *, out=None):
    pass

def atan(x, *, out=None):
    pass

def atanh(x, *, out=None):
    pass

def ceil(x, *, out=None):
    pass

def cos(x, *, out=None):
    pass

def cosh(x, *, out=None):
    pass

def divide(x1, x2, *, out=None):
    pass

def equal(x1, x2, *, out=None):
    pass

def exp(x, *, out=None):
    pass

def floor(x, *, out=None):
    pass

def greater(x1, x2, *, out=None):
    pass

def greater_equal(x1, x2, *, out=None):
    pass

def less(x1, x2, *, out=None):
    pass

def less_equal(x1, x2, *, out=None):
    pass

def log(x, *, out=None):
    pass

def logical_and(x1, x2, *, out=None):
    pass

def logical_not(x, *, out=None):
    pass

def logical_or(x1, x2, *, out=None):
    pass

def logical_xor(x1, x2, *, out=None):
    pass

def multiply(x1, x2, *, out=None):
    pass

def not_equal(x1, x2, *, out=None):
    pass

def round(x, *, out=None):
    pass

def sin(x, *, out=None):
    pass

def sinh(x, *, out=None):
    pass

def sqrt(x, *, out=None):
    pass

def subtract(x1, x2, *, out=None):
    pass

def tan(x, *, out=None):
    pass

def tanh(x, *, out=None):
    pass

def trunc(x, *, out=None):
    pass

_names = sorted([i for i in globals() if not i.startswith('_')])
