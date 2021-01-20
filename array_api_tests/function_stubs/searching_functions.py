"""
Function stubs for searching functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/searching_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

from ._types import array

def argmax(x: array, *, axis: int = None, keepdims: bool = False):
    pass

def argmin(x: array, *, axis: int = None, keepdims: bool = False):
    pass

def nonzero(x: array):
    pass

def where(condition: array, x1: array, x2: array):
    pass

__all__ = ['argmax', 'argmin', 'nonzero', 'where']
