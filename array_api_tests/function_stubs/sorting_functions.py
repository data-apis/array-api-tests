"""
Function stubs for sorting functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/sorting_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

from ._types import array

def argsort(x: array, *, axis: int = -1, descending: bool = False, stable: bool = True):
    pass

def sort(x: array, *, axis: int = -1, descending: bool = False, stable: bool = True):
    pass

__all__ = ['argsort', 'sort']
