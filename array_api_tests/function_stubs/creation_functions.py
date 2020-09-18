"""
Function stubs for creation functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/creation_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

def arange(start, *, stop=None, step=1, dtype=None):
    pass

def empty(shape, *, dtype=None):
    pass

def empty_like(x, *, dtype=None):
    pass

def eye(N, *, M=None, k=0, dtype=None):
    pass

def full(shape, fill_value, *, dtype=None):
    pass

def full_like(x, fill_value, *, dtype=None):
    pass

def linspace(start, stop, num, *, dtype=None, endpoint=True):
    pass

def ones(shape, *, dtype=None):
    pass

def ones_like(x, *, dtype=None):
    pass

def zeros(shape, *, dtype=None):
    pass

def zeros_like(x, *, dtype=None):
    pass

__all__ = ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']
