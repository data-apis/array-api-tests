"""
Function stubs for manipulation functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/manipulation_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

def concat(arrays, *, axis=0):
    pass

def expand_dims(x, axis):
    pass

def flip(x, *, axis=None):
    pass

def reshape(x, shape):
    pass

def roll(x, shift, *, axis=None):
    pass

def squeeze(x, *, axis=None):
    pass

def stack(arrays, *, axis=0):
    pass

__all__ = ['concat', 'expand_dims', 'flip', 'reshape', 'roll', 'squeeze', 'stack']
