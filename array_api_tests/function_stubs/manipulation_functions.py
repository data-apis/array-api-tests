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

from ._types import Optional, Tuple, Union, array

def concat(arrays: Tuple[array], *, axis: Optional[int] = 0) -> array:
    pass

def expand_dims(x: array, axis: int) -> array:
    pass

def flip(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    pass

def reshape(x: array, shape: Tuple[int, ...]) -> array:
    pass

def roll(x: array, shift: Union[int, Tuple[int, ...]], *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    pass

def squeeze(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    pass

def stack(arrays: Tuple[array], *, axis: Optional[int] = 0) -> array:
    pass

__all__ = ['concat', 'expand_dims', 'flip', 'reshape', 'roll', 'squeeze', 'stack']
