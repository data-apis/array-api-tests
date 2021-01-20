"""
Function stubs for statistical functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/statistical_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

from ._types import Optional, Tuple, Union, array

def max(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    pass

def mean(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    pass

def min(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    pass

def prod(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    pass

def std(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False):
    pass

def sum(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
    pass

def var(x: array, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False):
    pass

__all__ = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']
