"""
Function stubs for linear algebra functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/linear_algebra_functions.md

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""

from ._types import Literal, Optional, Tuple, Union, array
from .constants import inf

def cholesky():
    pass

def cross(x1: array, x2: array, *, axis: int = -1) -> array:
    pass

def det(x: array) -> array:
    pass

def diagonal(x: array, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    pass

def dot():
    pass

def eig():
    pass

def eigvalsh():
    pass

def einsum():
    pass

def inv(x: array) -> array:
    pass

def lstsq():
    pass

def matmul():
    pass

def matrix_power():
    pass

def matrix_rank():
    pass

def norm(x: array, *, axis: Optional[Union[int, Tuple[int, int]]] = None, keepdims: bool = False, ord: Optional[int, float, Literal[inf, -inf, 'fro', 'nuc']] = None) -> array:
    pass

def outer(x1: array, x2: array) -> array:
    pass

def pinv():
    pass

def qr():
    pass

def slogdet():
    pass

def solve():
    pass

def svd():
    pass

def trace(x: array, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    pass

def transpose(x: array, *, axes: Optional[Tuple[int, ...]] = None) -> array:
    pass

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'dot', 'eig', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'trace', 'transpose']
