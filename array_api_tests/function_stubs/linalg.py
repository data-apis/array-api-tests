"""
Function stubs for linear algebra functions (Extension).

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/linear_algebra_functions.md
"""

from __future__ import annotations


def cholesky(x, /, *, upper=False):
    pass

def cross(x1, x2, /, *, axis=-1):
    pass

def det(x, /):
    pass

def diagonal(x, /, *, axis1=0, axis2=1, offset=0):
    pass

def eig():
    pass

def eigh(x, /, *, upper=False):
    pass

def eigvals():
    pass

def eigvalsh(x, /, *, upper=False):
    pass

def einsum():
    pass

def inv(x, /):
    pass

def lstsq(x1, x2, /, *, rtol=None):
    pass

def matmul(x1, x2, /):
    pass

def matrix_power(x, n, /):
    pass

def matrix_rank(x, /, *, rtol=None):
    pass

def norm(x, /, *, axis=None, keepdims=False, ord=None):
    pass

def outer(x1, x2, /):
    pass

def pinv(x, /, *, rtol=None):
    pass

def qr(x, /, *, mode='reduced'):
    pass

def slogdet(x, /):
    pass

def solve(x1, x2, /):
    pass

def svd(x, /, *, full_matrices=True):
    pass

def tensordot(x1, x2, /, *, axes=2):
    pass

def svdvals(x, /):
    pass

def trace(x, /, *, axis1=0, axis2=1, offset=0):
    pass

def transpose(x, /, *, axes=None):
    pass

def vecdot(x1, x2, /, *, axis=None):
    pass

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'tensordot', 'svdvals', 'trace', 'transpose', 'vecdot']
