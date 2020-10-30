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

def cholesky():
    pass

def cross(x1, x2, *, axis=-1):
    pass

def det(x):
    pass

def diagonal(x, *, axis1=0, axis2=1, offset=0):
    pass

def dot():
    pass

def eig():
    pass

def eigvalsh():
    pass

def einsum():
    pass

def inv(x):
    pass

def lstsq():
    pass

def matmul():
    pass

def matrix_power():
    pass

def matrix_rank():
    pass

def norm(x, *, axis=None, keepdims=False, ord=None):
    pass

def outer(x1, x2):
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

def trace(x, *, axis1=0, axis2=1, offset=0):
    pass

def transpose(x, *, axes=None):
    pass

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'dot', 'eig', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'trace', 'transpose']
