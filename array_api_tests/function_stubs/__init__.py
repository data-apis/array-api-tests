"""
Stub definitions for functions defined in the spec

These are used to test function signatures.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

__all__ = []

from .array_object import __abs__, __add__, __and__, __eq__, __floordiv__, __ge__, __getitem__, __gt__, __invert__, __le__, __len__, __lshift__, __lt__, __matmul__, __mod__, __mul__, __ne__, __neg__, __or__, __pos__, __pow__, __rshift__, __setitem__, __sub__, __truediv__, __xor__, dtype, ndim, shape, size, T

__all__ += ['__abs__', '__add__', '__and__', '__eq__', '__floordiv__', '__ge__', '__getitem__', '__gt__', '__invert__', '__le__', '__len__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__or__', '__pos__', '__pow__', '__rshift__', '__setitem__', '__sub__', '__truediv__', '__xor__', 'dtype', 'ndim', 'shape', 'size', 'T']

from .constants import e, inf, nan, pi

__all__ += ['e', 'inf', 'nan', 'pi']

from .creation_functions import arange, empty, empty_like, eye, full, full_like, linspace, ones, ones_like, zeros, zeros_like

__all__ += ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']

from .elementwise_functions import abs, acos, acosh, add, asin, asinh, atan, atan2, atanh, bitwise_and, bitwise_left_shift, bitwise_invert, bitwise_or, bitwise_right_shift, bitwise_xor, ceil, cos, cosh, divide, equal, exp, expm1, floor, floor_divide, greater, greater_equal, isfinite, isinf, isnan, less, less_equal, log, log1p, log2, log10, logical_and, logical_not, logical_or, logical_xor, multiply, negative, not_equal, positive, pow, remainder, round, sign, sin, sinh, square, sqrt, subtract, tan, tanh, trunc

__all__ += ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor', 'floor_divide', 'greater', 'greater_equal', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'multiply', 'negative', 'not_equal', 'positive', 'pow', 'remainder', 'round', 'sign', 'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh', 'trunc']

from .linear_algebra_functions import cholesky, cross, det, diagonal, dot, eig, eigvalsh, einsum, inv, lstsq, matmul, matrix_power, matrix_rank, norm, outer, pinv, qr, slogdet, solve, svd, trace, transpose

__all__ += ['cholesky', 'cross', 'det', 'diagonal', 'dot', 'eig', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'trace', 'transpose']

from .manipulation_functions import concat, expand_dims, flip, reshape, roll, squeeze, stack

__all__ += ['concat', 'expand_dims', 'flip', 'reshape', 'roll', 'squeeze', 'stack']

from .searching_functions import argmax, argmin, nonzero, where

__all__ += ['argmax', 'argmin', 'nonzero', 'where']

from .set_functions import unique

__all__ += ['unique']

from .sorting_functions import argsort, sort

__all__ += ['argsort', 'sort']

from .statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

from .utility_functions import all, any

__all__ += ['all', 'any']
