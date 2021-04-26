"""
Function stubs for data type functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/data_type_functions.md
"""

from __future__ import annotations

from ._types import Union, array, dtype
from collections.abc import Sequence

def broadcast_arrays(*args):
    pass

def broadcast_to(x, shape, /):
    pass

def can_cast(from_, to, /):
    pass

def finfo(type: Union[dtype, array], /) -> finfo_object:
    pass

def iinfo(type: Union[dtype, array], /) -> iinfo_object:
    pass

def result_type(*arrays_and_dtypes: Sequence[Union[array, dtype]]) -> dtype:
    pass

__all__ = ['broadcast_arrays', 'broadcast_to', 'can_cast', 'finfo', 'iinfo', 'result_type']
