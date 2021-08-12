"""
Function stubs for creation functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/creation_functions.md
"""

from __future__ import annotations

from ._types import (List, Optional, SupportsBufferProtocol, SupportsDLPack, Tuple, Union, array,
                     device, dtype)
from collections.abc import Sequence

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def asarray(obj: Union[array, bool, int, float, NestedSequence[bool|int|float], SupportsDLPack, SupportsBufferProtocol], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, copy: Optional[bool] = None) -> array:
    pass

def empty(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def empty_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def eye(n_rows: int, n_cols: Optional[int] = None, /, *, k: Optional[int] = 0, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def from_dlpack(x: object, /) -> array:
    pass

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def full_like(x: array, /, fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *, dtype: Optional[dtype] = None, device: Optional[device] = None, endpoint: bool = True) -> array:
    pass

def meshgrid(*arrays: Sequence[array], indexing: str = 'xy') -> List[array, ...]:
    pass

def ones(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def ones_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def zeros_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

__all__ = ['arange', 'asarray', 'empty', 'empty_like', 'eye', 'from_dlpack', 'full', 'full_like', 'linspace', 'meshgrid', 'ones', 'ones_like', 'zeros', 'zeros_like']
