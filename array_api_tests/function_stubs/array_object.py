"""
Function stubs for array object.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/array_object.md
"""

from __future__ import annotations

from enum import IntEnum
from ._types import Any, Optional, PyCapsule, Tuple, Union, array, ellipsis

def __abs__(self: array, /) -> array:
    """
    Note: __abs__ is a method of the array object.
    """
    pass

def __add__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __add__ is a method of the array object.
    """
    pass

def __and__(self: array, other: Union[int, bool, array], /) -> array:
    """
    Note: __and__ is a method of the array object.
    """
    pass

def __array_namespace__(self: array, /, *, api_version: Optional[str] = None) -> object:
    """
    Note: __array_namespace__ is a method of the array object.
    """
    pass

def __bool__(self: array, /) -> bool:
    """
    Note: __bool__ is a method of the array object.
    """
    pass

def __dlpack__(self: array, /, *, stream: Optional[Union[int, Any]] = None) -> PyCapsule:
    """
    Note: __dlpack__ is a method of the array object.
    """
    pass

def __dlpack_device__(self: array, /) -> Tuple[IntEnum, int]:
    """
    Note: __dlpack_device__ is a method of the array object.
    """
    pass

def __eq__(self: array, other: Union[int, float, bool, array], /) -> array:
    """
    Note: __eq__ is a method of the array object.
    """
    pass

def __float__(self: array, /) -> float:
    """
    Note: __float__ is a method of the array object.
    """
    pass

def __floordiv__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __floordiv__ is a method of the array object.
    """
    pass

def __ge__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __ge__ is a method of the array object.
    """
    pass

def __getitem__(self: array, key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array], /) -> array:
    """
    Note: __getitem__ is a method of the array object.
    """
    pass

def __gt__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __gt__ is a method of the array object.
    """
    pass

def __index__(self: array, /) -> int:
    """
    Note: __index__ is a method of the array object.
    """
    pass

def __int__(self: array, /) -> int:
    """
    Note: __int__ is a method of the array object.
    """
    pass

def __invert__(self: array, /) -> array:
    """
    Note: __invert__ is a method of the array object.
    """
    pass

def __le__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __le__ is a method of the array object.
    """
    pass

def __lshift__(self: array, other: Union[int, array], /) -> array:
    """
    Note: __lshift__ is a method of the array object.
    """
    pass

def __lt__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __lt__ is a method of the array object.
    """
    pass

def __matmul__(self: array, other: array, /) -> array:
    """
    Note: __matmul__ is a method of the array object.
    """
    pass

def __mod__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __mod__ is a method of the array object.
    """
    pass

def __mul__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __mul__ is a method of the array object.
    """
    pass

def __ne__(self: array, other: Union[int, float, bool, array], /) -> array:
    """
    Note: __ne__ is a method of the array object.
    """
    pass

def __neg__(self: array, /) -> array:
    """
    Note: __neg__ is a method of the array object.
    """
    pass

def __or__(self: array, other: Union[int, bool, array], /) -> array:
    """
    Note: __or__ is a method of the array object.
    """
    pass

def __pos__(self: array, /) -> array:
    """
    Note: __pos__ is a method of the array object.
    """
    pass

def __pow__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __pow__ is a method of the array object.
    """
    pass

def __rshift__(self: array, other: Union[int, array], /) -> array:
    """
    Note: __rshift__ is a method of the array object.
    """
    pass

def __setitem__(self: array, key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array], value: Union[int, float, bool, array], /) -> None:
    """
    Note: __setitem__ is a method of the array object.
    """
    pass

def __sub__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __sub__ is a method of the array object.
    """
    pass

def __truediv__(self: array, other: Union[int, float, array], /) -> array:
    """
    Note: __truediv__ is a method of the array object.
    """
    pass

def __xor__(self: array, other: Union[int, bool, array], /) -> array:
    """
    Note: __xor__ is a method of the array object.
    """
    pass

def to_device(self: array, device: device, /, *, stream: Optional[Union[int, Any]] = None) -> array:
    """
    Note: to_device is a method of the array object.
    """
    pass

# Note: dtype is an attribute of the array object.
dtype: dtype = None

# Note: device is an attribute of the array object.
device: device = None

# Note: mT is an attribute of the array object.
mT: array = None

# Note: ndim is an attribute of the array object.
ndim: int = None

# Note: shape is an attribute of the array object.
shape: Tuple[Optional[int], ...] = None

# Note: size is an attribute of the array object.
size: Optional[int] = None

# Note: T is an attribute of the array object.
T: array = None

__all__ = ['__abs__', '__add__', '__and__', '__array_namespace__', '__bool__', '__dlpack__', '__dlpack_device__', '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', '__gt__', '__index__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__or__', '__pos__', '__pow__', '__rshift__', '__setitem__', '__sub__', '__truediv__', '__xor__', 'to_device', 'dtype', 'device', 'mT', 'ndim', 'shape', 'size', 'T']
