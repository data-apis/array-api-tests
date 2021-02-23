"""
This file defines the types for type annotations.

The type variables should be replaced with the actual types for a given
library, e.g., for NumPy TypeVar('array') would be replaced with ndarray.
"""

from typing import Literal, Optional, Tuple, Union, TypeVar

array = TypeVar('array')
device = TypeVar('device')
dtype = TypeVar('dtype')
SupportsDLPack = TypeVar('SupportsDLPack')
SupportsBufferProtocol = TypeVar('SupportsBufferProtocol')
PyCapsule = TypeVar('PyCapsule')
__all__ = ['Literal', 'Optional', 'Tuple', 'Union', 'array', 'device',
'dtype', 'SupportsDLPack', 'SupportsBufferProtocol', 'PyCapsule']

