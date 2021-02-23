"""
Function stubs for set functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/set_functions.md
"""

from __future__ import annotations

from ._types import Tuple, Union, array

def unique(x: array, /, *, return_counts: bool = False, return_index: bool = False, return_inverse: bool = False) -> Union[array, Tuple[array, ...]]:
    pass

__all__ = ['unique']
