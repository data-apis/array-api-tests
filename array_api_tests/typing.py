from typing import Tuple, Type, Union, Any

__all__ = [
    "DataType",
    "ScalarType",
    "Param",
]

DataType = Type[Any]
ScalarType = Union[Type[bool], Type[int], Type[float]]
Param = Tuple
