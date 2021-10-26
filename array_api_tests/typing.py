from typing import Tuple, Type, Union, Any

__all__ = [
    "DataType",
    "ScalarType",
    "Array",
    "Shape",
    "Param",
]

DataType = Type[Any]
ScalarType = Union[Type[bool], Type[int], Type[float]]
Array = Any
Shape = Tuple[int, ...]
Param = Tuple
