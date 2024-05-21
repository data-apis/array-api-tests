import os
import re
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, DefaultDict, Dict, List, NamedTuple, Sequence, Tuple, Union
from warnings import warn

from . import api_version
from . import xp
from .stubs import name_to_func
from .typing import DataType, ScalarType

__all__ = [
    "uint_names",
    "int_names",
    "all_int_names",
    "real_float_names",
    "real_names",
    "complex_names",
    "numeric_names",
    "dtype_names",
    "int_dtypes",
    "uint_dtypes",
    "all_int_dtypes",
    "real_float_dtypes",
    "real_dtypes",
    "numeric_dtypes",
    "all_dtypes",
    "all_float_dtypes",
    "bool_and_all_int_dtypes",
    "dtype_to_name",
    "kind_to_dtypes",
    "is_int_dtype",
    "is_float_dtype",
    "get_scalar_type",
    "dtype_ranges",
    "default_int",
    "default_uint",
    "default_float",
    "default_complex",
    "promotion_table",
    "dtype_nbits",
    "dtype_signed",
    "dtype_components",
    "func_in_dtypes",
    "func_returns_bool",
    "binary_op_to_symbol",
    "unary_op_to_symbol",
    "inplace_op_to_symbol",
    "op_to_func",
    "fmt_types",
]


class EqualityMapping(Mapping):
    """
    Mapping that uses equality for indexing

    Typical mappings (e.g. the built-in dict) use hashing for indexing. This
    isn't ideal for the Array API, as no __hash__() method is specified for
    dtype objects - but __eq__() is!

    See https://data-apis.org/array-api/latest/API_specification/data_types.html#data-type-objects
    """

    def __init__(self, key_value_pairs: Sequence[Tuple[Any, Any]]):
        keys = [k for k, _ in key_value_pairs]
        for i, key in enumerate(keys):
            if not (key == key):  # specifically checking __eq__, not __neq__
                raise ValueError(f"Key {key!r} does not have equality with itself")
            other_keys = keys[:]
            other_keys.pop(i)
            for other_key in other_keys:
                if key == other_key:
                    raise ValueError(f"Key {key!r} has equality with key {other_key!r}")
        self._key_value_pairs = key_value_pairs

    def __getitem__(self, key):
        for k, v in self._key_value_pairs:
            if key == k:
                return v
        else:
            raise KeyError(f"{key!r} not found")

    def __iter__(self):
        return (k for k, _ in self._key_value_pairs)

    def __len__(self):
        return len(self._key_value_pairs)

    def __str__(self):
        return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in self._key_value_pairs) + "}"

    def __repr__(self):
        return f"EqualityMapping({self})"


uint_names = ("uint8", "uint16", "uint32", "uint64")
int_names = ("int8", "int16", "int32", "int64")
all_int_names = uint_names + int_names
real_float_names = ("float32", "float64")
real_names = uint_names + int_names + real_float_names
complex_names = ("complex64", "complex128")
numeric_names = real_names + complex_names
dtype_names = ("bool",) + numeric_names

_skip_dtypes = os.getenv("ARRAY_API_TESTS_SKIP_DTYPES", '')
_skip_dtypes = _skip_dtypes.split(',')
skip_dtypes = []
for dtype in _skip_dtypes:
    if dtype and dtype not in dtype_names:
        raise ValueError(f"Invalid dtype name in ARRAY_API_TESTS_SKIP_DTYPES: {dtype}")
    skip_dtypes.append(dtype)

_name_to_dtype = {}
for name in dtype_names:
    if name in skip_dtypes:
        continue
    try:
        dtype = getattr(xp, name)
    except AttributeError:
        continue
    _name_to_dtype[name] = dtype
dtype_to_name = EqualityMapping([(d, n) for n, d in _name_to_dtype.items()])


def _make_dtype_tuple_from_names(names: List[str]) -> Tuple[DataType]:
    dtypes = []
    for name in names:
        try:
            dtype = _name_to_dtype[name]
        except KeyError:
            continue
        dtypes.append(dtype)
    return tuple(dtypes)


uint_dtypes = _make_dtype_tuple_from_names(uint_names)
int_dtypes = _make_dtype_tuple_from_names(int_names)
real_float_dtypes = _make_dtype_tuple_from_names(real_float_names)
all_int_dtypes = uint_dtypes + int_dtypes
real_dtypes = all_int_dtypes + real_float_dtypes
complex_dtypes = _make_dtype_tuple_from_names(complex_names)
numeric_dtypes = real_dtypes
if api_version > "2021.12":
    numeric_dtypes += complex_dtypes
all_dtypes = (xp.bool,) + numeric_dtypes
all_float_dtypes = real_float_dtypes
if api_version > "2021.12":
    all_float_dtypes += complex_dtypes
bool_and_all_int_dtypes = (xp.bool,) + all_int_dtypes


kind_to_dtypes = {
    "bool": [xp.bool],
    "signed integer": int_dtypes,
    "unsigned integer": uint_dtypes,
    "integral": all_int_dtypes,
    "real floating": real_float_dtypes,
    "complex floating": complex_dtypes,
    "numeric": numeric_dtypes,
}


def is_int_dtype(dtype):
    return dtype in all_int_dtypes


def is_float_dtype(dtype, *, include_complex=True):
    # None equals NumPy's xp.float64 object, so we specifically check it here.
    # xp.float64 is in fact an alias of np.dtype('float64'), and its equality
    # with None is meant to be deprecated at some point.
    # See https://github.com/numpy/numpy/issues/18434
    if dtype is None:
        return False
    valid_dtypes = real_float_dtypes
    if api_version > "2021.12" and include_complex:
        valid_dtypes += complex_dtypes
    return dtype in valid_dtypes

def get_scalar_type(dtype: DataType) -> ScalarType:
    if dtype in all_int_dtypes:
        return int
    elif dtype in real_float_dtypes:
        return float
    elif dtype in complex_dtypes:
        return complex
    else:
        return bool


def _make_dtype_mapping_from_names(mapping: Dict[str, Any]) -> EqualityMapping:
    dtype_value_pairs = []
    for name, value in mapping.items():
        assert isinstance(name, str) and name in dtype_names  # sanity check
        if name in _name_to_dtype:
            dtype = _name_to_dtype[name]
        else:
            continue
        dtype_value_pairs.append((dtype, value))
    return EqualityMapping(dtype_value_pairs)


class MinMax(NamedTuple):
    min: Union[int, float]
    max: Union[int, float]


dtype_ranges = _make_dtype_mapping_from_names(
    {
        "int8": MinMax(-128, +127),
        "int16": MinMax(-32_768, +32_767),
        "int32": MinMax(-2_147_483_648, +2_147_483_647),
        "int64": MinMax(-9_223_372_036_854_775_808, +9_223_372_036_854_775_807),
        "uint8": MinMax(0, +255),
        "uint16": MinMax(0, +65_535),
        "uint32": MinMax(0, +4_294_967_295),
        "uint64": MinMax(0, +18_446_744_073_709_551_615),
        "float32": MinMax(-3.4028234663852886e38, 3.4028234663852886e38),
        "float64": MinMax(-1.7976931348623157e308, 1.7976931348623157e308),
    }
)


r_nbits = re.compile(r"[a-z]+([0-9]+)")
_dtype_nbits: Dict[str, int] = {}
for name in numeric_names:
    m = r_nbits.fullmatch(name)
    assert m is not None  # sanity check / for mypy
    _dtype_nbits[name] = int(m.group(1))
dtype_nbits = _make_dtype_mapping_from_names(_dtype_nbits)


dtype_signed = _make_dtype_mapping_from_names(
    {**{name: True for name in int_names}, **{name: False for name in uint_names}}
)


dtype_components = _make_dtype_mapping_from_names(
    {"complex64": xp.float32, "complex128": xp.float64}
)

def as_real_dtype(dtype):
    """
    Return the corresponding real dtype for a given floating-point dtype.
    """
    if dtype in real_float_dtypes:
        return dtype
    elif dtype_to_name[dtype] in complex_names:
        return dtype_components[dtype]
    else:
        raise ValueError("as_real_dtype requires a floating-point dtype")

def accumulation_result_dtype(x_dtype, dtype_kwarg):
    """
    Result dtype logic for sum(), prod(), and trace()

    Note: may return None if a default uint cannot exist (e.g., for pytorch
    which doesn't support uint32 or uint64). See https://github.com/data-apis/array-api-tests/issues/106

    """
    if dtype_kwarg is None:
        if is_int_dtype(x_dtype):
            if x_dtype in uint_dtypes:
                default_dtype = default_uint
            else:
                default_dtype = default_int
            if default_dtype is None:
                _dtype = None
            else:
                m, M = dtype_ranges[x_dtype]
                d_m, d_M = dtype_ranges[default_dtype]
                if m < d_m or M > d_M:
                    _dtype = x_dtype
                else:
                    _dtype = default_dtype
        elif is_float_dtype(x_dtype, include_complex=False):
            if dtype_nbits[x_dtype] > dtype_nbits[default_float]:
                _dtype = x_dtype
            else:
                _dtype = default_float
        elif api_version > "2021.12":
            # Complex dtype
            if dtype_nbits[x_dtype] > dtype_nbits[default_complex]:
                _dtype = x_dtype
            else:
                _dtype = default_complex
        else:
            raise RuntimeError("Unexpected dtype. This indicates a bug in the test suite.")
    else:
        _dtype = dtype_kwarg

    return _dtype

if not hasattr(xp, "asarray"):
    default_int = xp.int32
    default_float = xp.float32
    # TODO: when api_version > '2021.12', just assign to xp.complex64,
    # otherwise default to None. Need array-api spec to be bumped first (#187).
    try:
        default_complex = xp.complex64
    except AttributeError:
        default_complex = None
    warn(
        "array module does not have attribute asarray. "
        "default int is assumed int32, default float is assumed float32"
    )
else:
    default_int = xp.asarray(int()).dtype
    if default_int not in int_dtypes:
        warn(f"inferred default int is {default_int!r}, which is not an int")
    default_float = xp.asarray(float()).dtype
    if default_float not in real_float_dtypes:
        warn(f"inferred default float is {default_float!r}, which is not a float")
    if api_version > "2021.12":
        default_complex = xp.asarray(complex()).dtype
        if default_complex not in complex_dtypes:
            warn(
                f"inferred default complex is {default_complex!r}, "
                "which is not a complex"
            )
    else:
        default_complex = None
if dtype_nbits[default_int] == 32:
    default_uint = _name_to_dtype.get("uint32")
else:
    default_uint = _name_to_dtype.get("uint64")

_promotion_table: Dict[Tuple[str, str], str] = {
    ("bool", "bool"): "bool",
    # ints
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    # uints
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint16"): "uint16",
    ("uint8", "uint32"): "uint32",
    ("uint8", "uint64"): "uint64",
    ("uint16", "uint16"): "uint16",
    ("uint16", "uint32"): "uint32",
    ("uint16", "uint64"): "uint64",
    ("uint32", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    ("uint64", "uint64"): "uint64",
    # ints and uints (mixed sign)
    ("int8", "uint8"): "int16",
    ("int8", "uint16"): "int32",
    ("int8", "uint32"): "int64",
    ("int16", "uint8"): "int16",
    ("int16", "uint16"): "int32",
    ("int16", "uint32"): "int64",
    ("int32", "uint8"): "int32",
    ("int32", "uint16"): "int32",
    ("int32", "uint32"): "int64",
    ("int64", "uint8"): "int64",
    ("int64", "uint16"): "int64",
    ("int64", "uint32"): "int64",
    # floats
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
    # complex
    ("complex64", "complex64"): "complex64",
    ("complex64", "complex128"): "complex128",
    ("complex128", "complex128"): "complex128",
}
_promotion_table.update({(d2, d1): res for (d1, d2), res in _promotion_table.items()})
_promotion_table_pairs: List[Tuple[Tuple[DataType, DataType], DataType]] = []
for (in_name1, in_name2), res_name in _promotion_table.items():
    if in_name1 not in _name_to_dtype or in_name2 not in _name_to_dtype or res_name not in _name_to_dtype:
        continue
    in_dtype1 = _name_to_dtype[in_name1]
    in_dtype2 = _name_to_dtype[in_name2]
    res_dtype = _name_to_dtype[res_name]

    _promotion_table_pairs.append(((in_dtype1, in_dtype2), res_dtype))
promotion_table = EqualityMapping(_promotion_table_pairs)


def result_type(*dtypes: DataType):
    if len(dtypes) == 0:
        raise ValueError()
    elif len(dtypes) == 1:
        return dtypes[0]
    result = promotion_table[dtypes[0], dtypes[1]]
    for i in range(2, len(dtypes)):
        result = promotion_table[result, dtypes[i]]
    return result


r_alias = re.compile("[aA]lias")
r_in_dtypes = re.compile("x1?: array\n.+have an? (.+) data type.")
r_int_note = re.compile(
    "If one or both of the input arrays have integer data types, "
    "the result is implementation-dependent"
)
category_to_dtypes = {
    "boolean": (xp.bool,),
    "integer": all_int_dtypes,
    "floating-point": real_float_dtypes,
    "real-valued": real_float_dtypes,
    "real-valued floating-point": real_float_dtypes,
    "complex floating-point": complex_dtypes,
    "numeric": numeric_dtypes,
    "integer or boolean": bool_and_all_int_dtypes,
}
func_in_dtypes: DefaultDict[str, Tuple[DataType, ...]] = defaultdict(lambda: all_dtypes)
for name, func in name_to_func.items():
    assert func.__doc__ is not None  # for mypy
    if m := r_in_dtypes.search(func.__doc__):
        dtype_category = m.group(1)
        if dtype_category == "numeric" and r_int_note.search(func.__doc__):
            dtype_category = "floating-point"
        dtypes = category_to_dtypes[dtype_category]
        func_in_dtypes[name] = dtypes


func_returns_bool = {
    # elementwise
    "abs": False,
    "acos": False,
    "acosh": False,
    "add": False,
    "asin": False,
    "asinh": False,
    "atan": False,
    "atan2": False,
    "atanh": False,
    "bitwise_and": False,
    "bitwise_invert": False,
    "bitwise_left_shift": False,
    "bitwise_or": False,
    "bitwise_right_shift": False,
    "bitwise_xor": False,
    "ceil": False,
    "cos": False,
    "cosh": False,
    "divide": False,
    "equal": True,
    "exp": False,
    "expm1": False,
    "floor": False,
    "floor_divide": False,
    "greater": True,
    "greater_equal": True,
    "isfinite": True,
    "isinf": True,
    "isnan": True,
    "less": True,
    "less_equal": True,
    "log": False,
    "logaddexp": False,
    "log10": False,
    "log1p": False,
    "log2": False,
    "logical_and": True,
    "logical_not": True,
    "logical_or": True,
    "logical_xor": True,
    "multiply": False,
    "negative": False,
    "not_equal": True,
    "positive": False,
    "pow": False,
    "remainder": False,
    "round": False,
    "sign": False,
    "sin": False,
    "sinh": False,
    "sqrt": False,
    "square": False,
    "subtract": False,
    "tan": False,
    "tanh": False,
    "trunc": False,
    # searching
    "where": False,
    # linalg
    "matmul": False,
}


unary_op_to_symbol = {
    "__invert__": "~",
    "__neg__": "-",
    "__pos__": "+",
}


binary_op_to_symbol = {
    "__add__": "+",
    "__and__": "&",
    "__eq__": "==",
    "__floordiv__": "//",
    "__ge__": ">=",
    "__gt__": ">",
    "__le__": "<=",
    "__lshift__": "<<",
    "__lt__": "<",
    "__matmul__": "@",
    "__mod__": "%",
    "__mul__": "*",
    "__ne__": "!=",
    "__or__": "|",
    "__pow__": "**",
    "__rshift__": ">>",
    "__sub__": "-",
    "__truediv__": "/",
    "__xor__": "^",
}


op_to_func = {
    "__abs__": "abs",
    "__add__": "add",
    "__and__": "bitwise_and",
    "__eq__": "equal",
    "__floordiv__": "floor_divide",
    "__ge__": "greater_equal",
    "__gt__": "greater",
    "__le__": "less_equal",
    "__lt__": "less",
    "__matmul__": "matmul",
    "__mod__": "remainder",
    "__mul__": "multiply",
    "__ne__": "not_equal",
    "__or__": "bitwise_or",
    "__pow__": "pow",
    "__lshift__": "bitwise_left_shift",
    "__rshift__": "bitwise_right_shift",
    "__sub__": "subtract",
    "__truediv__": "divide",
    "__xor__": "bitwise_xor",
    "__invert__": "bitwise_invert",
    "__neg__": "negative",
    "__pos__": "positive",
}


# Construct func_in_dtypes and func_returns bool
for op, elwise_func in op_to_func.items():
    func_in_dtypes[op] = func_in_dtypes[elwise_func]
    func_returns_bool[op] = func_returns_bool[elwise_func]
inplace_op_to_symbol = {}
for op, symbol in binary_op_to_symbol.items():
    if op == "__matmul__" or func_returns_bool[op]:
        continue
    iop = f"__i{op[2:]}"
    inplace_op_to_symbol[iop] = f"{symbol}="
    func_in_dtypes[iop] = func_in_dtypes[op]
    func_returns_bool[iop] = func_returns_bool[op]
func_in_dtypes["__bool__"] = (xp.bool,)
func_in_dtypes["__int__"] = all_int_dtypes
func_in_dtypes["__index__"] = all_int_dtypes
func_in_dtypes["__float__"] = real_float_dtypes
func_in_dtypes["from_dlpack"] = numeric_dtypes
func_in_dtypes["__dlpack__"] = numeric_dtypes


@lru_cache
def fmt_types(types: Tuple[Union[DataType, ScalarType], ...]) -> str:
    f_types = []
    for type_ in types:
        try:
            f_types.append(dtype_to_name[type_])
        except KeyError:
            # i.e. dtype is bool, int, or float
            f_types.append(type_.__name__)
    return ", ".join(f_types)
