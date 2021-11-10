from array_api_tests.algos import broadcast_shapes
import math
from inspect import getfullargspec
from typing import Any, Dict, Optional, Tuple, Union

from . import array_helpers as ah
from . import dtype_helpers as dh
from . import function_stubs
from .typing import Array, DataType, Scalar, Shape

__all__ = [
    "raises",
    "doesnt_raise",
    "nargs",
    "fmt_kw",
    "assert_dtype",
    "assert_kw_dtype",
    "assert_default_float",
    "assert_default_int",
    "assert_shape",
    "assert_result_shape",
    "assert_fill",
]


def raises(exceptions, function, message=""):
    """
    Like pytest.raises() except it allows custom error messages
    """
    try:
        function()
    except exceptions:
        return
    except Exception as e:
        if message:
            raise AssertionError(
                f"Unexpected exception {e!r} (expected {exceptions}): {message}"
            )
        raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions})")
    raise AssertionError(message)


def doesnt_raise(function, message=""):
    """
    The inverse of raises().

    Use doesnt_raise(function) to test that function() doesn't raise any
    exceptions. Returns the result of calling function.
    """
    if not callable(function):
        raise ValueError("doesnt_raise should take a lambda")
    try:
        return function()
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r}: {message}")
        raise AssertionError(f"Unexpected exception {e!r}")


def nargs(func_name):
    return len(getfullargspec(getattr(function_stubs, func_name)).args)


def fmt_kw(kw: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def assert_dtype(
    func_name: str,
    in_dtypes: Union[DataType, Tuple[DataType, ...]],
    out_dtype: DataType,
    expected: Optional[DataType] = None,
    *,
    repr_name: str = "out.dtype",
):
    if not isinstance(in_dtypes, tuple):
        in_dtypes = (in_dtypes,)
    f_in_dtypes = dh.fmt_types(in_dtypes)
    f_out_dtype = dh.dtype_to_name[out_dtype]
    if expected is None:
        expected = dh.result_type(*in_dtypes)
    f_expected = dh.dtype_to_name[expected]
    msg = (
        f"{repr_name}={f_out_dtype}, but should be {f_expected} "
        f"[{func_name}({f_in_dtypes})]"
    )
    assert out_dtype == expected, msg


def assert_kw_dtype(func_name: str, kw_dtype: DataType, out_dtype: DataType):
    f_kw_dtype = dh.dtype_to_name[kw_dtype]
    f_out_dtype = dh.dtype_to_name[out_dtype]
    msg = (
        f"out.dtype={f_out_dtype}, but should be {f_kw_dtype} "
        f"[{func_name}(dtype={f_kw_dtype})]"
    )
    assert out_dtype == kw_dtype, msg


def assert_default_float(func_name: str, dtype: DataType):
    f_dtype = dh.dtype_to_name[dtype]
    f_default = dh.dtype_to_name[dh.default_float]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"floating-point dtype {f_default} [{func_name}()]"
    )
    assert dtype == dh.default_float, msg


def assert_default_int(func_name: str, dtype: DataType):
    f_dtype = dh.dtype_to_name[dtype]
    f_default = dh.dtype_to_name[dh.default_int]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"integer dtype {f_default} [{func_name}()]"
    )
    assert dtype == dh.default_int, msg


def assert_shape(
    func_name: str,
    out_shape: Union[int, Shape],
    expected: Union[int, Shape],
    /,
    repr_name="out.shape",
    **kw,
):
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    if isinstance(expected, int):
        expected = (expected,)
    msg = (
        f"{repr_name}={out_shape}, but should be {expected} [{func_name}({fmt_kw(kw)})]"
    )
    assert out_shape == expected, msg


def assert_result_shape(
    func_name: str,
    in_shapes: Tuple[Shape],
    out_shape: Shape,
    /,
    expected: Optional[Shape] = None,
    *,
    repr_name="out.shape",
    **kw,
):
    if expected is None:
        expected = broadcast_shapes(*in_shapes)
    f_in_shapes = " . ".join(str(s) for s in in_shapes)
    f_sig = f" {f_in_shapes} "
    if kw:
        f_sig += f", {fmt_kw(kw)}"
    msg = f"{repr_name}={out_shape}, but should be {expected} [{func_name}({f_sig})]"
    assert out_shape == expected, msg


def assert_fill(
    func_name: str, fill_value: Scalar, dtype: DataType, out: Array, /, **kw
):
    msg = f"out not filled with {fill_value} [{func_name}({fmt_kw(kw)})]\n{out=}"
    if math.isnan(fill_value):
        assert ah.all(ah.isnan(out)), msg
    else:
        assert ah.all(ah.equal(out, ah.asarray(fill_value, dtype=dtype))), msg
