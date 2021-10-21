from inspect import getfullargspec
from typing import Tuple

from . import dtype_helpers as dh
from . import function_stubs
from .typing import DataType


def raises(exceptions, function, message=''):
    """
    Like pytest.raises() except it allows custom error messages
    """
    try:
        function()
    except exceptions:
        return
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions}): {message}")
        raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions})")
    raise AssertionError(message)

def doesnt_raise(function, message=''):
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


def assert_dtype(
    func_name: str,
    in_dtypes: Tuple[DataType, ...],
    out_name: str,
    out_dtype: DataType,
    expected: DataType
):
    f_in_dtypes = dh.fmt_types(in_dtypes)
    f_out_dtype = dh.dtype_to_name[out_dtype]
    f_expected = dh.dtype_to_name[expected]
    msg = (
        f"{out_name}={f_out_dtype}, but should be {f_expected} "
        f"[{func_name}({f_in_dtypes})]"
    )
    assert out_dtype == expected, msg


