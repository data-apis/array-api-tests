import pytest
from hypothesis import given

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from .typing import DataType


def make_dtype_id(dtype: DataType) -> str:
    return dh.dtype_to_name[dtype]


@pytest.mark.parametrize("dtype", dh.float_dtypes, ids=make_dtype_id)
def test_finfo(dtype):
    out = xp.finfo(dtype)
    f_func = f"[finfo({dh.dtype_to_name[dtype]})]"
    for attr, stype in [
        ("bits", int),
        ("eps", float),
        ("max", float),
        ("min", float),
        ("smallest_normal", float),
    ]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, stype
        ), f"type(out.{attr})={type(value)!r}, but should be {stype.__name__} {f_func}"
    # TODO: test values


@pytest.mark.parametrize("dtype", dh.all_int_dtypes, ids=make_dtype_id)
def test_iinfo(dtype):
    out = xp.iinfo(dtype)
    f_func = f"[iinfo({dh.dtype_to_name[dtype]})]"
    for attr in ["bits", "max", "min"]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, int
        ), f"type(out.{attr})={type(value)!r}, but should be int {f_func}"
    # TODO: test values


@given(hh.mutually_promotable_dtypes(None))
def test_result_type(dtypes):
    out = xp.result_type(*dtypes)
    ph.assert_dtype("result_type", dtypes, out, repr_name="out")
