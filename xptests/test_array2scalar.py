import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import xps
from .typing import DataType, Param

method_stype = {
    "__bool__": bool,
    "__int__": int,
    "__index__": int,
    "__float__": float,
}


def make_param(method_name: str, dtype: DataType) -> Param:
    stype = method_stype[method_name]
    if isinstance(dtype, xp._UndefinedStub):
        marks = pytest.mark.skip(reason=f"xp.{dtype.name} not defined")
    else:
        marks = ()
    return pytest.param(
        method_name,
        dtype,
        stype,
        id=f"{method_name}({dh.dtype_to_name[dtype]})",
        marks=marks,
    )


@pytest.mark.parametrize(
    "method_name, dtype, stype",
    [make_param("__bool__", xp.bool)]
    + [make_param("__int__", d) for d in dh.all_int_dtypes]
    + [make_param("__index__", d) for d in dh.all_int_dtypes]
    + [make_param("__float__", d) for d in dh.float_dtypes],
)
@given(data=st.data())
def test_0d_array_can_convert_to_scalar(method_name, dtype, stype, data):
    x = data.draw(xps.arrays(dtype, shape=()), label="x")
    method = getattr(x, method_name)
    out = method()
    assert isinstance(
        out, stype
    ), f"{method_name}({x})={out}, which is not a {stype.__name__} scalar"
