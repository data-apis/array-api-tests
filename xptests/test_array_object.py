import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .typing import DataType, Param, ScalarType


@given(hh.shapes(), st.data())
def test_getitem(shape, data):
    x = data.draw(xps.arrays(dtype=xps.scalar_dtypes(), shape=shape), label="x")
    key = data.draw(xps.indices(shape=shape), label="key")

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)

    _key = tuple(key) if isinstance(key, tuple) else (key,)
    if Ellipsis in _key:
        start_a = _key.index(Ellipsis)
        stop_a = start_a + (len(shape) - (len(_key) - 1))
        slices = tuple(slice(None, None) for _ in range(start_a, stop_a))
        _key = _key[:start_a] + slices + _key[start_a + 1 :]
    expected = []
    for a, i in enumerate(_key):
        if isinstance(i, slice):
            r = range(shape[a])[i]
            expected.append(len(r))
    expected = tuple(expected)
    ph.assert_shape("__getitem__", out.shape, expected)

    # TODO: fold in all remaining concepts from test_indexing.py


# TODO: test_setitem


def make_param(method_name: str, dtype: DataType, stype: ScalarType) -> Param:
    return pytest.param(
        method_name, dtype, stype, id=f"{method_name}({dh.dtype_to_name[dtype]})"
    )


@pytest.mark.parametrize(
    "method_name, dtype, stype",
    [make_param("__bool__", xp.bool, bool)]
    + [make_param("__int__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__index__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__float__", d, float) for d in dh.float_dtypes],
)
@given(data=st.data())
def test_duck_typing(method_name, dtype, stype, data):
    x = data.draw(xps.arrays(dtype, shape=()), label="x")
    method = getattr(x, method_name)
    out = method()
    assert isinstance(
        out, stype
    ), f"{method_name}({x})={out}, which is not a {stype.__name__} scalar"
