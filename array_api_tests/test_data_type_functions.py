from hypothesis import given

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph


@given(hh.mutually_promotable_dtypes(None))
def test_result_type(dtypes):
    out = xp.result_type(*dtypes)
    ph.assert_dtype("result_type", dtypes, out, repr_name="out")
