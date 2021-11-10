from hypothesis import given

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import xps


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_any(x):
    xp.any(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_all(x):
    xp.all(x)
    # TODO
