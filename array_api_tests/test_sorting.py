from hypothesis import given

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import xps


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_argsort(x):
    xp.argsort(x)
    # TODO


# TODO: generate 0d arrays, generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_dims=1)))
def test_sort(x):
    xp.sort(x)
    # TODO
