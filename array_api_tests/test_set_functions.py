from hypothesis import given

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import xps


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_all(x):
    xp.unique_all(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_counts(x):
    xp.unique_counts(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_inverse(x):
    xp.unique_inverse(x)
    # TODO


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_unique_values(x):
    xp.unique_values(x)
    # TODO
