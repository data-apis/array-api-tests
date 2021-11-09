from hypothesis import given

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import xps


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_min(x):
    xp.min(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_max(x):
    xp.max(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_mean(x):
    xp.mean(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_prod(x):
    xp.prod(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_std(x):
    xp.std(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(min_side=1)))
def test_sum(x):
    xp.sum(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes(min_side=1)))
def test_var(x):
    xp.var(x)
    # TODO
