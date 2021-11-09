from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import hypothesis_helpers as hh
from . import xps


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_argmin(x):
    xp.argmin(x)
    # TODO


# TODO: generate kwargs
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_argmax(x):
    xp.argmax(x)
    # TODO


# TODO: generate kwargs, skip if opted out
@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1)))
def test_nonzero(x):
    xp.nonzero(x)
    # TODO


# TODO: skip if opted out
@given(
    shapes=hh.mutually_broadcastable_shapes(3),
    dtypes=hh.mutually_promotable_dtypes(),
    data=st.data(),
)
def test_where(shapes, dtypes, data):
    cond = data.draw(xps.arrays(dtype=xp.bool, shape=shapes[0]), label="condition")
    x1 = data.draw(xps.arrays(dtype=dtypes[0], shape=shapes[1]), label="x1")
    x2 = data.draw(xps.arrays(dtype=dtypes[1], shape=shapes[2]), label="x2")
    xp.where(cond, x1, x2)
    # TODO
