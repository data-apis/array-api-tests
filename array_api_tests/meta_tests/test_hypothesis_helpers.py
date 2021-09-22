from math import prod

import pytest
from hypothesis import given

from .. import _array_module as xp
from .._array_module import _UndefinedStub
from ..array_helpers import dtype_objects
from ..hypothesis_helpers import (MAX_ARRAY_SIZE,
                                  mutually_promotable_dtype_pairs,
                                  shapes, two_broadcastable_shapes,
                                  two_mutually_broadcastable_shapes)

UNDEFINED_DTYPES = any(isinstance(d, _UndefinedStub) for d in dtype_objects)
pytestmark = [pytest.mark.skipif(UNDEFINED_DTYPES, reason="undefined dtypes")]


@given(mutually_promotable_dtype_pairs([xp.float32, xp.float64]))
def test_mutually_promotable_dtype_pairs(pairs):
    assert pairs in (
        (xp.float32, xp.float32),
        (xp.float32, xp.float64),
        (xp.float64, xp.float32),
        (xp.float64, xp.float64),
    )


def valid_shape(shape) -> bool:
    return (
        all(isinstance(side, int) for side in shape)
        and all(side >= 0 for side in shape)
        and prod(shape) < MAX_ARRAY_SIZE
    )


@given(shapes)
def test_shapes(shape):
    assert valid_shape(shape)


@given(two_mutually_broadcastable_shapes)
def test_two_mutually_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)


@given(two_broadcastable_shapes())
def test_two_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)

    from ..test_broadcasting import broadcast_shapes

    assert broadcast_shapes(pair[0], pair[1]) == pair[0]
