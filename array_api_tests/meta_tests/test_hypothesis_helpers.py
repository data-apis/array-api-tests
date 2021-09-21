import pytest
from hypothesis import given

from .. import _array_module as xp
from .._array_module import _UndefinedStub
from ..array_helpers import dtype_objects
from ..hypothesis_helpers import (mutually_promotable_dtype_pairs,
                                  promotable_dtypes)

UNDEFINED_DTYPES = any(isinstance(d, _UndefinedStub) for d in dtype_objects)
pytestmark = [pytest.mark.skipif(UNDEFINED_DTYPES, reason="undefined dtypes")]


@given(promotable_dtypes(xp.uint16))
def test_promotable_dtypes(dtype):
    assert dtype in (
        xp.uint8, xp.uint16, xp.uint32, xp.uint64, xp.int8, xp.int16, xp.int32, xp.int64
    )


@given(mutually_promotable_dtype_pairs([xp.float32, xp.float64]))
def test_mutually_promotable_dtype_pairs(pairs):
    assert pairs in (
        (xp.float32, xp.float32),
        (xp.float32, xp.float64),
        (xp.float64, xp.float32),
        (xp.float64, xp.float64),
    )

