from math import prod

import pytest
from hypothesis import given, strategies as st, assume

from .. import _array_module as xp
from .._array_module import _UndefinedStub
from .. import array_helpers as ah
from .. import hypothesis_helpers as hh

UNDEFINED_DTYPES = any(isinstance(d, _UndefinedStub) for d in ah.dtype_objects)
pytestmark = [pytest.mark.skipif(UNDEFINED_DTYPES, reason="undefined dtypes")]


@given(hh.mutually_promotable_dtypes([xp.float32, xp.float64]))
def test_mutually_promotable_dtypes(pairs):
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
        and prod(shape) < hh.MAX_ARRAY_SIZE
    )


@given(hh.shapes)
def test_shapes(shape):
    assert valid_shape(shape)


@given(hh.two_mutually_broadcastable_shapes)
def test_two_mutually_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)


@given(hh.two_broadcastable_shapes())
def test_two_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)

    from ..test_broadcasting import broadcast_shapes

    assert broadcast_shapes(pair[0], pair[1]) == pair[0]


def test_kwargs():
    results = []

    @given(hh.kwargs(n=st.integers(0, 10), c=st.from_regex("[a-f]")))
    def run(kw):
        results.append(kw)

    run()
    assert all(isinstance(kw, dict) for kw in results)
    for size in [0, 1, 2]:
        assert any(len(kw) == size for kw in results)

    n_results = [kw for kw in results if "n" in kw]
    assert len(n_results) > 0
    assert all(isinstance(kw["n"], int) for kw in n_results)

    c_results = [kw for kw in results if "c" in kw]
    assert len(c_results) > 0
    assert all(isinstance(kw["c"], str) for kw in c_results)

