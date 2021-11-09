"""
https://github.com/data-apis/array-api/blob/master/spec/API_specification/broadcasting.md
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from .. import _array_module as xp
from .. import dtype_helpers as dh
from .. import hypothesis_helpers as hh
from .. import pytest_helpers as ph
from .._array_module import _UndefinedStub
from ..algos import BroadcastError, _broadcast_shapes
from ..function_stubs import elementwise_functions


@pytest.mark.parametrize(
    "shape1, shape2, expected",
    [
        [(8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)],
        [(5, 4), (1,), (5, 4)],
        [(5, 4), (4,), (5, 4)],
        [(15, 3, 5), (15, 1, 5), (15, 3, 5)],
        [(15, 3, 5), (3, 5), (15, 3, 5)],
        [(15, 3, 5), (3, 1), (15, 3, 5)],
    ],
)
def test_broadcast_shapes(shape1, shape2, expected):
    assert _broadcast_shapes(shape1, shape2) == expected


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        [(3,), (4,)],  # dimension does not match
        [(2, 1), (8, 4, 3)],  # second dimension does not match
        [(15, 3, 5), (15, 3)],  # singleton dimensions can only be prepended
    ],
)
def test_broadcast_shapes_fails_on_bad_shapes(shape1, shape2):
    with pytest.raises(BroadcastError):
        _broadcast_shapes(shape1, shape2)


# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize(
    "func_name", [i for i in elementwise_functions.__all__ if ph.nargs(i) > 1]
)
@given(shape1=hh.shapes(), shape2=hh.shapes(), data=st.data())
def test_broadcasting_hypothesis(func_name, shape1, shape2, data):
    dtype = data.draw(st.sampled_from(dh.func_in_dtypes[func_name]), label="dtype")
    if hh.FILTER_UNDEFINED_DTYPES:
        assume(not isinstance(dtype, _UndefinedStub))
    func = getattr(xp, func_name)
    if isinstance(func, xp._UndefinedStub):
        func._raise()
    args = [xp.ones(shape1, dtype=dtype), xp.ones(shape2, dtype=dtype)]
    try:
        broadcast_shape = _broadcast_shapes(shape1, shape2)
    except BroadcastError:
        ph.raises(
            Exception,
            lambda: func(*args),
            f"{func_name} should raise an exception from not being able to broadcast inputs with hh.shapes {(shape1, shape2)}",
        )
    else:
        result = ph.doesnt_raise(
            lambda: func(*args),
            f"{func_name} raised an unexpected exception from broadcastable inputs with hh.shapes {(shape1, shape2)}",
        )
        assert result.shape == broadcast_shape, "broadcast hh.shapes incorrect"
