"""
https://github.com/data-apis/array-api/blob/master/spec/API_specification/broadcasting.md
"""

from functools import reduce

import pytest

from hypothesis import given

from .hypothesis_helpers import nonbroadcastable_ones_array_two_args
from .pytest_helpers import raises, doesnt_raise, nargs

from .function_stubs import elementwise_functions
from . import _array_module

# The spec does not specify what exception is raised on broadcast errors. We
# use a custom exception to distinguish it from potential bugs in
# broadcast_shapes().
class BroadcastError(Exception):
    pass

# The spec only specifies broadcasting for two shapes.
def broadcast_shapes(shape1, shape2):
    """
    Broadcast shapes `shape1` and `shape2`.

    The code in this function should follow the pseudocode in the spec as
    closely as possible.
    """
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None]*N
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise BroadcastError

        i = i - 1

    return tuple(shape)

def test_broadcast_shapes_explicit_spec():
    """
    Explicit broadcast shapes examples from the spec
    """
    shape1 = (8, 1, 6, 1)
    shape2 = (7, 1, 5)
    result = (8, 7, 6, 5)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (5, 4)
    shape2 = (1,)
    result = (5, 4)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (5, 4)
    shape2 = (4,)
    result = (5, 4)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (15, 3, 5)
    shape2 = (15, 1, 5)
    result = (15, 3, 5)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (15, 3, 5)
    shape2 = (3, 5)
    result = (15, 3, 5)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (15, 3, 5)
    shape2 = (3, 1)
    result = (15, 3, 5)
    assert broadcast_shapes(shape1, shape2) == result

    shape1 = (3,)
    shape2 = (4,)
    raises(BroadcastError, lambda: broadcast_shapes(shape1, shape2)) # dimension does not match

    shape1 = (2, 1)
    shape2 = (8, 4, 3)
    raises(BroadcastError, lambda: broadcast_shapes(shape1, shape2)) # second dimension does not match

    shape1 = (15, 3, 5)
    shape2 = (15, 3)
    raises(BroadcastError, lambda: broadcast_shapes(shape1, shape2)) # singleton dimensions can only be prepended, not appended

# TODO: Extend this to all functions (not just elementwise), and handle
# functions that take more than 2 args
@pytest.mark.parametrize('func_name', [i for i in
                                       elementwise_functions.__all__ if
                                       nargs(i) > 1])
@given(args=nonbroadcastable_ones_array_two_args)
def test_broadcasting_hypothesis(func_name, args):
    assert nargs(func_name) == 2
    func = getattr(_array_module, func_name)

    if isinstance(func, _array_module._UndefinedStub):
        func._raise()

    shapes = [i.shape for i in args]
    try:
        broadcast_shape = reduce(broadcast_shapes, shapes)
    except BroadcastError:
        raises(Exception, lambda: func(*args),
               f"{func_name} should raise an exception from not being able to broadcast inputs with shapes {shapes}")
    else:
        result = doesnt_raise(lambda: func(*args),
            f"{func_name} raised an unexpected exception from broadcastable inputs with shapes {shapes}")
        assert result.shape == broadcast_shape, "broadcast shapes incorrect"
