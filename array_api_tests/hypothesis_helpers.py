from functools import reduce
from operator import mul
from math import sqrt
import itertools
from typing import Tuple

from hypothesis import assume
from hypothesis.strategies import (lists, integers, sampled_from,
                                   shared, floats, just, composite, one_of,
                                   none, booleans)
from hypothesis.strategies._internal.strategies import SearchStrategy

from .pytest_helpers import nargs
from .array_helpers import ndindex
from . import dtype_helpers as dh
from ._array_module import (full, float32, float64, bool as bool_dtype,
                            _UndefinedStub, eye, broadcast_to)
from . import _array_module as xp
from . import xps

from .function_stubs import elementwise_functions


# Set this to True to not fail tests just because a dtype isn't implemented.
# If no compatible dtype is implemented for a given test, the test will fail
# with a hypothesis health check error. Note that this functionality will not
# work for floating point dtypes as those are assumed to be defined in other
# places in the tests.
FILTER_UNDEFINED_DTYPES = True

integer_dtypes = sampled_from(dh.all_int_dtypes)
floating_dtypes = sampled_from(dh.float_dtypes)
numeric_dtypes = sampled_from(dh.numeric_dtypes)
integer_or_boolean_dtypes = sampled_from(dh.bool_and_all_int_dtypes)
boolean_dtypes = just(xp.bool)
dtypes = sampled_from(dh.all_dtypes)

if FILTER_UNDEFINED_DTYPES:
    integer_dtypes = integer_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    floating_dtypes = floating_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    numeric_dtypes = numeric_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    integer_or_boolean_dtypes = integer_or_boolean_dtypes.filter(lambda x: not
                                                                 isinstance(x, _UndefinedStub))
    boolean_dtypes = boolean_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    dtypes = dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))

shared_dtypes = shared(dtypes, key="dtype")
shared_floating_dtypes = shared(floating_dtypes, key="dtype")


sorted_table = sorted(dh.promotion_table)
sorted_table = sorted(
    sorted_table, key=lambda ij: -1 if ij[0] == ij[1] else sorted_table.index(ij)
)
if FILTER_UNDEFINED_DTYPES:
    sorted_table = [(i, j) for i, j in sorted_table
                    if not isinstance(i, _UndefinedStub)
                    and not isinstance(j, _UndefinedStub)]


def mutually_promotable_dtypes(dtype_objs=dh.all_dtypes):
    # sort for shrinking (sampled_from shrinks to the earlier elements in the
    # list). Give pairs of the same dtypes first, then smaller dtypes,
    # preferring float, then int, then unsigned int. Note, this might not
    # always result in examples shrinking to these pairs because strategies
    # that draw from dtypes might not draw the same example from different
    # pairs (XXX: Can we redesign the strategies so that they can prefer
    # shrinking dtypes over values?)
    return sampled_from(
        [(i, j) for i, j in sorted_table if i in dtype_objs and j in dtype_objs]
    )

# shared() allows us to draw either the function or the function name and they
# will both correspond to the same function.

# TODO: Extend this to all functions, not just elementwise
elementwise_functions_names = shared(sampled_from(elementwise_functions.__all__))
array_functions_names = elementwise_functions_names
multiarg_array_functions_names = array_functions_names.filter(
    lambda func_name: nargs(func_name) > 1)

elementwise_function_objects = elementwise_functions_names.map(
    lambda i: getattr(xp, i))
array_functions = elementwise_function_objects
multiarg_array_functions = multiarg_array_functions_names.map(
    lambda i: getattr(xp, i))

# Limit the total size of an array shape
MAX_ARRAY_SIZE = 10000
# Size to use for 2-dim arrays
SQRT_MAX_ARRAY_SIZE = int(sqrt(MAX_ARRAY_SIZE))

# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)

# hypotheses.strategies.tuples only generates tuples of a fixed size
def tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return lists(elements, min_size=min_size, max_size=max_size,
                 unique_by=unique_by, unique=unique).map(tuple)

# Use this to avoid memory errors with NumPy.
# See https://github.com/numpy/numpy/issues/15753
shapes = xps.array_shapes(min_dims=0, min_side=0).filter(
    lambda shape: prod(i for i in shape if i) < MAX_ARRAY_SIZE
)

one_d_shapes = xps.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=SQRT_MAX_ARRAY_SIZE)

# Matrix shapes assume stacks of matrices
matrix_shapes = xps.array_shapes(min_dims=2, min_side=1).filter(
    lambda shape: prod(i for i in shape if i) < MAX_ARRAY_SIZE
)

square_matrix_shapes = matrix_shapes.filter(lambda shape: shape[-1] == shape[-2])

def mutually_broadcastable_shapes(num_shapes: int) -> SearchStrategy[Tuple[Tuple]]:
    return (
        xps.mutually_broadcastable_shapes(num_shapes)
        .map(lambda BS: BS.input_shapes)
        .filter(lambda shapes: all(
            prod(i for i in s if i > 0) < MAX_ARRAY_SIZE for s in shapes
        ))
    )

two_mutually_broadcastable_shapes = mutually_broadcastable_shapes(2)

# Note: This should become hermitian_matrices when complex dtypes are added
@composite
def symmetric_matrices(draw, dtypes=xps.floating_dtypes(), finite=True):
    shape = draw(square_matrix_shapes)
    dtype = draw(dtypes)
    elements = {'allow_nan': False, 'allow_infinity': False} if finite else None
    a = draw(xps.arrays(dtype=dtype, shape=shape, elements=elements))
    upper = xp.triu(a)
    lower = xp.triu(a, k=1).mT
    return upper + lower

@composite
def positive_definite_matrices(draw, dtypes=xps.floating_dtypes()):
    # For now just generate stacks of identity matrices
    # TODO: Generate arbitrary positive definite matrices, for instance, by
    # using something like
    # https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/datasets/_samples_generator.py#L1351.
    n = draw(integers(0))
    shape = draw(shapes) + (n, n)
    assume(prod(i for i in shape if i) < MAX_ARRAY_SIZE)
    dtype = draw(dtypes)
    return broadcast_to(eye(n, dtype=dtype), shape)

@composite
def invertible_matrices(draw, dtypes=xps.floating_dtypes()):
    # For now, just generate stacks of diagonal matrices.
    n = draw(integers(0, SQRT_MAX_ARRAY_SIZE),)
    stack_shape = draw(shapes)
    shape = stack_shape + (n, n)
    d = draw(xps.arrays(dtypes, shape=n*prod(stack_shape),
                        elements=dict(allow_nan=False, allow_infinity=False)))
    # Functions that require invertible matrices may do anything when it is
    # singular, including raising an exception, so we make sure the diagonals
    # are sufficiently nonzero to avoid any numerical issues.
    assume(xp.all(xp.abs(d) > 0.5))

    a = xp.zeros(shape)
    for j, (idx, i) in enumerate(itertools.product(ndindex(stack_shape), range(n))):
        a[idx + (i, i)] = d[j]
    return a

@composite
def two_broadcastable_shapes(draw):
    """
    This will produce two shapes (shape1, shape2) such that shape2 can be
    broadcast to shape1.
    """
    from .test_broadcasting import broadcast_shapes
    shape1, shape2 = draw(two_mutually_broadcastable_shapes)
    assume(broadcast_shapes(shape1, shape2) == shape1)
    return (shape1, shape2)

sizes = integers(0, MAX_ARRAY_SIZE)
sqrt_sizes = integers(0, SQRT_MAX_ARRAY_SIZE)

numeric_arrays = xps.arrays(
    dtype=shared(xps.floating_dtypes(), key='dtypes'),
    shape=shared(xps.array_shapes(), key='shapes'),
)

@composite
def scalars(draw, dtypes, finite=False):
    """
    Strategy to generate a scalar that matches a dtype strategy

    dtypes should be one of the shared_* dtypes strategies.
    """
    dtype = draw(dtypes)
    if dtype in dh.dtype_ranges:
        m, M = dh.dtype_ranges[dtype]
        return draw(integers(m, M))
    elif dtype == bool_dtype:
        return draw(booleans())
    elif dtype == float64:
        if finite:
            return draw(floats(allow_nan=False, allow_infinity=False))
        return draw(floats())
    elif dtype == float32:
        if finite:
            return draw(floats(width=32, allow_nan=False, allow_infinity=False))
        return draw(floats(width=32))
    else:
        raise ValueError(f"Unrecognized dtype {dtype}")

@composite
def array_scalars(draw, dtypes):
    dtype = draw(dtypes)
    return full((), draw(scalars(just(dtype))), dtype=dtype)

@composite
def python_integer_indices(draw, sizes):
    size = draw(sizes)
    if size == 0:
        assume(False)
    return draw(integers(-size, size - 1))

@composite
def integer_indices(draw, sizes):
    # Return either a Python integer or a 0-D array with some integer dtype
    idx = draw(python_integer_indices(sizes))
    dtype = draw(integer_dtypes)
    m, M = dh.dtype_ranges[dtype]
    if m <= idx <= M:
        return draw(one_of(just(idx),
                           just(full((), idx, dtype=dtype))))
    return idx

@composite
def slices(draw, sizes):
    size = draw(sizes)
    # The spec does not specify out of bounds behavior.
    max_step_size = draw(integers(1, max(1, size)))
    step = draw(one_of(integers(-max_step_size, -1), integers(1, max_step_size), none()))
    start = draw(one_of(integers(-size, max(0, size-1)), none()))
    if step is None or step > 0:
        stop = draw(one_of(integers(-size, size)), none())
    else:
        stop = draw(one_of(integers(-size - 1, size - 1)), none())
    s = slice(start, stop, step)
    l = list(range(size))
    sliced_list = l[s]
    if (sliced_list == []
        and size != 0
        and start is not None
        and stop is not None
        and stop != start
        ):
        # The spec does not specify behavior for out-of-bounds slices, except
        # for the case where stop == start.
        assume(False)
    return s

@composite
def multiaxis_indices(draw, shapes):
    res = []
    # Generate tuples no longer than the shape, with indices corresponding to
    # each dimension.
    shape = draw(shapes)
    n_entries = draw(integers(0, len(shape)))
    # from hypothesis import note
    # note(f"multiaxis_indices n_entries: {n_entries}")

    k = 0
    for i in range(n_entries):
        size = shape[k]
        idx = draw(one_of(
            integer_indices(just(size)),
            slices(just(size)),
            just(...)))
        if idx is ... and k >= 0:
            # If there is an ellipsis, index from the end of the shape
            k = k - n_entries
        k += 1
        res.append(idx)
    # Sometimes add more entries than necessary to test this.

    # Avoid using 'in', which might do == on an array.
    res_has_ellipsis = any(i is ... for i in res)
    if not res_has_ellipsis:
        if n_entries < len(shape):
            # The spec requires either an ellipsis or exactly as many indices
            # as dimensions.
            assume(False)
        elif n_entries == len(shape):
            # note("Adding extra")
            extra = draw(lists(one_of(integer_indices(sizes), slices(sizes)), min_size=0, max_size=3))
            res += extra
    return tuple(res)


def two_mutual_arrays(
    dtype_objs=dh.all_dtypes, two_shapes=two_mutually_broadcastable_shapes
):
    mutual_dtypes = shared(mutually_promotable_dtypes(dtype_objs))
    mutual_shapes = shared(two_shapes)
    arrays1 = xps.arrays(
        dtype=mutual_dtypes.map(lambda pair: pair[0]),
        shape=mutual_shapes.map(lambda pair: pair[0]),
    )
    arrays2 = xps.arrays(
        dtype=mutual_dtypes.map(lambda pair: pair[1]),
        shape=mutual_shapes.map(lambda pair: pair[1]),
    )
    return arrays1, arrays2

@composite
def kwargs(draw, **kw):
    """
    Strategy for keyword arguments

    For a signature like f(x, /, dtype=None, val=1) use

    @given(x=arrays(), kw=kwargs(a=none() | dtypes, val=integers()))
    def test_f(x, kw):
        res = f(x, **kw)

    kw may omit the keyword argument, meaning the default for f will be used.

    """
    result = {}
    for k, strat in kw.items():
        if draw(booleans()):
            result[k] = draw(strat)
    return result
