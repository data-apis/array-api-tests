from functools import reduce
from operator import mul
from math import sqrt

from hypothesis.strategies import (lists, integers, builds, sampled_from,
                                   shared, floats, just, composite, one_of,
                                   none, booleans)
from hypothesis.extra.numpy import mutually_broadcastable_shapes
from hypothesis import assume

from .pytest_helpers import nargs
from .array_helpers import (dtype_ranges, integer_dtype_objects,
                            floating_dtype_objects, numeric_dtype_objects,
                            boolean_dtype_objects,
                            integer_or_boolean_dtype_objects, dtype_objects)
from ._array_module import full, float32, float64, bool as bool_dtype, _UndefinedStub
from . import _array_module

from .function_stubs import elementwise_functions


# Set this to True to not fail tests just because a dtype isn't implemented.
# If no compatible dtype is implemented for a given test, the test will fail
# with a hypothesis health check error. Note that this functionality will not
# work for floating point dtypes as those are assumed to be defined in other
# places in the tests.
FILTER_UNDEFINED_DTYPES = True

integer_dtypes = sampled_from(integer_dtype_objects)
floating_dtypes = sampled_from(floating_dtype_objects)
numeric_dtypes = sampled_from(numeric_dtype_objects)
integer_or_boolean_dtypes = sampled_from(integer_or_boolean_dtype_objects)
boolean_dtypes = sampled_from(boolean_dtype_objects)
dtypes = sampled_from(dtype_objects)

if FILTER_UNDEFINED_DTYPES:
    integer_dtypes = integer_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    floating_dtypes = floating_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    numeric_dtypes = numeric_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    integer_or_boolean_dtypes = integer_or_boolean_dtypes.filter(lambda x: not
                                                                 isinstance(x, _UndefinedStub))
    boolean_dtypes = boolean_dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))
    dtypes = dtypes.filter(lambda x: not isinstance(x, _UndefinedStub))

shared_dtypes = shared(dtypes)

@composite
def mutually_promotable_dtypes(draw, dtype_objects=dtype_objects):
    from .test_type_promotion import dtype_mapping, promotion_table
    # sort for shrinking (sampled_from shrinks to the earlier elements in the
    # list). Give pairs of the same dtypes first, then smaller dtypes,
    # preferring float, then int, then unsigned int. Note, this might not
    # always result in examples shrinking to these pairs because strategies
    # that draw from dtypes might not draw the same example from different
    # pairs (XXX: Can we redesign the strategies so that they can prefer
    # shrinking dtypes over values?)
    sorted_table = sorted(promotion_table)
    sorted_table = sorted(sorted_table, key=lambda ij: -1 if ij[0] == ij[1] else sorted_table.index(ij))
    dtype_pairs = [(dtype_mapping[i], dtype_mapping[j]) for i, j in
                   sorted_table]

    filtered_dtype_pairs = [(i, j) for i, j in dtype_pairs if i in
                            dtype_objects and j in dtype_objects]
    if FILTER_UNDEFINED_DTYPES:
        filtered_dtype_pairs = [(i, j) for i, j in filtered_dtype_pairs
                                if not isinstance(i, _UndefinedStub)
                                and not isinstance(j, _UndefinedStub)]
    return draw(sampled_from(filtered_dtype_pairs))

# shared() allows us to draw either the function or the function name and they
# will both correspond to the same function.

# TODO: Extend this to all functions, not just elementwise
elementwise_functions_names = shared(sampled_from(elementwise_functions.__all__))
array_functions_names = elementwise_functions_names
multiarg_array_functions_names = array_functions_names.filter(
    lambda func_name: nargs(func_name) > 1)

elementwise_function_objects = elementwise_functions_names.map(
    lambda i: getattr(_array_module, i))
array_functions = elementwise_function_objects
multiarg_array_functions = multiarg_array_functions_names.map(
    lambda i: getattr(_array_module, i))

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

shapes = tuples(integers(0, 10)).filter(lambda shape: prod(shape) < MAX_ARRAY_SIZE)

# Use this to avoid memory errors with NumPy.
# See https://github.com/numpy/numpy/issues/15753
shapes = tuples(integers(0, 10)).filter(
             lambda shape: prod([i for i in shape if i]) < MAX_ARRAY_SIZE)

two_mutually_broadcastable_shapes = mutually_broadcastable_shapes(num_shapes=2)\
    .map(lambda S: S.input_shapes)\
    .filter(lambda S: all(prod([i for i in shape if i]) < MAX_ARRAY_SIZE for shape in S))

@composite
def two_broadcastable_shapes(draw, shapes=shapes):
    """
    This will produce two shapes (shape1, shape2) such that shape2 can be
    broadcast to shape1.

    """
    from .test_broadcasting import broadcast_shapes

    shape1, shape2 = draw(two_mutually_broadcastable_shapes)
    if broadcast_shapes(shape1, shape2) != shape1:
        assume(False)
    return (shape1, shape2)

sizes = integers(0, MAX_ARRAY_SIZE)
sqrt_sizes = integers(0, SQRT_MAX_ARRAY_SIZE)

# TODO: Generate general arrays here, rather than just scalars.
numeric_arrays = builds(full, just((1,)), floats())

@composite
def scalars(draw, dtypes, finite=False):
    """
    Strategy to generate a scalar that matches a dtype strategy

    dtypes should be one of the shared_* dtypes strategies.
    """
    dtype = draw(dtypes)
    if dtype in dtype_ranges:
        m, M = dtype_ranges[dtype]
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
    m, M = dtype_ranges[dtype]
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
    if n_entries == len(shape) and not res_has_ellipsis:
        # note("Adding extra")
        extra = draw(lists(one_of(integer_indices(sizes), slices(sizes)), min_size=0, max_size=3))
        res += extra
    return tuple(res)
