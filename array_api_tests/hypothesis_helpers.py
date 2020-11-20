from functools import reduce
from operator import mul

from hypothesis.strategies import (lists, integers, builds, sampled_from,
                                   shared, tuples as hypotheses_tuples,
                                   floats, just)

from .pytest_helpers import nargs
from ._array_module import _dtypes, ones, full
from . import _array_module

from .function_stubs import elementwise_functions

dtype_objects = [getattr(_array_module, t) for t in _dtypes]
dtypes = sampled_from(dtype_objects)
shared_dtypes = shared(dtypes)

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


ones_arrays = builds(ones, shapes, dtype=shared_dtypes)

nonbroadcastable_ones_array_two_args = hypotheses_tuples(ones_arrays, ones_arrays)

# TODO: Generate general arrays here, rather than just scalars.
numeric_arrays = builds(full, just((1,)), floats())
