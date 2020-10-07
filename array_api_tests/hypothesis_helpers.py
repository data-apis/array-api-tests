from functools import reduce
from operator import mul
from inspect import getfullargspec

from hypothesis.strategies import (lists, integers, builds, sampled_from,
shared, tuples as hypotheses_tuples)

from ._array_module import _dtypes, ones
from . import _array_module

from .function_stubs import elementwise_functions
from . import function_stubs

dtype_objects = [getattr(_array_module, t) for t in _dtypes]
dtypes = sampled_from(dtype_objects)

# shared() allows us to draw either the function or the function name and they
# will both correspond to the same function.

# TODO: Extend this to all functions, not just elementwise
elementwise_functions_names = shared(sampled_from(elementwise_functions.__all__))
array_function_names = elementwise_functions_names

elementwise_function_objects = elementwise_functions_names.map(
    lambda i: getattr(_array_module, i))
array_functions = elementwise_function_objects

# Limit the total size of an array shape
MAX_ARRAY_SIZE = 10000

# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)

# hypotheses.strategies.tuples only generates tuples of a fixed size
def tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return lists(elements, min_size=min_size, max_size=max_size,
                 unique_by=unique_by, unique=unique).map(tuple)

# Use this to avoid memory errors with NumPy.
# See https://github.com/numpy/numpy/issues/15753

# shapes = tuples(integers(0, 10)).filter(
#              lambda shape: prod([i for i in shape if i]) < MAX_ARRAY_SIZE)

shapes = tuples(integers(0, 10)).filter(lambda shape: prod(shape) < MAX_ARRAY_SIZE)

ones_arrays = builds(ones, shapes, dtype=dtypes)

def nargs(func_name):
    return len(getfullargspec(getattr(function_stubs, func_name)).args)

nonbroadcastable_ones_array_args = array_function_names.flatmap(
    lambda func_name: hypotheses_tuples(*[ones_arrays for i in range(nargs(func_name))]))
