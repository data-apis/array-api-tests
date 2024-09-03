from __future__ import annotations

import re
from contextlib import contextmanager
from functools import reduce, wraps
import math
from operator import mul
import struct
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from hypothesis import assume, reject
from hypothesis.strategies import (SearchStrategy, booleans, composite, floats,
                                   integers, just, lists, none, one_of,
                                   sampled_from, shared, builds)

from . import _array_module as xp, api_version
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import shape_helpers as sh
from . import xps
from ._array_module import _UndefinedStub
from ._array_module import bool as bool_dtype
from ._array_module import broadcast_to, eye, float32, float64, full
from .stubs import category_to_funcs
from .pytest_helpers import nargs
from .typing import Array, DataType, Scalar, Shape


def _float32ify(n: Union[int, float]) -> float:
    n = float(n)
    return struct.unpack("!f", struct.pack("!f", n))[0]


@wraps(xps.from_dtype)
def from_dtype(dtype, **kwargs) -> SearchStrategy[Scalar]:
    """xps.from_dtype() without the crazy large numbers."""
    if dtype == xp.bool:
        return xps.from_dtype(dtype, **kwargs)

    if dtype in dh.complex_dtypes:
        component_dtype = dh.dtype_components[dtype]
    else:
        component_dtype = dtype

    min_, max_ = dh.dtype_ranges[component_dtype]

    if "min_value" not in kwargs.keys() and min_ != 0:
        assert min_ < 0  # sanity check
        min_value = -1 * math.floor(math.sqrt(abs(min_)))
        if component_dtype == xp.float32:
            min_value = _float32ify(min_value)
        kwargs["min_value"] = min_value
    if "max_value" not in kwargs.keys():
        assert max_ > 0  # sanity check
        max_value = math.floor(math.sqrt(max_))
        if component_dtype == xp.float32:
            max_value = _float32ify(max_value)
        kwargs["max_value"] = max_value

    if dtype in dh.complex_dtypes:
        component_strat = xps.from_dtype(dh.dtype_components[dtype], **kwargs)
        return builds(complex, component_strat, component_strat)
    else:
        return xps.from_dtype(dtype, **kwargs)


@wraps(xps.arrays)
def arrays(dtype, *args, elements=None, **kwargs) -> SearchStrategy[Array]:
    """xps.arrays() without the crazy large numbers."""
    if isinstance(dtype, SearchStrategy):
        return dtype.flatmap(lambda d: arrays(d, *args, elements=elements, **kwargs))

    if elements is None:
        elements = from_dtype(dtype)
    elif isinstance(elements, Mapping):
        elements = from_dtype(dtype, **elements)

    return xps.arrays(dtype, *args, elements=elements, **kwargs)


_dtype_categories = [(xp.bool,), dh.uint_dtypes, dh.int_dtypes, dh.real_float_dtypes, dh.complex_dtypes]
_sorted_dtypes = [d for category in _dtype_categories for d in category]

def _dtypes_sorter(dtype_pair: Tuple[DataType, DataType]):
    dtype1, dtype2 = dtype_pair
    if dtype1 == dtype2:
        return _sorted_dtypes.index(dtype1)
    key = len(_sorted_dtypes)
    rank1 = _sorted_dtypes.index(dtype1)
    rank2 = _sorted_dtypes.index(dtype2)
    for category in _dtype_categories:
        if dtype1 in category and dtype2 in category:
            break
    else:
        key += len(_sorted_dtypes) ** 2
    key += 2 * (rank1 + rank2)
    if rank1 > rank2:
        key += 1
    return key

_promotable_dtypes = list(dh.promotion_table.keys())
_promotable_dtypes = [
    (d1, d2) for d1, d2 in _promotable_dtypes
    if not isinstance(d1, _UndefinedStub) or not isinstance(d2, _UndefinedStub)
]
promotable_dtypes: List[Tuple[DataType, DataType]] = sorted(_promotable_dtypes, key=_dtypes_sorter)

def mutually_promotable_dtypes(
    max_size: Optional[int] = 2,
    *,
    dtypes: Sequence[DataType] = dh.all_dtypes,
) -> SearchStrategy[Tuple[DataType, ...]]:
    dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
    assert len(dtypes) > 0, "all dtypes undefined"  # sanity check
    if max_size == 2:
        return sampled_from(
            [(i, j) for i, j in promotable_dtypes if i in dtypes and j in dtypes]
        )
    if isinstance(max_size, int) and max_size < 2:
        raise ValueError(f'{max_size=} should be >=2')
    strats = []
    category_samples = {
        category: [d for d in dtypes if d in category] for category in _dtype_categories
    }
    for samples in category_samples.values():
        if len(samples) > 0:
            strat = lists(sampled_from(samples), min_size=2, max_size=max_size)
            strats.append(strat)
    if len(category_samples[dh.uint_dtypes]) > 0 and len(category_samples[dh.int_dtypes]) > 0:
        mixed_samples = category_samples[dh.uint_dtypes] + category_samples[dh.int_dtypes]
        strat = lists(sampled_from(mixed_samples), min_size=2, max_size=max_size)
        if xp.uint64 in mixed_samples:
            strat = strat.filter(
                lambda l: not (xp.uint64 in l and any(d in dh.int_dtypes for d in l))
            )
    return one_of(strats).map(tuple)


class OnewayPromotableDtypes(NamedTuple):
    input_dtype: DataType
    result_dtype: DataType


@composite
def oneway_promotable_dtypes(
    draw, dtypes: Sequence[DataType]
) -> OnewayPromotableDtypes:
    """Return a strategy for input dtypes that promote to result dtypes."""
    d1, d2 = draw(mutually_promotable_dtypes(dtypes=dtypes))
    result_dtype = dh.result_type(d1, d2)
    if d1 == result_dtype:
        return OnewayPromotableDtypes(d2, d1)
    elif d2 == result_dtype:
        return OnewayPromotableDtypes(d1, d2)
    else:
        reject()


class OnewayBroadcastableShapes(NamedTuple):
    input_shape: Shape
    result_shape: Shape


@composite
def oneway_broadcastable_shapes(draw) -> OnewayBroadcastableShapes:
    """Return a strategy for input shapes that broadcast to result shapes."""
    result_shape = draw(shapes(min_side=1))
    input_shape = draw(
        xps.broadcastable_shapes(
            result_shape,
            # Override defaults so bad shapes are less likely to be generated.
            max_side=None if result_shape == () else max(result_shape),
            max_dims=len(result_shape),
        ).filter(lambda s: sh.broadcast_shapes(result_shape, s) == result_shape)
    )
    return OnewayBroadcastableShapes(input_shape, result_shape)


# Use these instead of xps.scalar_dtypes, etc. because it skips dtypes from
# ARRAY_API_TESTS_SKIP_DTYPES
all_dtypes = sampled_from(_sorted_dtypes)
int_dtypes = sampled_from(dh.all_int_dtypes)
uint_dtypes = sampled_from(dh.uint_dtypes)
real_dtypes = sampled_from(dh.real_dtypes)
# Warning: The hypothesis "floating_dtypes" is what we call
# "real_floating_dtypes"
floating_dtypes = sampled_from(dh.all_float_dtypes)
real_floating_dtypes = sampled_from(dh.real_float_dtypes)
numeric_dtypes = sampled_from(dh.numeric_dtypes)
# Note: this always returns complex dtypes, even if api_version < 2022.12
complex_dtypes: SearchStrategy[Any] | None = sampled_from(dh.complex_dtypes) if dh.complex_dtypes else None

def all_floating_dtypes() -> SearchStrategy[DataType]:
    strat = floating_dtypes
    if api_version >= "2022.12" and complex_dtypes is not None:
        strat |= complex_dtypes
    return strat


# shared() allows us to draw either the function or the function name and they
# will both correspond to the same function.

# TODO: Extend this to all functions, not just elementwise
elementwise_functions_names = shared(sampled_from([f.__name__ for f in category_to_funcs["elementwise"]]))
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
SQRT_MAX_ARRAY_SIZE = int(math.sqrt(MAX_ARRAY_SIZE))

# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)

# hypotheses.strategies.tuples only generates tuples of a fixed size
def tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return lists(elements, min_size=min_size, max_size=max_size,
                 unique_by=unique_by, unique=unique).map(tuple)

# Use this to avoid memory errors with NumPy.
# See https://github.com/numpy/numpy/issues/15753
# Note, the hypothesis default for max_dims is min_dims + 2 (i.e., 0 + 2)
def shapes(**kw):
    kw.setdefault('min_dims', 0)
    kw.setdefault('min_side', 0)
    return xps.array_shapes(**kw).filter(
        lambda shape: prod(i for i in shape if i) < MAX_ARRAY_SIZE
    )


one_d_shapes = xps.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=SQRT_MAX_ARRAY_SIZE)

# Matrix shapes assume stacks of matrices
@composite
def matrix_shapes(draw, stack_shapes=shapes()):
    stack_shape = draw(stack_shapes)
    mat_shape = draw(xps.array_shapes(max_dims=2, min_dims=2))
    shape = stack_shape + mat_shape
    assume(prod(i for i in shape if i) < MAX_ARRAY_SIZE)
    return shape

square_matrix_shapes = matrix_shapes().filter(lambda shape: shape[-1] == shape[-2])

@composite
def finite_matrices(draw, shape=matrix_shapes()):
    return draw(arrays(dtype=floating_dtypes,
                           shape=shape,
                           elements=dict(allow_nan=False,
                                         allow_infinity=False)))

rtol_shared_matrix_shapes = shared(matrix_shapes())
# Should we set a max_value here?
_rtol_float_kw = dict(allow_nan=False, allow_infinity=False, min_value=0)
rtols = one_of(floats(**_rtol_float_kw),
               arrays(dtype=real_floating_dtypes,
                          shape=rtol_shared_matrix_shapes.map(lambda shape:  shape[:-2]),
                          elements=_rtol_float_kw))


def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape: Shape = (),
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 0,
    max_side: Optional[int] = None,
) -> SearchStrategy[Tuple[Shape, ...]]:
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 5, 32)
    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 5
    return (
        xps.mutually_broadcastable_shapes(
            num_shapes,
            base_shape=base_shape,
            min_dims=min_dims,
            max_dims=max_dims,
            min_side=min_side,
            max_side=max_side,
        )
        .map(lambda BS: BS.input_shapes)
        .filter(lambda shapes: all(
            prod(i for i in s if i > 0) < MAX_ARRAY_SIZE for s in shapes
        ))
    )

two_mutually_broadcastable_shapes = mutually_broadcastable_shapes(2)

# TODO: Add support for complex Hermitian matrices
@composite
def symmetric_matrices(draw, dtypes=real_floating_dtypes, finite=True, bound=10.):
    shape = draw(square_matrix_shapes)
    dtype = draw(dtypes)
    if not isinstance(finite, bool):
        finite = draw(finite)
    elements = {'allow_nan': False, 'allow_infinity': False} if finite else None
    a = draw(arrays(dtype=dtype, shape=shape, elements=elements))
    at = ah._matrix_transpose(a)
    H = (a + at)*0.5
    if finite:
        assume(not xp.any(xp.isinf(H)))
    assume(xp.all((H == 0.) | ((1/bound <= xp.abs(H)) & (xp.abs(H) <= bound))))
    return H

@composite
def positive_definite_matrices(draw, dtypes=floating_dtypes):
    # For now just generate stacks of identity matrices
    # TODO: Generate arbitrary positive definite matrices, for instance, by
    # using something like
    # https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/datasets/_samples_generator.py#L1351.
    base_shape = draw(shapes())
    n = draw(integers(0, 8))  # 8 is an arbitrary small but interesting-enough value
    shape = base_shape + (n, n)
    assume(prod(i for i in shape if i) < MAX_ARRAY_SIZE)
    dtype = draw(dtypes)
    return broadcast_to(eye(n, dtype=dtype), shape)

@composite
def invertible_matrices(draw, dtypes=floating_dtypes, stack_shapes=shapes()):
    # For now, just generate stacks of diagonal matrices.
    stack_shape = draw(stack_shapes)
    n = draw(integers(0, SQRT_MAX_ARRAY_SIZE // max(math.prod(stack_shape), 1)),)
    dtype = draw(dtypes)
    elements = one_of(
        from_dtype(dtype, min_value=0.5, allow_nan=False, allow_infinity=False),
        from_dtype(dtype, max_value=-0.5, allow_nan=False, allow_infinity=False),
    )
    d = draw(arrays(dtype, shape=(*stack_shape, 1, n), elements=elements))

    # Functions that require invertible matrices may do anything when it is
    # singular, including raising an exception, so we make sure the diagonals
    # are sufficiently nonzero to avoid any numerical issues.
    assert xp.all(xp.abs(d) >= 0.5)

    diag_mask = xp.arange(n) == xp.reshape(xp.arange(n), (n, 1))
    return xp.where(diag_mask, d, xp.zeros_like(d))

# TODO: Better name
@composite
def two_broadcastable_shapes(draw):
    """
    This will produce two shapes (shape1, shape2) such that shape2 can be
    broadcast to shape1.
    """
    shape1, shape2 = draw(two_mutually_broadcastable_shapes)
    assume(sh.broadcast_shapes(shape1, shape2) == shape1)
    return (shape1, shape2)

sizes = integers(0, MAX_ARRAY_SIZE)
sqrt_sizes = integers(0, SQRT_MAX_ARRAY_SIZE)

numeric_arrays = arrays(
    dtype=shared(floating_dtypes, key='dtypes'),
    shape=shared(xps.array_shapes(), key='shapes'),
)

@composite
def scalars(draw, dtypes, finite=False):
    """
    Strategy to generate a scalar that matches a dtype strategy

    dtypes should be one of the shared_* dtypes strategies.
    """
    dtype = draw(dtypes)
    if dh.is_int_dtype(dtype):
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
    dtype = draw(int_dtypes | uint_dtypes)
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
    start = draw(one_of(integers(-size, size), none()))
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
    dtypes: Sequence[DataType] = dh.all_dtypes,
    two_shapes: SearchStrategy[Tuple[Shape, Shape]] = two_mutually_broadcastable_shapes,
) -> Tuple[SearchStrategy[Array], SearchStrategy[Array]]:
    if not isinstance(dtypes, Sequence):
        raise TypeError(f"{dtypes=} not a sequence")
    dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
    assert len(dtypes) > 0  # sanity check
    mutual_dtypes = shared(mutually_promotable_dtypes(dtypes=dtypes))
    mutual_shapes = shared(two_shapes)
    arrays1 = arrays(
        dtype=mutual_dtypes.map(lambda pair: pair[0]),
        shape=mutual_shapes.map(lambda pair: pair[0]),
    )
    arrays2 = arrays(
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


class KVD(NamedTuple):
    keyword: str
    value: Any
    default: Any


@composite
def specified_kwargs(draw, *keys_values_defaults: KVD):
    """Generates valid kwargs given expected defaults.

    When we can't realistically use hh.kwargs() and thus test whether xp infact
    defaults correctly, this strategy lets us remove generated arguments if they
    are of the default value anyway.
    """
    kw = {}
    for keyword, value, default in keys_values_defaults:
        if value is not default or draw(booleans()):
            kw[keyword] = value
    return kw


def axes(ndim: int) -> SearchStrategy[Optional[Union[int, Shape]]]:
    """Generate valid arguments for some axis keywords"""
    axes_strats = [none()]
    if ndim != 0:
        axes_strats.append(integers(-ndim, ndim - 1))
        axes_strats.append(xps.valid_tuple_axes(ndim))
    return one_of(axes_strats)


@contextmanager
def reject_overflow():
    try:
        yield
    except Exception as e:
        if isinstance(e, OverflowError) or re.search("[Oo]verflow", str(e)):
            reject()
        else:
            raise e
