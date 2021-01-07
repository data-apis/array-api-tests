from ._array_module import (arange, ceil, empty, _floating_dtypes, eye, full,
equal, all)
from .array_helpers import is_integer_dtype, dtype_ranges
from .hypothesis_helpers import (numeric_dtypes, dtypes, MAX_ARRAY_SIZE,
                                 shapes, sizes, sqrt_sizes, shared_dtypes,
                                 shared_scalars)

from hypothesis import assume, given
from hypothesis.strategies import integers, floats, one_of, none

int_range = integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE)
float_range = floats(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE, allow_nan=False)
@given(one_of(int_range, float_range),
       one_of(none(), int_range, float_range),
       one_of(none(), int_range, float_range).filter(lambda x: x != 0),
       one_of(none(), numeric_dtypes))
def test_arange(start, stop, step, dtype):
    if dtype in dtype_ranges:
        m, M = dtype_ranges[dtype]
        if (not (m <= start <= M)
            or isinstance(stop, int) and not (m <= stop <= M)
            or isinstance(step, int) and not (m <= step <= M)):
            assume(False)

    kwargs = {} if dtype is None else {'dtype': dtype}

    all_int = (is_integer_dtype(dtype)
               and isinstance(start, int)
               and (stop is None or isinstance(stop, int))
               and (step is None or isinstance(step, int)))

    if stop is None:
        # NB: "start" is really the stop
        # step is ignored in this case
        a = arange(start, **kwargs)
        if all_int:
            r = range(start)
    elif step is None:
        a = arange(start, stop, **kwargs)
        if all_int:
            r = range(start, stop)
    else:
        a = arange(start, stop, step, **kwargs)
        if all_int:
            r = range(start, stop, step)
    if dtype is None:
        # TODO: What is the correct dtype of a?
        pass
    else:
        assert a.dtype == dtype, "arange() produced an incorrect dtype"
    assert a.ndim == 1, "arange() should return a 1-dimensional array"
    if all_int:
        assert a.shape == (len(r),), "arange() produced incorrect shape"
        if len(r) <= MAX_ARRAY_SIZE:
            assert list(a) == list(r), "arange() produced incorrect values"
    else:
        # This is already implied by the len(r) test above
        if (stop is not None
            and step is not None
            and (step > 0 and stop >= start
                 or step < 0 and stop <= start)):
            assert a.size == ceil((stop-start)/step), "arange() produced an array of the incorrect size"

@given(one_of(shapes, sizes), one_of(none(), dtypes))
def test_empty(shape, dtype):
    if dtype is None:
        a = empty(shape)
        assert a.dtype in _floating_dtypes, "empty() should produce an array with the default floating point dtype"
    else:
        a = empty(shape, dtype=dtype)
        assert a.dtype == dtype

    if isinstance(shape, int):
        shape = (shape,)
    assert a.shape == shape, "empty() produced an array with an incorrect shape"

# TODO: implement empty_like (requires hypothesis arrays support)
def test_empty_like():
    pass

@given(sqrt_sizes, one_of(none(), sqrt_sizes), one_of(none(), integers()), numeric_dtypes)
def test_eye(N, M, k, dtype):
    kwargs = {k: v for k, v in {'M': M, 'k': k, 'dtype': dtype}.items() if v
              is not None}
    a = eye(N, **kwargs)
    if dtype is None:
        assert a.dtype in _floating_dtypes, "eye() should produce an array with the default floating point dtype"
    else:
        assert a.dtype == dtype

    if M is None:
        M = N
    assert a.shape == (N, M), "eye() produced an array with incorrect shape"

    if k is None:
        k = 0
    for i in range(N):
        for j in range(M):
            if j - i == k:
                assert a[i, j] == 1, "eye() did not produce a 1 on the diagonal"
            else:
                assert a[i, j] == 0, "eye() did not produce a 0 off the diagonal"

@given(shapes, shared_scalars(), one_of(none(), shared_dtypes))
def test_full(shape, fill_value, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}

    a = full(shape, fill_value, **kwargs)

    if dtype is None:
        # TODO: Should it actually match the fill_value?
        # assert a.dtype in _floating_dtypes, "eye() should produce an array with the default floating point dtype"
        pass
    else:
        assert a.dtype == dtype

    assert a.shape == shape, "full() produced an array with incorrect shape"
    assert all(equal(a, fill_value)), "full() array did not equal the fill value"
