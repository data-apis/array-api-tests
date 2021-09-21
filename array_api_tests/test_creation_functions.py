from ._array_module import (asarray, arange, ceil, empty, empty_like, eye, full,
                            full_like, equal, all, linspace, ones, ones_like,
                            zeros, zeros_like, isnan)
from .array_helpers import (is_integer_dtype, dtype_ranges,
                            assert_exactly_equal, isintegral, is_float_dtype)
from .hypothesis_helpers import (numeric_dtypes, dtypes, MAX_ARRAY_SIZE, promotable_dtypes,
                                 shapes, sizes, sqrt_sizes, shared_dtypes,
                                 scalars, xps)

from hypothesis import assume, given
from hypothesis.strategies import integers, floats, one_of, none, booleans, just

int_range = integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE)
float_range = floats(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE,
                     allow_nan=False)
@given(one_of(int_range, float_range),
       one_of(none(), int_range, float_range),
       one_of(none(), int_range, float_range).filter(lambda x: x != 0
                                                     and (abs(x) > 0.01 if isinstance(x, float) else True)),
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
            assert a.size == ceil(asarray((stop-start)/step)), "arange() produced an array of the incorrect size"

@given(one_of(shapes, sizes), one_of(none(), dtypes))
def test_empty(shape, dtype):
    if dtype is None:
        a = empty(shape)
        assert is_float_dtype(a.dtype), "empty() should produce an array with the default floating point dtype"
    else:
        a = empty(shape, dtype=dtype)
        assert a.dtype == dtype

    if isinstance(shape, int):
        shape = (shape,)
    assert a.shape == shape, "empty() produced an array with an incorrect shape"


@given(
    a=xps.arrays(
        dtype=shared_dtypes,
        shape=shapes,
    ),
    dtype=one_of(none(), promotable_dtypes(shared_dtypes)),
)
def test_empty_like(a, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}

    a_like = empty_like(a, **kwargs)

    if dtype is None:
        # TODO: Should it actually match a.dtype?
        # assert is_float_dtype(a_like.dtype), "empty_like() should produce an array with the default floating point dtype"
        pass
    else:
        assert a_like.dtype == dtype, "empty_like() produced an array with an incorrect dtype"

    assert a_like.shape == a.shape, "empty_like() produced an array with an incorrect shape"


# TODO: Use this method for all optional arguments
optional_marker = object()

@given(sqrt_sizes, one_of(just(optional_marker), none(), sqrt_sizes), one_of(none(), integers()), numeric_dtypes)
def test_eye(n_rows, n_cols, k, dtype):
    kwargs = {k: v for k, v in {'k': k, 'dtype': dtype}.items() if v
              is not None}
    if n_cols is optional_marker:
        a = eye(n_rows, **kwargs)
        n_cols = None
    else:
        a = eye(n_rows, n_cols, **kwargs)
    if dtype is None:
        assert is_float_dtype(a.dtype), "eye() should produce an array with the default floating point dtype"
    else:
        assert a.dtype == dtype, "eye() did not produce the correct dtype"

    if n_cols is None:
        n_cols = n_rows
    assert a.shape == (n_rows, n_cols), "eye() produced an array with incorrect shape"

    if k is None:
        k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if j - i == k:
                assert a[i, j] == 1, "eye() did not produce a 1 on the diagonal"
            else:
                assert a[i, j] == 0, "eye() did not produce a 0 off the diagonal"

@given(shapes, scalars(shared_dtypes), one_of(none(), shared_dtypes))
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
    if is_float_dtype(a.dtype) and isnan(asarray(fill_value)):
        assert all(isnan(a)), "full() array did not equal the fill value"
    else:
        assert all(equal(a, asarray(fill_value, **kwargs))), "full() array did not equal the fill value"

@given(
    a=xps.arrays(
        dtype=shared_dtypes,
        shape=shapes,
    ),
    fill_value=promotable_dtypes(shared_dtypes).flatmap(xps.from_dtype),
    dtype=one_of(none(), promotable_dtypes(shared_dtypes)),
)
def test_full_like(a, fill_value, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}

    a_like = full_like(a, fill_value, **kwargs)

    if dtype is None:
        # TODO: Should it actually match a.dtype?
        pass
    else:
        assert a_like.dtype == dtype

    assert a_like.shape == a.shape, "full_like() produced an array with incorrect shape"
    if is_float_dtype(a_like.dtype) and isnan(asarray(fill_value)):
        assert all(isnan(a_like)), "full_like() array did not equal the fill value"
    else:
        assert all(equal(a_like, asarray(fill_value, dtype=a_like.dtype))), "full_like() array did not equal the fill value"


@given(scalars(shared_dtypes, finite=True),
       scalars(shared_dtypes, finite=True),
       sizes,
       one_of(none(), shared_dtypes),
       one_of(none(), booleans()),)
def test_linspace(start, stop, num, dtype, endpoint):
    # Skip on int start or stop that cannot be exactly represented as a float,
    # since we do not have good approx_equal helpers yet.
    if ((dtype is None or is_float_dtype(dtype))
        and ((isinstance(start, int) and not isintegral(asarray(start, dtype=dtype)))
             or (isinstance(stop, int) and not isintegral(asarray(stop, dtype=dtype))))):
        assume(False)

    kwargs = {k: v for k, v in {'dtype': dtype, 'endpoint': endpoint}.items()
              if v is not None}
    a = linspace(start, stop, num, **kwargs)

    if dtype is None:
        assert is_float_dtype(a.dtype), "linspace() should produce an array with the default floating point dtype"
    else:
        assert a.dtype == dtype, "linspace() did not produce the correct dtype"

    assert a.shape == (num,), "linspace() did not produce an array with the correct shape"

    if endpoint in [None, True]:
        if num > 1:
            assert all(equal(a[-1], full((), stop, dtype=a.dtype))), "linspace() produced an array that does not include the endpoint"
    else:
        # linspace(..., num, endpoint=False) is the same as the first num
        # elements of linspace(..., num+1, endpoint=True)
        b = linspace(start, stop, num + 1, **{**kwargs, 'endpoint': True})
        assert_exactly_equal(b[:-1], a)

    if num > 0:
        # We need to cast start to dtype
        assert all(equal(a[0], full((), start, dtype=a.dtype))), "linspace() produced an array that does not start with the start"

        # TODO: This requires an assert_approx_equal function

        # n = num - 1 if endpoint in [None, True] else num
        # for i in range(1, num):
        #     assert all(equal(a[i], full((), i*(stop - start)/n + start, dtype=dtype))), f"linspace() produced an array with an incorrect value at index {i}"

@given(shapes, one_of(none(), dtypes))
def test_ones(shape, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}
    if dtype is None or is_float_dtype(dtype):
        ONE = 1.0
    elif is_integer_dtype(dtype):
        ONE = 1
    else:
        ONE = True

    a = ones(shape, **kwargs)

    if dtype is None:
        # TODO: Should it actually match the fill_value?
        # assert a.dtype in _floating_dtypes, "eye() should produce an array with the default floating point dtype"
        pass
    else:
        assert a.dtype == dtype

    assert a.shape == shape, "ones() produced an array with incorrect shape"
    assert all(equal(a, full((), ONE, **kwargs))), "ones() array did not equal 1"


@given(
    a=xps.arrays(
        dtype=shared_dtypes,
        shape=shapes,
    ),
    dtype=one_of(none(), promotable_dtypes(shared_dtypes)),
)
def test_ones_like(a, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}
    if kwargs is None or is_float_dtype(a.dtype):
        ONE = 1.0
    elif is_integer_dtype(a.dtype):
        ONE = 1
    else:
        ONE = True

    a_like = ones_like(a, **kwargs)

    if dtype is None:
        # TODO: Should it actually match a.dtype?
        pass
    else:
        assert a_like.dtype == dtype, "ones_like() produced an array with an incorrect dtype"

    assert a_like.shape == a.shape, "ones_like() produced an array with an incorrect shape"
    assert all(equal(a_like, full((), ONE, dtype=a_like.dtype))), "ones_like() array did not equal 1"


@given(shapes, one_of(none(), dtypes))
def test_zeros(shape, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}
    if dtype is None or is_float_dtype(dtype):
        ZERO = 0.0
    elif is_integer_dtype(dtype):
        ZERO = 0
    else:
        ZERO = False

    a = zeros(shape, **kwargs)

    if dtype is None:
        # TODO: Should it actually match the fill_value?
        # assert a.dtype in _floating_dtypes, "eye() should produce an array with the default floating point dtype"
        pass
    else:
        assert a.dtype == dtype

    assert a.shape == shape, "zeros() produced an array with incorrect shape"
    assert all(equal(a, full((), ZERO, **kwargs))), "zeros() array did not equal 0"


@given(
    a=xps.arrays(
        dtype=shared_dtypes,
        shape=shapes,
    ),
    dtype=one_of(none(), promotable_dtypes(shared_dtypes)),
)
def test_zeros_like(a, dtype):
    kwargs = {} if dtype is None else {'dtype': dtype}
    if dtype is None or is_float_dtype(a.dtype):
        ZERO = 0.0
    elif is_integer_dtype(a.dtype):
        ZERO = 0
    else:
        ZERO = False

    a_like = zeros_like(a, **kwargs)

    if dtype is None:
        # TODO: Should it actually match a.dtype?
        pass
    else:
        assert a_like.dtype == dtype, "zeros_like() produced an array with an incorrect dtype"

    assert a_like.shape == a.shape, "zeros_like() produced an array with an incorrect shape"
    assert all(equal(a_like, full((), ZERO, dtype=a_like.dtype))), "zeros_like() array did not equal 0"

