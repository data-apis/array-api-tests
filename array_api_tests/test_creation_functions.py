import math

from ._array_module import (asarray, arange, ceil, empty, empty_like, eye, full,
                            full_like, equal, all, linspace, ones, ones_like,
                            zeros, zeros_like, isnan)
from . import _array_module as xp
from .array_helpers import (is_integer_dtype, dtype_ranges,
                            assert_exactly_equal, isintegral, is_float_dtype)
from .hypothesis_helpers import (numeric_dtypes, dtypes, MAX_ARRAY_SIZE,
                                 shapes, sizes, sqrt_sizes, shared_dtypes,
                                 scalars, xps, kwargs)

from hypothesis import assume, given
from hypothesis.strategies import integers, floats, one_of, none, booleans, just, shared, composite


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

@given(shapes, kwargs(dtype=none() | shared_dtypes))
def test_empty(shape, kw):
    out = empty(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert is_float_dtype(out.dtype), f"empty() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but empty() returned an array with dtype {out.dtype}"
    if isinstance(shape, int):
        shape = (shape,)
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shapes),
    kw=kwargs(dtype=none() | xps.scalar_dtypes())
)
def test_empty_like(x, kw):
    out = empty_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but empty_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but empty_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, f"{x.shape=}, but empty_like() returned an array with shape {out.shape}"


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
        assert is_float_dtype(a.dtype), "eye() should return an array with the default floating point dtype"
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


@composite
def full_fill_values(draw):
    kw = draw(shared(kwargs(dtype=none() | xps.scalar_dtypes()), key="full_kw"))
    dtype = kw.get("dtype", None) or draw(xps.scalar_dtypes())
    return draw(xps.from_dtype(dtype))


@given(
    shape=shapes,
    fill_value=full_fill_values(),
    kw=shared(kwargs(dtype=none() | xps.scalar_dtypes()), key="full_kw"),
)
def test_full(shape, fill_value, kw):
    out = full(shape, fill_value, **kw)
    if kw.get("dtype", None):
        dtype = kw["dtype"]
    elif isinstance(fill_value, bool):
        dtype = xp.bool
    elif isinstance(fill_value, int):
        dtype = xp.int64
    else:
        dtype = xp.float64
    if kw.get("dtype", None) is None:
        if dtype == xp.float64:
            assert is_float_dtype(out.dtype), f"full() returned an array with dtype {out.dtype}, but should be the default float dtype"
        elif dtype == xp.int64:
            assert out.dtype == xp.int32 or out.dtype == xp.int64, f"full() returned an array with dtype {out.dtype}, but should be the default integer dtype"
        else:
            assert out.dtype == xp.bool, f"full() returned an array with dtype {out.dtype}, but should be the bool dtype"
    else:
        assert out.dtype == dtype
    assert out.shape == shape,  f"{shape=}, but full() returned an array with shape {out.shape}"
    if is_float_dtype(out.dtype) and math.isnan(fill_value):
        assert all(isnan(out)), "full() array did not equal the fill value"
    else:
        assert all(equal(out, asarray(fill_value, dtype=dtype))), "full() array did not equal the fill value"


@composite
def full_like_fill_values(draw):
    kw = draw(shared(kwargs(dtype=none() | xps.scalar_dtypes()), key="full_like_kw"))
    dtype = kw.get("dtype", None) or draw(shared_dtypes)
    return draw(xps.from_dtype(dtype))


@given(
    x=xps.arrays(dtype=shared_dtypes, shape=shapes),
    fill_value=full_like_fill_values(),
    kw=shared(kwargs(dtype=none() | xps.scalar_dtypes()), key="full_like_kw"),
)
def test_full_like(x, fill_value, kw):
    out = full_like(x, fill_value, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but full_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but full_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but full_like() returned an array with shape {out.shape}"
    if is_float_dtype(dtype) and math.isnan(fill_value):
        assert all(isnan(out)), "full_like() array did not equal the fill value"
    else:
        assert all(equal(out, asarray(fill_value, dtype=dtype))), "full_like() array did not equal the fill value"


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
        assert is_float_dtype(a.dtype), "linspace() should return an array with the default floating point dtype"
    else:
        assert a.dtype == dtype, "linspace() did not produce the correct dtype"

    assert a.shape == (num,), "linspace() did not return an array with the correct shape"

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


def make_one(dtype):
    if kwargs is None or is_float_dtype(dtype):
        return 1.0
    elif is_integer_dtype(dtype):
        return 1
    else:
        return True


@given(shapes, kwargs(dtype=none() | xps.scalar_dtypes()))
def test_ones(shape, kw):
    out = ones(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert is_float_dtype(out.dtype), f"ones() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but ones() returned an array with dtype {out.dtype}"
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"
    assert all(equal(out, full((), make_one(dtype), dtype=dtype))), "ones() array did not equal 1"


@given(
    x=xps.arrays(dtype=dtypes, shape=shapes),
    kw=kwargs(dtype=none() | xps.scalar_dtypes()),
)
def test_ones_like(x, kw):
    out = ones_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but ones_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but ones_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but ones_like() returned an array with shape {out.shape}"
    assert all(equal(out, full((), make_one(dtype), dtype=dtype))), "ones_like() array elements did not equal 1"


def make_zero(dtype):
    if is_float_dtype(dtype):
        return 0.0
    elif is_integer_dtype(dtype):
        return 0
    else:
        return False


@given(shapes, kwargs(dtype=none() | xps.scalar_dtypes()))
def test_zeros(shape, kw):
    out = zeros(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert is_float_dtype(out.dtype), "zeros() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but zeros() returned an array with dtype {out.dtype}"
    assert out.shape == shape, "zeros() produced an array with incorrect shape"
    assert all(equal(out, full((), make_zero(dtype), dtype=dtype))), "zeros() array did not equal 0"


@given(
    x=xps.arrays(dtype=dtypes, shape=shapes),
    kw=kwargs(dtype=none() | xps.scalar_dtypes()),
)
def test_zeros_like(x, kw):
    out = zeros_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but zeros_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but zeros_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but zeros_like() returned an array with shape {out.shape}"
    assert all(equal(out, full((), make_zero(dtype), dtype=out.dtype))), "zeros_like() array elements did not all equal 0"

