import math
from typing import Union
from itertools import takewhile, count

from ._array_module import (asarray, arange, empty, empty_like, eye, full,
                            full_like, equal, linspace, ones, ones_like,
                            zeros, zeros_like, isnan)
from . import _array_module as xp
from .array_helpers import assert_exactly_equal, isintegral
from .hypothesis_helpers import (numeric_dtypes, dtypes, MAX_ARRAY_SIZE,
                                 shapes, sizes, sqrt_sizes, shared_dtypes,
                                 scalars, kwargs)
from . import dtype_helpers as dh
from . import xps

from hypothesis import assume, given, strategies as st


int_min, int_max = dh.dtype_ranges[dh.default_int]
float_min = float(int_min * (MAX_ARRAY_SIZE - 1))
float_max = float(int_max * (MAX_ARRAY_SIZE - 1))


def reals(min_value=None, max_value=None) -> st.SearchStrategy[Union[int, float]]:
    round_ = int
    if min_value is not None and min_value > 0:
        round_ = math.ceil
    elif max_value is not None and max_value < 0:
        round_ = math.floor
    int_min_value = int_min if min_value is None else max(round_(min_value), int_min)
    int_max_value = int_max if max_value is None else min(round_(max_value), int_max)

    float_min_value = float_min if min_value is None else max(min_value, float_min)
    float_max_value = float_max if max_value is None else min(max_value, float_max)

    return st.one_of(
        st.integers(int_min_value, int_max_value),
        st.floats(float_min_value, float_max_value, allow_nan=False, allow_infinity=False)
    )

@given(start=reals(), dtype=st.none() | numeric_dtypes, data=st.data())
def test_arange(start, dtype, data):
    if data.draw(st.booleans(), label="stop is None"):
        _start = 0
        _stop = start
        stop = None
    else:
        _start = start
        _stop = data.draw(reals(), label="stop")
        stop = _stop

    tol = abs(_stop - _start) / (MAX_ARRAY_SIZE - 1)
    assume(-tol > int_min)
    assume(tol < int_max)
    if _stop - _start > 0:
        step_strat = reals(min_value=tol).filter(lambda n: n != 0)
    else:
        step_strat = reals(max_value=-tol).filter(lambda n: n != 0)
    step = data.draw(step_strat, label="step")

    all_int = all(arg is None or isinstance(arg, int) for arg in [start, stop, step])

    if dtype is None:
        if all_int:
            _dtype = dh.default_int
        else:
            _dtype = dh.default_float
    else:
        _dtype = dtype

    if dh.is_int_dtype(_dtype):
        m, M = dh.dtype_ranges[_dtype]
        assume(m <= _start <= M)
        assume(m <= _stop <= M)
        assume(m <= step <= M)

    if step > 0:
        condition = lambda x: x < _stop
    else:
        condition = lambda x: x > _stop
    scalar_type = int if dh.is_int_dtype(_dtype) else float
    elements = list(scalar_type(n) for n in takewhile(condition, count(_start, step)))
    size = len(elements)
    assert size < MAX_ARRAY_SIZE, f"{size=}, should be below {MAX_ARRAY_SIZE=}"

    out = arange(start, stop=stop, step=step, dtype=dtype)

    if dtype is None:
        if all_int:
            assert dh.is_int_dtype(out.dtype)
        else:
            assert dh.is_float_dtype(out.dtype)
    else:
        assert out.dtype == dtype
    assert out.ndim == 1
    if dh.is_int_dtype(step):
        assert out.size == size
    else:
        # We check size is roughly as expected to avoid edge cases e.g.
        #
        #     >>> xp.arange(2, step=0.333333333333333)
        #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0]
        #     >>> xp.arange(2, step=0.3333333333333333)
        #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66]
        #
        assert out.size in (size - 1, size, size + 1)
    # TODO test elements

@given(shapes(), kwargs(dtype=st.none() | shared_dtypes))
def test_empty(shape, kw):
    out = empty(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert dh.is_float_dtype(out.dtype), f"empty() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but empty() returned an array with dtype {out.dtype}"
    if isinstance(shape, int):
        shape = (shape,)
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shapes()),
    kw=kwargs(dtype=st.none() | xps.scalar_dtypes())
)
def test_empty_like(x, kw):
    out = empty_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but empty_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but empty_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, f"{x.shape=}, but empty_like() returned an array with shape {out.shape}"


# TODO: Use this method for xp.all optional arguments
optional_marker = object()

@given(sqrt_sizes, st.one_of(st.just(optional_marker), st.none(), sqrt_sizes), st.one_of(st.none(), st.integers()), numeric_dtypes)
def test_eye(n_rows, n_cols, k, dtype):
    kwargs = {k: v for k, v in {'k': k, 'dtype': dtype}.items() if v
              is not None}
    if n_cols is optional_marker:
        a = eye(n_rows, **kwargs)
        n_cols = None
    else:
        a = eye(n_rows, n_cols, **kwargs)
    if dtype is None:
        assert dh.is_float_dtype(a.dtype), "eye() should return an array with the default floating point dtype"
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


default_unsafe_dtypes = [xp.uint64]
if dh.default_int == xp.int32:
    default_unsafe_dtypes.extend([xp.uint32, xp.int64])
if dh.default_float == xp.float32:
    default_unsafe_dtypes.append(xp.float64)
default_safe_scalar_dtypes: st.SearchStrategy = xps.scalar_dtypes().filter(
    lambda d: d not in default_unsafe_dtypes
)


@st.composite
def full_fill_values(draw):
    kw = draw(st.shared(kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw"))
    dtype = kw.get("dtype", None) or draw(default_safe_scalar_dtypes)
    return draw(xps.from_dtype(dtype))


@given(
    shape=shapes(),
    fill_value=full_fill_values(),
    kw=st.shared(kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw"),
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
            assert dh.is_float_dtype(out.dtype), f"full() returned an array with dtype {out.dtype}, but should be the default float dtype"
        elif dtype == xp.int64:
            assert out.dtype == xp.int32 or out.dtype == xp.int64, f"full() returned an array with dtype {out.dtype}, but should be the default integer dtype"
        else:
            assert out.dtype == xp.bool, f"full() returned an array with dtype {out.dtype}, but should be the bool dtype"
    else:
        assert out.dtype == dtype
    assert out.shape == shape,  f"{shape=}, but full() returned an array with shape {out.shape}"
    if dh.is_float_dtype(out.dtype) and math.isnan(fill_value):
        assert xp.all(isnan(out)), "full() array did not equal the fill value"
    else:
        assert xp.all(equal(out, asarray(fill_value, dtype=dtype))), "full() array did not equal the fill value"


@st.composite
def full_like_fill_values(draw):
    kw = draw(st.shared(kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_like_kw"))
    dtype = kw.get("dtype", None) or draw(shared_dtypes)
    return draw(xps.from_dtype(dtype))


@given(
    x=xps.arrays(dtype=shared_dtypes, shape=shapes()),
    fill_value=full_like_fill_values(),
    kw=st.shared(kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_like_kw"),
)
def test_full_like(x, fill_value, kw):
    out = full_like(x, fill_value, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but full_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but full_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but full_like() returned an array with shape {out.shape}"
    if dh.is_float_dtype(dtype) and math.isnan(fill_value):
        assert xp.all(isnan(out)), "full_like() array did not equal the fill value"
    else:
        assert xp.all(equal(out, asarray(fill_value, dtype=dtype))), "full_like() array did not equal the fill value"


@given(scalars(shared_dtypes, finite=True),
       scalars(shared_dtypes, finite=True),
       sizes,
       st.one_of(st.none(), shared_dtypes),
       st.one_of(st.none(), st.booleans()),)
def test_linspace(start, stop, num, dtype, endpoint):
    # Skip on int start or stop that cannot be exactly represented as a float,
    # since we do not have good approx_equal helpers yet.
    if ((dtype is None or dh.is_float_dtype(dtype))
        and ((isinstance(start, int) and not isintegral(asarray(start, dtype=dtype)))
             or (isinstance(stop, int) and not isintegral(asarray(stop, dtype=dtype))))):
        assume(False)

    kwargs = {k: v for k, v in {'dtype': dtype, 'endpoint': endpoint}.items()
              if v is not None}
    a = linspace(start, stop, num, **kwargs)

    if dtype is None:
        assert dh.is_float_dtype(a.dtype), "linspace() should return an array with the default floating point dtype"
    else:
        assert a.dtype == dtype, "linspace() did not produce the correct dtype"

    assert a.shape == (num,), "linspace() did not return an array with the correct shape"

    if endpoint in [None, True]:
        if num > 1:
            assert xp.all(equal(a[-1], asarray(stop, dtype=a.dtype))), "linspace() produced an array that does not include the endpoint"
    else:
        # linspace(..., num, endpoint=False) is the same as the first num
        # elements of linspace(..., num+1, endpoint=True)
        b = linspace(start, stop, num + 1, **{**kwargs, 'endpoint': True})
        assert_exactly_equal(b[:-1], a)

    if num > 0:
        # We need to cast start to dtype
        assert xp.all(equal(a[0], asarray(start, dtype=a.dtype))), "linspace() produced an array that does not start with the start"

        # TODO: This requires an assert_approx_equal function

        # n = num - 1 if endpoint in [None, True] else num
        # for i in range(1, num):
        #     assert xp.all(equal(a[i], full((), i*(stop - start)/n + start, dtype=dtype))), f"linspace() produced an array with an incorrect value at index {i}"


def make_one(dtype):
    if kwargs is None or dh.is_float_dtype(dtype):
        return 1.0
    elif dh.is_int_dtype(dtype):
        return 1
    else:
        return True


@given(shapes(), kwargs(dtype=st.none() | xps.scalar_dtypes()))
def test_ones(shape, kw):
    out = ones(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert dh.is_float_dtype(out.dtype), f"ones() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but ones() returned an array with dtype {out.dtype}"
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"
    assert xp.all(equal(out, asarray(make_one(dtype), dtype=dtype))), "ones() array did not equal 1"


@given(
    x=xps.arrays(dtype=dtypes, shape=shapes()),
    kw=kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_ones_like(x, kw):
    out = ones_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but ones_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but ones_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but ones_like() returned an array with shape {out.shape}"
    assert xp.all(equal(out, asarray(make_one(dtype), dtype=dtype))), "ones_like() array elements did not equal 1"


def make_zero(dtype):
    if dh.is_float_dtype(dtype):
        return 0.0
    elif dh.is_int_dtype(dtype):
        return 0
    else:
        return False


@given(shapes(), kwargs(dtype=st.none() | xps.scalar_dtypes()))
def test_zeros(shape, kw):
    out = zeros(shape, **kw)
    dtype = kw.get("dtype", None) or xp.float64
    if kw.get("dtype", None) is None:
        assert dh.is_float_dtype(out.dtype), "zeros() returned an array with dtype {out.dtype}, but should be the default float dtype"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but zeros() returned an array with dtype {out.dtype}"
    assert out.shape == shape, "zeros() produced an array with incorrect shape"
    assert xp.all(equal(out, asarray(make_zero(dtype), dtype=dtype))), "zeros() array did not equal 0"


@given(
    x=xps.arrays(dtype=dtypes, shape=shapes()),
    kw=kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_zeros_like(x, kw):
    out = zeros_like(x, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        assert out.dtype == x.dtype, f"{x.dtype=!s}, but zeros_like() returned an array with dtype {out.dtype}"
    else:
        assert out.dtype == dtype, f"{dtype=!s}, but zeros_like() returned an array with dtype {out.dtype}"
    assert out.shape == x.shape, "{x.shape=}, but zeros_like() returned an array with shape {out.shape}"
    assert xp.all(equal(out, asarray(make_zero(dtype), dtype=out.dtype))), "zeros_like() array elements did not xp.all equal 0"

