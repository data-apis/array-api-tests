from array_api_tests.typing import DataType
import math
from typing import Union
from itertools import takewhile, count

from hypothesis import assume, given, strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import hypothesis_helpers as hh
from . import dtype_helpers as dh
from . import pytest_helpers as ph
from . import xps


def assert_default_float(func_name: str, dtype: DataType):
    f_dtype = dh.dtype_to_name[dtype]
    f_default = dh.dtype_to_name[dh.default_float]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"floating-point dtype {f_default} [{func_name}()]"
    )
    assert dtype == dh.default_float, msg


def assert_default_int(func_name: str, dtype: DataType):
    f_dtype = dh.dtype_to_name[dtype]
    f_default = dh.dtype_to_name[dh.default_int]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"integer dtype {f_default} [{func_name}()]"
    )
    assert dtype == dh.default_int, msg


def assert_kw_dtype(
    func_name: str,
    kw_dtype: DataType,
    out_dtype: DataType,
):
    f_kw_dtype = dh.dtype_to_name[kw_dtype]
    f_out_dtype = dh.dtype_to_name[out_dtype]
    msg = (
        f"out.dtype={f_out_dtype}, but should be {f_kw_dtype} "
        f"[{func_name}(dtype={f_kw_dtype})]"
    )
    assert out_dtype == kw_dtype, msg



# Testing xp.arange() requires bounding the start/stop/step arguments to only
# test argument combinations compliant with the Array API, as well as to not
# produce arrays with hh.sizes not supproted by an array module.
#
# We first make sure generated integers can be represented by an array module's
# default integer type, as even if a float array should be produced a module
# might represent integer arguments as 0d arrays.
#
# This means that float arguments also need to be bound, so that they do not
# require any integer arguments to be outside the representable bounds.
int_min, int_max = dh.dtype_ranges[dh.default_int]
float_min = float(int_min * (hh.MAX_ARRAY_SIZE - 1))
float_max = float(int_max * (hh.MAX_ARRAY_SIZE - 1))


def reals(min_value=None, max_value=None) -> st.SearchStrategy[Union[int, float]]:
    round_ = int
    if min_value is not None and min_value > 0:
        round_ = math.ceil
    elif max_value is not None and max_value < 0:
        round_ = math.floor
    int_min_value = int_min if min_value is None else max(round_(min_value), int_min)
    int_max_value = int_max if max_value is None else min(round_(max_value), int_max)
    return st.one_of(
        st.integers(int_min_value, int_max_value),
        # We do not assign float bounds to the floats() strategy, instead opting
        # to filter out-of-bound values. Passing such min/max values will modify
        # test case reduction behaviour so that simple bugs will become harder
        # for users to identify. Hypothesis plans to improve floats() behaviour
        # in https://github.com/HypothesisWorks/hypothesis/issues/2907
        st.floats(min_value, max_value, allow_nan=False, allow_infinity=False).filter(
            lambda n: float_min <= n <= float_max
        )
    )


@given(start=reals(), dtype=st.none() | hh.numeric_dtypes, data=st.data())
def test_arange(start, dtype, data):
    stop = data.draw(reals() | st.none(), label="stop")
    if stop is None:
        _start = 0
        _stop = start
        stop = None
    else:
        _start = start
        _stop = data.draw(reals(), label="stop")
        stop = _stop

    tol = abs(_stop - _start) / (hh.MAX_ARRAY_SIZE - 1)
    assume(-tol > int_min)
    assume(tol < int_max)
    step = data.draw(
        st.one_of(
            reals(min_value=tol).filter(lambda n: n != 0),
            reals(max_value=-tol).filter(lambda n: n != 0)
        ),
        label="step"
    )

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

    pos_range = _stop > _start
    pos_step = step > 0
    if _start != _stop and pos_range == pos_step:
        if pos_step:
            condition = lambda x: x <= _stop
        else:
            condition = lambda x: x >= _stop
        scalar_type = int if dh.is_int_dtype(_dtype) else float
        elements = list(scalar_type(n) for n in takewhile(condition, count(_start, step)))
    else:
        elements = []
    size = len(elements)
    assert size <= hh.MAX_ARRAY_SIZE, f"{size=}, should be no more than {hh.MAX_ARRAY_SIZE=}"

    out = xp.arange(start, stop=stop, step=step, dtype=dtype)

    if dtype is None:
        if all_int:
            assert_default_int("arange", out.dtype)
        else:
            assert_default_float("arange", out.dtype)
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
    assume(out.size == size)
    if dh.is_int_dtype(_dtype):
        ah.assert_exactly_equal(out, ah.asarray(elements, dtype=_dtype))
    else:
        pass # TODO: either emulate array module behaviour or assert a rough equals

@given(hh.shapes(), hh.kwargs(dtype=st.none() | hh.shared_dtypes))
def test_empty(shape, kw):
    out = xp.empty(shape, **kw)
    if kw.get("dtype", None) is None:
        assert_default_float("empty", out.dtype)
    else:
        assert_kw_dtype("empty", kw["dtype"], out.dtype)
    if isinstance(shape, int):
        shape = (shape,)
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes())
)
def test_empty_like(x, kw):
    out = xp.empty_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("empty_like", (x.dtype,), out.dtype)
    else:
        assert_kw_dtype("empty_like", kw["dtype"], out.dtype)
    assert out.shape == x.shape, f"{x.shape=}, but empty_like() returned an array with shape {out.shape}"


# TODO: Use this method for ah.all optional arguments
optional_marker = object()

@given(hh.sqrt_sizes, st.one_of(st.just(optional_marker), st.none(), hh.sqrt_sizes), st.one_of(st.none(), st.integers()), hh.numeric_dtypes)
def test_eye(n_rows, n_cols, k, dtype):
    kwargs = {k: v for k, v in {'k': k, 'dtype': dtype}.items() if v
              is not None}
    if n_cols is optional_marker:
        a = xp.eye(n_rows, **kwargs)
        n_cols = None
    else:
        a = xp.eye(n_rows, n_cols, **kwargs)
    if dtype is None:
        assert_default_float("arange", a.dtype)
    else:
        assert_kw_dtype("empty", dtype, a.dtype)

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
default_safe_dtypes: st.SearchStrategy = xps.scalar_dtypes().filter(
    lambda d: d not in default_unsafe_dtypes
)


@st.composite
def full_fill_values(draw):
    kw = draw(st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw"))
    dtype = kw.get("dtype", None) or draw(default_safe_dtypes)
    return draw(xps.from_dtype(dtype))


@given(
    shape=hh.shapes(),
    fill_value=full_fill_values(),
    kw=st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw"),
)
def test_full(shape, fill_value, kw):
    out = xp.full(shape, fill_value, **kw)
    if kw.get("dtype", None):
        dtype = kw["dtype"]
    elif isinstance(fill_value, bool):
        dtype = xp.bool
    elif isinstance(fill_value, int):
        dtype = dh.default_int
    else:
        dtype = dh.default_float
    if kw.get("dtype", None) is None:
        if isinstance(fill_value, bool):
            pass # TODO
        elif isinstance(fill_value, int):
            assert_default_int("full", out.dtype)
        else:
            assert_default_float("full", out.dtype)
    else:
        assert_kw_dtype("full", kw["dtype"], out.dtype)
    assert out.shape == shape,  f"{shape=}, but full() returned an array with shape {out.shape}"
    if dh.is_float_dtype(out.dtype) and math.isnan(fill_value):
        assert ah.all(ah.isnan(out)), "full() array did not equal the fill value"
    else:
        assert ah.all(ah.equal(out, ah.asarray(fill_value, dtype=dtype))), "full() array did not equal the fill value"


@st.composite
def full_like_fill_values(draw):
    kw = draw(st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_like_kw"))
    dtype = kw.get("dtype", None) or draw(hh.shared_dtypes)
    return draw(xps.from_dtype(dtype))


@given(
    x=xps.arrays(dtype=hh.shared_dtypes, shape=hh.shapes()),
    fill_value=full_like_fill_values(),
    kw=st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_like_kw"),
)
def test_full_like(x, fill_value, kw):
    out = xp.full_like(x, fill_value, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        ph.assert_dtype("full_like", (x.dtype,), out.dtype)
    else:
        assert_kw_dtype("full_like", kw["dtype"], out.dtype)
    assert out.shape == x.shape, "{x.shape=}, but full_like() returned an array with shape {out.shape}"
    if dh.is_float_dtype(dtype) and math.isnan(fill_value):
        assert ah.all(ah.isnan(out)), "full_like() array did not equal the fill value"
    else:
        assert ah.all(ah.equal(out, ah.asarray(fill_value, dtype=dtype))), "full_like() array did not equal the fill value"


@given(hh.scalars(hh.shared_dtypes, finite=True),
       hh.scalars(hh.shared_dtypes, finite=True),
       hh.sizes,
       st.one_of(st.none(), hh.shared_dtypes),
       st.one_of(st.none(), st.booleans()),)
def test_linspace(start, stop, num, dtype, endpoint):
    # Skip on int start or stop that cannot be exactly represented as a float,
    # since we do not have good approx_equal helpers yet.
    if ((dtype is None or dh.is_float_dtype(dtype))
        and ((isinstance(start, int) and not ah.isintegral(xp.asarray(start, dtype=dtype)))
             or (isinstance(stop, int) and not ah.isintegral(xp.asarray(stop, dtype=dtype))))):
        assume(False)

    kwargs = {k: v for k, v in {'dtype': dtype, 'endpoint': endpoint}.items()
              if v is not None}
    a = xp.linspace(start, stop, num, **kwargs)

    if dtype is None:
        assert_default_float("linspace", a.dtype)
    else:
        assert_kw_dtype("linspace", dtype, a.dtype)

    assert a.shape == (num,), "linspace() did not return an array with the correct shape"

    if endpoint in [None, True]:
        if num > 1:
            assert ah.all(ah.equal(a[-1], ah.asarray(stop, dtype=a.dtype))), "linspace() produced an array that does not include the endpoint"
    else:
        # linspace(..., num, endpoint=False) is the same as the first num
        # elements of linspace(..., num+1, endpoint=True)
        b = xp.linspace(start, stop, num + 1, **{**kwargs, 'endpoint': True})
        ah.assert_exactly_equal(b[:-1], a)

    if num > 0:
        # We need to cast start to dtype
        assert ah.all(ah.equal(a[0], ah.asarray(start, dtype=a.dtype))), "xp.linspace() produced an array that does not start with the start"

        # TODO: This requires an assert_approx_equal function

        # n = num - 1 if endpoint in [None, True] else num
        # for i in range(1, num):
        #     assert ah.all(ah.equal(a[i], ah.full((), i*(stop - start)/n + start, dtype=dtype))), f"linspace() produced an array with an incorrect value at index {i}"


def make_one(dtype):
    if dtype is None or dh.is_float_dtype(dtype):
        return 1.0
    elif dh.is_int_dtype(dtype):
        return 1
    else:
        return True


@given(hh.shapes(), hh.kwargs(dtype=st.none() | xps.scalar_dtypes()))
def test_ones(shape, kw):
    out = xp.ones(shape, **kw)
    if kw.get("dtype", None) is None:
        assert_default_float("ones", out.dtype)
    else:
        assert_kw_dtype("ones", kw["dtype"], out.dtype)
    assert out.shape == shape, f"{shape=}, but empty() returned an array with shape {out.shape}"
    dtype = kw.get("dtype", None) or dh.default_float
    assert ah.all(ah.equal(out, ah.asarray(make_one(dtype), dtype=dtype))), "ones() array did not equal 1"


@given(
    x=xps.arrays(dtype=hh.dtypes, shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_ones_like(x, kw):
    out = xp.ones_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("ones_like", (x.dtype,), out.dtype)
    else:
        assert_kw_dtype("ones_like", kw["dtype"], out.dtype)
    assert out.shape == x.shape, "{x.shape=}, but ones_like() returned an array with shape {out.shape}"
    dtype = kw.get("dtype", None) or x.dtype
    assert ah.all(ah.equal(out, ah.asarray(make_one(dtype), dtype=dtype))), "ones_like() array elements did not equal 1"


def make_zero(dtype):
    if dtype is None or dh.is_float_dtype(dtype):
        return 0.0
    elif dh.is_int_dtype(dtype):
        return 0
    else:
        return False


@given(hh.shapes(), hh.kwargs(dtype=st.none() | xps.scalar_dtypes()))
def test_zeros(shape, kw):
    out = xp.zeros(shape, **kw)
    if kw.get("dtype", None) is None:
        assert_default_float("zeros", out.dtype)
    else:
        assert_kw_dtype("zeros", kw["dtype"], out.dtype)
    assert out.shape == shape, "zeros() produced an array with incorrect shape"
    dtype = kw.get("dtype", None) or dh.default_float
    assert ah.all(ah.equal(out, ah.asarray(make_zero(dtype), dtype=dtype))), "zeros() array did not equal 0"


@given(
    x=xps.arrays(dtype=hh.dtypes, shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_zeros_like(x, kw):
    out = xp.zeros_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("zeros_like", (x.dtype,), out.dtype)
    else:
        assert_kw_dtype("zeros_like", kw["dtype"], out.dtype)
    assert out.shape == x.shape, "{x.shape=}, but xp.zeros_like() returned an array with shape {out.shape}"
    dtype = kw.get("dtype", None) or x.dtype
    assert ah.all(ah.equal(out, ah.asarray(make_zero(dtype), dtype=out.dtype))), "xp.zeros_like() array elements did not ah.all xp.equal 0"

