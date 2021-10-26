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
from .typing import Shape, DataType, Array


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


def assert_kw_dtype(func_name: str, kw_dtype: DataType, out_dtype: DataType):
    f_kw_dtype = dh.dtype_to_name[kw_dtype]
    f_out_dtype = dh.dtype_to_name[out_dtype]
    msg = (
        f"out.dtype={f_out_dtype}, but should be {f_kw_dtype} "
        f"[{func_name}(dtype={f_kw_dtype})]"
    )
    assert out_dtype == kw_dtype, msg


def assert_shape(func_name: str, out_shape: Shape, expected: Union[int, Shape], **kw):
    f_kw = ", ".join(f"{k}={v}" for k, v in kw.items())
    msg = f"out.shape={out_shape}, but should be {expected} [{func_name}({f_kw})]"
    if isinstance(expected, int):
        expected = (expected,)
    assert out_shape == expected, msg


def assert_fill(func_name: str, fill: float, dtype: DataType, out: Array, **kw):
    f_kw = ", ".join(f"{k}={v}" for k, v in kw.items())
    msg = f"out not filled with {fill} [{func_name}({f_kw})]\n" f"{out=}"
    if math.isnan(fill):
        assert ah.all(ah.isnan(out)), msg
    else:
        assert ah.all(ah.equal(out, ah.asarray(fill, dtype=dtype))), msg


# Testing xp.arange() requires bounding the start/stop/step arguments to only
# test argument combinations compliant with the Array API, as well as to not
# produce arrays with sizes not supproted by an array module.
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
        ),
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
            reals(max_value=-tol).filter(lambda n: n != 0),
        ),
        label="step",
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
        elements = list(
            scalar_type(n) for n in takewhile(condition, count(_start, step))
        )
    else:
        elements = []
    size = len(elements)
    assert (
        size <= hh.MAX_ARRAY_SIZE
    ), f"{size=}, should be no more than {hh.MAX_ARRAY_SIZE=}"

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
        pass  # TODO: either emulate array module behaviour or assert a rough equals


@given(hh.shapes(), hh.kwargs(dtype=st.none() | hh.shared_dtypes))
def test_empty(shape, kw):
    out = xp.empty(shape, **kw)
    if kw.get("dtype", None) is None:
        assert_default_float("empty", out.dtype)
    else:
        assert_kw_dtype("empty", kw["dtype"], out.dtype)
    assert_shape("empty", out.shape, shape, shape=shape)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_empty_like(x, kw):
    out = xp.empty_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("empty_like", (x.dtype,), out.dtype)
    else:
        assert_kw_dtype("empty_like", kw["dtype"], out.dtype)
    assert_shape("empty_like", out.shape, x.shape)


@given(
    n_rows=hh.sqrt_sizes,
    n_cols=st.none() | hh.sqrt_sizes,
    kw=hh.kwargs(
        k=st.integers(),
        dtype=xps.numeric_dtypes(),
    ),
)
def test_eye(n_rows, n_cols, kw):
    out = xp.eye(n_rows, n_cols, **kw)
    if kw.get("dtype", None) is None:
        assert_default_float("eye", out.dtype)
    else:
        assert_kw_dtype("eye", kw["dtype"], out.dtype)
    _n_cols = n_rows if n_cols is None else n_cols
    assert_shape("eye", out.shape, (n_rows, _n_cols), n_rows=n_rows, n_cols=n_cols)
    for i in range(n_rows):
        for j in range(_n_cols):
            if j - i == kw.get("k", 0):
                assert out[i, j] == 1, f"out[{i}, {j}]={out[i, j]}, should be 1 [eye()]"
            else:
                assert out[i, j] == 0, f"out[{i}, {j}]={out[i, j]}, should be 0 [eye()]"


default_unsafe_dtypes = [xp.uint64]
if dh.default_int == xp.int32:
    default_unsafe_dtypes.extend([xp.uint32, xp.int64])
if dh.default_float == xp.float32:
    default_unsafe_dtypes.append(xp.float64)
default_safe_dtypes: st.SearchStrategy = xps.scalar_dtypes().filter(
    lambda d: d not in default_unsafe_dtypes
)


@st.composite
def full_fill_values(draw) -> st.SearchStrategy[float]:
    kw = draw(
        st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw")
    )
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
            pass  # TODO
        elif isinstance(fill_value, int):
            assert_default_int("full", out.dtype)
        else:
            assert_default_float("full", out.dtype)
    else:
        assert_kw_dtype("full", kw["dtype"], out.dtype)
    assert_shape("full", out.shape, shape, shape=shape)
    assert_fill("full", fill_value, dtype, out, fill_value=fill_value)


@st.composite
def full_like_fill_values(draw):
    kw = draw(
        st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_like_kw")
    )
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
    assert_shape("full_like", out.shape, x.shape)
    assert_fill("full_like", fill_value, dtype, out, fill_value=fill_value)


finite_kw = {"allow_nan": False, "allow_infinity": False}


@st.composite
def int_stops(draw, start: int, min_gap: int, m: int, M: int):
    sign = draw(st.booleans().map(int))
    max_gap = abs(M - m)
    max_int = math.floor(math.sqrt(max_gap))
    gap = draw(st.just(0) | st.integers(1, max_int).map(lambda n: min_gap ** n))
    stop = start + sign * gap
    assume(m <= stop <= M)
    return stop


@given(
    num=hh.sizes,
    dtype=st.none() | xps.numeric_dtypes(),
    endpoint=st.booleans(),
    data=st.data(),
)
def test_linspace(num, dtype, endpoint, data):
    _dtype = dh.default_float if dtype is None else dtype

    start = data.draw(xps.from_dtype(_dtype, **finite_kw), label="start")
    if dh.is_float_dtype(_dtype):
        stop = data.draw(xps.from_dtype(_dtype, **finite_kw), label="stop")
        # avoid overflow errors
        delta = ah.asarray(stop - start, dtype=_dtype)
        assume(not ah.isnan(delta))
    else:
        if num == 0:
            stop = start
        else:
            min_gap = num
            if endpoint:
                min_gap += 1
            m, M = dh.dtype_ranges[_dtype]
            stop = data.draw(int_stops(start, min_gap, m, M), label="stop")

    out = xp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

    assert_shape("linspace", out.shape, num, start=stop, stop=stop, num=num)

    if endpoint:
        if num > 1:
            assert ah.equal(
                out[-1], ah.asarray(stop, dtype=out.dtype)
            ), f"out[-1]={out[-1]}, but should be {stop=} [linspace()]"
    else:
        # linspace(..., num, endpoint=True) should return an array equivalent to
        # the first num elements when endpoint=False
        expected = xp.linspace(start, stop, num + 1, dtype=dtype, endpoint=True)
        expected = expected[:-1]
        ah.assert_exactly_equal(out, expected)

    if num > 0:
        assert ah.equal(
            out[0], ah.asarray(start, dtype=out.dtype)
        ), f"out[0]={out[0]}, but should be {start=} [linspace()]"
        # TODO: array assertions ala test_arange


def make_one(dtype: DataType) -> Union[bool, float]:
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
    assert_shape("ones", out.shape, shape, shape=shape)
    dtype = kw.get("dtype", None) or dh.default_float
    assert_fill("ones", make_one(dtype), dtype, out)


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
    assert_shape("ones_like", out.shape, x.shape)
    dtype = kw.get("dtype", None) or x.dtype
    assert_fill("ones_like", make_one(dtype), dtype, out)


def make_zero(dtype: DataType) -> Union[bool, float]:
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
    assert_shape("zeros", out.shape, shape, shape=shape)
    dtype = kw.get("dtype", None) or dh.default_float
    assert_fill("zeros", make_zero(dtype), dtype, out)


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
    assert_shape("zeros_like", out.shape, x.shape)
    dtype = kw.get("dtype", None) or x.dtype
    assert_fill("zeros_like", make_zero(dtype), dtype, out)
