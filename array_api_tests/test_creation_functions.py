import math
from typing import Union, Any, Tuple, NamedTuple, Iterator
from itertools import count

from hypothesis import assume, given, strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import hypothesis_helpers as hh
from . import dtype_helpers as dh
from . import pytest_helpers as ph
from . import xps
from .typing import Shape, DataType, Array, Scalar


@st.composite
def specified_kwargs(draw, *keys_values_defaults: Tuple[str, Any, Any]):
    """Generates valid kwargs given expected defaults.

    When we can't realistically use hh.kwargs() and thus test whether xp infact
    defaults correctly, this strategy lets us remove generated arguments if they
    are of the default value anyway.
    """
    kw = {}
    for key, value, default in keys_values_defaults:
        if value is not default or draw(st.booleans()):
            kw[key] = value
    return kw


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


def assert_shape(
    func_name: str, out_shape: Union[int, Shape], expected: Union[int, Shape], /, **kw
):
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    if isinstance(expected, int):
        expected = (expected,)
    f_kw = ", ".join(f"{k}={v}" for k, v in kw.items())
    msg = f"out.shape={out_shape}, but should be {expected} [{func_name}({f_kw})]"
    assert out_shape == expected, msg


def assert_fill(
    func_name: str, fill_value: Scalar, dtype: DataType, out: Array, /, **kw
):
    f_kw = ", ".join(f"{k}={v}" for k, v in kw.items())
    msg = f"out not filled with {fill_value} [{func_name}({f_kw})]\n" f"{out=}"
    if math.isnan(fill_value):
        assert ah.all(ah.isnan(out)), msg
    else:
        assert ah.all(ah.equal(out, ah.asarray(fill_value, dtype=dtype))), msg


class frange(NamedTuple):
    start: float
    stop: float
    step: float

    def __iter__(self) -> Iterator[float]:
        pos_range = self.stop > self.start
        pos_step = self.step > 0
        if pos_step != pos_range:
            return
        if pos_range:
            for n in count(self.start, self.step):
                if n >= self.stop:
                    break
                yield n
        else:
            for n in count(self.start, self.step):
                if n <= self.stop:
                    break
                yield n

    def __len__(self) -> int:
        return max(math.ceil((self.stop - self.start) / self.step), 0)


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


@given(dtype=st.none() | hh.numeric_dtypes, data=st.data())
def test_arange(dtype, data):
    if dtype is None or dh.is_float_dtype(dtype):
        start = data.draw(reals(), label="start")
        stop = data.draw(reals() | st.none(), label="stop")
    else:
        start = data.draw(xps.from_dtype(dtype), label="start")
        stop = data.draw(xps.from_dtype(dtype), label="stop")
    if stop is None:
        _start = 0
        _stop = start
    else:
        _start = start
        _stop = stop

    # tol is the minimum tolerance for step values, used to avoid scenarios
    # where xp.arange() produces arrays that would be over MAX_ARRAY_SIZE.
    tol = max(abs(_stop - _start) / (math.sqrt(hh.MAX_ARRAY_SIZE)), 0.01)
    assert tol != 0, "tol must not equal 0"  # sanity check
    assume(-tol > int_min)
    assume(tol < int_max)
    if dtype is None or dh.is_float_dtype(dtype):
        step = data.draw(reals(min_value=tol) | reals(max_value=-tol), label="step")
    else:
        step_strats = []
        if dtype in dh.int_dtypes:
            step_min = min(math.floor(-tol), -1)
            step_strats.append(xps.from_dtype(dtype, max_value=step_min))
        step_max = max(math.ceil(tol), 1)
        step_strats.append(xps.from_dtype(dtype, min_value=step_max))
        step = data.draw(st.one_of(step_strats), label="step")
    assert step != 0, "step must not equal 0"  # sanity check

    all_int = all(arg is None or isinstance(arg, int) for arg in [start, stop, step])

    if dtype is None:
        if all_int:
            _dtype = dh.default_int
        else:
            _dtype = dh.default_float
    else:
        _dtype = dtype

    # sanity checks
    if dh.is_int_dtype(_dtype):
        m, M = dh.dtype_ranges[_dtype]
        assert m <= _start <= M
        assert m <= _stop <= M
        assert m <= step <= M

    r = frange(_start, _stop, step)
    size = len(r)
    assert (
        size <= hh.MAX_ARRAY_SIZE
    ), f"{size=} should be no more than {hh.MAX_ARRAY_SIZE}"  # sanity check

    kw = data.draw(
        specified_kwargs(
            ("stop", stop, None),
            ("step", step, None),
            ("dtype", dtype, None),
        ),
        label="kw",
    )
    out = xp.arange(start, **kw)

    if dtype is None:
        if all_int:
            assert_default_int("arange", out.dtype)
        else:
            assert_default_float("arange", out.dtype)
    else:
        assert out.dtype == dtype
    assert out.ndim == 1, f"{out.ndim=}, but should be 1 [linspace()]"
    f_func = f"[linspace({start=}, {stop=}, {step=})]"
    if dh.is_int_dtype(_dtype):
        assert out.size == size, f"{out.size=}, but should be {size} {f_func}"
    else:
        # We check size is roughly as expected to avoid edge cases e.g.
        #
        #     >>> xp.arange(2, step=0.333333333333333)
        #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0]
        #     >>> xp.arange(2, step=0.3333333333333333)
        #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66]
        #
        min_size = math.floor(size * 0.9)
        max_size = math.ceil(size * 1.1)
        assert (
            min_size <= out.size <= max_size
        ), f"{out.size=}, but should be roughly {size} {f_func}"
    assume(out.size == size)
    if dh.is_int_dtype(_dtype):
        ah.assert_exactly_equal(out, ah.asarray(list(r), dtype=_dtype))
    else:
        if out.size > 0:
            assert ah.equal(
                out[0], ah.asarray(_start, dtype=out.dtype)
            ), f"out[0]={out[0]}, but should be {_start} [linspace({start=}, {stop=})]"


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
    f_func = f"[eye({n_rows=}, {n_cols=})]"
    for i in range(n_rows):
        for j in range(_n_cols):
            f_indexed_out = f"out[{i}, {j}]={out[i, j]}"
            if j - i == kw.get("k", 0):
                assert out[i, j] == 1, f"{f_indexed_out}, should be 1 {f_func}"
            else:
                assert out[i, j] == 0, f"{f_indexed_out}, should be 0 {f_func}"


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


def int_stops(
    start: int, num, dtype: DataType, endpoint: bool
) -> st.SearchStrategy[int]:
    min_gap = num
    if endpoint:
        min_gap += 1
    m, M = dh.dtype_ranges[dtype]
    max_pos_gap = M - start
    max_neg_gap = start - m
    max_pos_mul = max_pos_gap // min_gap
    max_neg_mul = max_neg_gap // min_gap
    return st.one_of(
        st.integers(0, max_pos_mul).map(lambda n: start + min_gap * n),
        st.integers(0, max_neg_mul).map(lambda n: start - min_gap * n),
    )


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
        assume(not ah.isnan(ah.asarray(stop - start, dtype=_dtype)))
        assume(not ah.isnan(ah.asarray(start - stop, dtype=_dtype)))
    else:
        if num == 0:
            stop = start
        else:
            stop = data.draw(int_stops(start, num, _dtype, endpoint), label="stop")

    kw = data.draw(
        specified_kwargs(
            ("dtype", dtype, None),
            ("endpoint", endpoint, True),
        ),
        label="kw",
    )
    out = xp.linspace(start, stop, num, **kw)

    assert_shape("linspace", out.shape, num, start=stop, stop=stop, num=num)
    f_func = f"[linspace({start=}, {stop=}, {num=})]"
    if num > 0:
        assert ah.equal(
            out[0], ah.asarray(start, dtype=out.dtype)
        ), f"out[0]={out[0]}, but should be {start} {f_func}"
    if endpoint:
        if num > 1:
            assert ah.equal(
                out[-1], ah.asarray(stop, dtype=out.dtype)
            ), f"out[-1]={out[-1]}, but should be {stop} {f_func}"
    else:
        # linspace(..., num, endpoint=True) should return an array equivalent to
        # the first num elements when endpoint=False
        expected = xp.linspace(start, stop, num + 1, dtype=dtype, endpoint=True)
        expected = expected[:-1]
        ah.assert_exactly_equal(out, expected)


def make_one(dtype: DataType) -> Scalar:
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


def make_zero(dtype: DataType) -> Scalar:
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
