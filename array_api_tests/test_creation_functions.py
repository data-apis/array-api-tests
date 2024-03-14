import cmath
import math
from itertools import count
from typing import Iterator, NamedTuple, Union

from hypothesis import assume, given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import DataType, Scalar


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


# TODO: support testing complex dtypes
@given(dtype=st.none() | xps.real_dtypes(), data=st.data())
def test_arange(dtype, data):
    if dtype is None or dh.is_float_dtype(dtype):
        start = data.draw(reals(), label="start")
        stop = data.draw(reals() | st.none(), label="stop")
    else:
        start = data.draw(hh.from_dtype(dtype), label="start")
        stop = data.draw(hh.from_dtype(dtype), label="stop")
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
            step_strats.append(hh.from_dtype(dtype, max_value=step_min))
        step_max = max(math.ceil(tol), 1)
        step_strats.append(hh.from_dtype(dtype, min_value=step_max))
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
        # Ignore ridiculous distances so we don't fail like
        #
        #     >>> torch.arange(9132051521638391890, 0, -91320515216383920)
        #     RuntimeError: invalid size, possible overflow?
        #
        assume(abs(_start - _stop) < M // 2)

    r = frange(_start, _stop, step)
    size = len(r)
    assert (
        size <= hh.MAX_ARRAY_SIZE
    ), f"{size=} should be no more than {hh.MAX_ARRAY_SIZE}"  # sanity check

    args_samples = [(start, stop), (start, stop, step)]
    if stop is None:
        args_samples.insert(0, (start,))
    args = data.draw(st.sampled_from(args_samples), label="args")
    kvds = [hh.KVD("dtype", dtype, None)]
    if len(args) != 3:
        kvds.insert(0, hh.KVD("step", step, 1))
    kwargs = data.draw(hh.specified_kwargs(*kvds), label="kwargs")

    out = xp.arange(*args, **kwargs)

    if dtype is None:
        if all_int:
            ph.assert_default_int("arange", out.dtype)
        else:
            ph.assert_default_float("arange", out.dtype)
    else:
        ph.assert_kw_dtype("arange", kw_dtype=dtype, out_dtype=out.dtype)
    f_sig = ", ".join(str(n) for n in args)
    if len(kwargs) > 0:
        f_sig += f", {ph.fmt_kw(kwargs)}"
    f_func = f"[arange({f_sig})]"
    assert out.ndim == 1, f"{out.ndim=}, but should be 1 [{f_func}]"
    # We check size is roughly as expected to avoid edge cases e.g.
    #
    #     >>> xp.arange(2, step=0.333333333333333)
    #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0]
    #     >>> xp.arange(2, step=0.3333333333333333)
    #     [0.0, 0.33, 0.66, 1.0, 1.33, 1.66]
    #
    #     >>> start, stop, step = 0, 108086391056891901, 1080863910568919
    #     >>> x = xp.arange(start, stop, step, dtype=xp.uint64)
    #     >>> x.size
    #     100
    #     >>> r = range(start, stop, step)
    #     >>> len(r)
    #     101
    #
    min_size = math.floor(size * 0.9)
    max_size = max(math.ceil(size * 1.1), 1)
    out_size = math.prod(out.shape)
    assert (
        min_size <= out_size <= max_size
    ), f"prod(out.shape)={out_size}, but should be roughly {size} {f_func}"
    if dh.is_int_dtype(_dtype):
        elements = list(r)
        assume(out_size == len(elements))
        ph.assert_array_elements("arange", out=out, expected=xp.asarray(elements, dtype=_dtype))
    else:
        assume(out_size == size)
        if out_size > 0:
            assert xp.equal(
                out[0], xp.asarray(_start, dtype=out.dtype)
            ), f"out[0]={out[0]}, but should be {_start} {f_func}"


@given(shape=hh.shapes(min_side=1), data=st.data())
def test_asarray_scalars(shape, data):
    kw = data.draw(
        hh.kwargs(dtype=st.none() | xps.scalar_dtypes(), copy=st.none()), label="kw"
    )
    dtype = kw.get("dtype", None)
    if dtype is None:
        dtype_family = data.draw(
            st.sampled_from(
                [(xp.bool,), (xp.int32, xp.int64), (xp.float32, xp.float64)]
            ),
            label="expected out dtypes",
        )
        _dtype = dtype_family[0]
    else:
        _dtype = dtype
    if dh.is_float_dtype(_dtype):
        elements_strat = hh.from_dtype(_dtype) | hh.from_dtype(xp.int32)
    elif dh.is_int_dtype(_dtype):
        elements_strat = hh.from_dtype(_dtype) | st.booleans()
    else:
        elements_strat = hh.from_dtype(_dtype)
    size = math.prod(shape)
    obj_strat = st.lists(elements_strat, min_size=size, max_size=size)
    scalar_type = dh.get_scalar_type(_dtype)
    if dtype is None:
        # For asarray to infer the dtype we're testing, obj requires at least
        # one element to be the scalar equivalent of the inferred dtype, and so
        # we filter out invalid examples. Note we use type() as Python booleans
        # instance check with ints e.g. isinstance(False, int) == True.
        obj_strat = obj_strat.filter(lambda l: any(type(e) == scalar_type for e in l))
    _obj = data.draw(obj_strat, label="_obj")
    obj = sh.reshape(_obj, shape)
    note(f"{obj=}")

    out = xp.asarray(obj, **kw)

    if dtype is None:
        msg = f"out.dtype={dh.dtype_to_name[out.dtype]}, should be "
        if dtype_family == (xp.float32, xp.float64):
            msg += "default floating-point dtype (float32 or float64)"
        elif dtype_family == (xp.int32, xp.int64):
            msg += "default integer dtype (int32 or int64)"
        else:
            msg += "boolean dtype"
        msg += " [asarray()]"
        assert out.dtype in dtype_family, msg
    else:
        assert kw["dtype"] == _dtype  # sanity check
        ph.assert_kw_dtype("asarray", kw_dtype=_dtype, out_dtype=out.dtype)
    ph.assert_shape("asarray", out_shape=out.shape, expected=shape)
    for idx, v_expect in zip(sh.ndindex(out.shape), _obj):
        v = scalar_type(out[idx])
        ph.assert_scalar_equals("asarray", type_=scalar_type, idx=idx, out=v, expected=v_expect, kw=kw)


def scalar_eq(s1: Scalar, s2: Scalar) -> bool:
    if cmath.isnan(s1):
        return cmath.isnan(s2)
    else:
        return s1 == s2


@given(
    shape=hh.shapes(),
    dtypes=hh.oneway_promotable_dtypes(dh.all_dtypes),
    data=st.data(),
)
def test_asarray_arrays(shape, dtypes, data):
    x = data.draw(hh.arrays(dtype=dtypes.input_dtype, shape=shape), label="x")
    dtypes_strat = st.just(dtypes.input_dtype)
    if dtypes.input_dtype == dtypes.result_dtype:
        dtypes_strat |= st.none()
    kw = data.draw(
        hh.kwargs(dtype=dtypes_strat, copy=st.none() | st.booleans()),
        label="kw",
    )

    out = xp.asarray(x, **kw)

    dtype = kw.get("dtype", None)
    if dtype is None:
        ph.assert_dtype("asarray", in_dtype=x.dtype, out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("asarray", kw_dtype=dtype, out_dtype=out.dtype)
    ph.assert_shape("asarray", out_shape=out.shape, expected=x.shape)
    ph.assert_array_elements("asarray", out=out, expected=x, kw=kw)
    copy = kw.get("copy", None)
    if copy is not None:
        stype = dh.get_scalar_type(x.dtype)
        idx = data.draw(xps.indices(x.shape, max_dims=0), label="mutating idx")
        old_value = stype(x[idx])
        scalar_strat = hh.from_dtype(dtypes.input_dtype).filter(
            lambda n: not scalar_eq(n, old_value)
        )
        value = data.draw(
            scalar_strat | scalar_strat.map(lambda n: xp.asarray(n, dtype=x.dtype)),
            label="mutating value",
        )
        x[idx] = value
        note(f"mutated {x=}")
        # sanity check
        ph.assert_scalar_equals(
            "__setitem__", type_=stype, idx=idx, out=stype(x[idx]), expected=value, repr_name="x"
        )
        new_out_value = stype(out[idx])
        f_out = f"{sh.fmt_idx('out', idx)}={new_out_value}"
        if copy:
            assert scalar_eq(
                new_out_value, old_value
            ), f"{f_out}, but should be {old_value} even after x was mutated"
        else:
            assert scalar_eq(
                new_out_value, value
            ), f"{f_out}, but should be {value} after x was mutated"


@given(hh.shapes(), hh.kwargs(dtype=st.none() | xps.scalar_dtypes()))
def test_empty(shape, kw):
    out = xp.empty(shape, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_default_float("empty", out.dtype)
    else:
        ph.assert_kw_dtype("empty", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("empty", out_shape=out.shape, expected=shape, kw=dict(shape=shape))


@given(
    x=hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_empty_like(x, kw):
    out = xp.empty_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("empty_like", in_dtype=x.dtype, out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("empty_like", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("empty_like", out_shape=out.shape, expected=x.shape)


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
        ph.assert_default_float("eye", out.dtype)
    else:
        ph.assert_kw_dtype("eye", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    _n_cols = n_rows if n_cols is None else n_cols
    ph.assert_shape("eye", out_shape=out.shape, expected=(n_rows, _n_cols), kw=dict(n_rows=n_rows, n_cols=n_cols))
    k = kw.get("k", 0)
    expected = xp.asarray(
        [[1 if j - i == k else 0 for j in range(_n_cols)] for i in range(n_rows)],
        dtype=out.dtype  # Note: dtype already checked above.
    )
    if 0 in expected.shape:
        expected = xp.reshape(expected, (n_rows, _n_cols))
    ph.assert_array_elements("eye", out=out, expected=expected, kw=kw)


default_unsafe_dtypes = [xp.uint64]
if dh.default_int == xp.int32:
    default_unsafe_dtypes.extend([xp.uint32, xp.int64])
if dh.default_float == xp.float32:
    default_unsafe_dtypes.append(xp.float64)
if dh.default_complex == xp.complex64:
    default_unsafe_dtypes.append(xp.complex64)
default_safe_dtypes: st.SearchStrategy = xps.scalar_dtypes().filter(
    lambda d: d not in default_unsafe_dtypes
)


@st.composite
def full_fill_values(draw) -> Union[bool, int, float, complex]:
    kw = draw(
        st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw")
    )
    dtype = kw.get("dtype", None) or draw(default_safe_dtypes)
    return draw(hh.from_dtype(dtype))


@given(
    shape=hh.shapes(),
    fill_value=full_fill_values(),
    kw=st.shared(hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), key="full_kw"),
)
def test_full(shape, fill_value, kw):
    with hh.reject_overflow():
        out = xp.full(shape, fill_value, **kw)
    if kw.get("dtype", None):
        dtype = kw["dtype"]
    elif isinstance(fill_value, bool):
        dtype = xp.bool
    elif isinstance(fill_value, int):
        dtype = dh.default_int
    elif isinstance(fill_value, float):
        dtype = dh.default_float
    else:
        assert isinstance(fill_value, complex)  # sanity check
        dtype = dh.default_complex
        # Ignore large components so we don't fail like
        #
        #     >>> torch.fill(complex(0.0, 3.402823466385289e+38))
        #     RuntimeError: value cannot be converted to complex<float> without overflow
        #
        M = dh.dtype_ranges[dh.dtype_components[dtype]].max
        assume(all(abs(c) < math.sqrt(M) for c in [fill_value.real, fill_value.imag]))
    if kw.get("dtype", None) is None:
        if isinstance(fill_value, bool):
            assert out.dtype == xp.bool, f"{out.dtype=}, but should be bool [full()]"
        elif isinstance(fill_value, int):
            ph.assert_default_int("full", out.dtype)
        elif isinstance(fill_value, float):
            ph.assert_default_float("full", out.dtype)
        else:
            assert isinstance(fill_value, complex)  # sanity check
            ph.assert_default_complex("full", out.dtype)
    else:
        ph.assert_kw_dtype("full", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("full", out_shape=out.shape, expected=shape, kw=dict(shape=shape))
    ph.assert_fill("full", fill_value=fill_value, dtype=dtype, out=out, kw=dict(fill_value=fill_value))


@given(kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()), data=st.data())
def test_full_like(kw, data):
    dtype = kw.get("dtype", None) or data.draw(xps.scalar_dtypes(), label="dtype")
    x = data.draw(hh.arrays(dtype=dtype, shape=hh.shapes()), label="x")
    fill_value = data.draw(hh.from_dtype(dtype), label="fill_value")
    out = xp.full_like(x, fill_value, **kw)
    dtype = kw.get("dtype", None) or x.dtype
    if kw.get("dtype", None) is None:
        ph.assert_dtype("full_like", in_dtype=x.dtype, out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("full_like", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("full_like", out_shape=out.shape, expected=x.shape)
    ph.assert_fill("full_like", fill_value=fill_value, dtype=dtype, out=out, kw=dict(fill_value=fill_value))


finite_kw = {"allow_nan": False, "allow_infinity": False}


@given(
    num=hh.sizes,
    dtype=st.none() | xps.floating_dtypes(),
    endpoint=st.booleans(),
    data=st.data(),
)
def test_linspace(num, dtype, endpoint, data):
    _dtype = dh.default_float if dtype is None else dtype

    start = data.draw(hh.from_dtype(_dtype, **finite_kw), label="start")
    stop = data.draw(hh.from_dtype(_dtype, **finite_kw), label="stop")
    # avoid overflow errors
    assume(not xp.isnan(xp.asarray(stop - start, dtype=_dtype)))
    assume(not xp.isnan(xp.asarray(start - stop, dtype=_dtype)))
    # avoid generating very large distances
    # https://github.com/data-apis/array-api-tests/issues/125
    assume(abs(stop - start) < math.sqrt(dh.dtype_ranges[_dtype].max))

    kw = data.draw(
        hh.specified_kwargs(
            hh.KVD("dtype", dtype, None),
            hh.KVD("endpoint", endpoint, True),
        ),
        label="kw",
    )
    out = xp.linspace(start, stop, num, **kw)

    if dtype is None:
        ph.assert_default_float("linspace", out.dtype)
    else:
        ph.assert_kw_dtype("linspace", kw_dtype=dtype, out_dtype=out.dtype)
    ph.assert_shape("linspace", out_shape=out.shape, expected=num, kw=dict(start=start, stop=stop, num=num))
    f_func = f"[linspace({start}, {stop}, {num})]"
    if num > 0:
        assert xp.equal(
            out[0], xp.asarray(start, dtype=out.dtype)
        ), f"out[0]={out[0]}, but should be {start} {f_func}"
    if endpoint:
        if num > 1:
            assert xp.equal(
                out[-1], xp.asarray(stop, dtype=out.dtype)
            ), f"out[-1]={out[-1]}, but should be {stop} {f_func}"
    else:
        # linspace(..., num, endpoint=True) should return an array equivalent to
        # the first num elements when endpoint=False
        expected = xp.linspace(start, stop, num + 1, dtype=dtype, endpoint=True)
        expected = expected[:-1]
        ph.assert_array_elements("linspace", out=out, expected=expected)


@given(dtype=xps.numeric_dtypes(), data=st.data())
def test_meshgrid(dtype, data):
    # The number and size of generated arrays is arbitrarily limited to prevent
    # meshgrid() running out of memory.
    shapes = data.draw(
        st.integers(1, 5).flatmap(
            lambda n: hh.mutually_broadcastable_shapes(
                n, min_dims=1, max_dims=1, max_side=5
            )
        ),
        label="shapes",
    )
    arrays = []
    for i, shape in enumerate(shapes, 1):
        x = data.draw(hh.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    # sanity check
    assert math.prod(math.prod(x.shape) for x in arrays) <= hh.MAX_ARRAY_SIZE
    out = xp.meshgrid(*arrays)
    for i, x in enumerate(out):
        ph.assert_dtype("meshgrid", in_dtype=dtype, out_dtype=x.dtype, repr_name=f"out[{i}].dtype")


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
        ph.assert_default_float("ones", out.dtype)
    else:
        ph.assert_kw_dtype("ones", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("ones", out_shape=out.shape, expected=shape,
                    kw={'shape': shape, **kw})
    dtype = kw.get("dtype", None) or dh.default_float
    ph.assert_fill("ones", fill_value=make_one(dtype), dtype=dtype, out=out, kw=kw)


@given(
    x=hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_ones_like(x, kw):
    out = xp.ones_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("ones_like", in_dtype=x.dtype, out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("ones_like", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("ones_like", out_shape=out.shape, expected=x.shape, kw=kw)
    dtype = kw.get("dtype", None) or x.dtype
    ph.assert_fill("ones_like", fill_value=make_one(dtype), dtype=dtype,
                   out=out, kw=kw)


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
        ph.assert_default_float("zeros", out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("zeros", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("zeros", out_shape=out.shape, expected=shape, kw={'shape': shape, **kw})
    dtype = kw.get("dtype", None) or dh.default_float
    ph.assert_fill("zeros", fill_value=make_zero(dtype), dtype=dtype, out=out,
                   kw=kw)


@given(
    x=hh.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    kw=hh.kwargs(dtype=st.none() | xps.scalar_dtypes()),
)
def test_zeros_like(x, kw):
    out = xp.zeros_like(x, **kw)
    if kw.get("dtype", None) is None:
        ph.assert_dtype("zeros_like", in_dtype=x.dtype, out_dtype=out.dtype)
    else:
        ph.assert_kw_dtype("zeros_like", kw_dtype=kw["dtype"], out_dtype=out.dtype)
    ph.assert_shape("zeros_like", out_shape=out.shape, expected=x.shape,
                    kw=kw)
    dtype = kw.get("dtype", None) or x.dtype
    ph.assert_fill("zeros_like", fill_value=make_zero(dtype), dtype=dtype,
                   out=out, kw=kw)
