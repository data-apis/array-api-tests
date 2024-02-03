import math
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from array_api_tests.typing import Array, DataType

from . import api_version
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from . import xp

pytestmark = [
    pytest.mark.ci,
    pytest.mark.xp_extension("fft"),
    pytest.mark.min_version("2022.12"),
]


# Using xps.complex_dtypes() raises an AttributeError for 2021.12 instances of
# xps, hence this hack. TODO: figure out a better way to manage this!
if api_version < "2022.12":
    xps = MagicMock(xps)

fft_shapes_strat = hh.shapes(min_dims=1).filter(lambda s: math.prod(s) > 1)


def draw_n_axis_norm_kwargs(x: Array, data: st.DataObject, *, size_gt_1=False) -> tuple:
    size = math.prod(x.shape)
    n = data.draw(
        st.none() | st.integers((size // 2), math.ceil(size * 1.5)), label="n"
    )
    axis = data.draw(st.integers(-1, x.ndim - 1), label="axis")
    if size_gt_1:
        _axis = x.ndim - 1 if axis == -1 else axis
        assume(x.shape[_axis] > 1)
    norm = data.draw(st.sampled_from(["backward", "ortho", "forward"]), label="norm")
    kwargs = data.draw(
        hh.specified_kwargs(
            ("n", n, None),
            ("axis", axis, -1),
            ("norm", norm, "backward"),
        ),
        label="kwargs",
    )
    return n, axis, norm, kwargs


def draw_s_axes_norm_kwargs(x: Array, data: st.DataObject, *, size_gt_1=False) -> tuple:
    all_axes = list(range(x.ndim))
    axes = data.draw(
        st.none() | st.lists(st.sampled_from(all_axes), min_size=1, unique=True),
        label="axes",
    )
    _axes = all_axes if axes is None else axes
    axes_sides = [x.shape[axis] for axis in _axes]
    s_strat = st.tuples(
        *[st.integers(max(side // 2, 1), math.ceil(side * 1.5)) for side in axes_sides]
    )
    if axes is None:
        s_strat = st.none() | s_strat
    s = data.draw(s_strat, label="s")
    if size_gt_1:
        _s = x.shape if s is None else s
        for i in range(x.ndim):
            if i in _axes:
                side = _s[_axes.index(i)]
            else:
                side = x.shape[i]
                assume(side > 1)
    norm = data.draw(st.sampled_from(["backward", "ortho", "forward"]), label="norm")
    kwargs = data.draw(
        hh.specified_kwargs(
            ("s", s, None),
            ("axes", axes, None),
            ("norm", norm, "backward"),
        ),
        label="kwargs",
    )
    return s, axes, norm, kwargs


def assert_fft_dtype(func_name: str, *, in_dtype: DataType, out_dtype: DataType):
    if in_dtype == xp.float32:
        expected = xp.complex64
    elif in_dtype == xp.float64:
        expected = xp.complex128
    else:
        assert dh.is_float_dtype(in_dtype)  # sanity check
        expected = in_dtype
    ph.assert_dtype(
        func_name, in_dtype=in_dtype, out_dtype=out_dtype, expected=expected
    )


def assert_n_axis_shape(
    func_name: str,
    *,
    x: Array,
    n: Optional[int],
    axis: int,
    out: Array,
    size_gt_1: bool = False,
):
    _axis = len(x.shape) - 1 if axis == -1 else axis
    if n is None:
        if size_gt_1:
            axis_side = 2 * (x.shape[_axis] - 1)
        else:
            axis_side = x.shape[_axis]
    else:
        axis_side = n
    expected = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
    ph.assert_shape(func_name, out_shape=out.shape, expected=expected)


def assert_s_axes_shape(
    func_name: str,
    *,
    x: Array,
    s: Optional[List[int]],
    axes: Optional[List[int]],
    out: Array,
    size_gt_1: bool = False,
):
    _axes = sh.normalise_axis(axes, x.ndim)
    _s = x.shape if s is None else s
    expected = []
    for i in range(x.ndim):
        if i in _axes:
            side = _s[_axes.index(i)]
        else:
            side = x.shape[i]
        expected.append(side)
    if size_gt_1:
        last_axis = _axes[-1]
        expected[last_axis] = 2 * (expected[last_axis] - 1)
        assume(expected[last_axis] > 0)  # TODO: generate valid examples
    ph.assert_shape(func_name, out_shape=out.shape, expected=tuple(expected))


@given(
    x=hh.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_fft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    out = xp.fft.fft(x, **kwargs)

    assert_fft_dtype("fft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("fft", x=x, n=n, axis=axis, out=out)


@given(
    x=hh.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_ifft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    out = xp.fft.ifft(x, **kwargs)

    assert_fft_dtype("ifft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("ifft", x=x, n=n, axis=axis, out=out)


@given(
    x=hh.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_fftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    out = xp.fft.fftn(x, **kwargs)

    assert_fft_dtype("fftn", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_s_axes_shape("fftn", x=x, s=s, axes=axes, out=out)


@given(
    x=hh.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_ifftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    out = xp.fft.ifftn(x, **kwargs)

    assert_fft_dtype("ifftn", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_s_axes_shape("ifftn", x=x, s=s, axes=axes, out=out)


@given(
    x=hh.arrays(dtype=xps.floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_rfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    out = xp.fft.rfft(x, **kwargs)

    assert_fft_dtype("rfft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("rfft", x=x, n=n, axis=axis, out=out)


@given(
    x=hh.arrays(dtype=xps.complex_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_irfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data, size_gt_1=True)

    out = xp.fft.irfft(x, **kwargs)

    assert_fft_dtype("irfft", in_dtype=x.dtype, out_dtype=out.dtype)

    _axis = x.ndim - 1 if axis == -1 else axis
    if n is None:
        axis_side = 2 * (x.shape[_axis] - 1)
    else:
        axis_side = n
    expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
    ph.assert_shape("irfft", out_shape=out.shape, expected=expected_shape)


@given(
    x=hh.arrays(dtype=xps.floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_rfftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    out = xp.fft.rfftn(x, **kwargs)

    assert_fft_dtype("rfftn", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_s_axes_shape("rfftn", x=x, s=s, axes=axes, out=out)


@given(
    x=hh.arrays(
        dtype=xps.complex_dtypes(), shape=fft_shapes_strat.filter(lambda s: s[-1] > 1)
    ),
    data=st.data(),
)
def test_irfftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data, size_gt_1=True)

    out = xp.fft.irfftn(x, **kwargs)

    assert_fft_dtype("irfftn", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_s_axes_shape("rfftn", x=x, s=s, axes=axes, out=out, size_gt_1=True)


@given(
    x=hh.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_hfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data, size_gt_1=True)

    out = xp.fft.hfft(x, **kwargs)

    assert_fft_dtype("hfft", in_dtype=x.dtype, out_dtype=out.dtype)

    _axis = x.ndim - 1 if axis == -1 else axis
    if n is None:
        axis_side = 2 * (x.shape[_axis] - 1)
    else:
        axis_side = n
    expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
    ph.assert_shape("hfft", out_shape=out.shape, expected=expected_shape)


@given(
    x=hh.arrays(dtype=xps.floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_ihfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    out = xp.fft.ihfft(x, **kwargs)

    assert_fft_dtype("ihfft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("ihfft", x=x, n=n, axis=axis, out=out, size_gt_1=True)


@given( n=st.integers(1, 100), kw=hh.kwargs(d=st.floats(0.1, 5)))
def test_fftfreq(n, kw):
    out = xp.fft.fftfreq(n, **kw)
    ph.assert_shape("fftfreq", out_shape=out.shape, expected=(n,), kw={"n": n})


@given(n=st.integers(1, 100), kw=hh.kwargs(d=st.floats(0.1, 5)))
def test_rfftfreq(n, kw):
    out = xp.fft.rfftfreq(n, **kw)
    ph.assert_shape("rfftfreq", out_shape=out.shape, expected=(n // 2 + 1,), kw={"n": n})


@pytest.mark.parametrize("func_name", ["fftshift", "ifftshift"])
@given(x=hh.arrays(xps.floating_dtypes(), fft_shapes_strat), data=st.data())
def test_shift_func(func_name, x, data):
    func = getattr(xp.fft, func_name)
    axes = data.draw(
        st.none() | st.lists(st.sampled_from(list(range(x.ndim))), min_size=1, unique=True),
        label="axes",
    )
    out = func(x, axes=axes)
    ph.assert_dtype(func_name, in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape(func_name, out_shape=out.shape, expected=x.shape)
