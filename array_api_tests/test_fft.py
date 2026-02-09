import math
from typing import List, Optional

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from array_api_tests.typing import Array

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xp

pytestmark = [
    pytest.mark.xp_extension("fft"),
    pytest.mark.min_version("2022.12"),
]

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

    # Using `axes is None and s is not None` is disallowed by the spec
    assume(axes is not None or s is None)

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


def assert_n_axis_shape(
    func_name: str,
    *,
    x: Array,
    n: Optional[int],
    axis: int,
    out: Array,
):
    _axis = len(x.shape) - 1 if axis == -1 else axis
    if n is None:
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
):
    _axes = sh.normalize_axis(axes, x.ndim)
    _s = x.shape if s is None else s
    expected = []
    for i in range(x.ndim):
        if i in _axes:
            side = _s[_axes.index(i)]
        else:
            side = x.shape[i]
        expected.append(side)
    ph.assert_shape(func_name, out_shape=out.shape, expected=tuple(expected))


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_fft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.fft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.fft(x, **kwargs)

        ph.assert_dtype("fft", in_dtype=x.dtype, out_dtype=out.dtype)
        assert_n_axis_shape("fft", x=x, n=n, axis=axis, out=out)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_ifft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.ifft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.ifft(x, **kwargs)

        ph.assert_dtype("ifft", in_dtype=x.dtype, out_dtype=out.dtype)
        assert_n_axis_shape("ifft", x=x, n=n, axis=axis, out=out)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_fftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.fftn({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.fftn(x, **kwargs)

        ph.assert_dtype("fftn", in_dtype=x.dtype, out_dtype=out.dtype)
        assert_s_axes_shape("fftn", x=x, s=s, axes=axes, out=out)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_ifftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.ifftn({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.ifftn(x, **kwargs)

        ph.assert_dtype("ifftn", in_dtype=x.dtype, out_dtype=out.dtype)
        assert_s_axes_shape("ifftn", x=x, s=s, axes=axes, out=out)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.real_floating_dtypes, shape=fft_shapes_strat), data=st.data())
def test_rfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.rfft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.rfft(x, **kwargs)

        ph.assert_float_to_complex_dtype("rfft", in_dtype=x.dtype, out_dtype=out.dtype)

        _axis = x.ndim - 1 if axis == -1 else axis
        if n is None:
            axis_side = x.shape[_axis] // 2 + 1
        else:
            axis_side = n // 2 + 1
        expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
        ph.assert_shape("rfft", out_shape=out.shape, expected=expected_shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_irfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data, size_gt_1=True)

    repro_snippet = ph.format_snippet(f"xp.fft.irfft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.irfft(x, **kwargs)

        ph.assert_dtype(
            "irfft",
            in_dtype=x.dtype,
            out_dtype=out.dtype,
            expected=dh.dtype_components[x.dtype],
        )

        _axis = x.ndim - 1 if axis == -1 else axis
        if n is None:
            axis_side = 2 * (x.shape[_axis] - 1)
        else:
            axis_side = n
        expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
        ph.assert_shape("irfft", out_shape=out.shape, expected=expected_shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.real_floating_dtypes, shape=fft_shapes_strat), data=st.data())
def test_rfftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.rfftn({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.rfftn(x, **kwargs)

        ph.assert_float_to_complex_dtype("rfftn", in_dtype=x.dtype, out_dtype=out.dtype)

        _axes = sh.normalize_axis(axes, x.ndim)
        _s = x.shape if s is None else s
        expected = []
        for i in range(x.ndim):
            if i in _axes:
                side = _s[_axes.index(i)]
            else:
                side = x.shape[i]
            expected.append(side)
        expected[_axes[-1]] = _s[-1] // 2 + 1
        ph.assert_shape("rfftn", out_shape=out.shape, expected=tuple(expected))
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    x=hh.arrays(
        dtype=hh.complex_dtypes, shape=fft_shapes_strat.filter(lambda s: s[-1] > 1)
    ),
    data=st.data(),
)
def test_irfftn(x, data):
    s, axes, norm, kwargs = draw_s_axes_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.irfftn({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.irfftn(x, **kwargs)

        ph.assert_dtype(
            "irfftn",
            in_dtype=x.dtype,
            out_dtype=out.dtype,
            expected=dh.dtype_components[x.dtype],
        )

        _axes = sh.normalize_axis(axes, x.ndim)
        _s = x.shape if s is None else s
        expected = []
        for i in range(x.ndim):
            if i in _axes:
                side = _s[_axes.index(i)]
            else:
                side = x.shape[i]
            expected.append(side)
        expected[_axes[-1]] = 2*(_s[-1] - 1) if s is None else _s[-1]
        ph.assert_shape("irfftn", out_shape=out.shape, expected=tuple(expected))
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.complex_dtypes, shape=fft_shapes_strat), data=st.data())
def test_hfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data, size_gt_1=True)

    repro_snippet = ph.format_snippet(f"xp.fft.hfft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.hfft(x, **kwargs)

        ph.assert_dtype(
            "hfft",
            in_dtype=x.dtype,
            out_dtype=out.dtype,
            expected=dh.dtype_components[x.dtype],
        )

        _axis = x.ndim - 1 if axis == -1 else axis
        if n is None:
            axis_side = 2 * (x.shape[_axis] - 1)
        else:
            axis_side = n
        expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
        ph.assert_shape("hfft", out_shape=out.shape, expected=expected_shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(x=hh.arrays(dtype=hh.real_floating_dtypes, shape=fft_shapes_strat), data=st.data())
def test_ihfft(x, data):
    n, axis, norm, kwargs = draw_n_axis_norm_kwargs(x, data)

    repro_snippet = ph.format_snippet(f"xp.fft.ihfft({x!r}, **kwargs) with {kwargs = }")
    try:
        out = xp.fft.ihfft(x, **kwargs)

        ph.assert_float_to_complex_dtype("ihfft", in_dtype=x.dtype, out_dtype=out.dtype)

        _axis = x.ndim - 1 if axis == -1 else axis
        if n is None:
            axis_side = x.shape[_axis] // 2 + 1
        else:
            axis_side = n // 2 + 1
        expected_shape = x.shape[:_axis] + (axis_side,) + x.shape[_axis + 1 :]
        ph.assert_shape("ihfft", out_shape=out.shape, expected=expected_shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    n=st.integers(1, 100),
    kw=hh.kwargs(d=st.floats(0.1, 5), dtype=hh.real_floating_dtypes),
)
def test_fftfreq(n, kw):
    repro_snippet = ph.format_snippet(f"xp.fft.fftfreq({n!r}, **kw) with {kw = }")
    try:
        out = xp.fft.fftfreq(n, **kw)
        ph.assert_shape("fftfreq", out_shape=out.shape, expected=(n,), kw={"n": n})

        dt = kw.get("dtype", None)
        if dt is None:
            dt = xp.__array_namespace_info__().default_dtypes()["real floating"]
        assert out.dtype == dt
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    n=st.integers(1, 100),
    kw=hh.kwargs(d=st.floats(0.1, 5), dtype=hh.real_floating_dtypes)
)
def test_rfftfreq(n, kw):
    repro_snippet = ph.format_snippet(f"xp.fft.rfftfreq({n!r}, **kw) with {kw = }")
    try:
        out = xp.fft.rfftfreq(n, **kw)
        ph.assert_shape(
            "rfftfreq", out_shape=out.shape, expected=(n // 2 + 1,), kw={"n": n}
        )

        dt = kw.get("dtype", None)
        if dt is None:
            dt = xp.__array_namespace_info__().default_dtypes()["real floating"]
        assert out.dtype == dt
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.parametrize("func_name", ["fftshift", "ifftshift"])
@given(x=hh.arrays(hh.floating_dtypes, fft_shapes_strat), data=st.data())
def test_shift_func(func_name, x, data):
    func = getattr(xp.fft, func_name)
    axes = data.draw(
        st.none()
        | st.lists(st.sampled_from(list(range(x.ndim))), min_size=1, unique=True),
        label="axes",
    )

    repro_snippet = ph.format_snippet(f"xp.func_name({x!r}, {axes = })")
    try:
        out = func(x, axes=axes)
        ph.assert_dtype(func_name, in_dtype=x.dtype, out_dtype=out.dtype)
        ph.assert_shape(func_name, out_shape=out.shape, expected=x.shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise
