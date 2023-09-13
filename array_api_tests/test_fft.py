import math
from typing import Optional

import pytest
from hypothesis import given
from hypothesis import strategies as st

from array_api_tests.typing import Array, DataType

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from ._array_module import mod as xp

pytestmark = [
    pytest.mark.ci,
    pytest.mark.xp_extension("fft"),
    pytest.mark.min_version("draft"),
]


fft_shapes_strat = hh.shapes(min_dims=1).filter(lambda s: math.prod(s) > 1)


def n_axis_norm_kwargs(x: Array, data: st.DataObject) -> tuple:
    size = math.prod(x.shape)
    n = data.draw(st.none() | st.integers(size // 2, size * 2), label="n")
    axis = data.draw(st.integers(-1, x.ndim - 1), label="axis")
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
    func_name: str, *, x: Array, n: Optional[int], axis: int, out: Array
):
    if n is None:
        expected_shape = x.shape
    else:
        _axis = len(x.shape) - 1 if axis == -1 else axis
        expected_shape = x.shape[:_axis] + (n,) + x.shape[_axis + 1 :]
    ph.assert_shape(func_name, out_shape=out.shape, expected=expected_shape)


@given(
    x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_fft(x, data):
    n, axis, norm, kwargs = n_axis_norm_kwargs(x, data)

    out = xp.fft.fft(x, **kwargs)

    assert_fft_dtype("fft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("fft", x=x, n=n, axis=axis, out=out)


@given(
    x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_ifft(x, data):
    n, axis, norm, kwargs = n_axis_norm_kwargs(x, data)

    out = xp.fft.ifft(x, **kwargs)

    assert_fft_dtype("ifft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("ifft", x=x, n=n, axis=axis, out=out)


# TODO:
# test_fftn
# test_ifftn


@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_rfft(x, data):
    n, axis, norm, kwargs = n_axis_norm_kwargs(x, data)

    out = xp.fft.rfft(x, **kwargs)

    assert_fft_dtype("rfft", in_dtype=x.dtype, out_dtype=out.dtype)
    assert_n_axis_shape("rfft", x=x, n=n, axis=axis, out=out)


@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=fft_shapes_strat),
    data=st.data(),
)
def test_irfft(x, data):
    n, axis, norm, kwargs = n_axis_norm_kwargs(x, data)

    out = xp.fft.irfft(x, **kwargs)

    assert_fft_dtype("irfft", in_dtype=x.dtype, out_dtype=out.dtype)
    # TODO: assert shape


# TODO:
# test_rfftn
# test_irfftn
# test_hfft
# test_ihfft
# fftfreq
# rfftfreq
# fftshift
# ifftshift
