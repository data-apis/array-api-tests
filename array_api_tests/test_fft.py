import math

import pytest
from hypothesis import given

from array_api_tests.typing import DataType

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps

pytestmark = [
    pytest.mark.ci,
    pytest.mark.xp_extension("fft"),
    pytest.mark.min_version("draft"),
]


fft_shapes_strat = hh.shapes(min_dims=1).filter(lambda s: math.prod(s) > 1)


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


@given(x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat))
def test_fft(x):
    out = xp.fft.fft(x)
    assert_fft_dtype("fft", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape("fft", out_shape=out.shape, expected=x.shape)


@given(x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat))
def test_ifft(x):
    out = xp.fft.ifft(x)
    assert_fft_dtype("ifft", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape("ifft", out_shape=out.shape, expected=x.shape)


@given(x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat))
def test_fftn(x):
    out = xp.fft.fftn(x)
    assert_fft_dtype("fftn", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape("fftn", out_shape=out.shape, expected=x.shape)


@given(x=xps.arrays(dtype=hh.all_floating_dtypes(), shape=fft_shapes_strat))
def test_ifftn(x):
    out = xp.fft.ifftn(x)
    assert_fft_dtype("ifftn", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape("ifftn", out_shape=out.shape, expected=x.shape)
