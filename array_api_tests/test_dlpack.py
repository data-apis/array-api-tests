from enum import Enum

from hypothesis import given, strategies as st
from . import _array_module as xp
from . import pytest_helpers as ph
from . import hypothesis_helpers as hh

# dlpack Enum values,
# https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html

class DLPackDeviceEnum(Enum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10
    CUDA_MANAGED = 13
    ONE_API = 14


def _compatible_devices(devices):
    """Given a list of devices, filter out dlpack-incompatible ones."""
    # XXX: there seems to be no better way than try-catch for __dlpack_device__()

    # XXX: this process actually fails with CuPy because CuPy ignores the device= argument
    # cf https://github.com/data-apis/array-api-compat/issues/337 and
    # https://github.com/cupy/cupy/issues/9848
    # Luckily, CuPy only supports CUDA devices, and they are all compatible.
    compatible_ = []
    for device in devices:
        x = xp.empty(2, device=device)
        try:
            x.__dlpack_device__()
        except:
            # case in point: torch.device(type="meta") raises
            # ValueError: Unknown device type meta for Dlpack
            pass
        else:
            # no exception => device is compatible
            compatible_.append(device)
    return compatible_


@given(dtype=hh.all_dtypes, data=st.data())
def test_dlpack_device(dtype, data):
    """Test the array object __dlpack_device__ method."""
    # TODO: 1. generate inputs on non-default devices
    x = xp.empty(3, dtype=dtype)
    device_type, device_id  = x.__dlpack_device__()

    assert DLPackDeviceEnum(int(device_type))
    assert isinstance(device_id, int)


@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1, max_side=2)),
    copy_kw=hh.kwargs(
        copy=st.booleans() | st.none()
    ),
    max_version_kw=hh.kwargs(
        max_version=st.tuples(
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=0)
        )
    ),
    dl_device_kw=hh.kwargs(
        dl_device=st.tuples(  # XXX: the 2023.12 standard only mandates ... kDLCPU ?
            st.just(DLPackDeviceEnum.CPU.value),
            st.just(0)
        )
    ),
    data=st.data()
)
def test_dunder_dlpack(x, copy_kw, max_version_kw, dl_device_kw, data):
    repro_snippet = ph.format_snippet(
        f"x.__dlpack__ with {copy_kw = }, {max_version_kw = } and {dl_device_kw = }"
    )

    try:
        x.__dlpack__(**copy_kw, **max_version_kw, **dl_device_kw)
        # apparently, we cannot do anything with the DLPack capsule from python
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1, max_side=2)),
    copy_kw=hh.kwargs(copy=st.booleans()),
    data=st.data()
)
def test_from_dlpack(x, copy_kw, data):
    # TODO: 1. test copy;  2. generate inputs on non-default devices;
    #       3. test for copy=False cross-device transfers
    #       4. test 0D arrays / numpy scalars (the latter do not support dlpack ATM)

    copy = copy_kw["copy"] if copy_kw else None
    if copy is False:
        # XXX there is no way to tell if a no-copy cross-device transfer is meant to succeed
        devices = [x.device]
    else:
        devices = xp.__array_namespace_info__().devices()
        devices = _compatible_devices(devices)

    tgt_device_kw = data.draw(
        hh.kwargs(device=st.sampled_from(devices) | st.none())
    )
    tgt_device = tgt_device_kw['device'] if tgt_device_kw else None

    repro_snippet = ph.format_snippet(
        f"y = from_dlpack({x!r}, **tgt_device_kw, **copy_kw) with {tgt_device_kw=} and {copy_kw=}"
    )
    try:
        y = xp.from_dlpack(x, **tgt_device_kw, **copy_kw)

        if tgt_device is None:
            assert y.device == x.device
            assert xp.all(y == x)
        else:
            assert y.device == tgt_device

    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise
