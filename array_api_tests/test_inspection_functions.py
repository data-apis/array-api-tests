import pytest
from hypothesis import given, strategies as st
from array_api_tests.dtype_helpers import available_kinds, dtype_names

from . import xp

pytestmark = pytest.mark.min_version("2023.12")


class TestInspection:
    def test_capabilities(self):
        out = xp.__array_namespace_info__()

        capabilities = out.capabilities()
        assert isinstance(capabilities, dict)

        expected_attr = {"boolean indexing": bool, "data-dependent shapes": bool}
        if xp.__array_api_version__ >= "2024.12":
            expected_attr.update(**{"max dimensions": int})

        for attr, typ in expected_attr.items():
            assert attr in capabilities, f'capabilites is missing "{attr}".'
            assert isinstance(capabilities[attr], typ)

        assert capabilities.get("max dimensions", 100500) > 0

    def test_devices(self):
        out = xp.__array_namespace_info__()

        assert hasattr(out, "devices")
        assert hasattr(out, "default_device")

        assert isinstance(out.devices(), list)
        if out.default_device() is not None:
            # Per https://github.com/data-apis/array-api/issues/923
            # default_device() can return None. Otherwise, it must be a valid device.
            assert out.default_device() in out.devices()

    def test_default_dtypes(self):
        out = xp.__array_namespace_info__()

        for device in xp.__array_namespace_info__().devices():
            default_dtypes = out.default_dtypes(device=device)
            assert isinstance(default_dtypes, dict)
            expected_subset = (
                  {"real floating", "complex floating", "integral"}
                & available_kinds()
                | {"indexing"}
            )
            assert expected_subset.issubset(set(default_dtypes.keys()))


atomic_kinds = [
    "bool",
    "signed integer",
    "unsigned integer",
    "real floating",
    "complex floating",
]


@given(
    kind=st.one_of(
        st.none(),
        st.sampled_from(atomic_kinds + ["integral", "numeric"]),
        st.lists(st.sampled_from(atomic_kinds), unique=True, min_size=1).map(tuple),
    ),
    device=st.one_of(
        st.none(),
        st.sampled_from(xp.__array_namespace_info__().devices())
    )
)
def test_array_namespace_info_dtypes(kind, device):
    out = xp.__array_namespace_info__().dtypes(kind=kind, device=device)
    assert isinstance(out, dict)

    for name, dtyp in out.items():
        assert name in dtype_names
        xp.empty(1, dtype=dtyp, device=device)   # check `dtyp` is a valid dtype

