import pytest
from hypothesis import given, strategies as st

from . import xp

pytestmark = pytest.mark.min_version("2023.12")


def test_array_namespace_info():
    out = xp.__array_namespace_info__()

    capabilities = out.capabilities()
    assert isinstance(capabilities, dict)

    out.default_device()

    default_dtypes = out.default_dtypes()
    assert isinstance(default_dtypes, dict)
    assert {"real floating", "complex floating", "integral", "indexing"}.issubset(set(default_dtypes.keys()))

    devices = out.devices()
    assert isinstance(devices, list)
    
    
atomic_kinds = [
    "bool",
    "signed integer",
    "unsigned integer",
    "real floating",
    "complex floating",
]


@given(
    st.one_of(
        st.none(),
        st.sampled_from(atomic_kinds + ["integral", "numeric"]),
        st.lists(st.sampled_from(atomic_kinds), unique=True, min_size=1).map(tuple),
    )
)
def test_array_namespace_info_dtypes(kind):
    out = xp.__array_namespace_info__().dtypes(kind=kind)
    assert isinstance(out, dict)
