"""
https://data-apis.github.io/array-api/latest/API_specification/indexing.html

For these tests, we only need arrays where each element is distinct, so we use
arange().
"""

from hypothesis import given
from hypothesis.strategies import shared

from .hypothesis_helpers import slices, sizes, integer_indices
from ._array_module import arange

@given(shared(sizes, key='array_sizes'), integer_indices(shared(sizes, key='array_sizes')))
def test_integer_indexing(size, idx):
    # Test that indices on single dimensional arrays give the same result as
    # Python lists.

    a = arange(size)
    l = list(range(size))
    sliced_list = l[idx]
    sliced_array = a[idx]

    assert sliced_array.shape == ()
    assert sliced_array.dtype == a.dtype
    assert sliced_array == sliced_list

@given(shared(sizes, key='array_sizes'), slices(shared(sizes, key='array_sizes')))
def test_slicing(size, s):
    # Test that slices on arrays give the same result as Python lists.

    # Sanity check that the strategies are working properly
    if s.start is not None:
        assert -size <= s.start <= max(0, size - 1)
    if s.stop is not None:
        assert -size <= s.stop <= size

    a = arange(size)
    l = list(range(size))
    sliced_list = l[s]
    sliced_array = a[s]

    assert len(sliced_list) == sliced_array.size
    assert sliced_array.shape == (sliced_array.size,)
    assert sliced_array.dtype == a.dtype
    for i in range(len(sliced_list)):
        assert sliced_array[i] == sliced_list[i]
