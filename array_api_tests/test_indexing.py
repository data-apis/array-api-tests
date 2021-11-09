"""
https://data-apis.github.io/array-api/latest/API_specification/indexing.html

For these tests, we only need arrays where each element is distinct, so we use
arange().
"""

from hypothesis import given
from hypothesis.strategies import shared

from .array_helpers import assert_exactly_equal
from .hypothesis_helpers import (slices, sizes, integer_indices, shapes, prod,
                                 multiaxis_indices)
from .pytest_helpers import raises
from ._array_module import arange, reshape

# TODO: Add tests for __setitem__

@given(shared(sizes, key='array_sizes'), integer_indices(shared(sizes, key='array_sizes')))
def test_integer_indexing(size, idx):
    # Test that indices on single dimensional arrays give the same result as
    # Python lists. idx may be a Python integer or a 0-D array with integer dtype.

    # Sanity check that the strategies are working properly
    assert -size <= int(idx) <= max(0, size - 1), "Sanity check failed. This indicates a bug in the test suite"

    a = arange(size)
    l = list(range(size))
    # TODO: We can remove int() here if we add __index__ to the spec. See
    # https://github.com/data-apis/array-api/issues/231.
    sliced_list = l[int(idx)]
    sliced_array = a[idx]

    assert sliced_array.shape == (), "Integer indices should reduce the dimension by 1"
    assert sliced_array.dtype == a.dtype, "Integer indices should not change the dtype"
    assert sliced_array == sliced_list, "Integer index did not give the correct entry"

@given(shared(sizes, key='array_sizes'), slices(shared(sizes, key='array_sizes')))
def test_slicing(size, s):
    # Test that slices on arrays give the same result as Python lists.

    # Sanity check that the strategies are working properly
    if s.start is not None:
        assert -size <= s.start <= size, "Sanity check failed. This indicates a bug in the test suite"
    if s.stop is not None:
        if s.step is None or s.step > 0:
            assert -size <= s.stop <= size, "Sanity check failed. This indicates a bug in the test suite"
        else:
            assert -size - 1 <= s.stop <= size - 1, "Sanity check failed. This indicates a bug in the test suite"
    a = arange(size)
    l = list(range(size))
    sliced_list = l[s]
    sliced_array = a[s]

    assert len(sliced_list) == sliced_array.size, "Slice index did not give the same number of elements as slicing an equivalent Python list"
    assert sliced_array.shape == (sliced_array.size,), "Slice index did not give the correct shape"
    assert sliced_array.dtype == a.dtype, "Slice indices should not change the dtype"
    for i in range(len(sliced_list)):
        assert sliced_array[i] == sliced_list[i], "Slice index did not give the same elements as slicing an equivalent Python list"

@given(shared(shapes(), key='array_shapes'),
       multiaxis_indices(shapes=shared(shapes(), key='array_shapes')))
def test_multiaxis_indexing(shape, idx):
    # NOTE: Out of bounds indices (both integer and slices) are out of scope
    # for the spec. If you get a (valid) out of bounds error, it indicates a
    # bug in the multiaxis_indices strategy, which should only generate
    # indices that are not out of bounds.
    size = prod(shape)
    a = reshape(arange(size), shape)

    n_ellipses = len([i for i in idx if i is ...])
    if n_ellipses > 1:
        raises(IndexError, lambda: a[idx],
               "Indices with more than one ellipsis should raise IndexError")
        return
    elif len(idx) - n_ellipses > len(shape):
        raises(IndexError, lambda: a[idx],
               "Tuple indices with more single axis expressions than the shape should raise IndexError")
        return

    sliced_array = a[idx]
    equiv_idx = idx
    if n_ellipses or len(idx) < len(shape):
        # Would be
        #
        # ellipsis_i = idx.index(...) if n_ellipses else len(idx)
        #
        # except we have to be careful to not use == to compare array elements
        # of idx.
        ellipsis_i = len(idx)
        for i in range(len(idx)):
            if idx[i] is ...:
                ellipsis_i = i
                break

        equiv_idx = (idx[:ellipsis_i]
                     + (slice(None, None, None),)*(len(shape) - len(idx) + n_ellipses)
                     + idx[ellipsis_i + 1:])
        # Sanity check
        assert len(equiv_idx) == len(shape), "Sanity check failed. This indicates a bug in the test suite"
        sliced_array2 = a[equiv_idx]
        assert_exactly_equal(sliced_array, sliced_array2)

    # TODO: We don't check that the exact entries are correct. Instead we
    # check the shape and other properties, and assume the single dimension
    # tests above are sufficient for testing the exact behavior of integer
    # indices and slices.

    # Check that the new shape is what it should be
    newshape = []
    for i, s in enumerate(equiv_idx):
        # Slices should retain the dimension. Integers remove a dimension.
        if isinstance(s, slice):
            newshape.append(len(range(shape[i])[s]))
    assert sliced_array.shape == tuple(newshape), "Index did not give the correct resulting array shape"

    # Check that integer indices i chose the same elements as the slice i:i+1
    equiv_idx2 = []
    for i, size in zip(equiv_idx, shape):
        if isinstance(i, int):
            if i >= 0:
                i = slice(i, i + 1)
            else:
                i = slice(size + i, size + i + 1)
        equiv_idx2.append(i)
    equiv_idx2 = tuple(equiv_idx2)

    sliced_array2 = a[equiv_idx2]
    assert sliced_array2.size == sliced_array.size, "Integer index not choosing the same elements as an equivalent slice"
    assert_exactly_equal(reshape(sliced_array2, sliced_array.shape), sliced_array)
