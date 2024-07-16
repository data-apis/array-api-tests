from hypothesis import given

from array_api_tests import _array_module as xp
from array_api_tests.hypothesis_helpers import two_mutual_arrays
from array_api_tests.dtype_helpers import int_dtypes
from array_api_tests.shape_helpers import iter_indices, broadcast_shapes
from array_api_tests .array_helpers import exactly_equal, notequal, less

# TODO: These meta-tests currently only work with NumPy

def test_exactly_equal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([True, False, True, False, True, False, True, False])
    assert xp.all(xp.equal(exactly_equal(a, b), res))

def test_notequal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([False, True, False, False, False, True, False, True])
    assert xp.all(xp.equal(notequal(a, b), res))


@given(*two_mutual_arrays(dtypes=int_dtypes))
def test_less(x, y):
    res = less(x, y)

    for i, j, k in iter_indices(x.shape, y.shape, broadcast_shapes(x.shape, y.shape)):
        assert res[k] == (int(x[i]) < int(y[j]))
