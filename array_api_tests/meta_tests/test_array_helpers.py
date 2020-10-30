from ..array_helpers import exactly_equal
import numpy as np

# TODO: These meta-tests currently only work with NumPy

def test_exactly_equal():
    a = np.array([0, 0., -0., -0., np.nan, np.nan, 1, 1])
    b = np.array([0, -1, -0., 0., np.nan, 1, 1, 2])

    res = np.array([True, False, True, False, True, False, True, False])
    np.testing.assert_equal(exactly_equal(a, b), res)
