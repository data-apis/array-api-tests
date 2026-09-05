import pytest

from hypothesis import given

from array_api_tests .hypothesis_helpers import symmetric_matrices
from array_api_tests import array_helpers as ah
from array_api_tests import _array_module as xp
from array_api_tests.test_linalg import _test_matrix_transpose


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 3), (0, 3), (3, 0), (0, 0)])
def test_matrix_transpose_without_nested_arrays(monkeypatch, shape):
    size = 1
    for dim in shape:
        size *= dim
    x = xp.reshape(xp.arange(size, dtype=xp.int64), shape)
    original_asarray = xp.asarray

    def asarray(obj, **kwargs):
        def check_sequence(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    check_sequence(item)
            else:
                assert isinstance(value, (bool, int, float, complex))

        if isinstance(obj, (list, tuple)):
            check_sequence(obj)
        return original_asarray(obj, **kwargs)

    monkeypatch.setattr(xp, 'asarray', asarray)
    _test_matrix_transpose(xp, x)

@pytest.mark.xp_extension('linalg')
@given(x=symmetric_matrices(finite=True))
def test_symmetric_matrices(x):
    upper = xp.triu(x)
    lower = xp.tril(x)
    lowerT = ah._matrix_transpose(lower)

    ah.assert_exactly_equal(upper, lowerT)
