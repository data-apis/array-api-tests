import pytest

from hypothesis import given

from ..hypothesis_helpers import symmetric_matrices
from .. import array_helpers as ah
from .. import _array_module as xp

@pytest.mark.xp_extension('linalg')
@given(x=symmetric_matrices(finite=True))
def test_symmetric_matrices(x):
    upper = xp.triu(x)
    lower = xp.tril(x)
    lowerT = ah._matrix_transpose(lower)

    ah.assert_exactly_equal(upper, lowerT)
