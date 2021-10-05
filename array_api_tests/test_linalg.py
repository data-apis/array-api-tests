"""
Tests for linalg functions

https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html

and

https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html

Note: this file currently mixes both the required linear algebra functions and
functions from the linalg extension. The functions in the latter are not
required, but we don't yet have a clean way to disable only those tests (see https://github.com/data-apis/array-api-tests/issues/25).

"""

from hypothesis import given
from hypothesis.strategies import booleans

from .array_helpers import assert_exactly_equal, ndindex
from .hypothesis_helpers import (xps, shapes, kwargs, none, positive_definite_matrices)

from . import _array_module

# Standin strategy for not yet implemented tests
todo = none()

@given(
    x=positive_definite_matrices(),
    kw=kwargs(upper=booleans())
)
def test_cholesky(x, kw):
    res = _array_module.linalg.cholesky(x, **kw)

    assert res.shape == x.shape, "cholesky did not return the correct shape"
    assert res.dtype == x.dtype, "cholesky did not return the correct dtype"

    # Test that the result along stacks is the same
    for _idx in ndindex(x.shape[:-2]):
        idx = _idx + (slice(None), slice(None))
        res_stack = res[idx]
        x_stack = x[idx]
        decomp_res_stack = _array_module.linalg.cholesky(x_stack, **kw)
        assert_exactly_equal(res_stack, decomp_res_stack)

    # Test that the result is upper or lower triangular
    if kw.get('upper', False):
        assert_exactly_equal(res, _array_module.triu(res))
    else:
        assert_exactly_equal(res, _array_module.tril(res))

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(axis=todo)
)
def test_cross(x1, x2, kw):
    # res = _array_module.linalg.cross(x1, x2, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_det(x):
    # res = _array_module.linalg.det(x)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(offset=todo)
)
def test_diagonal(x, kw):
    # res = _array_module.linalg.diagonal(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_eigh(x):
    # res = _array_module.linalg.eigh(x)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_eigvalsh(x):
    # res = _array_module.linalg.eigvalsh(x)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_inv(x):
    # res = _array_module.linalg.inv(x)
    pass

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_matmul(x1, x2):
    # res = _array_module.linalg.matmul(x1, x2)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(axis=todo, keepdims=todo, ord=todo)
)
def test_matrix_norm(x, kw):
    # res = _array_module.linalg.matrix_norm(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    n=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_matrix_power(x, n):
    # res = _array_module.linalg.matrix_power(x, n)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(rtol=todo)
)
def test_matrix_rank(x, kw):
    # res = _array_module.linalg.matrix_rank(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_matrix_transpose(x):
    # res = _array_module.linalg.matrix_transpose(x)
    pass

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_outer(x1, x2):
    # res = _array_module.linalg.outer(x1, x2)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(rtol=todo)
)
def test_pinv(x, kw):
    # res = _array_module.linalg.pinv(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(mode=todo)
)
def test_qr(x, kw):
    # res = _array_module.linalg.qr(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_slogdet(x):
    # res = _array_module.linalg.slogdet(x)
    pass

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_solve(x1, x2):
    # res = _array_module.linalg.solve(x1, x2)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(full_matrices=todo)
)
def test_svd(x, kw):
    # res = _array_module.linalg.svd(x, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
)
def test_svdvals(x):
    # res = _array_module.linalg.svdvals(x)
    pass

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(axes=todo)
)
def test_tensordot(x1, x2, kw):
    # res = _array_module.linalg.tensordot(x1, x2, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(offset=todo)
)
def test_trace(x, kw):
    # res = _array_module.linalg.trace(x, **kw)
    pass

@given(
    x1=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    x2=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(axis=todo)
)
def test_vecdot(x1, x2, kw):
    # res = _array_module.linalg.vecdot(x1, x2, **kw)
    pass

@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes),
    kw=kwargs(axis=todo, keepdims=todo, ord=todo)
)
def test_vector_norm(x, kw):
    # res = _array_module.linalg.vector_norm(x, **kw)
    pass
