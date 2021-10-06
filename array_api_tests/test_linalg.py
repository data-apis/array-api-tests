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
from hypothesis.strategies import booleans, none, integers

from .array_helpers import assert_exactly_equal, ndindex, asarray
from .hypothesis_helpers import (xps, dtypes, shapes, kwargs, matrix_shapes,
                                 square_matrix_shapes,
                                 positive_definite_matrices, MAX_ARRAY_SIZE)

from . import _array_module

# Standin strategy for not yet implemented tests
todo = none()

def _test_stacks(f, x, kw, res=None, dims=2, true_val=None):
    """
    Test that f(x, **kw) maps across stacks of matrices

    dims is the number of dimensions f should have for a single n x m matrix
    stack.

    true_val may be a function such that true_val(x_stack) gives the true
    value for f on a stack
    """
    if res is None:
        res = f(x)
    for _idx in ndindex(x.shape[:-2]):
        idx = _idx + (slice(None),)*dims
        res_stack = res[idx]
        x_stack = x[idx]
        decomp_res_stack = f(x_stack, **kw)
        assert_exactly_equal(res_stack, decomp_res_stack)
        if true_val:
            assert_exactly_equal(decomp_res_stack, true_val(x_stack))

@given(
    x=positive_definite_matrices(),
    kw=kwargs(upper=booleans())
)
def test_cholesky(x, kw):
    res = _array_module.linalg.cholesky(x, **kw)

    assert res.shape == x.shape, "cholesky did not return the correct shape"
    assert res.dtype == x.dtype, "cholesky did not return the correct dtype"

    _test_stacks(_array_module.linalg.cholesky, x, kw, res)

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
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=square_matrix_shapes),
)
def test_det(x):
    res = _array_module.linalg.det(x)

    assert res.dtype == x.dtype, "det() did not return the correct dtype"
    assert res.shape == x.shape[:-2], "det() did not return the correct shape"

    # TODO: Test that res actually corresponds to the determinant of x

@given(
    x=xps.arrays(dtype=dtypes, shape=matrix_shapes),
    # offset may produce an overflow if it is too large. Supporting offsets
    # that are way larger than the array shape isn't very important.
    kw=kwargs(offset=integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE))
)
def test_diagonal(x, kw):
    res = _array_module.linalg.diagonal(x, **kw)

    assert res.dtype == x.dtype, "diagonal() returned the wrong dtype"

    n, m = x.shape[-2:]
    offset = kw.get('offset', 0)
    # Note: the spec does not specify that offset must be within the bounds of
    # the matrix. A large offset should just produce a size 0 in the last
    # dimension.
    if offset < 0:
        diag_size = min(n, m, max(n + offset, 0))
    elif offset == 0:
        diag_size = min(n, m)
    else:
        diag_size = min(n, m, max(m - offset, 0))

    assert res.shape == (*x.shape[:-2], diag_size), "diagonal() returned the wrong shape"

    def true_diag(x_stack):
        if offset >= 0:
            x_stack_diag = [x_stack[i + offset, i] for i in range(diag_size)]
        else:
            x_stack_diag = [x_stack[i, i - offset] for i in range(diag_size)]
        return asarray(x_stack_diag, dtype=x.dtype)

    _test_stacks(_array_module.linalg.diagonal, x, kw, res, dims=1, true_val=true_diag)

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
