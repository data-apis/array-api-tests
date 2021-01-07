from ._array_module import arange
from .array_helpers import is_integer_dtype, dtype_ranges
from .hypothesis_helpers import numeric_dtypes, MAX_ARRAY_SIZE

from hypothesis import assume, given
from hypothesis.strategies import integers, one_of, none

@given(integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE),
       one_of(integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE), none()),
       one_of(integers(-MAX_ARRAY_SIZE, -1), integers(1, MAX_ARRAY_SIZE), none()), numeric_dtypes)
def test_arange(start, stop, step, dtype):
    if dtype in dtype_ranges:
        m, M = dtype_ranges[dtype]
        if (not (m <= start <= M)
            or isinstance(stop, int) and not (m <= stop <= M)
            or isinstance(step, int) and not (m <= step <= M)):
            assume(False)

    if stop is None:
        # NB: "start" is really the stop
        # step is ignored in this case
        a = arange(start, dtype=dtype)
        if is_integer_dtype(dtype):
            r = range(start)
    elif step is None:
        a = arange(start, stop, dtype=dtype)
        if is_integer_dtype(dtype):
            r = range(start, stop)
    else:
        a = arange(start, stop, step, dtype=dtype)
        if is_integer_dtype(dtype):
            r = range(start, stop, step)
    assert a.dtype == dtype, "arange() produced an incorrect dtype"
    if is_integer_dtype(dtype):
        assert a.shape == (len(r),), "arange() produced incorrect shape"
        if len(r) <= MAX_ARRAY_SIZE:
            assert list(a) == list(r), "arange() produced incorrect values"
