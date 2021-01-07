from ._array_module import arange, ceil, empty, _floating_dtypes
from .array_helpers import is_integer_dtype, dtype_ranges
from .hypothesis_helpers import (numeric_dtypes, dtypes, MAX_ARRAY_SIZE, shapes, sizes)

from hypothesis import assume, given
from hypothesis.strategies import integers, floats, one_of, none

int_range = integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE)
float_range = floats(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE, allow_nan=False)
@given(one_of(int_range, float_range),
       one_of(int_range, float_range, none()),
       one_of(int_range, float_range, none()).filter(lambda x: x != 0),
       numeric_dtypes)
def test_arange(start, stop, step, dtype):
    if dtype in dtype_ranges:
        m, M = dtype_ranges[dtype]
        if (not (m <= start <= M)
            or isinstance(stop, int) and not (m <= stop <= M)
            or isinstance(step, int) and not (m <= step <= M)):
            assume(False)

    all_int = (is_integer_dtype(dtype)
               and isinstance(start, int)
               and (stop is None or isinstance(stop, int))
               and (step is None or isinstance(step, int)))

    if stop is None:
        # NB: "start" is really the stop
        # step is ignored in this case
        a = arange(start, dtype=dtype)
        if all_int:
            r = range(start)
    elif step is None:
        a = arange(start, stop, dtype=dtype)
        if all_int:
            r = range(start, stop)
    else:
        a = arange(start, stop, step, dtype=dtype)
        if all_int:
            r = range(start, stop, step)
    assert a.dtype == dtype, "arange() produced an incorrect dtype"
    assert a.ndim == 1, "arange() should return a 1-dimensional array"
    if all_int:
        assert a.shape == (len(r),), "arange() produced incorrect shape"
        if len(r) <= MAX_ARRAY_SIZE:
            assert list(a) == list(r), "arange() produced incorrect values"
    else:
        # This is already implied by the len(r) test above
        if (stop is not None
            and step is not None
            and (step > 0 and stop >= start
                 or step < 0 and stop <= start)):
            assert a.size == ceil((stop-start)/step), "arange() produced an array of the incorrect size"

@given(one_of(shapes, sizes), one_of(dtypes, none()))
def test_empty(shape, dtype):
    if dtype is None:
        a = empty(shape)
        assert a.dtype in _floating_dtypes, "empty() should produce an array with the default floating point dtype"
    else:
        a = empty(shape, dtype=dtype)
        assert a.dtype == dtype

    if isinstance(shape, int):
        shape = (shape,)
    assert a.shape == shape, "empty() produced an array with an incorrect shape"
