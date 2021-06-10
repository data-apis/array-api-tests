import math

from ._array_module import (e, inf, nan, pi, equal, isnan, abs, full, float32,
                            float64, less, isinf, greater, all)
from .array_helpers import one

def test_e():
    # Check that e is a Python scalar
    assert isinstance(e, float), "e is not a Python scalar"

    E = full((1,), e, dtype=float64)

    # We don't require any accuracy. This is just a smoke test to check that
    # 'e' is actually the constant e.
    assert all(less(abs(E - 2.71), one((1,), dtype=float64))), "e is not the constant e"

def test_pi():
    # Check that pi is a Python scalar
    assert isinstance(pi, float), "pi is not a Python scalar"

    PI = full((1,), pi, dtype=float64)

    # We don't require any accuracy. This is just a smoke test to check that
    # 'pi' is actually the constant π.
    assert all(less(abs(PI - 3.14), one((1,), dtype=float64))), "pi is not the constant π"

def test_inf():
    # Check that inf is a Python scalar
    assert isinstance(inf, float), "inf is not a Python scalar"

    INF = full((1,), inf, dtype=float64)
    zero = full((1,), 0.0, dtype=float64)

    assert math.isinf(inf), "inf is not infinity"
    assert all(isinf(INF)), "inf is not infinity"
    assert inf > 0, "inf is not positive"
    assert all(greater(INF, zero)), "inf is not positive"

def test_nan():
    # Check that nan is a Python scalar
    assert isinstance(nan, float), "nan is not a Python scalar"

    NAN = full((1,), nan, dtype=float64)

    assert math.isnan(nan), "nan is not Not a Number"
    assert all(isnan(NAN)), "nan is not Not a Number"

    assert nan != nan, "nan should be unequal to itself"
    assert not all(equal(NAN, NAN)), "nan should be unequal to itself"
