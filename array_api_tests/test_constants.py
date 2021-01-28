from ._array_module import (e, inf, nan, pi, equal, isnan, abs, full, float32,
                            float64, less, isinf, greater)
from .array_helpers import one

def test_e():
    # Check that e acts as a scalar
    E = full((1,), e, dtype=float64)

    # We don't require any accuracy. This is just a smoke test to check that
    # 'e' is actually the constant e.
    assert less(abs(E - 2.71), one((1,), dtype=float64)), "e is not the constant e"

def test_pi():
    # Check that pi acts as a scalar
    PI = full((1,), pi, dtype=float64)

    # We don't require any accuracy. This is just a smoke test to check that
    # 'pi' is actually the constant π.
    assert less(abs(PI - 3.14), one((1,), dtype=float64)), "pi is not the constant π"

def test_inf():
    # Check that inf acts as a scalar
    INF = full((1,), inf, dtype=float64)

    assert isinf(inf), "inf is not infinity"
    assert isinf(INF), "inf is not infinity"
    assert greater(inf, 0), "inf is not positive"
    assert greater(INF, 0), "inf is not positive"

def test_nan():
    # Check that nan acts as a scalar
    NAN = full((1,), nan, dtype=float64)

    assert isnan(nan), "nan is not Not a Number"
    assert isnan(NAN), "nan is not Not a Number"

    assert not equal(nan, nan), "nan should be unequal to itself"
    assert not equal(NAN, NAN), "nan should be unequal to itself"
