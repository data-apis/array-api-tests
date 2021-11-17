from hypothesis.extra import array_api

from . import dtype_helpers as dh
from ._array_module import mod as _xp

__all__ = ["xps"]


# For now we monkey patch the internal _from_dtype() method to work around a bug
# in st.floats() - see https://github.com/HypothesisWorks/hypothesis/issues/3153


broken_from_dtype = array_api._from_dtype


def _from_dtype(xp, dtype, **kwargs):
    strat = broken_from_dtype(_xp, dtype, **kwargs)
    if dh.is_float_dtype(dtype):
        smallest_normal = xp.finfo(dtype).smallest_normal
        strat = strat.filter(lambda n: abs(n) >= smallest_normal)
    return strat


array_api._from_dtype = _from_dtype


xps = array_api.make_strategies_namespace(_xp)
