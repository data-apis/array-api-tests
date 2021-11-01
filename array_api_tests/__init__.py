from hypothesis.extra.array_api import make_strategies_namespace

from ._array_module import mod as _xp


xps = make_strategies_namespace(_xp)


del _xp
del make_strategies_namespace
