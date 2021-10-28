from hypothesis.extra.array_api import make_strategies_namespace

from ._array_module import mod as xp


xps = make_strategies_namespace(xp)
