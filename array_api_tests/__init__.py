from hypothesis.extra.array_api import make_strategies_namespace

from . import _array_module as xp


xps = make_strategies_namespace(xp)
