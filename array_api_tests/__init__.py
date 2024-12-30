import os
from functools import wraps
from importlib import import_module

from hypothesis import strategies as st
from hypothesis.extra import array_api

from . import _version

__all__ = ["xp", "api_version", "xps"]


# You can comment the following out and instead import the specific array module
# you want to test, e.g. `import array_api_strict as xp`.
if "ARRAY_API_TESTS_MODULE" in os.environ:
    env_var = os.environ["ARRAY_API_TESTS_MODULE"]
    if env_var.startswith("exec(") and env_var.endswith(")"):
        script = env_var[5:][:-1]
        namespace = {}
        exec(script, namespace)
        xp = namespace["xp"]
        xp_name = xp.__name__
    else:
        xp_name = env_var
        _module, _sub = xp_name, None
        if "." in xp_name:
            _module, _sub = xp_name.split(".", 1)
        xp = import_module(_module)
        if _sub:
            try:
                xp = getattr(xp, _sub)
            except AttributeError:
                # _sub may be a submodule that needs to be imported. WE can't
                # do this in every case because some array modules are not
                # submodules that can be imported (like mxnet.nd).
                xp = import_module(xp_name)
else:
    raise RuntimeError(
        "No array module specified - either edit __init__.py or set the "
        "ARRAY_API_TESTS_MODULE environment variable."
    )


# If xp.bool is not available, like in some versions of NumPy and CuPy, try
# patching in xp.bool_.
try:
    xp.bool
except AttributeError as e:
    if hasattr(xp, "bool_"):
        xp.bool = xp.bool_
    else:
        raise e


# We monkey patch floats() to always disable subnormals as they are out-of-scope

_floats = st.floats


@wraps(_floats)
def floats(*a, **kw):
    kw["allow_subnormal"] = False
    return _floats(*a, **kw)


st.floats = floats


# We do the same with xps.from_dtype() - this is not strictly necessary, as
# the underlying floats() will never generate subnormals. We only do this
# because internal logic in xps.from_dtype() assumes xp.finfo() has its
# attributes as scalar floats, which is expected behaviour but disrupts many
# unrelated tests.
try:
    __from_dtype = array_api._from_dtype

    @wraps(__from_dtype)
    def _from_dtype(*a, **kw):
        kw["allow_subnormal"] = False
        return __from_dtype(*a, **kw)

    array_api._from_dtype = _from_dtype
except AttributeError:
    # Ignore monkey patching if Hypothesis changes the private API
    pass


api_version = os.getenv(
    "ARRAY_API_TESTS_VERSION", getattr(xp, "__array_api_version__", "2023.12")
)
xps = array_api.make_strategies_namespace(xp, api_version=api_version)

__version__ = _version.get_versions()["version"]
