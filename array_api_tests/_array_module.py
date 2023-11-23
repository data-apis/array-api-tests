from . import stubs, xp


class _UndefinedStub:
    """
    Standing for undefined names, so the tests can be imported even if they
    fail

    If this object appears in a test failure, it means a name is not defined
    in a function. This typically happens for things like dtype literals not
    being defined.

    """
    def __init__(self, name):
        self.name = name

    def _raise(self, *args, **kwargs):
        raise AssertionError(f"{self.name} is not defined in {xp.__name__}")

    def __repr__(self):
        return f"<undefined stub for {self.name!r}>"

    __call__ = _raise
    __getattr__ = _raise

_dtypes = [
    "bool",
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float32", "float64",
    "complex64", "complex128",
]
_constants = ["e", "inf", "nan", "pi"]
_funcs = [f.__name__ for funcs in stubs.category_to_funcs.values() for f in funcs]
_funcs += ["take", "isdtype", "conj", "imag", "real"]  # TODO: bump spec and update array-api-tests to new spec layout
_top_level_attrs = _dtypes + _constants + _funcs + stubs.EXTENSIONS + ["fft"]

for attr in _top_level_attrs:
    try:
        globals()[attr] = getattr(xp, attr)
    except AttributeError:
        globals()[attr] = _UndefinedStub(attr)
