"""
We're not interested in being 100% strict - instead we focus on areas which
could affect interop, e.g. with

    def add(x1, x2, /):
        ...

x1 and x2 don't need to be pos-only for the purposes of interoperability, but with

    def squeeze(x, /, axis):
        ...

axis has to be pos-or-keyword to support both styles

    >>> squeeze(x, 0)
    ...
    >>> squeeze(x, axis=0)
    ...

"""
from inspect import Parameter, Signature, signature
from types import FunctionType
from typing import Any, Callable, Dict, List, Literal, Sequence, get_args

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from hypothesis.strategies import DataObject

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps
from ._array_module import _UndefinedStub
from ._array_module import mod as xp
from .stubs import array_methods, category_to_funcs, extension_to_funcs
from .typing import DataType

pytestmark = pytest.mark.ci

ParameterKind = Literal[
    Parameter.POSITIONAL_ONLY,
    Parameter.VAR_POSITIONAL,
    Parameter.POSITIONAL_OR_KEYWORD,
    Parameter.KEYWORD_ONLY,
    Parameter.VAR_KEYWORD,
]
ALL_KINDS = get_args(ParameterKind)
VAR_KINDS = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
kind_to_str: Dict[ParameterKind, str] = {
    Parameter.POSITIONAL_OR_KEYWORD: "pos or kw argument",
    Parameter.POSITIONAL_ONLY: "pos-only argument",
    Parameter.KEYWORD_ONLY: "keyword-only argument",
    Parameter.VAR_POSITIONAL: "star-args (i.e. *args) argument",
    Parameter.VAR_KEYWORD: "star-kwargs (i.e. **kwargs) argument",
}


def _test_inspectable_func(sig: Signature, stub_sig: Signature):
    params = list(sig.parameters.values())
    stub_params = list(stub_sig.parameters.values())
    # We're not interested if the array module has additional arguments, so we
    # only iterate through the arguments listed in the spec.
    for i, stub_param in enumerate(stub_params):
        assert (
            len(params) >= i + 1
        ), f"Argument '{stub_param.name}' missing from signature"
        param = params[i]

        # We're not interested in the name if it isn't actually used
        if stub_param.kind not in [
            Parameter.POSITIONAL_ONLY,
            *VAR_KINDS,
        ]:
            assert (
                param.name == stub_param.name
            ), f"Expected argument '{param.name}' to be named '{stub_param.name}'"

        f_stub_kind = kind_to_str[stub_param.kind]
        if stub_param.kind in [Parameter.POSITIONAL_OR_KEYWORD, *VAR_KINDS]:
            assert param.kind == stub_param.kind, (
                f"{param.name} is a {kind_to_str[param.kind]}, "
                f"but should be a {f_stub_kind}"
            )
        else:
            # TODO: allow for kw-only args to be out-of-order
            assert param.kind in [stub_param.kind, Parameter.POSITIONAL_OR_KEYWORD,], (
                f"{param.name} is a {kind_to_str[param.kind]}, "
                f"but should be a {f_stub_kind} "
                f"(or at least a {kind_to_str[ParameterKind.POSITIONAL_OR_KEYWORD]})"
            )


def get_dtypes_strategy(func_name: str) -> st.SearchStrategy[DataType]:
    if func_name in dh.func_in_dtypes.keys():
        dtypes = dh.func_in_dtypes[func_name]
        if hh.FILTER_UNDEFINED_DTYPES:
            dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
        return st.sampled_from(dtypes)
    else:
        return xps.scalar_dtypes()


def make_pretty_func(func_name: str, args: Sequence[Any], kwargs: Dict[str, Any]):
    f_sig = f"{func_name}("
    f_sig += ", ".join(str(a) for a in args)
    if len(kwargs) != 0:
        if len(args) != 0:
            f_sig += ", "
        f_sig += ", ".join(f"{k}={v}" for k, v in kwargs.items())
    f_sig += ")"
    return f_sig


matrixy_funcs: List[FunctionType] = [
    *category_to_funcs["linear_algebra"], *extension_to_funcs["linalg"]
]
matrixy_names: List[str] = [f.__name__ for f in matrixy_funcs]
matrixy_names += ["__matmul__", "triu", "tril"]


@given(data=st.data())
@settings(max_examples=1)
def _test_uninspectable_func(
    func_name: str, func: Callable, stub_sig: Signature, data: DataObject
):
    skip_msg = (
        f"Signature for {func_name}() is not inspectable "
        "and is too troublesome to test for otherwise"
    )
    if func_name in [
        "__bool__",
        "__int__",
        "__index__",
        "__float__",
        "pow",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "sort",
        *matrixy_names,
    ]:
        pytest.skip(skip_msg)

    param_to_value: Dict[Parameter, Any] = {}
    for param in stub_sig.parameters.values():
        if param.kind in [Parameter.POSITIONAL_OR_KEYWORD, *VAR_KINDS]:
            pytest.skip(
                skip_msg + f" (because '{param.name}' is a {kind_to_str[param.kind]})"
            )
        elif param.default != Parameter.empty:
            value = param.default
        elif param.name in ["x", "x1"]:
            dtypes = get_dtypes_strategy(func_name)
            value = data.draw(
                xps.arrays(dtype=dtypes, shape=hh.shapes(min_side=1)), label=param.name
            )
        elif param.name == "x2":
            # sanity check
            assert "x1" in [p.name for p in param_to_value.keys()]
            x1 = next(v for p, v in param_to_value.items() if p.name == "x1")
            value = data.draw(
                xps.arrays(dtype=x1.dtype, shape=x1.shape), label=param.name
            )
        else:
            pytest.skip(
                skip_msg + f" (because no default was found for argument {param.name})"
            )
        param_to_value[param] = value

    args: List[Any] = [
        v for p, v in param_to_value.items() if p.kind == Parameter.POSITIONAL_ONLY
    ]
    kwargs: Dict[str, Any] = {
        p.name: v for p, v in param_to_value.items() if p.kind == Parameter.KEYWORD_ONLY
    }
    f_func = make_pretty_func(func_name, args, kwargs)
    note(f"trying {f_func}")
    func(*args, **kwargs)


def _test_func_signature(
    func: Callable, stub: FunctionType, ignore_first_stub_param: bool = False
):
    stub_sig = signature(stub)
    if ignore_first_stub_param:
        stub_params = list(stub_sig.parameters.values())
        del stub_params[0]
        stub_sig = Signature(
            parameters=stub_params, return_annotation=stub_sig.return_annotation
        )

    try:
        sig = signature(func)
        _test_inspectable_func(sig, stub_sig)
    except ValueError:
        _test_uninspectable_func(stub.__name__, func, stub_sig)


@pytest.mark.parametrize(
    "stub",
    [s for stubs in category_to_funcs.values() for s in stubs],
    ids=lambda f: f.__name__,
)
def test_func_signature(stub: FunctionType):
    assert hasattr(xp, stub.__name__), f"{stub.__name__} not found in array module"
    func = getattr(xp, stub.__name__)
    _test_func_signature(func, stub)


extension_and_stub_params = []
for ext, stubs in extension_to_funcs.items():
    for stub in stubs:
        p = pytest.param(
            ext, stub, id=f"{ext}.{stub.__name__}", marks=pytest.mark.xp_extension(ext)
        )
        extension_and_stub_params.append(p)


@pytest.mark.parametrize("extension, stub", extension_and_stub_params)
def test_extension_func_signature(extension: str, stub: FunctionType):
    mod = getattr(xp, extension)
    assert hasattr(
        mod, stub.__name__
    ), f"{stub.__name__} not found in {extension} extension"
    func = getattr(mod, stub.__name__)
    _test_func_signature(func, stub)


@pytest.mark.parametrize("stub", array_methods, ids=lambda f: f.__name__)
@given(st.data())
@settings(max_examples=1)
def test_array_method_signature(stub: FunctionType, data: DataObject):
    dtypes = get_dtypes_strategy(stub.__name__)
    x = data.draw(xps.arrays(dtype=dtypes, shape=hh.shapes(min_side=1)), label="x")
    assert hasattr(x, stub.__name__), f"{stub.__name__} not found in array object {x!r}"
    method = getattr(x, stub.__name__)
    # Ignore 'self' arg in stub, which won't be present in instantiated objects.
    _test_func_signature(method, stub, ignore_first_stub_param=True)
