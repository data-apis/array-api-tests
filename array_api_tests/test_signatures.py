from inspect import Parameter, Signature, signature
from types import FunctionType
from typing import Callable, Dict

import pytest
from hypothesis import given

from . import hypothesis_helpers as hh
from . import xps
from ._array_module import mod as xp
from .stubs import array_methods, category_to_funcs, extension_to_funcs

kind_to_str: Dict[Parameter, str] = {
    Parameter.POSITIONAL_OR_KEYWORD: "normal argument",
    Parameter.POSITIONAL_ONLY: "pos-only argument",
    Parameter.KEYWORD_ONLY: "keyword-only argument",
    Parameter.VAR_POSITIONAL: "star-args (i.e. *args) argument",
    Parameter.VAR_KEYWORD: "star-kwargs (i.e. **kwargs) argument",
}


def _test_signature(
    func: Callable, stub: FunctionType, ignore_first_stub_param: bool = False
):
    """
    Signature of function is correct enough to not affect interoperability

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
    try:
        sig = signature(func)
    except ValueError:
        pytest.skip(
            msg=f"type({stub.__name__})={type(func)} not supported by inspect.signature()"
        )
    params = list(sig.parameters.values())

    stub_sig = signature(stub)
    stub_params = list(stub_sig.parameters.values())
    if ignore_first_stub_param:
        stub_params = stub_params[1:]
        stub = Signature(
            parameters=stub_params, return_annotation=stub_sig.return_annotation
        )

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
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ]:
            assert (
                param.name == stub_param.name
            ), f"Expected argument '{param.name}' to be named '{stub_param.name}'"

        if (
            stub_param.name in ["x", "x1", "x2"]
            and stub_param.kind != Parameter.POSITIONAL_ONLY
        ):
            pytest.skip(
                f"faulty spec - argument {stub_param.name} should be a "
                f"{kind_to_str[Parameter.POSITIONAL_ONLY]}"
            )
        f_kind = kind_to_str[param.kind]
        f_stub_kind = kind_to_str[stub_param.kind]
        if stub_param.kind in [
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ]:
            assert (
                param.kind == stub_param.kind
            ), f"{param.name} is a {f_kind}, but should be a {f_stub_kind}"
        else:
            # TODO: allow for kw-only args to be out-of-order
            assert param.kind in [stub_param.kind, Parameter.POSITIONAL_OR_KEYWORD], (
                f"{param.name} is a {f_kind}, "
                f"but should be a {f_stub_kind} "
                f"(or at least a {kind_to_str[Parameter.POSITIONAL_OR_KEYWORD]})"
            )


@pytest.mark.parametrize(
    "stub",
    [s for stubs in category_to_funcs.values() for s in stubs],
    ids=lambda f: f.__name__,
)
def test_func_signature(stub: FunctionType):
    assert hasattr(xp, stub.__name__), f"{stub.__name__} not found in array module"
    func = getattr(xp, stub.__name__)
    _test_signature(func, stub)


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
    _test_signature(func, stub)


@pytest.mark.parametrize("stub", array_methods, ids=lambda f: f.__name__)
@given(x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()))
def test_array_method_signature(stub: FunctionType, x):
    assert hasattr(x, stub.__name__), f"{stub.__name__} not found in array object {x!r}"
    method = getattr(x, stub.__name__)
    # Ignore 'self' arg in stub, which won't be present in instantiated objects.
    _test_signature(method, stub, ignore_first_stub_param=True)
