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
from collections import defaultdict
from copy import copy
from inspect import Parameter, Signature, signature
from itertools import chain
from types import FunctionType
from typing import Any, Callable, DefaultDict, Dict, List, Literal, Sequence, get_args

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps
from ._array_module import _UndefinedStub
from ._array_module import mod as xp
from .stubs import array_methods, category_to_funcs, extension_to_funcs
from .typing import DataType, Shape

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
    Parameter.POSITIONAL_OR_KEYWORD: "normal argument",
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
        if sig is not None:
            assert (
                len(params) >= i + 1
            ), f"Argument '{stub_param.name}' missing from signature"
            param = params[i]

        # We're not interested in the name if it isn't actually used
        if sig is not None and stub_param.kind not in [
            Parameter.POSITIONAL_ONLY,
            *VAR_KINDS,
        ]:
            assert (
                param.name == stub_param.name
            ), f"Expected argument '{param.name}' to be named '{stub_param.name}'"

        f_stub_kind = kind_to_str[stub_param.kind]
        if stub_param.kind in [Parameter.POSITIONAL_OR_KEYWORD, *VAR_KINDS]:
            if sig is not None:
                assert param.kind == stub_param.kind, (
                    f"{param.name} is a {kind_to_str[param.kind]}, "
                    f"but should be a {f_stub_kind}"
                )
            else:
                pass
        else:
            # TODO: allow for kw-only args to be out-of-order
            if sig is not None:
                assert param.kind in [
                    stub_param.kind,
                    Parameter.POSITIONAL_OR_KEYWORD,
                ], (
                    f"{param.name} is a {kind_to_str[param.kind]}, "
                    f"but should be a {f_stub_kind} "
                    f"(or at least a {kind_to_str[ParameterKind.POSITIONAL_OR_KEYWORD]})"
                )
            else:
                pass


def shapes(**kw) -> st.SearchStrategy[Shape]:
    if "min_side" not in kw.keys():
        kw["min_side"] = 1
    return hh.shapes(**kw)


matrixy_funcs: List[str] = [
    f.__name__
    for f in chain(category_to_funcs["linear_algebra"], extension_to_funcs["linalg"])
]
matrixy_funcs += ["__matmul__", "triu", "tril"]
func_to_shapes: DefaultDict[str, st.SearchStrategy[Shape]] = defaultdict(
    shapes,
    {
        **{k: st.just(()) for k in ["__bool__", "__int__", "__index__", "__float__"]},
        "sort": shapes(min_dims=1),  # for axis=-1,
        **{k: shapes(min_dims=2) for k in matrixy_funcs},
        # Overwrite min_dims=2 shapes for some matrixy functions
        "cross": shapes(min_side=3, max_side=3, min_dims=3, max_dims=3),
        "outer": shapes(min_dims=1, max_dims=1),
    },
)


def get_dtypes_strategy(func_name: str) -> st.SearchStrategy[DataType]:
    if func_name in dh.func_in_dtypes.keys():
        dtypes = dh.func_in_dtypes[func_name]
        if hh.FILTER_UNDEFINED_DTYPES:
            dtypes = [d for d in dtypes if not isinstance(d, _UndefinedStub)]
        return st.sampled_from(dtypes)
    else:
        return xps.scalar_dtypes()


func_to_example_values: Dict[str, Dict[ParameterKind, Dict[str, Any]]] = {
    "broadcast_to": {
        Parameter.POSITIONAL_ONLY: {"x": xp.asarray([0, 1])},
        Parameter.POSITIONAL_OR_KEYWORD: {"shape": (1, 2)},
    },
    "cholesky": {
        Parameter.POSITIONAL_ONLY: {"x": xp.asarray([[1.0, 0.0], [0.0, 1.0]])}
    },
    "inv": {Parameter.POSITIONAL_ONLY: {"x": xp.asarray([[1.0, 0.0], [0.0, 1.0]])}},
}


def make_pretty_func(func_name: str, args: Sequence[Any], kwargs: Dict[str, Any]):
    f_sig = f"{func_name}("
    f_sig += ", ".join(str(a) for a in args)
    if len(kwargs) != 0:
        if len(args) != 0:
            f_sig += ", "
        f_sig += ", ".join(f"{k}={v}" for k, v in kwargs.items())
    f_sig += ")"
    return f_sig


@given(data=st.data())
def _test_uninspectable_func(func_name: str, func: Callable, stub_sig: Signature, data):
    example_values: Dict[ParameterKind, Dict[str, Any]] = func_to_example_values.get(
        func_name, {}
    )
    for kind in ALL_KINDS:
        example_values.setdefault(kind, {})

    for param in stub_sig.parameters.values():
        for name_to_value in example_values.values():
            if param.name in name_to_value.keys():
                continue

        if param.default != Parameter.empty:
            example_value = param.default
        elif param.name in ["x", "x1"]:
            dtypes = get_dtypes_strategy(func_name)
            shapes = func_to_shapes[func_name]
            example_value = data.draw(
                xps.arrays(dtype=dtypes, shape=shapes), label=param.name
            )
        elif param.name == "x2":
            # sanity check
            assert "x1" in example_values[Parameter.POSITIONAL_ONLY].keys()
            x1 = example_values[Parameter.POSITIONAL_ONLY]["x1"]
            example_value = data.draw(
                xps.arrays(dtype=x1.dtype, shape=x1.shape), label="x2"
            )
        elif param.name == "axes":
            example_value = ()
        elif param.name == "shape":
            example_value = ()
        else:
            pytest.skip(f"No example value for argument '{param.name}'")

        if param.kind in VAR_KINDS:
            pytest.skip("TODO")
        example_values[param.kind][param.name] = example_value

    if len(example_values[Parameter.POSITIONAL_OR_KEYWORD]) == 0:
        f_func = make_pretty_func(
            func_name,
            example_values[Parameter.POSITIONAL_ONLY].values(),
            example_values[Parameter.KEYWORD_ONLY],
        )
        note(f"trying {f_func}")
        func(
            *example_values[Parameter.POSITIONAL_ONLY].values(),
            **example_values[Parameter.KEYWORD_ONLY],
        )
    else:
        either_argname_value_pairs = list(
            example_values[Parameter.POSITIONAL_OR_KEYWORD].items()
        )
        n_either_args = len(either_argname_value_pairs)
        for n_extra_args in reversed(range(n_either_args + 1)):
            extra_args = [v for _, v in either_argname_value_pairs[:n_extra_args]]
            if n_extra_args < n_either_args:
                extra_kwargs = dict(either_argname_value_pairs[n_extra_args:])
            else:
                extra_kwargs = {}
            args = list(example_values[Parameter.POSITIONAL_ONLY].values())
            args += extra_args
            kwargs = copy(example_values[Parameter.KEYWORD_ONLY])
            if len(extra_kwargs) != 0:
                kwargs.update(extra_kwargs)
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
@given(data=st.data())
def test_array_method_signature(stub: FunctionType, data):
    dtypes = get_dtypes_strategy(stub.__name__)
    shapes = func_to_shapes[stub.__name__]
    x = data.draw(xps.arrays(dtype=dtypes, shape=shapes), label="x")
    assert hasattr(x, stub.__name__), f"{stub.__name__} not found in array object {x!r}"
    method = getattr(x, stub.__name__)
    # Ignore 'self' arg in stub, which won't be present in instantiated objects.
    _test_func_signature(method, stub, ignore_first_stub_param=True)
