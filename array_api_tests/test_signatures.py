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
from inspect import Parameter, Signature, signature
from itertools import chain
from types import FunctionType
from typing import Callable, DefaultDict, Dict, List

import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import xps
from ._array_module import _UndefinedStub
from ._array_module import mod as xp
from .stubs import array_methods, category_to_funcs, extension_to_funcs
from .typing import DataType, Shape

pytestmark = pytest.mark.ci


kind_to_str: Dict[Parameter, str] = {
    Parameter.POSITIONAL_OR_KEYWORD: "normal argument",
    Parameter.POSITIONAL_ONLY: "pos-only argument",
    Parameter.KEYWORD_ONLY: "keyword-only argument",
    Parameter.VAR_POSITIONAL: "star-args (i.e. *args) argument",
    Parameter.VAR_KEYWORD: "star-kwargs (i.e. **kwargs) argument",
}

VAR_KINDS = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)


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
                    f"(or at least a {kind_to_str[Parameter.POSITIONAL_OR_KEYWORD]})"
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
        # Override for some matrixy functions
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


@given(data=st.data())
def _test_uninspectable_func(func_name: str, func: Callable, stub_sig: Signature, data):
    if func_name in ["cholesky", "inv"]:
        func(xp.asarray([[1.0, 0.0], [0.0, 1.0]]))
        return
    elif func_name == "solve":
        func(xp.asarray([[1.0, 2.0], [3.0, 5.0]]), xp.asarray([1.0, 2.0]))
        return

    pos_argname_to_example_value = {}
    normal_argname_to_example_value = {}
    kw_argname_to_example_value = {}
    for stub_param in stub_sig.parameters.values():
        if stub_param.name in ["x", "x1"]:
            dtypes = get_dtypes_strategy(func_name)
            shapes = func_to_shapes[func_name]
            example_value = data.draw(
                xps.arrays(dtype=dtypes, shape=shapes), label=stub_param.name
            )
        elif stub_param.name == "x2":
            assert "x1" in pos_argname_to_example_value.keys()  # sanity check
            x1 = pos_argname_to_example_value["x1"]
            example_value = data.draw(
                xps.arrays(dtype=x1.dtype, shape=x1.shape), label="x2"
            )
        else:
            if stub_param.default != Parameter.empty:
                example_value = stub_param.default
            else:
                pytest.skip(f"No example value for argument '{stub_param.name}'")

        if stub_param.kind == Parameter.POSITIONAL_ONLY:
            pos_argname_to_example_value[stub_param.name] = example_value
        elif stub_param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            normal_argname_to_example_value[stub_param.name] = example_value
        elif stub_param.kind == Parameter.KEYWORD_ONLY:
            kw_argname_to_example_value[stub_param.name] = example_value
        else:
            pytest.skip()

    if len(normal_argname_to_example_value) == 0:
        func(*pos_argname_to_example_value.values(), **kw_argname_to_example_value)
    else:
        pass  # TODO


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
