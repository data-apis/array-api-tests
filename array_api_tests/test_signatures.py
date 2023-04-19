"""
Tests for function/method signatures compliance

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
from types import FunctionType
from typing import Any, Callable, Dict, Literal, get_args
from warnings import warn

import pytest

from . import dtype_helpers as dh
from ._array_module import mod as xp
from .stubs import array_methods, category_to_funcs, extension_to_funcs, name_to_func

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

    non_kwonly_stub_params = [
        p for p in stub_params if p.kind != Parameter.KEYWORD_ONLY
    ]
    # sanity check
    assert non_kwonly_stub_params == stub_params[: len(non_kwonly_stub_params)]
    # We're not interested if the array module has additional arguments, so we
    # only iterate through the arguments listed in the spec.
    for i, stub_param in enumerate(non_kwonly_stub_params):
        assert (
            len(params) >= i + 1
        ), f"Argument '{stub_param.name}' missing from signature"
        param = params[i]

        # We're not interested in the name if it isn't actually used
        if stub_param.kind not in [Parameter.POSITIONAL_ONLY, *VAR_KINDS]:
            assert (
                param.name == stub_param.name
            ), f"Expected argument '{param.name}' to be named '{stub_param.name}'"

        if stub_param.kind in [Parameter.POSITIONAL_OR_KEYWORD, *VAR_KINDS]:
            f_stub_kind = kind_to_str[stub_param.kind]
            assert param.kind == stub_param.kind, (
                f"{param.name} is a {kind_to_str[param.kind]}, "
                f"but should be a {f_stub_kind}"
            )

    kwonly_stub_params = stub_params[len(non_kwonly_stub_params) :]
    for stub_param in kwonly_stub_params:
        assert (
            stub_param.name in sig.parameters.keys()
        ), f"Argument '{stub_param.name}' missing from signature"
        param = next(p for p in params if p.name == stub_param.name)
        f_stub_kind = kind_to_str[stub_param.kind]
        assert param.kind in [stub_param.kind, Parameter.POSITIONAL_OR_KEYWORD,], (
            f"{param.name} is a {kind_to_str[param.kind]}, "
            f"but should be a {f_stub_kind} "
            f"(or at least a {kind_to_str[ParameterKind.POSITIONAL_OR_KEYWORD]})"
        )


def make_pretty_func(func_name: str, *args: Any, **kwargs: Any) -> str:
    f_sig = f"{func_name}("
    f_sig += ", ".join(str(a) for a in args)
    if len(kwargs) != 0:
        if len(args) != 0:
            f_sig += ", "
        f_sig += ", ".join(f"{k}={v}" for k, v in kwargs.items())
    f_sig += ")"
    return f_sig


# We test uninspectable signatures by passing valid, manually-defined arguments
# to the signature's function/method.
#
# Arguments which require use of the array module are specified as string
# expressions to be eval()'d on runtime. This is as opposed to just using the
# array module whilst setting up the tests, which is prone to halt the entire
# test suite if an array module doesn't support a given expression.
func_to_specified_args = defaultdict(
    dict,
    {
        "permute_dims": {"axes": 0},
        "reshape": {"shape": (1, 5)},
        "broadcast_to": {"shape": (1, 5)},
        "asarray": {"obj": [0, 1, 2, 3, 4]},
        "full_like": {"fill_value": 42},
        "matrix_power": {"n": 2},
    },
)
func_to_specified_arg_exprs = defaultdict(
    dict,
    {
        "stack": {"arrays": "[xp.ones((5,)), xp.ones((5,))]"},
        "iinfo": {"type": "xp.int64"},
        "finfo": {"type": "xp.float64"},
        "cholesky": {"x": "xp.asarray([[1, 0], [0, 1]], dtype=xp.float64)"},
        "inv": {"x": "xp.asarray([[1, 2], [3, 4]], dtype=xp.float64)"},
        "solve": {
            a: "xp.asarray([[1, 2], [3, 4]], dtype=xp.float64)" for a in ["x1", "x2"]
        },
    },
)
# We default most array arguments heuristically. As functions/methods work only
# with arrays of certain dtypes and shapes, we specify only supported arrays
# respective to the function.
casty_names = ["__bool__", "__int__", "__float__", "__complex__", "__index__"]
matrixy_names = [
    f.__name__
    for f in category_to_funcs["linear_algebra"] + extension_to_funcs["linalg"]
]
matrixy_names += ["__matmul__", "triu", "tril"]
for func_name, func in name_to_func.items():
    stub_sig = signature(func)
    array_argnames = set(stub_sig.parameters.keys()) & {"x", "x1", "x2", "other"}
    if func in array_methods:
        array_argnames.add("self")
    array_argnames -= set(func_to_specified_arg_exprs[func_name].keys())
    if len(array_argnames) > 0:
        in_dtypes = dh.func_in_dtypes[func_name]
        for dtype_name in ["float64", "bool", "int64", "complex128"]:
            # We try float64 first because uninspectable numerical functions
            # tend to support float inputs first-and-foremost (i.e. PyTorch)
            try:
                dtype = getattr(xp, dtype_name)
            except AttributeError:
                pass
            else:
                if dtype in in_dtypes:
                    if func_name in casty_names:
                        shape = ()
                    elif func_name in matrixy_names:
                        shape = (3, 3)
                    else:
                        shape = (5,)
                    fallback_array_expr = f"xp.ones({shape}, dtype=xp.{dtype_name})"
                    break
        else:
            warn(
                f"{dh.func_in_dtypes['{func_name}']}={in_dtypes} seemingly does "
                "not contain any assumed dtypes, so skipping specifying fallback array."
            )
            continue
        for argname in array_argnames:
            func_to_specified_arg_exprs[func_name][argname] = fallback_array_expr


def _test_uninspectable_func(func_name: str, func: Callable, stub_sig: Signature):
    params = list(stub_sig.parameters.values())

    if len(params) == 0:
        func()
        return

    uninspectable_msg = (
        f"Note {func_name}() is not inspectable so arguments are passed "
        "manually to test the signature."
    )

    argname_to_arg = func_to_specified_args[func_name]
    argname_to_expr = func_to_specified_arg_exprs[func_name]
    for argname, expr in argname_to_expr.items():
        assert argname not in argname_to_arg.keys()  # sanity check
        try:
            argname_to_arg[argname] = eval(expr, {"xp": xp})
        except Exception as e:
            pytest.skip(
                f"Exception occured when evaluating {argname}={expr}: {e}\n"
                f"{uninspectable_msg}"
            )

    posargs = []
    posorkw_args = {}
    kwargs = {}
    no_arg_msg = (
        "We have no argument specified for '{}'. Please ensure you're using "
        "the latest version of array-api-tests, then open an issue if one "
        f"doesn't already exist. {uninspectable_msg}"
    )
    for param in params:
        if param.kind == Parameter.POSITIONAL_ONLY:
            try:
                posargs.append(argname_to_arg[param.name])
            except KeyError:
                pytest.skip(no_arg_msg.format(param.name))
        elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            if param.default == Parameter.empty:
                try:
                    posorkw_args[param.name] = argname_to_arg[param.name]
                except KeyError:
                    pytest.skip(no_arg_msg.format(param.name))
            else:
                assert argname_to_arg[param.name]
                posorkw_args[param.name] = param.default
        elif param.kind == Parameter.KEYWORD_ONLY:
            assert param.default != Parameter.empty  # sanity check
            kwargs[param.name] = param.default
        else:
            assert param.kind in VAR_KINDS  # sanity check
            pytest.skip(no_arg_msg.format(param.name))
    if len(posorkw_args) == 0:
        func(*posargs, **kwargs)
    else:
        func(*posargs, **posorkw_args, **kwargs)
        # TODO: test all positional and keyword permutations of pos-or-kw args


def _test_func_signature(func: Callable, stub: FunctionType, is_method=False):
    stub_sig = signature(stub)
    # If testing against array, ignore 'self' arg in stub as it won't be present
    # in func (which should be a method).
    if is_method:
        stub_params = list(stub_sig.parameters.values())
        if stub_params[0].name == "self":
            del stub_params[0]
        stub_sig = Signature(
            parameters=stub_params, return_annotation=stub_sig.return_annotation
        )

    try:
        sig = signature(func)
    except ValueError:
        try:
            _test_uninspectable_func(stub.__name__, func, stub_sig)
        except Exception as e:
            raise e from None  # suppress parent exception for cleaner pytest output
    else:
        _test_inspectable_func(sig, stub_sig)


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
def test_array_method_signature(stub: FunctionType):
    x_expr = func_to_specified_arg_exprs[stub.__name__]["self"]
    try:
        x = eval(x_expr, {"xp": xp})
    except Exception as e:
        pytest.skip(f"Exception occured when evaluating x={x_expr}: {e}")
    assert hasattr(x, stub.__name__), f"{stub.__name__} not found in array object {x!r}"
    method = getattr(x, stub.__name__)
    _test_func_signature(method, stub, is_method=True)
