import inspect

import pytest

from ._array_module import mod, mod_name, ones, eye, float64, bool, int64
from .pytest_helpers import raises, doesnt_raise
from .test_type_promotion import elementwise_function_input_types, operators_to_functions

from . import function_stubs


def stub_module(name):
    submodules = [m for m in dir(function_stubs) if
                  inspect.ismodule(getattr(function_stubs, m)) and not
                  m.startswith('_')]
    for m in submodules:
        if name in getattr(function_stubs, m).__all__:
            return m

def array_method(name):
    return stub_module(name) == 'array_object'

def function_category(name):
    return stub_module(name).rsplit('_', 1)[0].replace('_', ' ')

def example_argument(arg, func_name, dtype):
    """
    Get an example argument for the argument arg for the function func_name

    The full tests for function behavior is in other files. We just need to
    have an example input for each argument name that should work so that we
    can check if the argument is implemented at all.

    """
    # Note: for keyword arguments that have a default, this should be
    # different from the default, as the default argument is tested separately
    # (it can have the same behavior as the default, just not literally the
    # same value).
    known_args = dict(
        M=1,
        N=1,
        arrays=(ones((1, 3, 3), dtype=dtype), ones((1, 3, 3), dtype=dtype)),
        # These cannot be the same as each other, which is why all our test
        # arrays have to have at least 3 dimensions.
        axis1=2,
        axis2=2,
        axis=1,
        axes=(2, 1, 0),
        condition=ones((1, 3, 3), dtype=bool),
        correction=1.0,
        descending=True,
        dtype=float64,
        endpoint=False,
        fill_value=1.0,
        k=1,
        keepdims=True,
        key=0,
        num=2,
        offset=1,
        ord=1,
        other=ones((1, 3, 3), dtype=dtype),
        return_counts=True,
        return_index=True,
        return_inverse=True,
        self=ones((1, 3, 3), dtype=dtype),
        shape=(1, 3, 3),
        shift=1,
        sorted=False,
        stable=False,
        start=0,
        step=2,
        stop=1,
        value=0,
        x1=ones((1, 3, 3), dtype=dtype),
        x2=ones((1, 3, 3), dtype=dtype),
        x=ones((1, 3, 3), dtype=dtype),
    )

    if arg in known_args:
        # Special cases:

        # squeeze() requires an axis of size 1, but other functions such as
        # cross() require axes of size >1
        if func_name == 'squeeze' and arg == 'axis':
            return 0
        # ones() is not invertible
        elif func_name == 'inv' and arg == 'x':
            return eye(3)
        return known_args[arg]
    else:
        raise RuntimeError(f"Don't know how to test argument {arg}. Please update test_signatures.py")

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_has_names(name):
    if array_method(name):
        arr = ones((1,))
        if getattr(function_stubs.array_object, name) is None:
            assert hasattr(arr, name), f"The array object is missing the attribute {name}"
        else:
            assert hasattr(arr, name), f"The array object is missing the method {name}()"
    else:
        assert hasattr(mod, name), f"{mod_name} is missing the {function_category(name)} function {name}()"

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_function_positional_args(name):
    # Note: We can't actually test that positional arguments are
    # positional-only, as that would require knowing the argument name and
    # checking that it can't be used as a keyword argument. But argument name
    # inspection does not work for most array library functions that are not
    # written in pure Python (e.g., it won't work for numpy ufuncs).

    dtype = None
    if (name.startswith('__i') and name not in ['__int__', '__invert__']
        or name.startswith('__r') and name != '__rshift__'):
        n = operators_to_functions[name[:2] + name[3:]]
    else:
        n = operators_to_functions.get(name, name)
    if 'boolean' in elementwise_function_input_types.get(n, 'floating'):
        dtype = bool
    elif 'integer' in elementwise_function_input_types.get(n, 'floating'):
        dtype = int64

    if array_method(name):
        if name == '__bool__':
            _mod = ones((), dtype=bool)
        elif name == '__int__':
            _mod = ones((), dtype=int64)
        elif name == '__float__':
            _mod = ones((), dtype=float64)
        else:
            _mod = example_argument('self', name, dtype)
    else:
        _mod = mod

    if not hasattr(_mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    stub_func = getattr(function_stubs, name)
    if stub_func is None:
        # TODO: Can we make this skip the parameterization entirely?
        pytest.skip(f"{name} is not a function, skipping.")
    mod_func = getattr(_mod, name)
    argspec = inspect.getfullargspec(stub_func)
    args = argspec.args
    if name.startswith('__'):
        args = args[1:]
    nargs = [len(args)]
    if argspec.defaults:
        # The actual default values are checked in the specific tests
        nargs.extend([len(args) - i for i in range(1, len(argspec.defaults) + 1)])

    args = [example_argument(arg, name, dtype) for arg in args]
    if not args:
        args = [example_argument('x', name, dtype)]
    else:
        # Duplicate the last positional argument for the n+1 test.
        args = args + [args[-1]]

    for n in range(nargs[0]+2):
        if n in nargs:
            doesnt_raise(lambda: mod_func(*args[:n]))
        elif argspec.varargs:
            pass
        else:
            # NumPy ufuncs raise ValueError instead of TypeError
            raises((TypeError, ValueError), lambda: mod_func(*args[:n]), f"{name}() should not accept {n} positional arguments")

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_function_keyword_only_args(name):
    if array_method(name):
        _mod = ones((1,))
    else:
        _mod = mod

    if not hasattr(_mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    stub_func = getattr(function_stubs, name)
    if stub_func is None:
        # TODO: Can we make this skip the parameterization entirely?
        pytest.skip(f"{name} is not a function, skipping.")
    mod_func = getattr(_mod, name)
    argspec = inspect.getfullargspec(stub_func)
    args = argspec.args
    if name.startswith('__'):
        args = args[1:]
    kwonlyargs = argspec.kwonlyargs
    kwonlydefaults = argspec.kwonlydefaults
    dtype = None

    args = [example_argument(arg, name, dtype) for arg in args]

    for arg in kwonlyargs:
        value = example_argument(arg, name, dtype)
        # The "only" part of keyword-only is tested by the positional test above.
        doesnt_raise(lambda: mod_func(*args, **{arg: value}),
                     f"{name}() should accept the keyword-only argument {arg!r}")

        # Make sure the default is accepted. These tests are not granular
        # enough to test that the default is actually the default, i.e., gives
        # the same value if the keyword isn't passed. That is tested in the
        # specific function tests.
        if arg in kwonlydefaults:
            default_value = kwonlydefaults[arg]
            doesnt_raise(lambda: mod_func(*args, **{arg: default_value}),
                         f"{name}() should accept the default value {default_value!r} for the keyword-only argument {arg!r}")
