import inspect

import pytest

from ._array_module import mod, mod_name, array, float64

from . import function_stubs

def function_category(name):
    func_stub = getattr(function_stubs, name)
    return func_stub.__module__.split('_')[0]

def example_argument(arg):
    """
    Get an example argument for the argument arg

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
        # These cannot be the same as each other, which is why all our test
        # arrays have to have at least 3 dimensions.
        axis1=2,
        axis2=2,
        axis=1,
        axes=(2, 1, 0),
        condition=array([[[True]]]),
        correction=1.0,
        dtype=float64,
        endpoint=False,
        fill_value=1.0,
        k=1,
        keepdims=True,
        num=2,
        offset=1,
        ord=1,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        shape=(1, 1, 1),
        sorted=False,
        start=0,
        step=2,
        stop=1,
        x1=array([[[1.]]]),
        x2=array([[[1.]]]),
        x=array([[[1.]]]),
    )

    if arg in known_args:
        return known_args[arg]
    else:
        raise RuntimeError(f"Don't know how to test argument {arg}. Please update test_signatures.py")

def raises(exceptions, function, message=''):
    """
    Like pytest.raises() except it allows custom error messages
    """
    try:
        function()
    except exceptions:
        return
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions}): {message}")
        raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions})")
    raise AssertionError(message)

def doesnt_raise(function, message=''):
    """
    The inverse of raises().

    Use doesnt_raise(function) to test that function() doesn't raise any
    exceptions.
    """
    if not callable(function):
        raise ValueError("doesnt_raise should take a lambda")
    try:
        function()
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r}: {message}")
        raise AssertionError(f"Unexpected exception {e!r}")

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_has_names(name):
    assert hasattr(mod, name), f"{mod_name} is missing the {function_category(name)} function {name}()"

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_function_positional_args(name):
    if not hasattr(mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    stub_func = getattr(function_stubs, name)
    mod_func = getattr(mod, name)
    args = inspect.getfullargspec(stub_func).args
    nargs = len(args)

    args = [example_argument(name) for name in args]
    # Duplicate the last positional argument for the n+1 test.
    args = args + [args[-1]]

    for n in range(nargs+2):
        if n == nargs:
            doesnt_raise(lambda: mod_func(*args[:n]))
        else:
            # NumPy ufuncs raise ValueError instead of TypeError
            raises((TypeError, ValueError), lambda: mod_func(*args[:n]), f"{name}() should not accept {n} positional arguments")

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_function_keyword_only_args(name):
    if not hasattr(mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    stub_func = getattr(function_stubs, name)
    mod_func = getattr(mod, name)
    args = inspect.getfullargspec(stub_func).args
    kwonlyargs = inspect.getfullargspec(stub_func).kwonlyargs

    args = [example_argument(name) for name in args]

    for arg in kwonlyargs:
        value = example_argument(arg)
        # The "only" part of keyword-only is tested by the positional test above.
        doesnt_raise(lambda: mod_func(*args, **{arg: value}),
                     f"{name}() should accept the keyword-only argument {arg!r}")
