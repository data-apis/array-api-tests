import inspect

import pytest

from ._array_module import mod, mod_name

from .function_stubs import elementwise

def raises(exceptions, function, message=''):
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
    if not callable(function):
        raise ValueError("doesnt_raise should take a lambda")
    try:
        function()
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r}: {message}")
        raise AssertionError(f"Unexpected exception {e!r}")

@pytest.mark.parametrize('name', elementwise._names)
def test_has_names_elementwise(name):
    assert hasattr(mod, name), f"{mod_name} is missing the elementwise function {name}"

@pytest.mark.parametrize('name', elementwise._names)
def test_function_parameters(name):
    if not hasattr(mod, name):
        pytest.skip(f"{mod_name} does not have {name}, skipping.")
    stub_func = getattr(elementwise, name)
    mod_func = getattr(mod, name)
    signature = inspect.signature(stub_func)
