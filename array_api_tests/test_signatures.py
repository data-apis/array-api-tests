import inspect

import pytest

from ._array_module import mod, mod_name

from .function_stubs import elementwise

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
