import inspect

import pytest

from ._array_module import mod, mod_name

from .function_stubs import elementwise

@pytest.mark.parametrize('name', [
    pytest.param(name,
                 marks=pytest.mark.dependency(name=f"{name}_exists"))
    for name in elementwise._names])
def test_has_names_elementwise(name):
    assert hasattr(mod, name), f"{mod_name} is missing the elementwise function {name}"


@pytest.mark.parametrize('name', [
    pytest.param(name,
                 marks=pytest.mark.dependency(name=f"{name}_parameters",
                                             depends=[f"{name}_exists"]))
    for name in elementwise._names])
def test_function_parameters(name):
    stub_func = getattr(elementwise, name)
    mod_func = getattr(mod, name)
    signature = inspect.signature(stub_func)
