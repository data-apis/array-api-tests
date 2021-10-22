from ..test_signatures import extension_module


def test_extension_module_is_extension():
    assert extension_module('linalg')


def test_extension_func_is_not_extension():
    assert not extension_module('linalg.cross')
