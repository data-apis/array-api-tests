from ..test_signatures import extension_module
from ..conftest import xp_has_ext


def test_extension_module_is_extension():
    assert extension_module('linalg')


def test_extension_func_is_not_extension():
    assert not extension_module('linalg.cross')


def test_xp_has_ext():
    assert not xp_has_ext('nonexistent_extension')
