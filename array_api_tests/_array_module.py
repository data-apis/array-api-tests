import os
from importlib import import_module

# Replace this with a specific array module to test it, for example,
#
# import numpy as array_module
array_module = None

if array_module is None:
    if 'ARRAY_API_TESTS_MODULE' in os.environ:
        mod_name = os.environ['ARRAY_API_TESTS_MODULE']
        _module, _sub = mod_name, None
        if '.' in mod_name:
            _module, _sub = mod_name.split('.', 1)
        mod = import_module(_module)
        if _sub:
            try:
                mod = getattr(mod, _sub)
            except AttributeError:
                # _sub may be a submodule that needs to be imported. WE can't
                # do this in every case because some array modules are not
                # submodules that can be imported (like mxnet.nd).
                mod = import_module(mod_name)
    else:
        raise RuntimeError("No array module specified. Either edit _array_module.py or set the ARRAY_API_TESTS_MODULE environment variable")
else:
    mod = array_module
    mod_name = mod.__name__
# Names from the spec. This is what should actually be imported from this
# file.

try:
    array = mod.array
except AttributeError:
    def array(*args, **kwargs):
        raise AssertionError(f"array is not defined in {mod_name}")

try:
    dtype = mod.dtype
except AttributeError:
    def dtype(*args, **kwargs):
        raise AssertionError(f"dtype is not defined in {mod_name}")

try:
    float64 = mod.float64
except AttributeError:
    def float64(*args, **kwargs):
        raise AssertionError(f"dtype is not defined in {mod_name}")

try:
    add = mod.add
except AttributeError:
    def add(*args, **kwargs):
        raise AssertionError(f"add is not defined in {mod_name}")
