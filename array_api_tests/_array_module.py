import os
from importlib import import_module

# Replace this with a specific array module to test it, for example,
#
# import numpy as array_module
array_module = None

if array_module is None:
    if 'ARRAY_API_TESTS_MODULE' in os.environ:
        mod_name = os.environ['ARRAY_API_TESTS_MODULE']
        _module, _sub = mod_name.split('.', 1)
        mod = import_module(_module)
        if _sub:
            mod = getattr(mod, _sub)
    else:
        raise RuntimeError("No array module specified. Either edit _array_module.py or set the ARRAY_API_TESTS_MODULE environment variable")
else:
    mod = array_module
    mod_name = mod.__name__
# Names from the spec. This is what should actually be imported from this
# file.

array = mod.array
dtype = mod.dtype

add = mod.add
