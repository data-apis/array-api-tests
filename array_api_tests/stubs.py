import inspect
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Dict, List

from . import api_version

__all__ = [
    "name_to_func",
    "array_methods",
    "array_attributes",
    "category_to_funcs",
    "EXTENSIONS",
    "extension_to_funcs",
]

spec_module = "_" + api_version.replace('.', '_')

spec_dir = Path(__file__).parent.parent / "array-api" / "spec" / api_version / "API_specification"
assert spec_dir.exists(), f"{spec_dir} not found - try `git submodule update --init`"
sigs_dir = Path(__file__).parent.parent / "array-api" / "src" / "array_api_stubs" / spec_module
assert sigs_dir.exists()

sigs_abs_path: str = str(sigs_dir.parent.parent.resolve())
sys.path.append(sigs_abs_path)
assert find_spec(f"array_api_stubs.{spec_module}") is not None

name_to_mod: Dict[str, ModuleType] = {}
for path in sigs_dir.glob("*.py"):
    name = path.name.replace(".py", "")
    name_to_mod[name] = import_module(f"array_api_stubs.{spec_module}.{name}")

array = name_to_mod["array_object"].array
array_methods = [
    f for n, f in inspect.getmembers(array, predicate=inspect.isfunction)
    if n != "__init__"  # probably exists for Sphinx
]
array_attributes = [
    n for n, f in inspect.getmembers(array, predicate=lambda x: isinstance(x, property))
]

category_to_funcs: Dict[str, List[FunctionType]] = {}
for name, mod in name_to_mod.items():
    if name.endswith("_functions") or name == "info":  # info functions file just named info.py
        category = name.replace("_functions", "")
        objects = [getattr(mod, name) for name in mod.__all__]
        assert all(isinstance(o, FunctionType) for o in objects)  # sanity check
        category_to_funcs[category] = objects

all_funcs = []
for funcs in [array_methods, *category_to_funcs.values()]:
    all_funcs.extend(funcs)
name_to_func: Dict[str, FunctionType] = {f.__name__: f for f in all_funcs}

EXTENSIONS: List[str] = ["linalg"]  # TODO: add "fft" once stubs available
extension_to_funcs: Dict[str, List[FunctionType]] = {}
for ext in EXTENSIONS:
    mod = name_to_mod[ext]
    objects = [getattr(mod, name) for name in mod.__all__]
    assert all(isinstance(o, FunctionType) for o in objects)  # sanity check
    funcs = []
    for func in objects:
        if "Alias" in func.__doc__:
            funcs.append(name_to_func[func.__name__])
        else:
            funcs.append(func)
    extension_to_funcs[ext] = funcs

for funcs in extension_to_funcs.values():
    for func in funcs:
        if func.__name__ not in name_to_func.keys():
            name_to_func[func.__name__] = func

# sanity check public attributes are not empty
for attr in __all__:
    assert len(locals()[attr]) != 0, f"{attr} is empty"
