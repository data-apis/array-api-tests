import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Dict, List

__all__ = ["category_to_funcs", "array", "extension_to_funcs"]


spec_dir = Path(__file__).parent / "array-api" / "spec" / "API_specification"
assert spec_dir.exists(), f"{spec_dir} not found - try `git pull --recurse-submodules`"
sigs_dir = spec_dir / "signatures"
assert sigs_dir.exists()

spec_abs_path: str = str(spec_dir.resolve())
sys.path.append(spec_abs_path)
assert find_spec("signatures") is not None

name_to_mod: Dict[str, ModuleType] = {}
for path in sigs_dir.glob("*.py"):
    name = path.name.replace(".py", "")
    name_to_mod[name] = import_module(f"signatures.{name}")


category_to_funcs: Dict[str, List[FunctionType]] = {}
for name, mod in name_to_mod.items():
    if name.endswith("_functions"):
        category = name.replace("_functions", "")
        objects = [getattr(mod, name) for name in mod.__all__]
        assert all(isinstance(o, FunctionType) for o in objects)
        category_to_funcs[category] = objects


array = name_to_mod["array_object"].array


EXTENSIONS = ["linalg"]
extension_to_funcs: Dict[str, List[FunctionType]] = {}
for ext in EXTENSIONS:
    mod = name_to_mod[ext]
    objects = [getattr(mod, name) for name in mod.__all__]
    assert all(isinstance(o, FunctionType) for o in objects)
    extension_to_funcs[ext] = objects


sys.path.remove(spec_abs_path)
