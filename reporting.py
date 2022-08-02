from array_api_tests.dtype_helpers import dtype_to_name
from array_api_tests import _array_module as xp
from array_api_tests import __version__

from types import BuiltinFunctionType, FunctionType
import dataclasses
import json

from hypothesis.strategies import SearchStrategy

from pytest import mark, fixture

def to_json_serializable(o):
    if o in dtype_to_name:
        return dtype_to_name[o]
    if isinstance(o, (BuiltinFunctionType, FunctionType)):
        return o.__name__
    if dataclasses.is_dataclass(o):
        return to_json_serializable(dataclasses.asdict(o))
    if isinstance(o, SearchStrategy):
        return repr(o)
    if isinstance(o, dict):
        return {to_json_serializable(k): to_json_serializable(v) for k, v in o.items()}
    if isinstance(o, tuple):
        if hasattr(o, '_asdict'): # namedtuple
            return to_json_serializable(o._asdict())
        return tuple(to_json_serializable(i) for i in o)
    if isinstance(o, list):
        return [to_json_serializable(i) for i in o]

    return o

@mark.optionalhook
def pytest_metadata(metadata):
    """
    Additional global metadata for --json-report.
    """
    metadata['array_api_tests_module'] = xp.mod_name
    metadata['array_api_tests_version'] = __version__

# This is dynamically decorated as a fixture in pytest_collection_modifyitems
# when --json-report is used.
def add_api_name_to_metadata(request, json_metadata):
    """
    Additional per-test metadata for --json-report
    """
    def add_metadata(name, obj):
        obj = to_json_serializable(obj)
        # Ensure everything is JSON serializable. If this errors, it means the
        # given type needs to be added to to_json_serializable above.
        json.dumps(obj)
        json_metadata[name] = obj

    test_module = request.module.__name__
    if test_module.startswith('array_api_tests.meta'):
        return

    test_function = request.function.__name__
    assert test_function.startswith('test_'), 'unexpected test function name'

    if test_module == 'array_api_tests.test_has_names':
        array_api_function_name = None
    else:
        array_api_function_name = test_function[len('test_'):]

    add_metadata('test_module', test_module)
    add_metadata('test_function', test_function)
    add_metadata('array_api_function_name', array_api_function_name)

    if hasattr(request.node, 'callspec'):
        params = request.node.callspec.params
        add_metadata('params', params)

    def finalizer():
        # TODO: This metadata is all in the form of error strings. It might be
        # nice to extract the hypothesis failing inputs directly somehow.
        if hasattr(request.node, 'hypothesis_report_information'):
            add_metadata('hypothesis_report_information', request.node.hypothesis_report_information)
        if hasattr(request.node, 'hypothesis_statistics'):
            add_metadata('hypothesis_statistics', request.node.hypothesis_statistics)

    request.addfinalizer(finalizer)
