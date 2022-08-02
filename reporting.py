from array_api_tests.test_operators_and_elementwise_functions import (UnaryParamContext, BinaryParamContext)
from array_api_tests.dtype_helpers import dtype_to_name
from array_api_tests import _array_module as xp

from pytest import mark, fixture

def to_json_serializable(o):
    if o in dtype_to_name:
        return dtype_to_name[o]
    if isinstance(o, UnaryParamContext):
        return {'func_name': o.func_name}
    if isinstance(o, BinaryParamContext):
        return {
            'func_name': o.func_name,
            'left_sym': o.left_sym,
            'right_sym': o.right_sym,
            'right_is_scalar': o.right_is_scalar,
            'res_name': o.res_name,
        }
    if isinstance(o, dict):
        return {to_json_serializable(k): to_json_serializable(v) for k, v in o.items()}
    if isinstance(o, tuple):
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

@fixture(autouse=True)
def add_api_name_to_metadata(request, json_metadata):
    """
    Additional per-test metadata for --json-report
    """
    test_module = request.module.__name__
    if test_module.startswith('array_api_tests.meta'):
        return

    test_function = request.function.__name__
    assert test_function.startswith('test_'), 'unexpected test function name'

    if test_module == 'array_api_tests.test_has_names':
        array_api_function_name = None
    else:
        array_api_function_name = test_function[len('test_'):]

    json_metadata['test_module'] = test_module
    json_metadata['test_function'] = test_function
    json_metadata['array_api_function_name'] = array_api_function_name

    if hasattr(request.node, 'callspec'):
        json_metadata['params'] = to_json_serializable(request.node.callspec.params)
