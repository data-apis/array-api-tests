from functools import lru_cache
from pathlib import Path
import argparse
import warnings
import os

from hypothesis import settings
from hypothesis.errors import InvalidArgument
from pytest import mark

from array_api_tests import _array_module as xp
from array_api_tests import api_version
from array_api_tests._array_module import _UndefinedStub
from array_api_tests.stubs import EXTENSIONS
from array_api_tests import xp_name, xp as array_module

from reporting import pytest_metadata, pytest_json_modifyreport, add_extra_json_metadata # noqa

def pytest_report_header(config):
    disabled_extensions = config.getoption("--disable-extension")
    enabled_extensions = sorted({
        ext for ext in EXTENSIONS + ['fft'] if ext not in disabled_extensions and xp_has_ext(ext)
    })

    try:
        array_module_version = array_module.__version__
    except AttributeError:
        array_module_version = "version unknown"

    # make it easier to catch typos in environment variables (ARRAY_API_*** instead of ARRAY_API_TESTS_*** etc)
    env_vars = "\n".join([f"{k} = {v}" for k, v in os.environ.items() if 'ARRAY_API' in k])
    env_vars = f"Environment variables:\n{'-'*22}\n{env_vars}\n\n"

    header1 = f"Array API Tests Module: {xp_name} ({array_module_version}). API Version: {api_version}. Enabled Extensions: {', '.join(enabled_extensions)}"
    return env_vars + header1

def pytest_addoption(parser):
    # Hypothesis max examples
    # See https://github.com/HypothesisWorks/hypothesis/issues/2434
    parser.addoption(
        "--hypothesis-max-examples",
        "--max-examples",
        action="store",
        default=100,
        type=int,
        help="set the Hypothesis max_examples setting",
    )
    # Hypothesis deadline
    parser.addoption(
        "--hypothesis-disable-deadline",
        "--disable-deadline",
        action="store_true",
        help="disable the Hypothesis deadline",
    )
    # Hypothesis derandomize
    parser.addoption(
        "--hypothesis-derandomize",
        "--derandomize",
        action="store_true",
        help="set the Hypothesis derandomize parameter",
    )
    # disable extensions
    parser.addoption(
        "--disable-extension",
        metavar="ext",
        nargs="+",
        default=[],
        help="disable testing for Array API extension(s)",
    )
    # data-dependent shape
    parser.addoption(
        "--disable-data-dependent-shapes",
        "--disable-dds",
        action="store_true",
        help="disable testing functions with output shapes dependent on input",
    )
    # CI
    parser.addoption("--ci", action="store_true", help=argparse.SUPPRESS )  # deprecated
    parser.addoption(
        "--skips-file",
        action="store",
        help="file with tests to skip. Defaults to skips.txt"
    )
    parser.addoption(
        "--xfails-file",
        action="store",
        help="file with tests to skip. Defaults to xfails.txt"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "xp_extension(ext): tests an Array API extension"
    )
    config.addinivalue_line(
        "markers", "data_dependent_shapes: output shapes are dependent on inputs"
    )
    config.addinivalue_line(
        "markers",
        "min_version(api_version): run when greater or equal to api_version",
    )
    config.addinivalue_line(
        "markers",
        "unvectorized: asserts against values via element-wise iteration (not performative!)",
    )
    # Hypothesis
    deadline = None if config.getoption("--hypothesis-disable-deadline") else 800
    settings.register_profile(
        "array-api-tests",
        max_examples=config.getoption("--hypothesis-max-examples"),
        derandomize=config.getoption("--hypothesis-derandomize"),
        deadline=deadline,
    )
    settings.load_profile("array-api-tests")
    # CI
    if config.getoption("--ci"):
        warnings.warn(
            "Custom pytest option --ci is deprecated as any tests not for CI "
            "are now located in meta_tests/"
        )


@lru_cache
def xp_has_ext(ext: str) -> bool:
    try:
        return not isinstance(getattr(xp, ext), _UndefinedStub)
    except AttributeError:
        return False


def check_id_match(id_, pattern):
    id_ = id_.removeprefix('array-api-tests/')

    if id_ == pattern:
        return True
    
    if id_.startswith(pattern.removesuffix("/") + "/"):
        return True
    
    if pattern.endswith(".py") and id_.startswith(pattern):
        return True
    
    if id_.split("::", maxsplit=2)[0] == pattern:
        return True
    
    if id_.split("[", maxsplit=2)[0] == pattern:
        return True

    return False


def get_xfail_mark():
    """Skip or xfail tests from the xfails-file.txt."""
    m = os.environ.get("ARRAY_API_TESTS_XFAIL_MARK", "xfail")
    if m == "xfail":
        return mark.xfail
    elif m == "skip":
        return mark.skip
    else:
        raise ValueError(
            f'ARRAY_API_TESTS_XFAIL_MARK value should be one of "skip" or "xfail" '
            f'got {m} instead.'
        )


def pytest_collection_modifyitems(config, items):
    # 1. Prepare for iterating over items
    # -----------------------------------

    skips_file = skips_path = config.getoption('--skips-file')
    if skips_file is None:
        skips_file = Path(__file__).parent / "skips.txt"
        if skips_file.exists():
            skips_path = skips_file

    skip_ids = []
    if skips_path:
        with open(os.path.expanduser(skips_path)) as f:
            for line in f:
                if line.startswith("array_api_tests"):
                    id_ = line.strip("\n")
                    skip_ids.append(id_)

    xfails_file = xfails_path = config.getoption('--xfails-file')
    if xfails_file is None:
        xfails_file = Path(__file__).parent / "xfails.txt"
        if xfails_file.exists():
            xfails_path = xfails_file

    xfail_ids = []
    if xfails_path:
        with open(os.path.expanduser(xfails_path)) as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                id_ = line.strip("\n")
                xfail_ids.append(id_)

    skip_id_matched = {id_: False for id_ in skip_ids}
    xfail_id_matched = {id_: False for id_ in xfail_ids}

    disabled_exts = config.getoption("--disable-extension")
    disabled_dds = config.getoption("--disable-data-dependent-shapes")
    unvectorized_max_examples = max(1, config.getoption("--hypothesis-max-examples")//10)

    # 2. Iterate through items and apply markers accordingly
    # ------------------------------------------------------

    xfail_mark = get_xfail_mark()

    for item in items:
        markers = list(item.iter_markers())
        # skip if specified in skips file
        for id_ in skip_ids:
            if check_id_match(item.nodeid, id_):
                item.add_marker(mark.skip(reason=f"--skips-file ({skips_file})"))
                skip_id_matched[id_] = True
                break
        # xfail if specified in xfails file
        for id_ in xfail_ids:
            if check_id_match(item.nodeid, id_):
                item.add_marker(xfail_mark(reason=f"--xfails-file ({xfails_file})"))
                xfail_id_matched[id_] = True
                break
        # skip if disabled or non-existent extension
        ext_mark = next((m for m in markers if m.name == "xp_extension"), None)
        if ext_mark is not None:
            ext = ext_mark.args[0]
            if ext in disabled_exts:
                item.add_marker(
                    mark.skip(reason=f"{ext} disabled in --disable-extensions")
                )
            elif not xp_has_ext(ext):
                item.add_marker(mark.skip(reason=f"{ext} not found in array module"))
        # skip if disabled by dds flag
        if disabled_dds:
            for m in markers:
                if m.name == "data_dependent_shapes":
                    item.add_marker(
                        mark.skip(reason="disabled via --disable-data-dependent-shapes")
                    )
                    break
        # skip if test is for greater api_version
        ver_mark = next((m for m in markers if m.name == "min_version"), None)
        if ver_mark is not None:
            min_version = ver_mark.args[0]
            if api_version < min_version:
                item.add_marker(
                    mark.skip(
                        reason=f"requires ARRAY_API_TESTS_VERSION >= {min_version}"
                    )
                )
        # reduce max generated Hypothesis example for unvectorized tests
        if any(m.name == "unvectorized" for m in markers):
            # TODO: limit generated examples when settings already applied

            # account for both test functions and test methods of test classes
            test_func = getattr(item.obj, "__func__", item.obj)

            # https://groups.google.com/g/hypothesis-users/c/6K6WPR5knAs
            if not hasattr(test_func, "_hypothesis_internal_settings_applied"):
                try:
                    sett = settings(max_examples=unvectorized_max_examples)
                    test_func._hypothesis_internal_use_settings = sett
                except InvalidArgument as e:
                    warnings.warn(
                        f"Tried decorating {item.name} with settings() but got "
                        f"hypothesis.errors.InvalidArgument: {e}"
                    )


    # 3. Warn on bad skipped/xfailed ids
    # ----------------------------------

    bad_ids_end_msg = (
        "Note the relevant tests might not have been collected by pytest, or "
        "another specified id might have already matched a test."
    )
    bad_skip_ids = [id_ for id_, matched in skip_id_matched.items() if not matched]
    if bad_skip_ids:
        f_bad_ids = "\n".join(f"    {id_}" for id_ in bad_skip_ids)
        warnings.warn(
            f"{len(bad_skip_ids)} ids in skips file don't match any collected tests: \n"
            f"{f_bad_ids}\n"
            f"(skips file: {skips_file})\n"
            f"{bad_ids_end_msg}"
        )
    bad_xfail_ids = [id_ for id_, matched in xfail_id_matched.items() if not matched]
    if bad_xfail_ids:
        f_bad_ids = "\n".join(f"    {id_}" for id_ in bad_xfail_ids)
        warnings.warn(
            f"{len(bad_xfail_ids)} ids in xfails file don't match any collected tests: \n"
            f"{f_bad_ids}\n"
            f"(xfails file: {xfails_file})\n"
            f"{bad_ids_end_msg}"
        )
