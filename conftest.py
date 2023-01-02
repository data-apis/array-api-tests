from functools import lru_cache
from pathlib import Path

from hypothesis import settings
from pytest import mark

from array_api_tests import _array_module as xp
from array_api_tests._array_module import _UndefinedStub

from reporting import pytest_metadata, pytest_json_modifyreport, add_extra_json_metadata # noqa

settings.register_profile("xp_default", deadline=800)

def pytest_addoption(parser):
    # Hypothesis max examples
    # See https://github.com/HypothesisWorks/hypothesis/issues/2434
    parser.addoption(
        "--hypothesis-max-examples",
        "--max-examples",
        action="store",
        default=None,
        help="set the Hypothesis max_examples setting",
    )
    # Hypothesis deadline
    parser.addoption(
        "--hypothesis-disable-deadline",
        "--disable-deadline",
        action="store_true",
        help="disable the Hypothesis deadline",
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
    parser.addoption(
        "--ci",
        action="store_true",
        help="run just the tests appropriate for CI",
    )
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
    config.addinivalue_line("markers", "ci: primary test")
    # Hypothesis
    hypothesis_max_examples = config.getoption("--hypothesis-max-examples")
    disable_deadline = config.getoption("--hypothesis-disable-deadline")
    profile_settings = {}
    if hypothesis_max_examples is not None:
        profile_settings["max_examples"] = int(hypothesis_max_examples)
    if disable_deadline is not None:
        profile_settings["deadline"] = None
    if profile_settings:
        settings.register_profile("xp_override", **profile_settings)
        settings.load_profile("xp_override")
    else:
        settings.load_profile("xp_default")


@lru_cache
def xp_has_ext(ext: str) -> bool:
    try:
        return not isinstance(getattr(xp, ext), _UndefinedStub)
    except AttributeError:
        return False


def pytest_collection_modifyitems(config, items):
    skips_file = skips_path = config.getoption('--skips-file')
    if skips_file is None:
        skips_file = Path(__file__).parent / "skips.txt"
        if skips_file.exists():
            skips_path = skips_file

    skip_ids = []
    if skips_path:
        with open(skips_path) as f:
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
        with open(xfails_path) as f:
            for line in f:
                if line.startswith("array_api_tests"):
                    id_ = line.strip("\n")
                    xfail_ids.append(id_)

    disabled_exts = config.getoption("--disable-extension")
    disabled_dds = config.getoption("--disable-data-dependent-shapes")
    ci = config.getoption("--ci")
    for item in items:
        markers = list(item.iter_markers())
        # skip if specified in skips file
        for id_ in skip_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(mark.skip(reason="skips file"))
                break
        # xfail if specified in xfails file
        for id_ in xfail_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(mark.xfails(reason="xfails file"))
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
        # skip if test not appropriate for CI
        if ci:
            ci_mark = next((m for m in markers if m.name == "ci"), None)
            if ci_mark is None:
                item.add_marker(mark.skip(reason="disabled via --ci"))
