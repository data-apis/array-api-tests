from functools import lru_cache

from hypothesis import settings
from pytest import mark

from . import _array_module as xp
from ._array_module import _UndefinedStub


settings.register_profile('xp_default', deadline=800)


def pytest_addoption(parser):
    # Enable extensions
    parser.addoption(
        '--ext',
        '--disable-extensions',
        nargs='+',
        default=[],
        help='disable testing for Array API extensions',
    )
    # Hypothesis max examples
    # See https://github.com/HypothesisWorks/hypothesis/issues/2434
    parser.addoption(
        '--hypothesis-max-examples',
        '--max-examples',
        action='store',
        default=None,
        help='set the Hypothesis max_examples setting',
    )
    # Hypothesis deadline
    parser.addoption(
        '--hypothesis-disable-deadline',
        '--disable-deadline',
        action='store_true',
        help='disable the Hypothesis deadline',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'xp_extension(ext): tests an Array API extension'
    )
    # Configure Hypothesis
    hypothesis_max_examples = config.getoption('--hypothesis-max-examples')
    disable_deadline = config.getoption('--hypothesis-disable-deadline')
    profile_settings = {}
    if hypothesis_max_examples is not None:
        profile_settings['max_examples'] = int(hypothesis_max_examples)
    if disable_deadline is not None:
        profile_settings['deadline'] = None
    if profile_settings:
        settings.register_profile('xp_override', **profile_settings)
        settings.load_profile('xp_override')
    else:
        settings.load_profile('xp_default')


@lru_cache
def xp_has_ext(ext: str) -> bool:
    try:
        return not isinstance(getattr(xp, ext), _UndefinedStub)
    except AttributeError:
        return False


def pytest_collection_modifyitems(config, items):
    disabled_exts = config.getoption('--disable-extensions')
    for item in items:
        if 'xp_extension' in item.keywords:
            ext = item.keywords['xp_extension'].args[0]
            if ext in disabled_exts:
                item.add_marker(
                    mark.skip(reason=f'{ext} disabled in --disable-extensions')
                )
            elif not xp_has_ext(ext):
                item.add_marker(mark.skip(reason=f'{ext} not found in array module'))
