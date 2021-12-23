from functools import lru_cache
from pathlib import Path

from pytest import mark
from hypothesis import settings

from xptests import _array_module as xp
from xptests._array_module import _UndefinedStub


settings.register_profile('xp_default', deadline=800)


def pytest_addoption(parser):
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
    # enable/disable extensions
    parser.addoption(
        '--enable-extension',
        metavar='ext',
        nargs='+',
        default=[],
        help='enable testing for Array API extension(s)',
    )
    parser.addoption(
        '--disable-extension',
        metavar='ext',
        nargs='+',
        default=[],
        help='disable testing for Array API extension(s)',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'xp_extension(ext): tests an Array API extension'
    )
    # Hypothesis
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


xfail_ids = []
xfails_path = Path(__file__).parent / 'xfails.txt'
if xfails_path.exists():
    with open(xfails_path) as f:
        for line in f:
            if line.startswith('xptests'):
                id_ = line.strip('\n')
                xfail_ids.append(id_)


def pytest_collection_modifyitems(config, items):
    enabled_exts = config.getoption('--enable-extension')
    disabled_exts = config.getoption('--disable-extension')
    for ext in enabled_exts:
        if ext in disabled_exts:
            raise ValueError(f'{ext=} both enabled and disabled')
    for item in items:
        # enable/disable extensions
        try:
            ext_mark = next(m for m in item.iter_markers() if m.name == 'xp_extension')
            ext = ext_mark.args[0]
            if ext in disabled_exts:
                item.add_marker(
                    mark.skip(reason=f'{ext} disabled in --disable-extensions')
                )
            elif not ext in enabled_exts and not xp_has_ext(ext):
                item.add_marker(mark.skip(reason=f'{ext} not found in array module'))
        except StopIteration:
            pass
        # workflow xfail_ids
        for id_ in xfail_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(mark.xfail(reason='xfails.txt'))
