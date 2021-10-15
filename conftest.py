from hypothesis import settings

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


def pytest_configure(config):
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
