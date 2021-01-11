from hypothesis import settings

# Add a --hypothesis-max-examples flag to pytest. See
# https://github.com/HypothesisWorks/hypothesis/issues/2434#issuecomment-630309150

def pytest_addoption(parser):
    # Add an option to change the Hypothesis max_examples setting.
    parser.addoption(
        "--hypothesis-max-examples",
        "--max-examples",
        action="store",
        default=None,
        help="set the Hypothesis max_examples setting",
    )

    # Add an option to disable the Hypothesis deadline
    parser.addoption(
        "--hypothesis-disable-deadline",
        "--disable-deadline",
        action="store_true",
        help="disable the Hypothesis deadline",
    )


def pytest_configure(config):
    # Set Hypothesis max_examples.
    hypothesis_max_examples = config.getoption("--hypothesis-max-examples")
    disable_deadline = config.getoption('--hypothesis-disable-deadline')
    profile_settings = {}
    if hypothesis_max_examples is not None:
        profile_settings['max_examples'] = int(hypothesis_max_examples)
    if disable_deadline is not None:
        profile_settings['deadline'] = None
    if profile_settings:
        import hypothesis

        hypothesis.settings.register_profile(
            "array-api-tests-hypothesis-overridden", **profile_settings,
        )

        hypothesis.settings.load_profile("array-api-tests-hypothesis-overridden")


settings.register_profile('array_api_tests_hypothesis_profile', deadline=800)
settings.load_profile('array_api_tests_hypothesis_profile')
