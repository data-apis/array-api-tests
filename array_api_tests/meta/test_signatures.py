from inspect import signature

import pytest

from ..test_signatures import _test_inspectable_func


@pytest.mark.xfail("not implemented")
def test_kwonly():
    def func(*, foo=None, bar=None):
        pass

    sig = signature(func)
    _test_inspectable_func(sig, sig)

    def reversed_func(*, bar=None, foo=None):
        pass

    reversed_sig = signature(reversed_func)
    _test_inspectable_func(sig, reversed_sig)
