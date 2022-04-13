import pytest

from ..dtype_helpers import EqualityMapping


def test_raises_on_distinct_eq_key():
    with pytest.raises(ValueError):
        EqualityMapping({float("nan"): "foo"})


def test_raises_on_indistinct_eq_keys():
    class AlwaysEq:
        def __init__(self, hash):
            self._hash = hash

        def __eq__(self, other):
            return True

        def __hash__(self):
            return self._hash

    with pytest.raises(ValueError):
        EqualityMapping({AlwaysEq(0): "foo", AlwaysEq(1): "bar"})


def test_key_error():
    mapping = EqualityMapping({"foo": "bar"})
    with pytest.raises(KeyError):
        mapping["nonexistent key"]
