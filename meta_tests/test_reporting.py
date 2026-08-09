from array_api_tests.dtype_helpers import EqualityMapping
import reporting


class MLXDtype:
    def __eq__(self, other):
        if other is None:
            raise TypeError("MLX dtype equality does not support None")
        return self is other


def test_to_json_serializable_none_does_not_compare_to_dtypes(monkeypatch):
    dtype = MLXDtype()
    monkeypatch.setattr(reporting, "dtype_to_name", EqualityMapping([(dtype, "float16")]))

    assert reporting.to_json_serializable(None) is None
