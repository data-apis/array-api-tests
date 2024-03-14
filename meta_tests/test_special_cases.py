import math

from array_api_tests .test_special_cases import parse_result


def test_parse_result():
    check_result, _ = parse_result(
        "an implementation-dependent approximation to ``+3π/4``"
    )
    assert check_result(3 * math.pi / 4)
