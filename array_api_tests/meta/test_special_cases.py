import math

from ..test_special_cases import parse_result


def test_parse_result():
    s_result = "an implementation-dependent approximation to ``+3π/4``"
    assert parse_result(s_result).value == 3 * math.pi / 4
