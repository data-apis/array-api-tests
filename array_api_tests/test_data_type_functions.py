import struct
from typing import Union

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import DataType


# TODO: test with complex dtypes
def non_complex_dtypes():
    return xps.boolean_dtypes() | hh.real_dtypes


def float32(n: Union[int, float]) -> float:
    return struct.unpack("!f", struct.pack("!f", float(n)))[0]


def _float_match_complex(complex_dtype):
    if complex_dtype == xp.complex64:
        return xp.float32
    elif complex_dtype == xp.complex128:
        return xp.float64
    else:
        return dh.default_float


@given(
    x_dtype=hh.all_dtypes,
    dtype=hh.all_dtypes,
    kw=hh.kwargs(copy=st.booleans()),
    data=st.data(),
)
def test_astype(x_dtype, dtype, kw, data):
    _complex_dtypes = (xp.complex64, xp.complex128)

    if xp.bool in (x_dtype, dtype):
        elements_strat = hh.from_dtype(x_dtype)
    else:

        if dh.is_int_dtype(x_dtype):
            cast = int
        elif x_dtype in (xp.float32, xp.complex64):
            cast = float32
        else:
            cast = float

        real_dtype = x_dtype
        if x_dtype in _complex_dtypes:
            real_dtype = _float_match_complex(x_dtype)
        m1, M1 = dh.dtype_ranges[real_dtype]

        real_dtype = dtype
        if dtype in _complex_dtypes:
            real_dtype = _float_match_complex(x_dtype)
        m2, M2 = dh.dtype_ranges[real_dtype]

        min_value = cast(max(m1, m2))
        max_value = cast(min(M1, M2))

        elements_strat = hh.from_dtype(
            x_dtype,
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    x = data.draw(
        hh.arrays(dtype=x_dtype, shape=hh.shapes(), elements=elements_strat), label="x"
    )

    # according to the spec, "Casting a complex floating-point array to a real-valued
    # data type should not be permitted."
    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.astype.html#astype
    assume(not ((x_dtype in _complex_dtypes) and (dtype not in _complex_dtypes)))

    out = xp.astype(x, dtype, **kw)

    ph.assert_kw_dtype("astype", kw_dtype=dtype, out_dtype=out.dtype)
    ph.assert_shape("astype", out_shape=out.shape, expected=x.shape, kw=kw)
    # TODO: test values
    # Check copy is respected (only if input dtype is same as output dtype)
    if dtype == x_dtype:
        ph.assert_kw_copy("astype", x, out, data, kw.get("copy", None))


@given(
    shapes=st.integers(1, 5).flatmap(hh.mutually_broadcastable_shapes), data=st.data()
)
def test_broadcast_arrays(shapes, data):
    arrays = []
    for c, shape in enumerate(shapes, 1):
        x = data.draw(hh.arrays(dtype=hh.all_dtypes, shape=shape), label=f"x{c}")
        arrays.append(x)

    out = xp.broadcast_arrays(*arrays)

    expected_shape = sh.broadcast_shapes(*shapes)
    for i, x in enumerate(arrays):
        ph.assert_dtype(
            "broadcast_arrays",
            in_dtype=x.dtype,
            out_dtype=out[i].dtype,
            repr_name=f"out[{i}].dtype"
        )
        ph.assert_result_shape(
            "broadcast_arrays",
            in_shapes=shapes,
            out_shape=out[i].shape,
            expected=expected_shape,
            repr_name=f"out[{i}].shape",
        )
    # TODO: test values


@given(x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes()), data=st.data())
def test_broadcast_to(x, data):
    shape = data.draw(
        hh.mutually_broadcastable_shapes(1, base_shape=x.shape)
        .map(lambda S: S[0])
        .filter(lambda s: sh.broadcast_shapes(x.shape, s) == s),
        label="shape",
    )

    out = xp.broadcast_to(x, shape)

    ph.assert_dtype("broadcast_to", in_dtype=x.dtype, out_dtype=out.dtype)
    ph.assert_shape("broadcast_to", out_shape=out.shape, expected=shape)
    # TODO: test values


@given(_from=hh.all_dtypes, to=hh.all_dtypes)
def test_can_cast(_from, to):
    out = xp.can_cast(_from, to)

    expected = False
    for other in dh.all_dtypes:
        if dh.promotion_table.get((_from, other)) == to:
            expected = True
            break

    f_func = f"[can_cast({dh.dtype_to_name[_from]}, {dh.dtype_to_name[to]})]"
    if expected:
        # cross-kind casting is not explicitly disallowed. We can only test
        # the cases where it should return True. TODO: if expected=False,
        # check that the array library actually allows such casts.
        assert out == expected, f"{out=}, but should be {expected} {f_func}"


@pytest.mark.parametrize("dtype", dh.real_float_dtypes + dh.complex_dtypes)
def test_finfo(dtype):
    for arg in (
        dtype,
        xp.asarray(1, dtype=dtype),
        # np.float64 and np.asarray(1, dtype=np.float64).dtype are different
        xp.asarray(1, dtype=dtype).dtype,
    ):
        repro_snippet = ph.format_snippet(f"xp.finfo({arg})")
        try:
            out = xp.finfo(arg)
            assert isinstance(out.bits, int)
            assert isinstance(out.eps, float)
            assert isinstance(out.max, float)
            assert isinstance(out.min, float)
            assert isinstance(out.smallest_normal, float)
        except Exception as exc:
            exc.add_note(repro_snippet)
            raise

@pytest.mark.min_version("2022.12")
@pytest.mark.parametrize("dtype", dh.real_float_dtypes + dh.complex_dtypes)
def test_finfo_dtype(dtype):
    out = xp.finfo(dtype)

    if dtype == xp.complex64:
        assert out.dtype == xp.float32
    elif dtype == xp.complex128:
        assert out.dtype == xp.float64
    else:
        assert out.dtype == dtype

    # Guard vs. numpy.dtype.__eq__ lax comparison
    assert not isinstance(out.dtype, str)
    assert out.dtype is not float
    assert out.dtype is not complex


@pytest.mark.parametrize("dtype", dh.int_dtypes + dh.uint_dtypes)
def test_iinfo(dtype):
    for arg in (
        dtype,
        xp.asarray(1, dtype=dtype),
        # np.int64 and np.asarray(1, dtype=np.int64).dtype are different
        xp.asarray(1, dtype=dtype).dtype,
    ):
        out = xp.iinfo(arg)
        assert isinstance(out.bits, int)
        assert isinstance(out.max, int)
        assert isinstance(out.min, int)


@pytest.mark.min_version("2022.12")
@pytest.mark.parametrize("dtype", dh.int_dtypes + dh.uint_dtypes)
def test_iinfo_dtype(dtype):
    out = xp.iinfo(dtype)
    assert out.dtype == dtype
    # Guard vs. numpy.dtype.__eq__ lax comparison
    assert not isinstance(out.dtype, str)
    assert out.dtype is not int


def atomic_kinds() -> st.SearchStrategy[Union[DataType, str]]:
    return hh.all_dtypes | st.sampled_from(list(dh.kind_to_dtypes.keys()))


@pytest.mark.min_version("2022.12")
@given(
    dtype=hh.all_dtypes,
    kind=atomic_kinds() | st.lists(atomic_kinds(), min_size=1).map(tuple),
)
def test_isdtype(dtype, kind):
    out = xp.isdtype(dtype, kind)

    assert isinstance(out, bool), f"{type(out)=}, but should be bool [isdtype()]"
    _kinds = kind if isinstance(kind, tuple) else (kind,)
    expected = False
    for _kind in _kinds:
        if isinstance(_kind, str):
            if dtype in dh.kind_to_dtypes[_kind]:
                expected = True
                break
        else:
            if dtype == _kind:
                expected = True
                break
    assert out == expected, f"{out=}, but should be {expected} [isdtype()]"


@pytest.mark.min_version("2024.12")
class TestResultType:
    @given(dtypes=hh.mutually_promotable_dtypes(None))
    def test_result_type(self, dtypes):
        out = xp.result_type(*dtypes)
        ph.assert_dtype("result_type", in_dtype=dtypes, out_dtype=out, repr_name="out")

    @given(pair=hh.pair_of_mutually_promotable_dtypes(None))
    def test_shuffled(self, pair):
        """Test that result_type is insensitive to the order of arguments."""
        s1, s2 = pair
        out1 = xp.result_type(*s1)
        out2 = xp.result_type(*s2)
        assert out1 == out2

    @given(pair=hh.pair_of_mutually_promotable_dtypes(2), data=st.data())
    def test_arrays_and_dtypes(self, pair, data):
        s1, s2 = pair
        a2 = tuple(xp.empty(1, dtype=dt) for dt in s2)
        a_and_dt = data.draw(st.permutations(s1 + a2))
        out = xp.result_type(*a_and_dt)
        ph.assert_dtype("result_type", in_dtype=s1+s2, out_dtype=out, repr_name="out")

    @given(dtypes=hh.mutually_promotable_dtypes(2), data=st.data())
    def test_with_scalars(self, dtypes, data):
        out = xp.result_type(*dtypes)

        if out == xp.bool:
            scalars = [True]
        elif out in dh.all_int_dtypes:
            scalars = [1]
        elif out in dh.real_dtypes:
            scalars = [1, 1.0]
        elif out in dh.numeric_dtypes:
            scalars = [1, 1.0, 1j]        # numeric_types - real_types == complex_types
        else:
            raise ValueError(f"unknown dtype {out = }.")

        scalar = data.draw(st.sampled_from(scalars))
        inputs = data.draw(st.permutations(dtypes + (scalar,)))

        out_scalar = xp.result_type(*inputs)
        assert out_scalar == out

        # retry with arrays
        arrays = tuple(xp.empty(1, dtype=dt) for dt in dtypes)
        inputs = data.draw(st.permutations(arrays + (scalar,)))
        out_scalar = xp.result_type(*inputs)
        assert out_scalar == out

