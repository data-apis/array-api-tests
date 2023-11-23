import struct
from typing import Union

import pytest
from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from . import xp as _xp
from .typing import DataType

pytestmark = pytest.mark.ci


# TODO: test with complex dtypes
def non_complex_dtypes():
    return xps.boolean_dtypes() | xps.real_dtypes()


def float32(n: Union[int, float]) -> float:
    return struct.unpack("!f", struct.pack("!f", float(n)))[0]


@given(
    x_dtype=non_complex_dtypes(),
    dtype=non_complex_dtypes(),
    kw=hh.kwargs(copy=st.booleans()),
    data=st.data(),
)
def test_astype(x_dtype, dtype, kw, data):
    if xp.bool in (x_dtype, dtype):
        elements_strat = xps.from_dtype(x_dtype)
    else:
        m1, M1 = dh.dtype_ranges[x_dtype]
        m2, M2 = dh.dtype_ranges[dtype]
        if dh.is_int_dtype(x_dtype):
            cast = int
        elif x_dtype == xp.float32:
            cast = float32
        else:
            cast = float
        min_value = cast(max(m1, m2))
        max_value = cast(min(M1, M2))
        elements_strat = xps.from_dtype(
            x_dtype,
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    x = data.draw(
        xps.arrays(dtype=x_dtype, shape=hh.shapes(), elements=elements_strat), label="x"
    )

    out = xp.astype(x, dtype, **kw)

    ph.assert_kw_dtype("astype", kw_dtype=dtype, out_dtype=out.dtype)
    ph.assert_shape("astype", out_shape=out.shape, expected=x.shape, kw=kw)
    # TODO: test values
    # TODO: test copy


@given(
    shapes=st.integers(1, 5).flatmap(hh.mutually_broadcastable_shapes), data=st.data()
)
def test_broadcast_arrays(shapes, data):
    arrays = []
    for c, shape in enumerate(shapes, 1):
        x = data.draw(xps.arrays(dtype=xps.scalar_dtypes(), shape=shape), label=f"x{c}")
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


@given(x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()), data=st.data())
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


@given(_from=non_complex_dtypes(), to=non_complex_dtypes(), data=st.data())
def test_can_cast(_from, to, data):
    from_ = data.draw(
        st.just(_from) | xps.arrays(dtype=_from, shape=hh.shapes()), label="from_"
    )

    out = xp.can_cast(from_, to)

    f_func = f"[can_cast({dh.dtype_to_name[_from]}, {dh.dtype_to_name[to]})]"
    assert isinstance(out, bool), f"{type(out)=}, but should be bool {f_func}"
    if _from == xp.bool:
        expected = to == xp.bool
    else:
        same_family = None
        for dtypes in [dh.all_int_dtypes, dh.real_float_dtypes, dh.complex_dtypes]:
            if _from in dtypes:
                same_family = to in dtypes
                break
        assert same_family is not None  # sanity check
        if same_family:
            from_min, from_max = dh.dtype_ranges[_from]
            to_min, to_max = dh.dtype_ranges[to]
            expected = from_min >= to_min and from_max <= to_max
        else:
            expected = False
    if expected:
        # cross-kind casting is not explicitly disallowed. We can only test
        # the cases where it should return True. TODO: if expected=False,
        # check that the array library actually allows such casts.
        assert out == expected, f"{out=}, but should be {expected} {f_func}"


@pytest.mark.parametrize("dtype_name", dh.real_float_names)
def test_finfo(dtype_name):
    try:
        dtype = getattr(_xp, dtype_name)
    except AttributeError as e:
        pytest.skip(str(e))
    out = xp.finfo(dtype)
    f_func = f"[finfo({dh.dtype_to_name[dtype]})]"
    for attr, stype in [
        ("bits", int),
        ("eps", float),
        ("max", float),
        ("min", float),
        ("smallest_normal", float),
    ]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, stype
        ), f"type(out.{attr})={type(value)!r}, but should be {stype.__name__} {f_func}"
    assert hasattr(out, "dtype"), f"out has no attribute 'dtype' {f_func}"
    # TODO: test values


@pytest.mark.parametrize("dtype_name", dh.all_int_names)
def test_iinfo(dtype_name):
    try:
        dtype = getattr(_xp, dtype_name)
    except AttributeError as e:
        pytest.skip(str(e))
    out = xp.iinfo(dtype)
    f_func = f"[iinfo({dh.dtype_to_name[dtype]})]"
    for attr in ["bits", "max", "min"]:
        assert hasattr(out, attr), f"out has no attribute '{attr}' {f_func}"
        value = getattr(out, attr)
        assert isinstance(
            value, int
        ), f"type(out.{attr})={type(value)!r}, but should be int {f_func}"
    assert hasattr(out, "dtype"), f"out has no attribute 'dtype' {f_func}"
    # TODO: test values


def atomic_kinds() -> st.SearchStrategy[Union[DataType, str]]:
    return xps.scalar_dtypes() | st.sampled_from(list(dh.kind_to_dtypes.keys()))


@pytest.mark.min_version("2022.12")
@given(
    dtype=xps.scalar_dtypes(),
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


@given(hh.mutually_promotable_dtypes(None))
def test_result_type(dtypes):
    out = xp.result_type(*dtypes)
    ph.assert_dtype("result_type", in_dtype=dtypes, out_dtype=out, repr_name="out")
