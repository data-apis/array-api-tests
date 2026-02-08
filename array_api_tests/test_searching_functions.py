import math

import pytest
from hypothesis import given, note, assume
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps


pytestmark = pytest.mark.unvectorized


@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmax(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    repro_snippet = ph.format_snippet(f"xp.argmax({x!r}, **kw) with {kw = }")
    try:
        out = xp.argmax(x, **kw)

        ph.assert_default_index("argmax", out.dtype)
        axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
        ph.assert_keepdimable_shape(
            "argmax", in_shape=x.shape, out_shape=out.shape, axes=axes, keepdims=keepdims, kw=kw
        )
        scalar_type = dh.get_scalar_type(x.dtype)
        for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
            max_i = int(out[out_idx])
            elements = []
            for idx in indices:
                s = scalar_type(x[idx])
                elements.append(s)
            expected = max(range(len(elements)), key=elements.__getitem__)
            ph.assert_scalar_equals("argmax", type_=int, idx=out_idx, out=max_i,
                                    expected=expected, kw=kw)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argmin(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=st.none() | st.integers(-x.ndim, max(x.ndim - 1, 0)),
            keepdims=st.booleans(),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    repro_snippet = ph.format_snippet(f"xp.argmin({x!r}, **kw) with {kw = }")
    try:
        out = xp.argmin(x, **kw)

        ph.assert_default_index("argmin", out.dtype)
        axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
        ph.assert_keepdimable_shape(
            "argmin", in_shape=x.shape, out_shape=out.shape, axes=axes, keepdims=keepdims, kw=kw
        )
        scalar_type = dh.get_scalar_type(x.dtype)
        for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
            min_i = int(out[out_idx])
            elements = []
            for idx in indices:
                s = scalar_type(x[idx])
                elements.append(s)
            expected = min(range(len(elements)), key=elements.__getitem__)
            ph.assert_scalar_equals("argmin", type_=int, idx=out_idx, out=min_i, expected=expected)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


# XXX: the strategy for x is problematic on JAX unless JAX_ENABLE_X64 is on
# the problem is tha for ints >iinfo(int32) it runs into essentially this:
#  >>> jnp.asarray[2147483648], dtype=jnp.int64)
#  .... https://github.com/jax-ml/jax/pull/6047 ...
# Explicitly limiting the range in elements(...) runs into problems with
# hypothesis where floating-point numbers are not exactly representable.
@pytest.mark.min_version("2024.12")
@given(
    x=hh.arrays(
        dtype=hh.all_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_count_nonzero(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=hh.axes(x.ndim),
            keepdims=st.booleans(),
        ),
        label="kw",
    )
    keepdims = kw.get("keepdims", False)

    assume(kw.get("axis", None) != ())  # TODO clarify in the spec

    repro_snippet = ph.format_snippet(f"xp.count_nonzero({x!r}, **kw) with {kw = }")
    try:
        out = xp.count_nonzero(x, **kw)

        ph.assert_default_index("count_nonzero", out.dtype)
        axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
        ph.assert_keepdimable_shape(
            "count_nonzero", in_shape=x.shape, out_shape=out.shape, axes=axes, keepdims=keepdims, kw=kw
        )
        scalar_type = dh.get_scalar_type(x.dtype)

        for indices, out_idx in zip(sh.axes_ndindex(x.shape, axes), sh.ndindex(out.shape)):
            count = int(out[out_idx])
            elements = []
            for idx in indices:
                s = scalar_type(x[idx])
                elements.append(s)
            expected = sum(el != 0 for el in elements)
            ph.assert_scalar_equals("count_nonzero", type_=int, idx=out_idx, out=count, expected=expected)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(hh.arrays(dtype=hh.all_dtypes, shape=()))
def test_nonzero_zerodim_error(x):
    with pytest.raises(Exception):
        xp.nonzero(x)


@pytest.mark.data_dependent_shapes
@given(hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1, min_side=1)))
def test_nonzero(x):
    repro_snippet = ph.format_snippet(f"xp.nonzero({x!r})")
    try:
        out = xp.nonzero(x)
        assert len(out) == x.ndim, f"{len(out)=}, but should be {x.ndim=}"
        out_size = math.prod(out[0].shape)
        for i in range(len(out)):
            assert out[i].ndim == 1, f"out[{i}].ndim={x.ndim}, but should be 1"
            size_at = math.prod(out[i].shape)
            assert size_at == out_size, (
                f"prod(out[{i}].shape)={size_at}, "
                f"but should be prod(out[0].shape)={out_size}"
            )
            ph.assert_default_index("nonzero", out[i].dtype, repr_name=f"out[{i}].dtype")
        indices = []
        if x.dtype == xp.bool:
            for idx in sh.ndindex(x.shape):
                if x[idx]:
                    indices.append(idx)
        else:
            for idx in sh.ndindex(x.shape):
                if x[idx] != 0:
                    indices.append(idx)

        for i in range(out_size):
            idx = tuple(int(x[i]) for x in out)
            f_idx = f"Extrapolated index (x[{i}] for x in out)={idx}"
            f_element = f"x[{idx}]={x[idx]}"
            assert idx in indices, f"{f_idx} results in {f_element}, a zero element"
            assert (
                idx == indices[i]
            ), f"{f_idx} is in the wrong position, should be {indices.index(idx)}"
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@given(
    shapes=hh.mutually_broadcastable_shapes(3),
    dtypes=hh.mutually_promotable_dtypes(),
    data=st.data(),
)
def test_where(shapes, dtypes, data):
    cond = data.draw(hh.arrays(dtype=xp.bool, shape=shapes[0]), label="condition")
    x1 = data.draw(hh.arrays(dtype=dtypes[0], shape=shapes[1]), label="x1")
    x2 = data.draw(hh.arrays(dtype=dtypes[1], shape=shapes[2]), label="x2")

    repro_snippet = ph.format_snippet(f"xp.where({cond!r}, {x1!r}, {x2!r})")
    try:
        out = xp.where(cond, x1, x2)

        shape = sh.broadcast_shapes(*shapes)
        ph.assert_shape("where", out_shape=out.shape, expected=shape)
        # TODO: generate indices without broadcasting arrays
        _cond = xp.broadcast_to(cond, shape)
        _x1 = xp.broadcast_to(x1, shape)
        _x2 = xp.broadcast_to(x2, shape)
        for idx in sh.ndindex(shape):
            if _cond[idx]:
                ph.assert_0d_equals(
                    "where",
                    x_repr=f"_x1[{idx}]",
                    x_val=_x1[idx],
                    out_repr=f"out[{idx}]",
                    out_val=out[idx]
                )
            else:
                ph.assert_0d_equals(
                    "where",
                    x_repr=f"_x2[{idx}]",
                    x_val=_x2[idx],
                    out_repr=f"out[{idx}]",
                    out_val=out[idx]
                )
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.min_version("2023.12")
@given(data=st.data())
def test_searchsorted(data):
    # TODO: Allow different dtypes for x1 and x2
    x1_dtype = data.draw(st.sampled_from(dh.real_dtypes))
    _x1 = data.draw(
        st.lists(
            xps.from_dtype(x1_dtype, allow_nan=False, allow_infinity=False),
            min_size=1,
            unique=True
        ),
        label="_x1",
    )
    x1 = xp.asarray(_x1, dtype=x1_dtype)
    if data.draw(st.booleans(), label="use sorter?"):
        sorter = xp.argsort(x1)
    else:
        sorter = None
        x1 = xp.sort(x1)
    note(f"{x1=}")

    x2 = data.draw(
        st.lists(st.sampled_from(_x1), unique=True, min_size=1).map(
            lambda o: xp.asarray(o, dtype=x1_dtype)
        ),
        label="x2",
    )
    # make x2.ndim > 1, if it makes sense
    factors = hh._factorize(x2.shape[0])
    if len(factors) > 1:
        x2 = xp.reshape(x2, tuple(factors))

    kw = data.draw(hh.kwargs(side=st.sampled_from(["left", "right"])))

    repro_snippet = ph.format_snippet(
        f"xp.searchsorted({x1!r}, {x2!r}, sorter={sorter!r}, **kw) with {kw=}"
    )
    try:
        out = xp.searchsorted(x1, x2, sorter=sorter, **kw)

        ph.assert_dtype(
            "searchsorted",
            in_dtype=[x1.dtype, x2.dtype],
            out_dtype=out.dtype,
            expected=xp.__array_namespace_info__().default_dtypes()["indexing"],
        )
        # TODO: values testing
        ph.assert_shape("searchsorted", out_shape=out.shape, expected=x2.shape)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise
