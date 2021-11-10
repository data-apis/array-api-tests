import math

from hypothesis import given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps

shared_shapes = st.shared(hh.shapes(), key="shape")


@given(
    shape=hh.shapes(min_dims=1),
    dtypes=hh.mutually_promotable_dtypes(None, dtypes=dh.numeric_dtypes),
    kw=hh.kwargs(axis=st.just(0) | st.none()),  # TODO: test with axis >= 1
    data=st.data(),
)
def test_concat(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    out = xp.concat(arrays, **kw)
    ph.assert_dtype("concat", dtypes, out.dtype)
    shapes = tuple(x.shape for x in arrays)
    if kw.get("axis", 0) == 0:
        pass  # TODO: assert expected shape
    elif kw["axis"] is None:
        size = sum(math.prod(s) for s in shapes)
        ph.assert_result_shape("concat", shapes, out.shape, (size,), **kw)
    # TODO: assert out elements match input arrays


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    axis=shared_shapes.flatmap(lambda s: st.integers(-len(s), len(s))),
)
def test_expand_dims(x, axis):
    out = xp.expand_dims(x, axis=axis)

    ph.assert_dtype("expand_dims", x.dtype, out.dtype)

    shape = [side for side in x.shape]
    index = axis if axis >= 0 else x.ndim + axis + 1
    shape.insert(index, 1)
    shape = tuple(shape)
    ph.assert_result_shape("expand_dims", (x.shape,), out.shape, shape)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    kw=hh.kwargs(
        axis=st.one_of(
            st.none(),
            shared_shapes.flatmap(
                lambda s: st.none()
                if len(s) == 0
                else st.integers(-len(s) + 1, len(s) - 1),
            ),
        )
    ),
)
def test_flip(x, kw):
    xp.flip(x, **kw)
    # TODO


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    axes=shared_shapes.flatmap(
        lambda s: st.lists(
            st.integers(0, max(len(s) - 1, 0)),
            min_size=len(s),
            max_size=len(s),
            unique=True,
        ).map(tuple)
    ),
)
def test_permute_dims(x, axes):
    xp.permute_dims(x, axes)
    # TODO


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    shape=shared_shapes,  # TODO: test more compatible shapes
)
def test_reshape(x, shape):
    xp.reshape(x, shape)
    # TODO


@given(
    # TODO: axis arguments, update shift respectively
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    shift=shared_shapes.flatmap(lambda s: st.integers(0, max(math.prod(s) - 1, 0))),
)
def test_roll(x, shift):
    xp.roll(x, shift)
    # TODO


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes),
    axis=shared_shapes.flatmap(
        lambda s: st.just(0)
        if len(s) == 0
        else st.integers(-len(s) + 1, len(s) - 1).filter(lambda i: s[i] == 1)
    ),  # TODO: tuple of axis i.e. axes
)
def test_squeeze(x, axis):
    xp.squeeze(x, axis)
    # TODO


@given(
    shape=hh.shapes(),
    dtypes=hh.mutually_promotable_dtypes(None),
    data=st.data(),
)
def test_stack(shape, dtypes, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)
    out = xp.stack(arrays)
    ph.assert_dtype("stack", dtypes, out.dtype)
    # TODO
