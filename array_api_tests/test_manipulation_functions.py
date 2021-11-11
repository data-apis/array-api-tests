import math
from collections import deque
from typing import Iterable, Union

from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .typing import Array, Shape

MAX_SIDE = hh.MAX_ARRAY_SIZE // 64
MAX_DIMS = min(hh.MAX_ARRAY_SIZE // MAX_SIDE, 32)  # NumPy only supports up to 32 dims


def shared_shapes(*args, **kwargs) -> st.SearchStrategy[Shape]:
    key = "shape"
    if args:
        key += " " + " ".join(args)
    if kwargs:
        key += " " + ph.fmt_kw(kwargs)
    return st.shared(hh.shapes(*args, **kwargs), key="shape")


def assert_array_ndindex(
    func_name: str,
    x: Array,
    x_indices: Iterable[Union[int, Shape]],
    out: Array,
    out_indices: Iterable[Union[int, Shape]],
):
    msg_suffix = f" [{func_name}()]\n  {x=}\n{out=}"
    for x_idx, out_idx in zip(x_indices, out_indices):
        msg = f"out[{out_idx}]={out[out_idx]}, should be x[{x_idx}]={x[x_idx]}"
        msg += msg_suffix
        if dh.is_float_dtype(x.dtype) and xp.isnan(x[x_idx]):
            assert xp.isnan(out[out_idx]), msg
        else:
            assert out[out_idx] == x[x_idx], msg


@st.composite
def concat_shapes(draw, shape, axis):
    shape = list(shape)
    shape[axis] = draw(st.integers(1, MAX_SIDE))
    return tuple(shape)


@given(
    dtypes=hh.mutually_promotable_dtypes(None, dtypes=dh.numeric_dtypes),
    kw=hh.kwargs(axis=st.none() | st.integers(-MAX_DIMS, MAX_DIMS - 1)),
    data=st.data(),
)
def test_concat(dtypes, kw, data):
    axis = kw.get("axis", 0)
    if axis is None:
        shape_strat = hh.shapes()
    else:
        _axis = axis if axis >= 0 else abs(axis) - 1
        shape_strat = shared_shapes(min_dims=_axis + 1).flatmap(
            lambda s: concat_shapes(s, axis)
        )
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape_strat), label=f"x{i}")
        arrays.append(x)

    out = xp.concat(arrays, **kw)

    ph.assert_dtype("concat", dtypes, out.dtype)

    shapes = tuple(x.shape for x in arrays)
    axis = kw.get("axis", 0)
    if axis is None:
        size = sum(math.prod(s) for s in shapes)
        shape = (size,)
    else:
        shape = list(shapes[0])
        for other_shape in shapes[1:]:
            shape[axis] += other_shape[axis]
        shape = tuple(shape)
    ph.assert_result_shape("concat", shapes, out.shape, shape, **kw)

    # TODO: adjust indices with nonzero axis
    if axis is None or axis == 0:
        out_indices = ah.ndindex(out.shape)
        for i, x in enumerate(arrays, 1):
            msg_suffix = f" [concat({ph.fmt_kw(kw)})]\nx{i}={x!r}\n{out=}"
            for x_idx in ah.ndindex(x.shape):
                out_idx = next(out_indices)
                msg = (
                    f"out[{out_idx}]={out[out_idx]}, should be x{i}[{x_idx}]={x[x_idx]}"
                )
                msg += msg_suffix
                if dh.is_float_dtype(x.dtype) and xp.isnan(x[x_idx]):
                    assert xp.isnan(out[out_idx]), msg
                else:
                    assert out[out_idx] == x[x_idx], msg


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes()),
    axis=shared_shapes().flatmap(lambda s: st.integers(-len(s) - 1, len(s))),
)
def test_expand_dims(x, axis):
    out = xp.expand_dims(x, axis=axis)

    ph.assert_dtype("expand_dims", x.dtype, out.dtype)

    shape = [side for side in x.shape]
    index = axis if axis >= 0 else x.ndim + axis + 1
    shape.insert(index, 1)
    shape = tuple(shape)
    ph.assert_result_shape("expand_dims", (x.shape,), out.shape, shape)

    assert_array_ndindex(
        "expand_dims", x, ah.ndindex(x.shape), out, ah.ndindex(out.shape)
    )


@given(
    x=xps.arrays(
        dtype=xps.scalar_dtypes(), shape=hh.shapes(min_side=1).filter(lambda s: 1 in s)
    ),
    data=st.data(),
)
def test_squeeze(x, data):
    # TODO: generate valid negative axis (which keep uniqueness)
    squeezable_axes = st.sampled_from(
        [i for i, side in enumerate(x.shape) if side == 1]
    )
    axis = data.draw(
        squeezable_axes | st.lists(squeezable_axes, unique=True).map(tuple),
        label="axis",
    )

    out = xp.squeeze(x, axis)

    ph.assert_dtype("squeeze", x.dtype, out.dtype)

    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis
    shape = []
    for i, side in enumerate(x.shape):
        if i not in axes:
            shape.append(side)
    shape = tuple(shape)
    ph.assert_result_shape("squeeze", (x.shape,), out.shape, shape, axis=axis)

    assert_array_ndindex("squeeze", x, ah.ndindex(x.shape), out, ah.ndindex(out.shape))


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes()),
    data=st.data(),
)
def test_flip(x, data):
    if x.ndim == 0:
        axis_strat = st.none()
    else:
        axis_strat = (
            st.none() | st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim)
        )
    kw = data.draw(hh.kwargs(axis=axis_strat), label="kw")

    out = xp.flip(x, **kw)

    ph.assert_dtype("flip", x.dtype, out.dtype)

    # TODO: test all axis scenarios
    if kw.get("axis", None) is None:
        indices = list(ah.ndindex(x.shape))
        reverse_indices = indices[::-1]
        assert_array_ndindex("flip", x, indices, out, reverse_indices)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes(min_dims=1)),
    axes=shared_shapes(min_dims=1).flatmap(
        lambda s: st.lists(
            st.integers(0, len(s) - 1),
            min_size=len(s),
            max_size=len(s),
            unique=True,
        ).map(tuple)
    ),
)
def test_permute_dims(x, axes):
    out = xp.permute_dims(x, axes)

    ph.assert_dtype("permute_dims", x.dtype, out.dtype)

    shape = [None for _ in range(len(axes))]
    for i, dim in enumerate(axes):
        side = x.shape[dim]
        shape[i] = side
    assert all(isinstance(side, int) for side in shape)  # sanity check
    shape = tuple(shape)
    ph.assert_result_shape("permute_dims", (x.shape,), out.shape, shape, axes=axes)

    # TODO: test elements


@st.composite
def reshape_shapes(draw, shape):
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(st.integers(0)).filter(lambda s: math.prod(s) == size))
    assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(st.integers(0, len(rshape) - 1))
        rshape[index] = -1
    return tuple(rshape)


@given(
    x=xps.arrays(dtype=xps.scalar_dtypes(), shape=hh.shapes(max_side=MAX_SIDE)),
    data=st.data(),
)
def test_reshape(x, data):
    shape = data.draw(reshape_shapes(x.shape))

    out = xp.reshape(x, shape)

    ph.assert_dtype("reshape", x.dtype, out.dtype)

    _shape = list(shape)
    if any(side == -1 for side in shape):
        size = math.prod(x.shape)
        rsize = math.prod(shape) * -1
        _shape[shape.index(-1)] = size / rsize
    _shape = tuple(_shape)
    ph.assert_result_shape("reshape", (x.shape,), out.shape, _shape, shape=shape)

    assert_array_ndindex("reshape", x, ah.ndindex(x.shape), out, ah.ndindex(out.shape))


@given(xps.arrays(dtype=xps.scalar_dtypes(), shape=shared_shapes()), st.data())
def test_roll(x, data):
    shift = data.draw(
        st.integers() | st.lists(st.integers(), max_size=x.ndim).map(tuple),
        label="shift",
    )
    axis_strats = [st.none()]
    if x.shape != ():
        axis_strats.append(st.integers(-x.ndim, x.ndim - 1))
        if isinstance(shift, int):
            axis_strats.append(xps.valid_tuple_axes(x.ndim))
    kw = data.draw(hh.kwargs(axis=st.one_of(axis_strats)), label="kw")

    out = xp.roll(x, shift, **kw)

    ph.assert_dtype("roll", x.dtype, out.dtype)

    ph.assert_result_shape("roll", (x.shape,), out.shape)

    # TODO: test all shift/axis scenarios
    if isinstance(shift, int) and kw.get("axis", None) is None:
        indices = list(ah.ndindex(x.shape))
        shifted_indices = deque(indices)
        shifted_indices.rotate(-shift)
        assert_array_ndindex("roll", x, indices, out, shifted_indices)


@given(
    shape=shared_shapes(min_dims=1),
    dtypes=hh.mutually_promotable_dtypes(None),
    kw=hh.kwargs(
        axis=shared_shapes(min_dims=1).flatmap(
            lambda s: st.integers(-len(s), len(s) - 1)
        )
    ),
    data=st.data(),
)
def test_stack(shape, dtypes, kw, data):
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(xps.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)

    out = xp.stack(arrays, **kw)

    ph.assert_dtype("stack", dtypes, out.dtype)

    axis = kw.get("axis", 0)
    _axis = axis if axis >= 0 else len(shape) + axis + 1
    _shape = list(shape)
    _shape.insert(_axis, len(arrays))
    _shape = tuple(_shape)
    ph.assert_result_shape(
        "stack", tuple(x.shape for x in arrays), out.shape, _shape, **kw
    )

    # TODO: adjust indices with nonzero axis
    if axis == 0:
        out_indices = ah.ndindex(out.shape)
        for i, x in enumerate(arrays, 1):
            msg_suffix = f" [stack({ph.fmt_kw(kw)})]\nx{i}={x!r}\n{out=}"
            for x_idx in ah.ndindex(x.shape):
                out_idx = next(out_indices)
                msg = (
                    f"out[{out_idx}]={out[out_idx]}, should be x{i}[{x_idx}]={x[x_idx]}"
                )
                msg += msg_suffix
                if dh.is_float_dtype(x.dtype) and xp.isnan(x[x_idx]):
                    assert xp.isnan(out[out_idx]), msg
                else:
                    assert out[out_idx] == x[x_idx], msg
