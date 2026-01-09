import math
from collections import deque
from typing import Iterable, Iterator, Tuple, Union

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import Array, Shape


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
    *,
    x_indices: Iterable[Union[int, Shape]],
    out: Array,
    out_indices: Iterable[Union[int, Shape]],
    kw: dict = {},
):
    msg_suffix = f" [{func_name}({ph.fmt_kw(kw)})]\n  {x=}\n{out=}"
    for x_idx, out_idx in zip(x_indices, out_indices):
        msg = f"out[{out_idx}]={out[out_idx]}, should be x[{x_idx}]={x[x_idx]}"
        msg += msg_suffix
        if dh.is_float_dtype(x.dtype) and xp.isnan(x[x_idx]):
            assert xp.isnan(out[out_idx]), msg
        else:
            assert out[out_idx] == x[x_idx], msg


@pytest.mark.unvectorized
@given(
    dtypes=hh.mutually_promotable_dtypes(None, dtypes=dh.numeric_dtypes),
    base_shape=hh.shapes(),
    data=st.data(),
)
def test_concat(dtypes, base_shape, data):
    axis_strat = st.none()
    ndim = len(base_shape)
    if ndim > 0:
        axis_strat |= st.integers(-ndim, ndim - 1)
    kw = data.draw(
        axis_strat.flatmap(lambda a: hh.specified_kwargs(("axis", a, 0))), label="kw"
    )
    axis = kw.get("axis", 0)
    if axis is None:
        _axis = None
        shape_strat = hh.shapes()
    else:
        _axis = axis if axis >= 0 else len(base_shape) + axis
        shape_strat = st.integers(0, hh.MAX_SIDE).map(
            lambda i: base_shape[:_axis] + (i,) + base_shape[_axis + 1 :]
        )
    arrays = []
    for i, dtype in enumerate(dtypes, 1):
        x = data.draw(hh.arrays(dtype=dtype, shape=shape_strat), label=f"x{i}")
        arrays.append(x)

    repro_snippet = ph.format_snippet(f"xp.concat({arrays!r}, **kw) with {kw = }")
    try:
        out = xp.concat(arrays, **kw)

        ph.assert_dtype("concat", in_dtype=dtypes, out_dtype=out.dtype)

        shapes = tuple(x.shape for x in arrays)
        if _axis is None:
            size = sum(math.prod(s) for s in shapes)
            shape = (size,)
        else:
            shape = list(shapes[0])
            for other_shape in shapes[1:]:
                shape[_axis] += other_shape[_axis]
            shape = tuple(shape)
        ph.assert_result_shape("concat", in_shapes=shapes, out_shape=out.shape, expected=shape, kw=kw)

        if _axis is None:
            out_indices = (i for i in range(math.prod(out.shape)))
            for x_num, x in enumerate(arrays, 1):
                for x_idx in sh.ndindex(x.shape):
                    out_i = next(out_indices)
                    ph.assert_0d_equals(
                        "concat",
                        x_repr=f"x{x_num}[{x_idx}]",
                        x_val=x[x_idx],
                        out_repr=f"out[{out_i}]",
                        out_val=out[out_i],
                        kw=kw,
                    )
        else:
            out_indices = sh.ndindex(out.shape)
            for idx in sh.axis_ndindex(shapes[0], _axis):
                f_idx = ", ".join(str(i) if isinstance(i, int) else ":" for i in idx)
                for x_num, x in enumerate(arrays, 1):
                    indexed_x = x[idx]
                    for x_idx in sh.ndindex(indexed_x.shape):
                        out_idx = next(out_indices)
                        ph.assert_0d_equals(
                            "concat",
                            x_repr=f"x{x_num}[{f_idx}][{x_idx}]",
                            x_val=indexed_x[x_idx],
                            out_repr=f"out[{out_idx}]",
                            out_val=out[out_idx],
                            kw=kw,
                        )
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=shared_shapes()),
    axis=shared_shapes().flatmap(
        # Generate both valid and invalid axis
        lambda s: st.integers(2 * (-len(s) - 1), 2 * len(s))
    ),
)
def test_expand_dims(x, axis):
    if axis < -x.ndim - 1 or axis > x.ndim:
        with pytest.raises(IndexError):
            xp.expand_dims(x, axis=axis)
        return

    repro_snippet = ph.format_snippet(f"xp.expand_dims({x!r}, axis={axis!r})")
    try:
        out = xp.expand_dims(x, axis=axis)

        ph.assert_dtype("expand_dims", in_dtype=x.dtype, out_dtype=out.dtype)

        shape = [side for side in x.shape]
        index = axis if axis >= 0 else x.ndim + axis + 1
        shape.insert(index, 1)
        shape = tuple(shape)
        ph.assert_result_shape("expand_dims", in_shapes=[x.shape], out_shape=out.shape, expected=shape)

        assert_array_ndindex(
            "expand_dims", x, x_indices=sh.ndindex(x.shape), out=out, out_indices=sh.ndindex(out.shape)
        )
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.min_version("2023.12")
@given(x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1)), data=st.data())
def test_moveaxis(x, data):
    source = data.draw(
        st.integers(-x.ndim, x.ndim - 1) | xps.valid_tuple_axes(x.ndim), label="source"
    )
    if isinstance(source, int):
        destination = data.draw(st.integers(-x.ndim, x.ndim - 1), label="destination")
    else:
        assert isinstance(source, tuple)  # sanity check
        destination = data.draw(
            st.lists(
                st.integers(-x.ndim, x.ndim - 1),
                min_size=len(source),
                max_size=len(source),
                unique_by=lambda n: n if n >= 0 else x.ndim + n,
            ).map(tuple),
            label="destination"
        )

    repro_snippet = ph.format_snippet(f"xp.moveaxis({x!r}, {source!r}, {destination!r})")
    try:
        out = xp.moveaxis(x, source, destination)

        ph.assert_dtype("moveaxis", in_dtype=x.dtype, out_dtype=out.dtype)

        _source = sh.normalize_axis(source, x.ndim)
        _destination = sh.normalize_axis(destination, x.ndim)

        new_axes = [n for n in range(x.ndim) if n not in _source]

        for dest, src in sorted(zip(_destination, _source)):
            new_axes.insert(dest, src)

        expected_shape = tuple(x.shape[i] for i in new_axes)

        ph.assert_result_shape("moveaxis", in_shapes=[x.shape],
                               out_shape=out.shape, expected=expected_shape,
                               kw={"source": source, "destination": destination})

        indices = list(sh.ndindex(x.shape))
        permuted_indices = [tuple(idx[axis] for axis in new_axes) for idx in indices]
        assert_array_ndindex(
            "moveaxis", x, x_indices=sh.ndindex(x.shape), out=out, out_indices=permuted_indices
        )
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.all_dtypes, shape=hh.shapes(min_side=1).filter(lambda s: 1 in s)
    ),
    data=st.data(),
)
def test_squeeze(x, data):
    axes = st.integers(-x.ndim, x.ndim - 1)
    axis = data.draw(
        axes
        | st.lists(axes, unique_by=lambda i: i if i >= 0 else i + x.ndim).map(tuple),
        label="axis",
    )

    axes = (axis,) if isinstance(axis, int) else axis
    axes = sh.normalize_axis(axes, x.ndim)

    squeezable_axes = [i for i, side in enumerate(x.shape) if side == 1]
    if any(i not in squeezable_axes for i in axes):
        with pytest.raises(ValueError):
            xp.squeeze(x, axis)
        return

    repro_snippet = ph.format_snippet(f"xp.squeeze({x!r}, {axis!r})")
    try:
        out = xp.squeeze(x, axis)

        ph.assert_dtype("squeeze", in_dtype=x.dtype, out_dtype=out.dtype)

        shape = []
        for i, side in enumerate(x.shape):
            if i not in axes:
                shape.append(side)
        shape = tuple(shape)
        ph.assert_result_shape("squeeze", in_shapes=[x.shape], out_shape=out.shape, expected=shape, kw=dict(axis=axis))

        assert_array_ndindex("squeeze", x, x_indices=sh.ndindex(x.shape), out=out, out_indices=sh.ndindex(out.shape))
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes()),
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

    repro_snippet = ph.format_snippet(f"xp.flip({x!r}, **kw) with {kw=}")
    try:
        out = xp.flip(x, **kw)

        ph.assert_dtype("flip", in_dtype=x.dtype, out_dtype=out.dtype)

        _axes = sh.normalize_axis(kw.get("axis", None), x.ndim)
        for indices in sh.axes_ndindex(x.shape, _axes):
            reverse_indices = indices[::-1]
            assert_array_ndindex("flip", x, x_indices=indices, out=out,
                                 out_indices=reverse_indices, kw=kw)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=shared_shapes(min_dims=1)),
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
    repro_snippet = ph.format_snippet(f"xp.permute_dims({x!r},{axes!r})")
    try:
        out = xp.permute_dims(x, axes)

        ph.assert_dtype("permute_dims", in_dtype=x.dtype, out_dtype=out.dtype)

        shape = [None for _ in range(len(axes))]
        for i, dim in enumerate(axes):
            side = x.shape[dim]
            shape[i] = side
        shape = tuple(shape)
        ph.assert_result_shape("permute_dims", in_shapes=[x.shape], out_shape=out.shape, expected=shape, kw=dict(axes=axes))

        indices = list(sh.ndindex(x.shape))
        permuted_indices = [tuple(idx[axis] for axis in axes) for idx in indices]
        assert_array_ndindex("permute_dims", x, x_indices=indices, out=out,
                             out_indices=permuted_indices)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise

@pytest.mark.min_version("2023.12")
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=shared_shapes(min_dims=1)),
    kw=hh.kwargs(
        axis=st.none() | shared_shapes(min_dims=1).flatmap(
            lambda s: st.integers(-len(s), len(s) - 1)
        )
    ),
    data=st.data(),
)
def test_repeat(x, kw, data):
    shape = x.shape
    axis = kw.get("axis", None)
    size = math.prod(shape) if axis is None else shape[axis]
    repeat_strat = st.integers(1, 10)
    repeats = data.draw(repeat_strat
                        | hh.arrays(dtype=hh.int_dtypes, elements=repeat_strat,
                                    shape=st.sampled_from([(1,), (size,)])),
        label="repeats")
    if isinstance(repeats, int):
        n_repititions = size*repeats
    else:
        if repeats.shape == (1,):
            n_repititions = size*int(repeats[0])
        else:
            n_repititions = int(xp.sum(repeats))

    assume(n_repititions <= hh.SQRT_MAX_ARRAY_SIZE)

    repro_snippet = ph.format_snippet(f"xp.repeat({x!r},{repeats!r}, **kw) with {kw=}")
    try:
        out = xp.repeat(x, repeats, **kw)

        ph.assert_dtype("repeat", in_dtype=x.dtype, out_dtype=out.dtype)
        if axis is None:
            expected_shape = (n_repititions,)
        else:
            expected_shape = list(shape)
            expected_shape[axis] = n_repititions
            expected_shape = tuple(expected_shape)
        ph.assert_shape("repeat", out_shape=out.shape, expected=expected_shape)

        # Test values

        if isinstance(repeats, int):
            repeats_array = xp.full(size, repeats, dtype=xp.int32)
        else:
            repeats_array = repeats

        if kw.get("axis") is None:
            x = xp.reshape(x, (-1,))
            axis = 0

        for idx, in sh.iter_indices(x.shape, skip_axes=axis):
            x_slice = x[idx]
            out_slice = out[idx]
            start = 0
            for i, count in enumerate(repeats_array):
                end = start + count
                ph.assert_array_elements("repeat", out=out_slice[start:end],
                                         expected=xp.full((int(count),), x_slice[i], dtype=x.dtype),
                                         kw=kw)
                start = end

    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


reshape_shape = st.shared(hh.shapes(), key="reshape_shape")

@pytest.mark.has_setup_funcs
@pytest.mark.unvectorized
@given(
    x=hh.arrays(dtype=hh.all_dtypes, shape=reshape_shape),
    shape=hh.reshape_shapes(reshape_shape),
)
def test_reshape(x, shape):
    repro_snippet = ph.format_snippet(f"xp.reshape({x!r},{shape!r})")
    try: 
        out = xp.reshape(x, shape)

        ph.assert_dtype("reshape", in_dtype=x.dtype, out_dtype=out.dtype)

        _shape = list(shape)
        if any(side == -1 for side in shape):
            size = math.prod(x.shape)
            rsize = math.prod(shape) * -1
            _shape[shape.index(-1)] = size / rsize
        _shape = tuple(_shape)
        ph.assert_result_shape("reshape", in_shapes=[x.shape], out_shape=out.shape, expected=_shape, kw=dict(shape=shape))

        assert_array_ndindex("reshape", x, x_indices=sh.ndindex(x.shape), out=out, out_indices=sh.ndindex(out.shape))
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


def roll_ndindex(shape: Shape, shifts: Tuple[int], axes: Tuple[int]) -> Iterator[Shape]:
    assert len(shifts) == len(axes)  # sanity check
    all_shifts = [0 for _ in shape]
    for s, a in zip(shifts, axes):
        all_shifts[a] = s
    for idx in sh.ndindex(shape):
        yield tuple((i + sh) % si for i, sh, si in zip(idx, all_shifts, shape))


@pytest.mark.unvectorized
@given(hh.arrays(dtype=hh.all_dtypes, shape=shared_shapes()), st.data())
def test_roll(x, data):
    shift_strat = st.integers(-hh.MAX_ARRAY_SIZE, hh.MAX_ARRAY_SIZE)
    if x.ndim > 0:
        shift_strat = shift_strat | st.lists(
            shift_strat, min_size=1, max_size=x.ndim
        ).map(tuple)
    shift = data.draw(shift_strat, label="shift")
    if isinstance(shift, tuple):
        axis_strat = xps.valid_tuple_axes(x.ndim).filter(lambda t: len(t) == len(shift))
        kw_strat = axis_strat.map(lambda t: {"axis": t})
    else:
        axis_strat = st.none()
        if x.ndim != 0:
            axis_strat |= st.integers(-x.ndim, x.ndim - 1)
        kw_strat = hh.kwargs(axis=axis_strat)
    kw = data.draw(kw_strat, label="kw")

    repro_snippet = ph.format_snippet(f"xp.roll({x!r},{shift!r}, **kw) with {kw=}")
    try:
        out = xp.roll(x, shift, **kw)

        kw = {"shift": shift, **kw}  # for error messages

        ph.assert_dtype("roll", in_dtype=x.dtype, out_dtype=out.dtype)

        ph.assert_result_shape("roll", in_shapes=[x.shape], out_shape=out.shape, kw=kw)

        if kw.get("axis", None) is None:
            assert isinstance(shift, int)  # sanity check
            indices = list(sh.ndindex(x.shape))
            shifted_indices = deque(indices)
            shifted_indices.rotate(-shift)
            assert_array_ndindex("roll", x, x_indices=indices, out=out, out_indices=shifted_indices, kw=kw)
        else:
            shifts = (shift,) if isinstance(shift, int) else shift
            axes = sh.normalize_axis(kw["axis"], x.ndim)
            shifted_indices = roll_ndindex(x.shape, shifts, axes)
            assert_array_ndindex("roll", x, x_indices=sh.ndindex(x.shape), out=out, out_indices=shifted_indices, kw=kw)
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.unvectorized
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
        x = data.draw(hh.arrays(dtype=dtype, shape=shape), label=f"x{i}")
        arrays.append(x)

    repro_snippet = ph.format_snippet(f"xp.stack({arrays!r}, **kw) with {kw=}")
    try: 
        out = xp.stack(arrays, **kw)

        ph.assert_dtype("stack", in_dtype=dtypes, out_dtype=out.dtype)

        axis = kw.get("axis", 0)
        _axis = axis if axis >= 0 else len(shape) + axis + 1
        _shape = list(shape)
        _shape.insert(_axis, len(arrays))
        _shape = tuple(_shape)
        ph.assert_result_shape(
            "stack", in_shapes=tuple(x.shape for x in arrays), out_shape=out.shape, expected=_shape, kw=kw
        )

        out_indices = sh.ndindex(out.shape)
        for idx in sh.axis_ndindex(arrays[0].shape, axis=_axis):
            f_idx = ", ".join(str(i) if isinstance(i, int) else ":" for i in idx)
            for x_num, x in enumerate(arrays, 1):
                indexed_x = x[idx]
                for x_idx in sh.ndindex(indexed_x.shape):
                    out_idx = next(out_indices)
                    ph.assert_0d_equals(
                        "stack",
                        x_repr=f"x{x_num}[{f_idx}][{x_idx}]",
                        x_val=indexed_x[x_idx],
                        out_repr=f"out[{out_idx}]",
                        out_val=out[out_idx],
                        kw=kw,
                    )
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise


@pytest.mark.min_version("2023.12")
@given(x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes()), data=st.data())
def test_tile(x, data):
    repetitions = data.draw(
        st.lists(st.integers(1, 4), min_size=1, max_size=x.ndim + 1).map(tuple),
        label="repetitions"
    )
    repro_snippet = ph.format_snippet(f"xp.tile({x!r}, {repetitions!r})")
    try:
        out = xp.tile(x, repetitions)
        ph.assert_dtype("tile", in_dtype=x.dtype, out_dtype=out.dtype)
        # TODO: values testing

        # shape check; the notation is from the Array API docs
        N, M = len(x.shape), len(repetitions)
        if N > M:
            S = x.shape
            R = (1,)*(N - M) + repetitions
        else:
            S = (1,)*(M - N) + x.shape
            R = repetitions

        assert out.shape == tuple(r*s for r, s in zip(R, S))
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise

@pytest.mark.min_version("2023.12")
@given(x=hh.arrays(dtype=hh.all_dtypes, shape=hh.shapes(min_dims=1)), data=st.data())
def test_unstack(x, data):
    axis = data.draw(st.integers(min_value=-x.ndim, max_value=x.ndim - 1), label="axis")
    kw = data.draw(hh.specified_kwargs(("axis", axis, 0)), label="kw")

    repro_snippet = ph.format_snippet(f"xp.unstack({x!r}, **kw) with {kw=}")
    try:
        out = xp.unstack(x, **kw)

        assert isinstance(out, tuple)
        assert len(out) == x.shape[axis]
        expected_shape = list(x.shape)
        expected_shape.pop(axis)
        expected_shape = tuple(expected_shape)
        for i in range(x.shape[axis]):
            arr = out[i]
            ph.assert_result_shape("unstack", in_shapes=[x.shape],
                                   out_shape=arr.shape, expected=expected_shape,
                                   kw=kw, repr_name=f"out[{i}].shape")

            ph.assert_dtype("unstack", in_dtype=x.dtype, out_dtype=arr.dtype,
                            repr_name=f"out[{i}].dtype")

            idx = [slice(None)] * x.ndim
            idx[axis] = i
            ph.assert_array_elements("unstack", out=arr, expected=x[tuple(idx)], kw=kw, out_repr=f"out[{i}]")
    except Exception as exc:
        ph.add_note(exc, repro_snippet)
        raise
