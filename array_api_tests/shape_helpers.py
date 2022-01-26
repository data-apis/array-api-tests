import math
from functools import lru_cache
from itertools import product
from typing import Iterator, List, Optional, Tuple, Union

from ndindex import iter_indices as _iter_indices

from .typing import AtomicIndex, Index, Scalar, Shape

__all__ = [
    "normalise_axis",
    "ndindex",
    "axis_ndindex",
    "axes_ndindex",
    "reshape",
    "fmt_idx",
]


def normalise_axis(
    axis: Optional[Union[int, Tuple[int, ...]]], ndim: int
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
    return axes


def ndindex(shape):
    """Yield every index of shape"""
    return (indices[0] for indices in iter_indices(shape))


def iter_indices(*shapes, skip_axes=()):
    """Wrapper for ndindex.iter_indices()"""
    gen = _iter_indices(*shapes, skip_axes=skip_axes)
    return ([i.raw for i in indices] for indices in gen)


def axis_ndindex(
    shape: Shape, axis: int
) -> Iterator[Tuple[Tuple[Union[int, slice], ...], ...]]:
    """Generate indices that index all elements in dimensions beyond `axis`"""
    assert axis >= 0  # sanity check
    axis_indices = [range(side) for side in shape[:axis]]
    for _ in range(axis, len(shape)):
        axis_indices.append([slice(None, None)])
    yield from product(*axis_indices)


def axes_ndindex(shape: Shape, axes: Tuple[int, ...]) -> Iterator[List[Shape]]:
    """Generate indices that index all elements except in `axes` dimensions"""
    base_indices = []
    axes_indices = []
    for axis, side in enumerate(shape):
        if axis in axes:
            base_indices.append([None])
            axes_indices.append(range(side))
        else:
            base_indices.append(range(side))
            axes_indices.append([None])
    for base_idx in product(*base_indices):
        indices = []
        for idx in product(*axes_indices):
            idx = list(idx)
            for axis, side in enumerate(idx):
                if axis not in axes:
                    idx[axis] = base_idx[axis]
            idx = tuple(idx)
            indices.append(idx)
        yield list(indices)


def reshape(flat_seq: List[Scalar], shape: Shape) -> Union[Scalar, List]:
    """Reshape a flat sequence"""
    if any(s == 0 for s in shape):
        raise ValueError(
            f"{shape=} contains 0-sided dimensions, "
            f"but that's not representable in lists"
        )
    if len(shape) == 0:
        assert len(flat_seq) == 1  # sanity check
        return flat_seq[0]
    elif len(shape) == 1:
        return flat_seq
    size = len(flat_seq)
    n = math.prod(shape[1:])
    return [reshape(flat_seq[i * n : (i + 1) * n], shape[1:]) for i in range(size // n)]


def fmt_i(i: AtomicIndex) -> str:
    if isinstance(i, int):
        return str(i)
    elif isinstance(i, slice):
        res = ""
        if i.start is not None:
            res += str(i.start)
        res += ":"
        if i.stop is not None:
            res += str(i.stop)
        if i.step is not None:
            res += f":{i.step}"
        return res
    else:
        return "..."


@lru_cache
def fmt_idx(sym: str, idx: Index) -> str:
    if idx == ():
        return sym
    res = f"{sym}["
    _idx = idx if isinstance(idx, tuple) else (idx,)
    if len(_idx) == 1:
        res += fmt_i(_idx[0])
    else:
        res += ", ".join(fmt_i(i) for i in _idx)
    res += "]"
    return res
