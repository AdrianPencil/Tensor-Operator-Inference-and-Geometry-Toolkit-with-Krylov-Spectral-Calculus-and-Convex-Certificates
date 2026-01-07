"""
Layout and memory as math objects (minimal).

This module defines a compact layout descriptor:
- shape
- strides (in elements)
- C-contiguity checks (row-major)

Used to connect operator algebra and vectorization to compute constraints.
"""

from dataclasses import dataclass
from typing import Tuple

__all__ = ["Layout", "c_strides", "is_c_contiguous"]


@dataclass(frozen=True)
class Layout:
    """
    Tensor layout description.
    """

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]


def c_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Row-major (C) strides in elements.
    """
    s = tuple(int(x) for x in shape)
    if len(s) == 0:
        return ()
    strides = [1]
    for dim in reversed(s[1:]):
        strides.append(strides[-1] * int(dim))
    return tuple(reversed(strides))


def is_c_contiguous(layout: Layout) -> bool:
    """
    Check if a layout matches row-major contiguous strides.
    """
    return tuple(layout.strides) == c_strides(tuple(layout.shape))
