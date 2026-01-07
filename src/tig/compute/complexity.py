"""
Operator complexity proxies (minimal).

Provides simple FLOP/byte models for common tensor operations.
These are used for compute contracts and reporting, not exact hardware timing.
"""

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "Complexity",
    "flops_matmul",
    "bytes_tensor",
    "arithmetic_intensity",
]


@dataclass(frozen=True)
class Complexity:
    """
    Simple complexity container.
    """

    flops: int
    bytes_moved: int


def flops_matmul(m: int, n: int, k: int) -> int:
    """
    Dense GEMM flops: ~2 m n k (multiply-add count).
    """
    return 2 * int(m) * int(n) * int(k)


def bytes_tensor(shape: Tuple[int, ...], dtype_bytes: int = 8) -> int:
    """
    Bytes for a dense tensor of given shape and dtype size (default float64 = 8 bytes).
    """
    num = 1
    for s in shape:
        num *= int(s)
    return int(num) * int(dtype_bytes)


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    """
    Flops per byte (a simple roofline-style statistic).
    """
    b = int(bytes_moved)
    if b <= 0:
        return float("inf")
    return float(flops) / float(b)
