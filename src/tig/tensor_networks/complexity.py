"""
Tensor network complexity (minimal).

Provides simple FLOP-count proxies for:
- einsum contractions (rough)
- MPS/MPO contractions (rough)

These are used for reporting and benchmarking contracts, not as exact counts.
"""

from dataclasses import dataclass
from typing import Sequence

__all__ = ["EinsumCost", "einsum_cost_proxy", "mps_inner_cost_proxy"]


@dataclass(frozen=True)
class EinsumCost:
    """
    Rough contraction cost proxy.
    """

    flops: int


def einsum_cost_proxy(input_sizes: Sequence[int]) -> EinsumCost:
    """
    Proxy: product of input sizes (very rough upper bound).
    """
    flops = 1
    for s in input_sizes:
        flops *= int(s)
    return EinsumCost(flops=int(flops))


def mps_inner_cost_proxy(length: int, phys_dim: int, bond_dim: int) -> int:
    """
    Proxy for MPS inner product cost O(L * d * r^3).
    """
    l = int(length)
    d = int(phys_dim)
    r = int(bond_dim)
    return l * d * (r**3)
