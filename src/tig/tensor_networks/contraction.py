"""
Tensor network contraction (minimal).

This module provides:
- a small contraction wrapper built around einsum strings
- optional NetworkX graph representation for contraction structure (if installed)

Advanced contraction-order optimization is intentionally out of scope here.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import tensorflow as tf

from tig.core.einsum import einsum
from tig.core.types import TensorLike, to_tensor

__all__ = ["EinsumContraction", "einsum_graph_optional"]


@dataclass(frozen=True)
class EinsumContraction:
    """
    Contraction defined by an einsum expression.
    """

    expr: str

    def __call__(self, *tensors: TensorLike) -> tf.Tensor:
        ts = [to_tensor(t) for t in tensors]
        return einsum(self.expr, *ts)


def einsum_graph_optional(expr: str):
    """
    Optional NetworkX graph representation of an einsum contraction.

    Returns a NetworkX graph if networkx is installed, otherwise returns None.
    """
    try:
        import networkx as nx
    except Exception:
        return None

    g = nx.Graph()
    g.add_node("einsum", kind="op", expr=expr)
    return g
