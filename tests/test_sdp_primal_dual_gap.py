"""
SDP certificate utilities sanity.
"""

from tig.convex.certificates import primal_dual_gap


def test_primal_dual_gap_positive_when_primal_ge_dual() -> None:
    gap = primal_dual_gap(primal_value=2.0, dual_value=1.5)
    assert gap >= 0.0


def test_primal_dual_gap_negative_when_dual_exceeds_primal() -> None:
    gap = primal_dual_gap(primal_value=1.0, dual_value=1.2)
    assert gap < 0.0
