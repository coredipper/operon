"""Tests for graph-based morphogen diffusion (Paper §6.5.2 / §6.5.3)."""

import pytest

from operon_ai.coordination.diffusion import (
    DiffusionField,
    DiffusionParams,
    MorphogenSource,
)
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType


# ── TestMorphogenSource ──────────────────────────────────────────────────


class TestMorphogenSource:
    def test_frozen(self):
        s = MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5)
        with pytest.raises(AttributeError):
            s.node_id = "B"  # type: ignore[misc]

    def test_emission_rate(self):
        s = MorphogenSource("A", MorphogenType.COMPLEXITY, 0.3)
        assert s.emission_rate == 0.3
        assert s.max_concentration == 1.0  # default


# ── TestDiffusionParams ─────────────────────────────────────────────────


class TestDiffusionParams:
    def test_frozen(self):
        p = DiffusionParams()
        with pytest.raises(AttributeError):
            p.diffusion_rate = 0.5  # type: ignore[misc]

    def test_defaults(self):
        p = DiffusionParams()
        assert p.diffusion_rate == 0.1
        assert p.decay_rate == 0.05
        assert p.min_concentration == 0.001


# ── TestDiffusionField ──────────────────────────────────────────────────


class TestDiffusionField:
    def test_add_node(self):
        f = DiffusionField()
        f.add_node("A")
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.0

    def test_add_edge(self):
        f = DiffusionField()
        f.add_node("A")
        f.add_node("B")
        f.add_edge("A", "B")
        # Bidirectional by default
        assert "B" in f._adjacency["A"]
        assert "A" in f._adjacency["B"]

    def test_unknown_node_raises(self):
        f = DiffusionField()
        with pytest.raises(KeyError, match="Unknown node"):
            f.add_edge("A", "B")

    def test_set_get_concentration(self):
        f = DiffusionField()
        f.add_node("A")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 0.7)
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.7

    def test_default_is_zero(self):
        f = DiffusionField()
        f.add_node("A")
        assert f.get_concentration("A", MorphogenType.BUDGET) == 0.0

    def test_add_source_unknown_node_raises(self):
        f = DiffusionField()
        with pytest.raises(KeyError, match="Unknown node"):
            f.add_source(MorphogenSource("X", MorphogenType.COMPLEXITY, 0.5))


# ── TestDiffusionStep ───────────────────────────────────────────────────


class TestDiffusionStep:
    def test_source_emission(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.0, decay_rate=0.0))
        f.add_node("A")
        f.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5))
        f.step()
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.5

    def test_diffusion_to_neighbor(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.5, decay_rate=0.0))
        f.add_node("A")
        f.add_node("B")
        f.add_edge("A", "B")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 1.0)
        f.step()
        # A loses 0.5 to B, both decay 0%
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == pytest.approx(0.5)
        assert f.get_concentration("B", MorphogenType.COMPLEXITY) == pytest.approx(0.5)

    def test_decay(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.0, decay_rate=0.5))
        f.add_node("A")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 1.0)
        f.step()
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == pytest.approx(0.5)

    def test_clamp_at_one(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.0, decay_rate=0.0))
        f.add_node("A")
        f.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.8))
        f.set_concentration("A", MorphogenType.COMPLEXITY, 0.5)
        f.step()
        # 0.5 + 0.8 = 1.3, clamped to 1.0 by source max_concentration
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) <= 1.0

    def test_snap_to_zero(self):
        f = DiffusionField(params=DiffusionParams(
            diffusion_rate=0.0, decay_rate=0.99, min_concentration=0.01
        ))
        f.add_node("A")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 0.1)
        f.step()
        # 0.1 * (1 - 0.99) = 0.001 < 0.01 → snapped to 0
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.0

    def test_gradient_forms(self):
        """Source > near > far after diffusion with decay.

        Uses default params (diffusion=0.1, decay=0.05) which produce
        a stable gradient where the source concentration exceeds its
        neighbors.
        """
        f = DiffusionField()  # default params
        for n in ["source", "near", "far"]:
            f.add_node(n)
        f.add_edge("source", "near")
        f.add_edge("near", "far")
        f.add_source(MorphogenSource("source", MorphogenType.COMPLEXITY, 0.5))
        f.run(50)
        src = f.get_concentration("source", MorphogenType.COMPLEXITY)
        near = f.get_concentration("near", MorphogenType.COMPLEXITY)
        far = f.get_concentration("far", MorphogenType.COMPLEXITY)
        assert src > near > far > 0

    def test_no_diffusion_without_edges(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.5, decay_rate=0.0))
        f.add_node("A")
        f.add_node("B")
        # No edge between A and B
        f.set_concentration("A", MorphogenType.COMPLEXITY, 1.0)
        f.step()
        # A keeps all (no outflow because no neighbors)
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == pytest.approx(1.0)
        assert f.get_concentration("B", MorphogenType.COMPLEXITY) == 0.0


# ── TestDiffusionIntegration ────────────────────────────────────────────


class TestDiffusionIntegration:
    def test_get_local_gradient_returns_morphogen_gradient(self):
        f = DiffusionField()
        f.add_node("A")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 0.8)
        g = f.get_local_gradient("A")
        assert isinstance(g, MorphogenGradient)
        assert g.get(MorphogenType.COMPLEXITY) == pytest.approx(0.8)

    def test_from_adjacency(self):
        f = DiffusionField.from_adjacency({
            "A": ["B"],
            "B": ["A", "C"],
            "C": ["B"],
        })
        assert "B" in f._adjacency["A"]
        assert "A" in f._adjacency["B"]
        assert "C" in f._adjacency["B"]

    def test_snapshot(self):
        f = DiffusionField()
        f.add_node("A")
        f.set_concentration("A", MorphogenType.COMPLEXITY, 0.5)
        snap = f.snapshot()
        assert snap == {"A": {"complexity": 0.5}}

    def test_remove_source(self):
        f = DiffusionField(params=DiffusionParams(diffusion_rate=0.0, decay_rate=0.0))
        f.add_node("A")
        f.add_source(MorphogenSource("A", MorphogenType.COMPLEXITY, 0.5))
        f.step()
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.5
        f.remove_source("A", MorphogenType.COMPLEXITY)
        # Source removed, but concentration remains
        assert f.get_concentration("A", MorphogenType.COMPLEXITY) == 0.5
