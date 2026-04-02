"""Tests for Ralph compiler."""

import pytest

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.ralph_compiler import organism_to_ralph, managed_to_ralph
from operon_ai.convergence.types import RuntimeConfig


def _make_organism(n_stages=3):
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))
    stages = []
    for i in range(n_stages):
        mode = "fixed" if i == n_stages - 1 else "fuzzy"
        stages.append(SkillStage(
            name=f"stage_{i}", role=f"role_{i}",
            instructions=f"Do step {i}", mode=mode,
            handler=lambda t: f"done_{i}",
        ))
    return skill_organism(stages=stages, fast_nucleus=fast, deep_nucleus=deep)


class TestOrganismToRalph:
    def test_produces_valid_dict(self):
        result = organism_to_ralph(_make_organism())
        assert "hats" in result
        assert "events" in result
        assert "backend" in result
        assert "backpressure" in result

    def test_hats_match_stages(self):
        result = organism_to_ralph(_make_organism(3))
        assert len(result["hats"]) == 3
        assert result["hats"][0]["name"] == "stage_0"

    def test_hat_pattern_from_mode(self):
        result = organism_to_ralph(_make_organism(3))
        assert result["hats"][0]["pattern"] == "code-assist"  # fuzzy
        assert result["hats"][2]["pattern"] == "review"  # fixed

    def test_events_follow_stage_order(self):
        result = organism_to_ralph(_make_organism(3))
        assert len(result["events"]) == 2
        assert result["events"][0]["from"] == "stage_0"
        assert result["events"][0]["to"] == "stage_1"

    def test_backend_parameter(self):
        result = organism_to_ralph(_make_organism(), backend="gemini")
        assert result["backend"] == "gemini"

    def test_backpressure_when_review_stage(self):
        result = organism_to_ralph(_make_organism(3))
        assert "tests" in result["backpressure"]

    def test_system_prompt_from_instructions(self):
        result = organism_to_ralph(_make_organism())
        assert result["hats"][0]["system_prompt"] == "Do step 0"

    def test_empty_stages_raises(self):
        """Organism with no stages raises ValueError (at construction or compilation)."""
        with pytest.raises(ValueError):
            fast = Nucleus(provider=MockProvider(responses={}))
            org = skill_organism(stages=[], fast_nucleus=fast, deep_nucleus=fast)
            organism_to_ralph(org)
