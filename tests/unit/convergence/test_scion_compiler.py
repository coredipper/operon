"""Tests for Scion compiler."""

import pytest

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.scion_compiler import organism_to_scion, managed_to_scion
from operon_ai.convergence.types import RuntimeConfig


def _make_organism(n_stages=3):
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))
    stages = []
    for i in range(n_stages):
        stages.append(SkillStage(
            name=f"stage_{i}", role=f"role_{i}",
            instructions=f"Do step {i}", mode="fuzzy",
            handler=lambda t: f"done_{i}",
        ))
    return skill_organism(stages=stages, fast_nucleus=fast, deep_nucleus=deep)


class TestOrganismToScion:
    def test_produces_valid_dict(self):
        result = organism_to_scion(_make_organism())
        assert "grove" in result
        assert "agents" in result
        assert "messaging" in result
        assert "watcher" in result

    def test_grove_config(self):
        result = organism_to_scion(_make_organism(), grove_name="my-project", runtime="podman")
        assert result["grove"]["name"] == "my-project"
        assert result["grove"]["runtime"] == "podman"

    def test_agents_per_stage_plus_watcher(self):
        result = organism_to_scion(_make_organism(3))
        assert len(result["agents"]) == 4  # 3 stages + 1 watcher

    def test_watcher_agent_present(self):
        result = organism_to_scion(_make_organism())
        watcher_agents = [a for a in result["agents"] if a["name"] == "operon-watcher"]
        assert len(watcher_agents) == 1
        assert result["watcher"]["enabled"] is True
        assert result["watcher"]["telemetry"] == "otel"

    def test_isolation_defaults(self):
        result = organism_to_scion(_make_organism())
        stage_agents = [a for a in result["agents"] if a["name"] != "operon-watcher"]
        for agent in stage_agents:
            assert agent["isolation"]["git_worktree"] is True
            assert agent["isolation"]["credentials"] == "isolated"

    def test_messaging_follows_stage_order(self):
        result = organism_to_scion(_make_organism(3))
        assert len(result["messaging"]) == 2
        assert result["messaging"][0]["from"] == "stage_0"
        assert result["messaging"][0]["to"] == "stage_1"

    def test_runtime_from_config(self):
        cfg = RuntimeConfig(sandbox="k8s")
        result = organism_to_scion(_make_organism(), config=cfg)
        assert result["grove"]["runtime"] == "k8s"

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError):
            fast = Nucleus(provider=MockProvider(responses={}))
            org = skill_organism(stages=[], fast_nucleus=fast, deep_nucleus=fast)
            organism_to_scion(org)
