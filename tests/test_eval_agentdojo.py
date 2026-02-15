"""Tests for the AgentDojo immune detection suite."""
from __future__ import annotations

import random

from eval.suites.agentdojo_immune import AgentDojoImmuneConfig, run_agentdojo_immune


class TestAgentDojoImmune:
    def test_returns_expected_keys(self):
        config = AgentDojoImmuneConfig(agents=10, compromised=2, eval_observations=5)
        rng = random.Random(42)
        result = run_agentdojo_immune(config, rng)
        assert "agents" in result
        assert "compromised" in result
        assert "sensitivity" in result
        assert "false_positive_rate" in result

    def test_sensitivity_has_counter_fields(self):
        config = AgentDojoImmuneConfig(agents=10, compromised=2, eval_observations=5)
        rng = random.Random(42)
        result = run_agentdojo_immune(config, rng)
        for key in ("success", "total", "rate", "wilson_95"):
            assert key in result["sensitivity"]
            assert key in result["false_positive_rate"]

    def test_sensitivity_total_equals_compromised(self):
        config = AgentDojoImmuneConfig(agents=10, compromised=3, eval_observations=5)
        rng = random.Random(42)
        result = run_agentdojo_immune(config, rng)
        assert result["sensitivity"]["total"] == 3

    def test_false_positive_total_equals_clean(self):
        config = AgentDojoImmuneConfig(agents=10, compromised=3, eval_observations=5)
        rng = random.Random(42)
        result = run_agentdojo_immune(config, rng)
        assert result["false_positive_rate"]["total"] == 7

    def test_deterministic(self):
        config = AgentDojoImmuneConfig(agents=8, compromised=2, eval_observations=5)
        r1 = run_agentdojo_immune(config, random.Random(99))
        r2 = run_agentdojo_immune(config, random.Random(99))
        assert r1 == r2
