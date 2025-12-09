"""Tests for topology patterns: CoherentFeedForwardLoop, QuorumSensing."""

import pytest
from io import StringIO
import sys

from operon_ai.state.metabolism import ATP_Store
from operon_ai.topology.loops import CoherentFeedForwardLoop
from operon_ai.topology.quorum import QuorumSensing


class TestCoherentFeedForwardLoop:
    """Tests for the CFFL (guardrail) topology."""

    def test_cffl_initialization(self):
        """CFFL initializes with executor and assessor agents."""
        budget = ATP_Store(budget=1000)
        cffl = CoherentFeedForwardLoop(budget=budget)

        assert cffl.executor is not None
        assert cffl.assessor is not None
        assert cffl.executor.role == "Executor"
        assert cffl.assessor.role == "RiskAssessor"

    def test_safe_request_processed(self, capsys):
        """Safe requests are processed by both agents."""
        budget = ATP_Store(budget=1000)
        cffl = CoherentFeedForwardLoop(budget=budget)

        cffl.run("Please calculate 2 + 2")

        captured = capsys.readouterr()
        # Both agents should express (process the signal)
        assert "Gene_Z" in captured.out  # Executor
        assert "Gene_Y" in captured.out  # Risk Assessor

    def test_dangerous_request_blocked(self, capsys):
        """Dangerous requests are blocked by the risk assessor."""
        budget = ATP_Store(budget=1000)
        cffl = CoherentFeedForwardLoop(budget=budget)

        cffl.run("Destroy the production database")

        captured = capsys.readouterr()
        assert "BLOCKED" in captured.out

    def test_budget_consumed_by_both_agents(self):
        """Both executor and assessor consume from shared budget."""
        budget = ATP_Store(budget=100)
        initial_budget = budget.atp

        cffl = CoherentFeedForwardLoop(budget=budget)
        cffl.run("Simple request")

        # Both agents should have consumed ATP (10 each by default)
        assert budget.atp < initial_budget

    def test_insufficient_budget_causes_failure(self, capsys):
        """CFFL fails gracefully when budget is exhausted."""
        budget = ATP_Store(budget=5)  # Not enough for two agents
        cffl = CoherentFeedForwardLoop(budget=budget)

        cffl.run("Any request")

        captured = capsys.readouterr()
        # Should indicate ATP/metabolic failure
        assert "APOPTOSIS" in captured.out or "ATP" in captured.out or "RUNTIME ERROR" in captured.out


class TestQuorumSensing:
    """Tests for the QuorumSensing (consensus) topology."""

    def test_quorum_initialization(self):
        """QuorumSensing creates the specified number of agents."""
        budget = ATP_Store(budget=1000)
        quorum = QuorumSensing(n_agents=5, budget=budget)

        assert len(quorum.colony) == 5
        for agent in quorum.colony:
            assert agent.role == "Voter"

    def test_quorum_reached_with_safe_request(self, capsys):
        """Safe requests should reach quorum."""
        budget = ATP_Store(budget=1000)
        quorum = QuorumSensing(n_agents=3, budget=budget)

        quorum.run_vote("Calculate 5 * 5")

        captured = capsys.readouterr()
        # With safe requests, should reach quorum or execute
        assert "QUORUM" in captured.out or "Concentration" in captured.out

    def test_quorum_shows_concentration(self, capsys):
        """QuorumSensing displays vote concentration."""
        budget = ATP_Store(budget=1000)
        quorum = QuorumSensing(n_agents=5, budget=budget)

        quorum.run_vote("Test request")

        captured = capsys.readouterr()
        assert "Concentration" in captured.out
        # Should show votes/total format
        assert "/" in captured.out

    def test_quorum_with_single_agent(self, capsys):
        """QuorumSensing works with single agent (trivial case)."""
        budget = ATP_Store(budget=1000)
        quorum = QuorumSensing(n_agents=1, budget=budget)

        quorum.run_vote("Simple request")

        captured = capsys.readouterr()
        assert "1/1" in captured.out or "Concentration" in captured.out

    def test_budget_consumed_by_all_agents(self):
        """All voting agents consume from the shared budget."""
        budget = ATP_Store(budget=1000)
        initial = budget.atp

        quorum = QuorumSensing(n_agents=5, budget=budget)
        quorum.run_vote("Test")

        # 5 agents * 10 ATP each = 50 ATP consumed
        assert budget.atp <= initial - 50

    def test_quorum_fails_with_insufficient_budget(self, capsys):
        """Quorum degrades gracefully with limited budget."""
        budget = ATP_Store(budget=25)  # Only enough for ~2 agents
        quorum = QuorumSensing(n_agents=5, budget=budget)

        quorum.run_vote("Test request")

        captured = capsys.readouterr()
        # Some agents will fail due to ATP exhaustion
        assert "APOPTOSIS" in captured.out or "ATP" in captured.out
