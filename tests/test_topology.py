"""Tests for topology patterns: CFFL, NegativeFeedback, QuorumSensing, Cascade."""

import pytest
from operon_ai.state.metabolism import ATP_Store
from operon_ai.topology.loops import (
    CoherentFeedForwardLoop,
    NegativeFeedbackLoop,
    GateLogic,
    CircuitState,
    LoopResult,
)
from operon_ai.topology.quorum import (
    QuorumSensing,
    VotingStrategy,
    VoteType,
    AgentProfile,
)
from operon_ai.topology.cascade import (
    Cascade,
    CascadeStage,
    CascadeMode,
    StageStatus,
)


class TestCoherentFeedForwardLoop:
    """Tests for the CFFL (guardrail) topology."""

    def test_cffl_initialization(self):
        """CFFL initializes with executor and assessor agents."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)

        assert cffl.executor is not None
        assert cffl.assessor is not None
        assert cffl.executor.role == "Executor"
        assert cffl.assessor.role == "RiskAssessor"

    def test_cffl_default_gate_logic(self):
        """CFFL defaults to AND gate logic."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)
        assert cffl.gate_logic == GateLogic.AND

    def test_cffl_run_returns_loop_result(self):
        """run() returns a LoopResult object."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)
        result = cffl.run("Test request")

        assert isinstance(result, LoopResult)
        assert result.gate_logic == GateLogic.AND

    def test_cffl_circuit_breaker_initial_state(self):
        """Circuit breaker starts in CLOSED state."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)
        stats = cffl.get_circuit_breaker_stats()

        assert stats.state == CircuitState.CLOSED
        assert stats.failure_count == 0

    def test_cffl_caching(self):
        """CFFL caches identical requests."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, enable_cache=True, silent=True)

        result1 = cffl.run("Cached request")
        result2 = cffl.run("Cached request")

        assert result2.cached is True

    def test_cffl_cache_disabled(self):
        """CFFL doesn't cache when caching is disabled."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, enable_cache=False, silent=True)

        result1 = cffl.run("Request")
        result2 = cffl.run("Request")

        assert result2.cached is False

    def test_cffl_statistics(self):
        """CFFL tracks statistics correctly."""
        budget = ATP_Store(budget=1000, silent=True)
        cffl = CoherentFeedForwardLoop(budget=budget, silent=True)

        cffl.run("Request 1")
        cffl.run("Request 2")

        stats = cffl.get_statistics()
        assert stats['total_requests'] == 2
        assert 'block_rate' in stats
        assert 'cache_size' in stats

    def test_cffl_gate_logic_options(self):
        """CFFL accepts different gate logic configurations."""
        budget = ATP_Store(budget=1000, silent=True)

        for logic in [GateLogic.AND, GateLogic.OR, GateLogic.EXECUTOR_PRIORITY]:
            cffl = CoherentFeedForwardLoop(budget=budget, gate_logic=logic, silent=True)
            assert cffl.gate_logic == logic


class TestNegativeFeedbackLoop:
    """Tests for the Negative Feedback Loop."""

    def test_nfl_initialization(self):
        """NegativeFeedbackLoop initializes with setpoint."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, silent=True)
        assert nfl.setpoint == 100.0

    def test_nfl_measure_error(self):
        """measure() calculates and returns correction."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, gain=1.0, damping=0.0, silent=True)

        correction = nfl.measure(80.0)  # 20 below setpoint
        assert correction > 0  # Should be positive (upward correction)

    def test_nfl_apply_correction(self):
        """apply() returns corrected value."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, gain=0.5, damping=0.0, silent=True)

        corrected = nfl.apply(80.0)
        assert corrected > 80.0  # Should move toward setpoint

    def test_nfl_setpoint_change(self):
        """Setpoint can be changed dynamically."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, silent=True)
        nfl.set_setpoint(200.0)
        assert nfl.setpoint == 200.0

    def test_nfl_error_calculation(self):
        """get_error() returns current error from setpoint."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, silent=True)
        nfl.measure(75.0)  # Set current value

        error = nfl.get_error()
        assert error == 25.0  # 100 - 75

    def test_nfl_statistics(self):
        """Statistics track corrections correctly."""
        nfl = NegativeFeedbackLoop(setpoint=100.0, silent=True)

        nfl.measure(90.0)
        nfl.measure(95.0)
        nfl.measure(98.0)

        stats = nfl.get_statistics()
        assert stats['corrections_count'] == 3
        assert 'total_correction' in stats
        assert 'average_correction' in stats

    def test_nfl_min_correction_threshold(self):
        """Small corrections below threshold are ignored."""
        nfl = NegativeFeedbackLoop(
            setpoint=100.0,
            gain=0.1,
            min_correction=5.0,  # Minimum 5 unit correction
            silent=True
        )

        # Very small error, correction would be < 5
        correction = nfl.measure(99.0)
        assert correction == 0.0  # Below threshold

    def test_nfl_max_correction_limit(self):
        """Large corrections are clamped to maximum."""
        nfl = NegativeFeedbackLoop(
            setpoint=100.0,
            gain=1.0,
            max_correction=10.0,  # Maximum 10 unit correction
            silent=True
        )

        correction = nfl.measure(0.0)  # Huge error of 100
        assert abs(correction) <= 10.0


class TestQuorumSensing:
    """Tests for the QuorumSensing (consensus) topology."""

    def test_quorum_initialization(self):
        """QuorumSensing creates the specified number of agents."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=5, budget=budget, silent=True)

        assert len(quorum.colony) == 5
        for profile in quorum.colony:
            assert isinstance(profile, AgentProfile)
            assert profile.agent.role == "Voter"

    def test_quorum_default_strategy(self):
        """QuorumSensing defaults to MAJORITY strategy."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)
        assert quorum.strategy == VotingStrategy.MAJORITY

    def test_quorum_set_strategy(self):
        """Strategy can be changed dynamically."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)

        quorum.set_strategy(VotingStrategy.UNANIMOUS)
        assert quorum.strategy == VotingStrategy.UNANIMOUS

    def test_quorum_run_vote(self):
        """run_vote() returns a QuorumResult."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)

        result = quorum.run_vote("Should we proceed?")

        assert result.total_votes == 3
        assert result.strategy == VotingStrategy.MAJORITY
        assert result.permit_votes + result.block_votes + result.abstain_votes == result.total_votes

    def test_quorum_add_agent(self):
        """Agents can be added to the colony."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=2, budget=budget, silent=True)

        profile = quorum.add_agent("NewAgent", weight=2.0)

        assert len(quorum.colony) == 3
        assert profile.weight == 2.0

    def test_quorum_remove_agent(self):
        """Agents can be removed from the colony."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)

        # Remove the first agent
        name = quorum.colony[0].agent.name
        success = quorum.remove_agent(name)

        assert success is True
        assert len(quorum.colony) == 2

    def test_quorum_set_agent_weight(self):
        """Agent weights can be modified."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=2, budget=budget, silent=True)

        name = quorum.colony[0].agent.name
        success = quorum.set_agent_weight(name, 5.0)

        assert success is True
        assert quorum.colony[0].weight == 5.0

    def test_quorum_statistics(self):
        """Statistics are tracked correctly."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)

        quorum.run_vote("Vote 1")
        quorum.run_vote("Vote 2")

        stats = quorum.get_statistics()
        assert stats['n_agents'] == 3
        assert stats['total_votes'] == 6  # 3 agents × 2 votes
        assert 'success_rate' in stats

    def test_quorum_agent_rankings(self):
        """Agent rankings return sorted by effective weight."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=3, budget=budget, silent=True)

        # Modify weights
        quorum.set_agent_weight(quorum.colony[0].agent.name, 3.0)
        quorum.set_agent_weight(quorum.colony[1].agent.name, 1.0)
        quorum.set_agent_weight(quorum.colony[2].agent.name, 2.0)

        rankings = quorum.get_agent_rankings()

        assert len(rankings) == 3
        # Should be sorted by effective weight (descending)
        assert rankings[0]['effective_weight'] >= rankings[1]['effective_weight']

    def test_quorum_vote_history(self):
        """Vote history is recorded."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(n_agents=2, budget=budget, silent=True)

        quorum.run_vote("First vote")
        quorum.run_vote("Second vote")

        history = quorum.get_vote_history()
        assert len(history) == 2


class TestCascade:
    """Tests for the Cascade (signal amplification) topology."""

    def test_cascade_initialization(self):
        """Cascade initializes with name and default mode."""
        cascade = Cascade("TestCascade", silent=True)
        assert cascade.name == "TestCascade"
        assert cascade.mode == CascadeMode.SEQUENTIAL

    def test_cascade_add_stage(self):
        """Stages can be added to cascade."""
        cascade = Cascade("Test", silent=True)

        stage = CascadeStage(
            name="uppercase",
            processor=lambda x: x.upper(),
        )
        cascade.add_stage(stage)

        stats = cascade.get_statistics()
        assert stats['stages_count'] == 1
        assert 'uppercase' in stats['stage_names']

    def test_cascade_run_single_stage(self):
        """Cascade processes single stage correctly."""
        cascade = Cascade("Test", silent=True)
        cascade.add_stage(CascadeStage(
            name="double",
            processor=lambda x: x * 2,
        ))

        result = cascade.run(5)

        assert result.success is True
        assert result.final_output == 10
        assert result.stages_completed == 1

    def test_cascade_run_multiple_stages(self):
        """Cascade processes multiple stages in sequence."""
        cascade = Cascade("StringProcessor", silent=True)

        cascade.add_stage(CascadeStage(
            name="strip",
            processor=lambda x: x.strip(),
        ))
        cascade.add_stage(CascadeStage(
            name="lower",
            processor=lambda x: x.lower(),
        ))
        cascade.add_stage(CascadeStage(
            name="replace",
            processor=lambda x: x.replace(" ", "_"),
        ))

        result = cascade.run("  HELLO WORLD  ")

        assert result.success is True
        assert result.final_output == "hello_world"
        assert result.stages_completed == 3

    def test_cascade_amplification(self):
        """Cascade tracks amplification factor."""
        cascade = Cascade("Amplifier", silent=True)

        cascade.add_stage(CascadeStage(
            name="double",
            processor=lambda x: x * 2,
            amplification=2.0,
        ))
        cascade.add_stage(CascadeStage(
            name="triple",
            processor=lambda x: x * 3,
            amplification=3.0,
        ))

        result = cascade.run(1)

        assert result.final_output == 6  # 1 * 2 * 3
        assert result.total_amplification == 6.0  # 2.0 * 3.0

    def test_cascade_checkpoint_blocks(self):
        """Checkpoint can block cascade progression."""
        cascade = Cascade("Gated", halt_on_failure=True, silent=True)

        cascade.add_stage(CascadeStage(
            name="process",
            processor=lambda x: x,
        ))
        cascade.add_stage(CascadeStage(
            name="gate",
            processor=lambda x: x,
            checkpoint=lambda x: x > 10,  # Only proceed if > 10
        ))
        cascade.add_stage(CascadeStage(
            name="final",
            processor=lambda x: x * 2,
        ))

        result = cascade.run(5)  # Will fail checkpoint

        assert result.success is False
        assert result.stages_completed < 3

    def test_cascade_stage_results(self):
        """Stage results are recorded correctly."""
        cascade = Cascade("Test", silent=True)

        cascade.add_stage(CascadeStage(name="s1", processor=lambda x: x + 1))
        cascade.add_stage(CascadeStage(name="s2", processor=lambda x: x + 1))

        result = cascade.run(0)

        assert len(result.stage_results) == 2
        assert result.stage_results[0].stage_name == "s1"
        assert result.stage_results[1].stage_name == "s2"

    def test_cascade_error_handling(self):
        """Cascade handles stage errors correctly."""
        cascade = Cascade("ErrorTest", halt_on_failure=True, silent=True)

        cascade.add_stage(CascadeStage(
            name="error_stage",
            processor=lambda x: 1 / 0,  # Will raise ZeroDivisionError
        ))

        result = cascade.run(1)

        assert result.success is False

    def test_cascade_statistics(self):
        """Cascade tracks statistics."""
        cascade = Cascade("Stats", silent=True)
        cascade.add_stage(CascadeStage(name="s1", processor=lambda x: x))

        cascade.run("test")
        cascade.run("test2")

        stats = cascade.get_statistics()
        assert stats['runs_count'] == 2
        assert 'success_rate' in stats
        assert stats['successful_runs'] == 2


class TestVotingStrategies:
    """Tests for different QuorumSensing voting strategies."""

    def test_majority_strategy(self):
        """MAJORITY strategy is configured correctly."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=3,
            budget=budget,
            strategy=VotingStrategy.MAJORITY,
            silent=True
        )

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.MAJORITY
        assert result.total_votes == 3  # 3 agents voted

    def test_supermajority_strategy(self):
        """SUPERMAJORITY strategy is configured correctly."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=3,
            budget=budget,
            strategy=VotingStrategy.SUPERMAJORITY,
            silent=True
        )

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.SUPERMAJORITY
        assert result.total_votes == 3

    def test_weighted_strategy(self):
        """WEIGHTED uses weight-adjusted voting."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=3,
            budget=budget,
            strategy=VotingStrategy.WEIGHTED,
            silent=True
        )

        # Give first agent much higher weight
        quorum.set_agent_weight(quorum.colony[0].agent.name, 10.0)

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.WEIGHTED

    def test_confidence_strategy(self):
        """CONFIDENCE uses confidence-weighted aggregation."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=3,
            budget=budget,
            strategy=VotingStrategy.CONFIDENCE,
            silent=True
        )

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.CONFIDENCE
        assert 0 <= result.confidence_score <= 1

    def test_bayesian_strategy(self):
        """BAYESIAN uses Bayesian belief aggregation."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=5,
            budget=budget,
            strategy=VotingStrategy.BAYESIAN,
            silent=True
        )

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.BAYESIAN
        # Weighted score is the posterior probability
        assert 0 <= result.weighted_score <= 1

    def test_threshold_strategy(self):
        """THRESHOLD requires fixed count of permits."""
        budget = ATP_Store(budget=1000, silent=True)
        quorum = QuorumSensing(
            n_agents=5,
            budget=budget,
            strategy=VotingStrategy.THRESHOLD,
            threshold=3,  # Need exactly 3 permits
            silent=True
        )

        result = quorum.run_vote("Test")
        assert result.strategy == VotingStrategy.THRESHOLD
