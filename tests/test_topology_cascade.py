import pytest
from unittest.mock import MagicMock

from operon_ai.state.metabolism import ATP_Store
from operon_ai.topology.cascade import (
    Cascade,
    CascadeStage,
    CascadeMode,
    StageStatus,
    AgentCascade,
    MAPKCascade,
)
from operon_ai.core.types import Signal, ActionProtein


class TestCascadeExtended:
    """Additional tests for the Cascade topology."""

    def test_cascade_checkpoint_error(self):
        """Test that a checkpoint error is handled correctly and halts the cascade if requested."""
        cascade = Cascade("CheckpointErrorTest", silent=True, halt_on_failure=True)

        def failing_checkpoint(x):
            raise ValueError("Intentional Checkpoint Error")

        cascade.add_stage(CascadeStage(
            name="stage1",
            processor=lambda x: x,
            checkpoint=failing_checkpoint
        ))

        result = cascade.run("initial")

        assert result.success is False
        assert len(result.stage_results) == 1
        assert result.stage_results[0].status == StageStatus.FAILED
        assert result.stage_results[0].error == "Intentional Checkpoint Error"

    def test_cascade_run_parallel(self):
        """Test that run_parallel executes stages and aggregates results."""
        cascade = Cascade("ParallelTest", silent=True)

        cascade.add_stage(CascadeStage(
            name="stage1",
            processor=lambda x: f"{x}_1"
        ))
        cascade.add_stage(CascadeStage(
            name="stage2",
            processor=lambda x: f"{x}_2"
        ))

        result = cascade.run_parallel("base")

        assert result.success is True
        assert result.stages_completed == 2
        # Order of outputs may vary in parallel execution, so we check using set
        assert set(result.final_output) == {"base_1", "base_2"}
        assert len(result.stage_results) == 2

    def test_cascade_error_recovery(self):
        """Test that a stage can recover from an error using on_error handler."""
        cascade = Cascade("ErrorRecovery", silent=True)

        def failing_processor(x):
            raise ValueError("Intentional Error")

        def error_handler(e):
            return "recovered"

        cascade.add_stage(CascadeStage(
            name="stage1",
            processor=failing_processor,
            on_error=error_handler
        ))

        result = cascade.run("test")

        assert result.success is True
        assert result.final_output == "recovered"
        assert result.stage_results[0].status == StageStatus.COMPLETED

    def test_cascade_non_required_stage(self):
        """Test that failure in a non-required stage doesn't halt the cascade."""
        cascade = Cascade("NonRequiredStage", silent=True, halt_on_failure=True)

        def failing_processor(x):
            raise ValueError("Intentional Error")

        cascade.add_stage(CascadeStage(
            name="stage1",
            processor=lambda x: "start",
        ))
        cascade.add_stage(CascadeStage(
            name="stage2_optional",
            processor=failing_processor,
            required=False
        ))
        cascade.add_stage(CascadeStage(
            name="stage3",
            processor=lambda x: f"{x}_end",
        ))

        result = cascade.run("initial")

        assert result.success is False  # A stage failed/skipped so overall success is false because not all stages completed
        # Actually wait, let's check the logic:
        # success = completed_stages == len(self._stages) and blocked_at is None
        # So success will be False, but final_output might be None if success is False
        assert result.stages_completed == 2
        assert result.stage_results[1].status == StageStatus.SKIPPED
        assert result.stage_results[2].status == StageStatus.COMPLETED

    def test_cascade_statistics_history(self):
        cascade = Cascade("HistoryTest", silent=True)
        cascade.add_stage(CascadeStage(name="s1", processor=lambda x: x))
        cascade.run("test1")
        cascade.run("test2")
        history = cascade.get_history(1)
        assert len(history) == 1
        assert history[0].stage_results[0].input_signal == "test2"


class TestAgentCascade:
    """Tests for the AgentCascade."""

    def test_agent_cascade_add_agent_stage(self):
        budget = ATP_Store(budget=1000, silent=True)
        cascade = AgentCascade("AgentCascade", budget=budget, silent=True)

        # We need to mock BioAgent.express to avoid LLM calls
        cascade.add_agent_stage(
            agent_name="MockAgent1",
            role="Processor",
            amplification=2.0
        )

        # Access the agent and mock express
        agent = cascade._agents[0]
        mock_protein = ActionProtein(action_type="test", payload="processed", confidence=1.0)
        agent.express = MagicMock(return_value=mock_protein)

        result = cascade.run("input_signal")

        assert result.success is True
        assert result.final_output == "processed"
        assert result.total_amplification == 2.0
        agent.express.assert_called_once()


class TestMAPKCascade:
    """Tests for MAPKCascade."""

    def test_mapk_cascade_execution(self):
        cascade = MAPKCascade("TestMAPK", max_amplification=1000.0, silent=True)

        result = cascade.run("initial_signal")

        assert result.success is True
        assert result.stages_completed == 3
        assert result.final_output.get("signal") == "initial_signal"
        assert result.final_output.get("tier") == 3
        assert result.final_output.get("response") == "ACTIVATED"
        assert result.total_amplification == 10.0 * 10.0 * 10.0

    def test_mapk_cascade_blocked(self):
        cascade = MAPKCascade("TestMAPK", max_amplification=1000.0, silent=True)

        # Override the first stage to not activate
        cascade._stages[0].processor = lambda x: {"signal": x, "tier": 1, "active": False}

        result = cascade.run("initial_signal")

        assert result.success is False
        assert result.stages_completed == 1
        assert result.stage_results[1].status == StageStatus.BLOCKED
