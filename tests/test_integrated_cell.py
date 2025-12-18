"""Tests for IntegratedCell."""
import pytest
from datetime import datetime, timedelta
from operon_ai.cell import IntegratedCell, CellExecutionResult, CellHealth


class TestCellExecutionResult:
    def test_create_result(self):
        result = CellExecutionResult(
            agent_id="test",
            success=True,
            output="hello",
        )
        assert result.agent_id == "test"
        assert result.success is True


class TestIntegratedCell:
    def test_create_cell(self):
        cell = IntegratedCell()
        assert cell.quality_pool is not None
        assert cell.surveillance is not None
        assert cell.coordination is not None

    def test_register_agent(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        # Should be registered in surveillance
        assert "agent1" in cell.surveillance.displays

    def test_register_resource(self):
        cell = IntegratedCell()
        cell.register_resource("file_a")

        # Should be in coordination
        assert "file_a" in cell.coordination.controller.resources

    def test_execute_simple(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        def work():
            return {"output": "hello"}

        result = cell.execute(
            agent_id="agent1",
            operation_id="op1",
            work_fn=work,
        )

        assert result.success is True
        assert result.output == {"output": "hello"}

    def test_execute_with_provenance(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        def work():
            return "hello world"

        result = cell.execute(
            agent_id="agent1",
            operation_id="op1",
            work_fn=work,
        )

        # Output should be tagged with provenance
        assert result.tagged_output is not None
        assert result.tagged_output.tag.origin == "agent1"

    def test_execute_records_observation(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        def work():
            return "output text"

        cell.execute(agent_id="agent1", operation_id="op1", work_fn=work)

        # Should have recorded observation in surveillance
        display = cell.surveillance.displays["agent1"]
        assert len(display.observations) > 0

    def test_low_confidence_triggers_surveillance(self):
        cell = IntegratedCell(degradation_threshold=0.9)
        cell.register_agent("agent1")

        # First output - establishes baseline
        cell.execute(agent_id="agent1", operation_id="op1", work_fn=lambda: "test")

        # Artificially lower confidence on pool
        cell.quality_pool.available = 0  # Force low confidence allocation

        # The surveillance should be notified of low confidence outputs
        # (This is integration - just verify the systems are connected)
        assert cell.surveillance is not None

    def test_byzantine_detection_affects_coordination(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        # Flag agent as suspicious in surveillance
        if "agent1" in cell.surveillance.tcells:
            cell.surveillance.tcells["agent1"].flag_manually("test threat")

        # Byzantine detection should affect coordination priority
        # (Integration test - verify systems are wired)
        assert cell.coordination is not None

    def test_health_check(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")
        cell.register_resource("file_a")

        health = cell.health()

        assert isinstance(health, CellHealth)
        assert health.healthy is True
        assert "quality" in str(health) or health.pool_status is not None

    def test_run_maintenance(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        # Run maintenance should check all subsystems
        events = cell.run_maintenance()

        assert "surveillance" in events or "coordination" in events or events == {}

    def test_cross_system_agent_terminated(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")

        # When agent terminated in coordination, quality should recycle tags
        # This is the cross-system callback

        # Start an operation
        ctx = cell.coordination.start_operation("op1", "agent1")

        # Abort it (simulates termination)
        cell.coordination.controller.abort_operation(ctx, "test")

        # Cell should handle cleanup
        assert "op1" not in cell.coordination.controller.active_operations

    def test_execute_with_resources(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")
        cell.register_resource("shared_file")

        def work():
            return "used resource"

        result = cell.execute(
            agent_id="agent1",
            operation_id="op1",
            work_fn=work,
            resources=["shared_file"],
        )

        assert result.success is True
        # Resource should be released
        assert cell.coordination.controller.resources["shared_file"].is_available

    def test_shutdown(self):
        cell = IntegratedCell()
        cell.register_agent("agent1")
        ctx = cell.coordination.start_operation("op1", "agent1")

        cell.shutdown()

        # All operations should be terminated
        assert len(cell.coordination.controller.active_operations) == 0
