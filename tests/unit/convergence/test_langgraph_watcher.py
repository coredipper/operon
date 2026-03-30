"""Tests for LangGraph watcher node."""

from operon_ai.convergence.langgraph_watcher import (
    create_watcher_config,
    operon_watcher_node,
)


class TestCreateWatcherConfig:
    def test_defaults(self):
        cfg = create_watcher_config()
        assert cfg["max_intervention_rate"] == 0.5
        assert cfg["max_retries_per_stage"] == 1

    def test_custom(self):
        cfg = create_watcher_config(max_intervention_rate=0.3, max_retries_per_stage=2)
        assert cfg["max_intervention_rate"] == 0.3
        assert cfg["max_retries_per_stage"] == 2


class TestOperonWatcherNode:
    def test_empty_state(self):
        result = operon_watcher_node({})
        assert result["should_halt"] is False
        assert result["watcher_summary"]["total_stages"] == 0
        assert result["watcher_signals"] == []
        assert result["watcher_interventions"] == []

    def test_success_stage(self):
        state = {
            "stage_results": [{"stage_name": "plan", "output": "ok", "action_type": "EXECUTE"}],
        }
        result = operon_watcher_node(state)
        assert len(result["watcher_signals"]) == 1
        assert result["watcher_signals"][0]["category"] == "epistemic"
        assert result["should_halt"] is False

    def test_failure_triggers_retry_lowercase(self):
        """Failure triggers retry with lowercase action (matching InterventionKind)."""
        state = {
            "stage_results": [{"stage_name": "code", "output": "", "action_type": "FAILURE"}],
        }
        result = operon_watcher_node(state)
        retry = [i for i in result["watcher_interventions"] if i["action"] == "retry"]
        assert len(retry) == 1

    def test_max_retries_triggers_escalate(self):
        """When retries exhausted, escalate instead of retry (with high enough threshold to avoid halt)."""
        state = {
            "stage_results": [
                {"stage_name": "ok1", "action_type": "EXECUTE"},
                {"stage_name": "ok2", "action_type": "EXECUTE"},
                {"stage_name": "code", "output": "", "action_type": "FAILURE"},
            ],
            "watcher_interventions": [
                {"stage_name": "code", "action": "retry"},
            ],
        }
        cfg = create_watcher_config(max_retries_per_stage=1, max_intervention_rate=0.9)
        result = operon_watcher_node(state, watcher_config=cfg)
        assert any(i["action"] == "escalate" for i in result["watcher_interventions"])

    def test_convergence_uses_pre_intervention_count(self):
        """First failure should NOT immediately halt — convergence checks pre-intervention count."""
        state = {
            "stage_results": [{"stage_name": "s1", "action_type": "FAILURE"}],
        }
        cfg = create_watcher_config(max_intervention_rate=0.5)
        result = operon_watcher_node(state, watcher_config=cfg)
        # Pre-intervention count is 0, rate = 0/1 = 0.0, should NOT halt.
        assert result["should_halt"] is False
        assert any(i["action"] == "retry" for i in result["watcher_interventions"])

    def test_high_pre_intervention_rate_halts(self):
        """Halt when pre-existing interventions already exceed threshold."""
        state = {
            "stage_results": [
                {"stage_name": "s1", "action_type": "EXECUTE"},
                {"stage_name": "s2", "action_type": "EXECUTE"},
            ],
            "watcher_interventions": [
                {"stage_name": "s0", "action": "retry"},
                {"stage_name": "s0", "action": "escalate"},
            ],
        }
        cfg = create_watcher_config(max_intervention_rate=0.5)
        result = operon_watcher_node(state, watcher_config=cfg)
        assert result["should_halt"] is True

    def test_cursor_prevents_duplicate_processing(self):
        """Re-invoking with same state should not duplicate signals."""
        state = {
            "stage_results": [{"stage_name": "a", "action_type": "EXECUTE"}],
        }
        result1 = operon_watcher_node(state)
        assert len(result1["watcher_signals"]) == 1
        # Re-invoke with cursor from first call.
        state2 = {**state, **result1}
        result2 = operon_watcher_node(state2)
        assert len(result2["watcher_signals"]) == 1  # no duplicates

    def test_summary_fields(self):
        state = {
            "stage_results": [{"stage_name": "a", "action_type": "EXECUTE"}],
        }
        result = operon_watcher_node(state)
        s = result["watcher_summary"]
        assert "total_stages" in s
        assert "total_signals" in s
        assert "intervention_rate" in s
        assert "convergent" in s
