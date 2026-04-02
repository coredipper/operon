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
        """When retries exhausted, escalate instead of retry."""
        # Pre-process 3 stages to keep intervention rate low, then add failure.
        state1 = {
            "stage_results": [
                {"stage_name": "ok1", "action_type": "EXECUTE"},
                {"stage_name": "ok2", "action_type": "EXECUTE"},
                {"stage_name": "ok3", "action_type": "EXECUTE"},
            ],
        }
        result1 = operon_watcher_node(state1)
        # Now add a failure with an existing retry.
        state2 = {
            "stage_results": state1["stage_results"] + [
                {"stage_name": "code", "action_type": "FAILURE"},
            ],
            "watcher_signals": result1["watcher_signals"],
            "watcher_interventions": [{"stage_name": "code", "action": "retry"}],
            "_watcher_cursor": result1["_watcher_cursor"],
        }
        cfg = create_watcher_config(max_retries_per_stage=1)
        result2 = operon_watcher_node(state2, watcher_config=cfg)
        assert any(i["action"] == "escalate" for i in result2["watcher_interventions"])

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

    def test_summary_convergent_uses_configured_threshold(self):
        """Summary 'convergent' should use configured max_rate, not hardcoded 0.5.

        1 intervention / 3 stages = 0.33. With max_rate=0.3, convergent should
        be False even though 0.33 < 0.5 (the old hardcoded threshold).
        """
        # Pre-process 3 stages so cursor is past them.
        state1 = {"stage_results": [
            {"stage_name": "a", "action_type": "EXECUTE"},
            {"stage_name": "b", "action_type": "EXECUTE"},
            {"stage_name": "c", "action_type": "EXECUTE"},
        ]}
        result1 = operon_watcher_node(state1)
        # Inject 1 intervention into the accumulated state (rate = 1/3 = 0.33).
        state2 = {
            "stage_results": state1["stage_results"],
            "watcher_signals": result1["watcher_signals"],
            "watcher_interventions": [{"stage_name": "x", "action": "retry"}],
            "_watcher_cursor": result1["_watcher_cursor"],
        }
        cfg = create_watcher_config(max_intervention_rate=0.3)
        result2 = operon_watcher_node(state2, watcher_config=cfg)
        # No new results to process, so should_halt=False, but summary
        # should reflect that 0.33 > 0.3 → not convergent.
        assert result2["should_halt"] is False
        assert result2["watcher_summary"]["convergent"] is False

    def test_batch_processing_incremental(self):
        """Multiple new results processed incrementally with cursor advancing."""
        state = {"stage_results": [
            {"stage_name": "s1", "action_type": "EXECUTE"},
            {"stage_name": "s2", "action_type": "EXECUTE"},
            {"stage_name": "s3", "action_type": "EXECUTE"},
        ]}
        result = operon_watcher_node(state)
        assert len(result["watcher_signals"]) == 3
        assert result["_watcher_cursor"] == 3
        # Add more with existing cursor.
        state2 = {
            "stage_results": state["stage_results"] + [
                {"stage_name": "s4", "action_type": "FAILURE"},
            ],
            **{k: v for k, v in result.items() if k != "stage_results"},
        }
        result2 = operon_watcher_node(state2)
        assert len(result2["watcher_signals"]) == 4
        assert result2["_watcher_cursor"] == 4

    def test_batch_halt_stops_at_correct_cursor(self):
        """When halt triggers mid-batch, cursor stops at the halting item."""
        state = {
            "stage_results": [
                {"stage_name": "s1", "action_type": "EXECUTE"},
                {"stage_name": "s2", "action_type": "EXECUTE"},
            ],
            "watcher_interventions": [
                {"stage_name": "x", "action": "retry"},
                {"stage_name": "y", "action": "retry"},
            ],
        }
        cfg = create_watcher_config(max_intervention_rate=0.5)
        result = operon_watcher_node(state, watcher_config=cfg)
        assert result["should_halt"] is True
        assert result["_watcher_cursor"] == 1  # only first result processed
