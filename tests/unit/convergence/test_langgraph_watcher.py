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

    def test_failure_triggers_retry(self):
        state = {
            "stage_results": [{"stage_name": "code", "output": "", "action_type": "FAILURE"}],
        }
        result = operon_watcher_node(state)
        assert any(i["action"] == "RETRY" for i in result["watcher_interventions"])

    def test_max_retries_triggers_escalate(self):
        state = {
            "stage_results": [{"stage_name": "code", "output": "", "action_type": "FAILURE"}],
            "watcher_interventions": [
                {"stage_name": "code", "action": "RETRY"},
            ],
        }
        cfg = create_watcher_config(max_retries_per_stage=1)
        result = operon_watcher_node(state, watcher_config=cfg)
        assert any(i["action"] == "ESCALATE" for i in result["watcher_interventions"])

    def test_high_intervention_rate_halts(self):
        state = {
            "stage_results": [
                {"stage_name": "s1", "action_type": "EXECUTE"},
            ],
            "watcher_interventions": [
                {"stage_name": "s0", "action": "RETRY"},
            ],
        }
        cfg = create_watcher_config(max_intervention_rate=0.5)
        result = operon_watcher_node(state, watcher_config=cfg)
        # 1 pre-existing + possibly 0 new = 1 intervention / 1 stage = 1.0 > 0.5
        assert result["should_halt"] is True

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
