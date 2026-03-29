from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from operon_ai.patterns.heartbeat import HeartbeatDaemon
from operon_ai.patterns.watcher import WatcherComponent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeConsolidationResult:
    templates_created: int = 1
    memories_promoted: int = 2


class _FakeConsolidation:
    """Minimal stand-in for SleepConsolidation."""

    def __init__(self):
        self.call_count = 0

    def consolidate(self) -> _FakeConsolidationResult:
        self.call_count += 1
        return _FakeConsolidationResult()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_heartbeat_is_watcher_component():
    daemon = HeartbeatDaemon()
    assert isinstance(daemon, WatcherComponent)


def test_on_run_start_increments_counter():
    daemon = HeartbeatDaemon()
    for _ in range(3):
        daemon.on_run_start("task", {})
    assert daemon._run_count_since_consolidate == 3


def test_on_run_start_clears_signals():
    """Parent on_run_start still clears per-run state."""
    daemon = HeartbeatDaemon()
    # Inject a fake signal to verify it gets cleared
    daemon.signals.append(SimpleNamespace(category="fake", source="test"))
    daemon.on_run_start("task", {})
    assert len(daemon.signals) == 0
    assert daemon._run_count_since_consolidate == 1


def test_heartbeat_no_consolidation():
    daemon = HeartbeatDaemon(consolidation=None)
    for _ in range(5):
        daemon.on_run_start("task", {})

    result = daemon.heartbeat()
    assert result["triggered"] is False
    assert result["reason"] == "no consolidation configured"


def test_heartbeat_not_enough_runs():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(consolidation=fake, min_runs_before_consolidate=5)
    for _ in range(3):
        daemon.on_run_start("task", {})

    result = daemon.heartbeat()
    assert result["triggered"] is False
    assert "only 3 runs" in result["reason"]
    assert fake.call_count == 0


def test_heartbeat_triggers_when_conditions_met():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=3,
    )
    for _ in range(3):
        daemon.on_run_start("task", {})

    result = daemon.heartbeat()
    assert result["triggered"] is True
    assert result["reason"] == "conditions met"
    assert isinstance(result["result"], _FakeConsolidationResult)
    assert fake.call_count == 1


def test_heartbeat_resets_counter_after_trigger():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=2,
    )
    for _ in range(2):
        daemon.on_run_start("task", {})

    daemon.heartbeat()
    assert daemon._run_count_since_consolidate == 0


def test_heartbeat_respects_interval():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=1,
        heartbeat_interval_s=600.0,
    )

    # First heartbeat triggers (no prior consolidation)
    daemon.on_run_start("task", {})
    result1 = daemon.heartbeat()
    assert result1["triggered"] is True

    # Simulate more runs but too soon since last consolidation
    for _ in range(3):
        daemon.on_run_start("task", {})

    # Manually set _last_consolidation to 60s ago (below 600s threshold)
    daemon._last_consolidation = datetime.now(UTC) - timedelta(seconds=60)
    result2 = daemon.heartbeat()
    assert result2["triggered"] is False
    assert "too soon" in result2["reason"]
    assert fake.call_count == 1  # Only the first call


def test_heartbeat_triggers_after_interval_elapses():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=1,
        heartbeat_interval_s=60.0,
    )

    # First trigger
    daemon.on_run_start("task", {})
    daemon.heartbeat()

    # Simulate enough time passing
    daemon._last_consolidation = datetime.now(UTC) - timedelta(seconds=120)
    daemon.on_run_start("task", {})

    result = daemon.heartbeat()
    assert result["triggered"] is True
    assert fake.call_count == 2


def test_summary_includes_heartbeat_stats():
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=2,
    )
    for _ in range(2):
        daemon.on_run_start("task", {})

    daemon.heartbeat()

    s = daemon.summary()
    assert "heartbeat" in s
    hb = s["heartbeat"]
    assert hb["runs_since_consolidate"] == 0
    assert hb["last_consolidation"] is not None
    assert hb["total_heartbeats_triggered"] == 1

    # Watcher base keys are still present
    assert "total_signals" in s
    assert "total_interventions" in s


def test_summary_before_any_heartbeat():
    daemon = HeartbeatDaemon()
    s = daemon.summary()
    hb = s["heartbeat"]
    assert hb["runs_since_consolidate"] == 0
    assert hb["last_consolidation"] is None
    assert hb["total_heartbeats_triggered"] == 0


def test_heartbeat_results_accumulate():
    """Multiple triggered heartbeats build up _heartbeat_results."""
    fake = _FakeConsolidation()
    daemon = HeartbeatDaemon(
        consolidation=fake,
        min_runs_before_consolidate=1,
        heartbeat_interval_s=0.0,  # No interval gating
    )

    for _ in range(3):
        daemon.on_run_start("task", {})
        daemon.heartbeat()

    assert len(daemon._heartbeat_results) == 3
    assert fake.call_count == 3
