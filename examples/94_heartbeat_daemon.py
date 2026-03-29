"""
Example 94 — HeartbeatDaemon
===============================

Demonstrates the HeartbeatDaemon: a WatcherComponent that triggers
SleepConsolidation after enough runs accumulate.

Usage:
    python examples/94_heartbeat_daemon.py
"""

from types import SimpleNamespace
from operon_ai.patterns.heartbeat import HeartbeatDaemon
from operon_ai.patterns.watcher import WatcherComponent

# Mock consolidation that tracks calls
call_count = 0
def mock_consolidate():
    global call_count
    call_count += 1
    return SimpleNamespace(templates_created=1, memories_promoted=2)

mock_consolidation = SimpleNamespace(consolidate=mock_consolidate)

# Create daemon with low threshold for demo
daemon = HeartbeatDaemon(
    consolidation=mock_consolidation,
    min_runs_before_consolidate=3,
    heartbeat_interval_s=0,  # no cooldown for demo
)

print("=== HeartbeatDaemon ===")
print(f"  isinstance(WatcherComponent): {isinstance(daemon, WatcherComponent)}")

# Simulate 2 runs — not enough for consolidation
for i in range(2):
    daemon.on_run_start(f"task_{i}", {})
result = daemon.heartbeat()
print(f"\n  After 2 runs: triggered={result['triggered']}, reason={result['reason']}")

# Third run — now conditions are met
daemon.on_run_start("task_2", {})
result = daemon.heartbeat()
print(f"  After 3 runs: triggered={result['triggered']}, reason={result['reason']}")

# Summary includes heartbeat stats
summary = daemon.summary()
print(f"\n  Summary heartbeat: {summary.get('heartbeat', {})}")

# --test
assert isinstance(daemon, WatcherComponent)
assert result["triggered"] is True
assert call_count == 1
assert summary["heartbeat"]["total_heartbeats_triggered"] == 1
print("\n--- all assertions passed ---")
