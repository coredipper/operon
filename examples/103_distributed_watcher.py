"""
Example 103 — Distributed Watcher
====================================

Demonstrates transport-agnostic watcher signal distribution across
single-process (InMemory) and webhook (HTTP) transports.

Usage:
    python examples/103_distributed_watcher.py
"""

from operon_ai.convergence.distributed_watcher import (
    DistributedWatcher,
    HttpTransport,
    InMemoryTransport,
)

# ---------------------------------------------------------------------------
# 1. In-memory transport (single-process)
# ---------------------------------------------------------------------------

transport = InMemoryTransport()
received_signals = []
transport.subscribe_signals(lambda s: received_signals.append(s))

watcher = DistributedWatcher(transport=transport, organism_id="org-alpha")

# Simulate 3 stage results
watcher.publish_stage_result("research", [{"type": "epistemic", "value": 0.7}])
watcher.publish_stage_result("plan", [{"type": "epistemic", "value": 0.5}])
watcher.publish_stage_result(
    "execute",
    [{"type": "somatic", "value": 0.3}],
    intervention={"action": "RETRY", "reason": "low confidence"},
)

print("=== In-Memory Transport ===")
print(f"  Signals published: {watcher.summary()['signals_published']}")
print(f"  Interventions published: {watcher.summary()['interventions_published']}")
print(f"  Signals received by subscriber: {len(received_signals)}")
print()

# ---------------------------------------------------------------------------
# 2. HTTP transport (webhook-based)
# ---------------------------------------------------------------------------

http = HttpTransport(
    signal_url="https://monitoring.example.com/signals",
    intervention_url="https://monitoring.example.com/interventions",
)
http_watcher = DistributedWatcher(transport=http, organism_id="org-beta")

http_watcher.publish_stage_result("analyze", [{"type": "epistemic"}])
http_watcher.publish_heartbeat({"status": "healthy", "runs_since_consolidate": 5})

pending = http.get_pending_requests()
print("=== HTTP Transport ===")
print(f"  Pending requests: {len(pending)}")
for req in pending:
    print(f"    {req['method']} {req['url']} → {list(req['body'].keys())}")
print()

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------

print("=== Summary ===")
print(f"  org-alpha: {watcher.summary()}")
print(f"  org-beta: {http_watcher.summary()}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert watcher.summary()["signals_published"] == 3
assert watcher.summary()["interventions_published"] == 1
assert len(received_signals) == 3
assert len(pending) == 2
assert http_watcher.summary()["signals_published"] == 2  # 1 signal + 1 heartbeat
print("\n--- all assertions passed ---")
