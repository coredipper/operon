"""Tests for distributed watcher transport layer."""

from operon_ai.convergence.distributed_watcher import (
    DistributedWatcher,
    HttpTransport,
    InMemoryTransport,
)


class TestInMemoryTransport:
    def test_publish_signal_logged(self):
        t = InMemoryTransport()
        t.publish_signal({"type": "epistemic", "value": 0.5})
        assert len(t._signal_log) == 1
        assert t._signal_log[0]["type"] == "epistemic"

    def test_publish_intervention_logged(self):
        t = InMemoryTransport()
        t.publish_intervention({"action": "RETRY"})
        assert len(t._intervention_log) == 1

    def test_subscribe_signals_callback(self):
        t = InMemoryTransport()
        received = []
        t.subscribe_signals(lambda s: received.append(s))
        t.publish_signal({"x": 1})
        assert len(received) == 1
        assert received[0]["x"] == 1

    def test_subscribe_interventions_callback(self):
        t = InMemoryTransport()
        received = []
        t.subscribe_interventions(lambda i: received.append(i))
        t.publish_intervention({"action": "HALT"})
        assert len(received) == 1

    def test_multiple_subscribers(self):
        t = InMemoryTransport()
        a, b = [], []
        t.subscribe_signals(lambda s: a.append(s))
        t.subscribe_signals(lambda s: b.append(s))
        t.publish_signal({"v": 1})
        assert len(a) == 1 and len(b) == 1


class TestHttpTransport:
    def test_publish_signal_produces_request(self):
        t = HttpTransport(signal_url="https://example.com/signals")
        t.publish_signal({"type": "somatic"})
        pending = t.get_pending_requests()
        assert len(pending) == 1
        assert pending[0]["url"] == "https://example.com/signals"
        assert pending[0]["method"] == "POST"
        assert pending[0]["body"]["type"] == "somatic"

    def test_publish_intervention_produces_request(self):
        t = HttpTransport(intervention_url="https://example.com/interventions")
        t.publish_intervention({"action": "ESCALATE"})
        pending = t.get_pending_requests()
        assert len(pending) == 1
        assert pending[0]["url"] == "https://example.com/interventions"

    def test_get_pending_clears(self):
        t = HttpTransport(signal_url="http://x")
        t.publish_signal({"v": 1})
        t.get_pending_requests()
        assert len(t.get_pending_requests()) == 0


class TestDistributedWatcher:
    def test_publish_stage_result_signals(self):
        t = InMemoryTransport()
        dw = DistributedWatcher(transport=t, organism_id="org1")
        dw.publish_stage_result("research", [{"type": "epistemic"}])
        assert len(t._signal_log) == 1
        assert t._signal_log[0]["organism_id"] == "org1"
        assert t._signal_log[0]["stage_name"] == "research"

    def test_publish_stage_result_with_intervention(self):
        t = InMemoryTransport()
        dw = DistributedWatcher(transport=t)
        dw.publish_stage_result("plan", [], intervention={"action": "RETRY"})
        assert len(t._intervention_log) == 1
        assert t._intervention_log[0]["action"] == "RETRY"

    def test_publish_heartbeat(self):
        t = InMemoryTransport()
        dw = DistributedWatcher(transport=t, organism_id="org2")
        dw.publish_heartbeat({"status": "healthy"})
        assert len(t._signal_log) == 1
        assert t._signal_log[0]["type"] == "heartbeat"

    def test_summary(self):
        t = InMemoryTransport()
        dw = DistributedWatcher(transport=t, organism_id="test")
        dw.publish_stage_result("s1", [{"a": 1}, {"b": 2}])
        dw.publish_stage_result("s2", [{"c": 3}], intervention={"action": "HALT"})
        s = dw.summary()
        assert s["organism_id"] == "test"
        assert s["signals_published"] == 3
        assert s["interventions_published"] == 1
