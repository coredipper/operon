"""Tests for Regulatory T-Cell tolerance."""
import pytest
from datetime import datetime, timedelta
from operon_ai.surveillance.types import Signal1, Signal2, ThreatLevel, ResponseAction
from operon_ai.surveillance.tcell import ImmuneResponse
from operon_ai.surveillance.treg import (
    RegulatoryTCell, SuppressionRule, ToleranceRecord, SuppressionResult,
)


def make_response(
    agent_id: str = "test",
    threat_level: ThreatLevel = ThreatLevel.CONFIRMED,
    action: ResponseAction = ResponseAction.ISOLATE,
    violations: list[str] = None,
) -> ImmuneResponse:
    """Helper to create test immune responses."""
    return ImmuneResponse(
        agent_id=agent_id,
        threat_level=threat_level,
        action=action,
        signal1=Signal1.NON_SELF,
        signal2=Signal2.CANARY_FAILED,
        violations=violations or ["output_length out of bounds"],
    )


class TestSuppressionRule:
    def test_create_rule(self):
        rule = SuppressionRule(
            name="after_update",
            condition=lambda resp, rec: rec.recent_update,
            duration=timedelta(hours=1),
        )
        assert rule.name == "after_update"

    def test_rule_matches(self):
        rule = SuppressionRule(
            name="known_variance",
            condition=lambda resp, rec: "output_length" in str(resp.violations),
        )
        record = ToleranceRecord(agent_id="test")
        response = make_response(violations=["output_length out of bounds"])
        assert rule.condition(response, record) is True

    def test_rule_not_matches(self):
        rule = SuppressionRule(
            name="known_variance",
            condition=lambda resp, rec: "output_length" in str(resp.violations),
        )
        record = ToleranceRecord(agent_id="test")
        response = make_response(violations=["vocabulary_hash unknown"])
        assert rule.condition(response, record) is False


class TestToleranceRecord:
    def test_create_record(self):
        record = ToleranceRecord(agent_id="test")
        assert record.agent_id == "test"
        assert record.clean_inspections == 0
        assert record.recent_update is False

    def test_record_clean_inspection(self):
        record = ToleranceRecord(agent_id="test")
        record.record_inspection(clean=True)
        assert record.clean_inspections == 1
        assert record.total_inspections == 1

    def test_record_dirty_inspection_resets_streak(self):
        record = ToleranceRecord(agent_id="test")
        record.record_inspection(clean=True)
        record.record_inspection(clean=True)
        record.record_inspection(clean=False)
        assert record.clean_inspections == 0  # Streak reset
        assert record.total_inspections == 3

    def test_mark_updated(self):
        record = ToleranceRecord(agent_id="test")
        record.mark_updated()
        assert record.recent_update is True
        assert record.last_update is not None

    def test_update_expires(self):
        record = ToleranceRecord(agent_id="test", update_tolerance_duration=timedelta(seconds=1))
        record.mark_updated()
        assert record.recent_update is True

        # Manually set update time to past
        record.last_update = datetime.utcnow() - timedelta(seconds=2)
        assert record.recent_update is False  # Expired


class TestRegulatoryTCell:
    def test_create_treg(self):
        treg = RegulatoryTCell()
        assert treg.stability_threshold == 100

    def test_evaluate_no_suppression(self):
        treg = RegulatoryTCell()
        record = ToleranceRecord(agent_id="test")
        response = make_response()

        result = treg.evaluate(response, record)
        assert result.suppressed is False
        assert result.modified_action == ResponseAction.ISOLATE

    def test_evaluate_suppresses_after_update(self):
        rule = SuppressionRule(
            name="post_update",
            condition=lambda resp, rec: rec.recent_update,
            max_severity=ThreatLevel.CONFIRMED,  # Can suppress up to CONFIRMED
        )
        treg = RegulatoryTCell(rules=[rule])

        record = ToleranceRecord(agent_id="test")
        record.mark_updated()

        response = make_response(threat_level=ThreatLevel.CONFIRMED)
        result = treg.evaluate(response, record)

        assert result.suppressed is True
        assert result.modified_action == ResponseAction.MONITOR  # Downgraded
        assert result.suppression_reason == "post_update"

    def test_evaluate_does_not_suppress_critical(self):
        rule = SuppressionRule(
            name="post_update",
            condition=lambda resp, rec: rec.recent_update,
            max_severity=ThreatLevel.CONFIRMED,  # Cannot suppress CRITICAL
        )
        treg = RegulatoryTCell(rules=[rule])

        record = ToleranceRecord(agent_id="test")
        record.mark_updated()

        response = make_response(threat_level=ThreatLevel.CRITICAL)
        result = treg.evaluate(response, record)

        assert result.suppressed is False
        assert result.modified_action == ResponseAction.ISOLATE  # Not changed

    def test_stable_agent_auto_tolerance(self):
        treg = RegulatoryTCell(stability_threshold=5)

        record = ToleranceRecord(agent_id="test")
        for _ in range(5):
            record.record_inspection(clean=True)

        assert record.is_stable(threshold=5)

        response = make_response(threat_level=ThreatLevel.SUSPICIOUS)
        result = treg.evaluate(response, record)

        # Stable agents get leniency for SUSPICIOUS threats
        assert result.suppressed is True
        assert result.modified_action == ResponseAction.IGNORE

    def test_known_violation_tolerance(self):
        rule = SuppressionRule(
            name="known_output_variance",
            condition=lambda resp, rec: (
                len(resp.violations) == 1 and
                "output_length" in resp.violations[0]
            ),
            max_severity=ThreatLevel.SUSPICIOUS,
        )
        treg = RegulatoryTCell(rules=[rule])
        record = ToleranceRecord(agent_id="test")

        response = make_response(
            threat_level=ThreatLevel.SUSPICIOUS,
            violations=["output_length out of bounds: 150.0 not in [90.0, 110.0]"],
        )
        result = treg.evaluate(response, record)

        assert result.suppressed is True
        assert result.suppression_reason == "known_output_variance"

    def test_register_agent(self):
        treg = RegulatoryTCell()
        treg.register_agent("new_agent")

        assert "new_agent" in treg.records
        record = treg.get_record("new_agent")
        assert record.agent_id == "new_agent"

    def test_multiple_rules_first_match_wins(self):
        rule1 = SuppressionRule(
            name="rule1",
            condition=lambda resp, rec: True,  # Always matches
            max_severity=ThreatLevel.CONFIRMED,
        )
        rule2 = SuppressionRule(
            name="rule2",
            condition=lambda resp, rec: True,  # Also matches
            max_severity=ThreatLevel.CONFIRMED,
        )
        treg = RegulatoryTCell(rules=[rule1, rule2])
        record = ToleranceRecord(agent_id="test")
        response = make_response(threat_level=ThreatLevel.CONFIRMED)

        result = treg.evaluate(response, record)
        assert result.suppression_reason == "rule1"  # First match
