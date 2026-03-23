"""Tests for SocialLearning and TrustRegistry."""

from operon_ai import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
    SocialLearning,
    PeerExchange,
    TrustRegistry,
    AdoptionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=2, subtask_count=2, required_roles=("worker",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _tmpl(tid="t1", tags=()):
    return PatternTemplate(
        template_id=tid, name=f"test-{tid}", topology="skill_organism",
        stage_specs=({"name": "s1"},), intervention_policy={},
        fingerprint=_fp(), tags=tags,
    )


def _record(tid="t1", success=True):
    return PatternRunRecord(
        record_id=PatternLibrary.make_id(), template_id=tid,
        fingerprint=_fp(), success=success, latency_ms=100.0, tokens_used=500,
    )


def _seeded_library(tid="t1", successes=3, failures=0, tags=()):
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid=tid, tags=tags))
    for _ in range(successes):
        lib.record_run(_record(tid=tid, success=True))
    for _ in range(failures):
        lib.record_run(_record(tid=tid, success=False))
    return lib


# ---------------------------------------------------------------------------
# TrustRegistry
# ---------------------------------------------------------------------------


def test_default_trust_for_unknown_peer():
    tr = TrustRegistry(default_trust=0.5)
    assert tr.trust_score("unknown") == 0.5


def test_trust_increases_on_success():
    tr = TrustRegistry(default_trust=0.5, decay_alpha=0.3)
    new = tr.record_outcome("peer1", "t1", success=True)
    assert new > 0.5


def test_trust_decreases_on_failure():
    tr = TrustRegistry(default_trust=0.5, decay_alpha=0.3)
    new = tr.record_outcome("peer1", "t1", success=False)
    assert new < 0.5


def test_ema_weights_recent_outcomes():
    tr = TrustRegistry(default_trust=0.5, decay_alpha=0.5)
    tr.record_outcome("p", "t", True)   # 0.5*1 + 0.5*0.5 = 0.75
    tr.record_outcome("p", "t", False)  # 0.5*0 + 0.5*0.75 = 0.375
    assert tr.trust_score("p") < 0.5  # recent failure dominates


def test_is_trusted_returns_false_below_min():
    tr = TrustRegistry(default_trust=0.1, min_trust_to_adopt=0.2)
    assert tr.is_trusted("peer1") is False


def test_is_trusted_returns_true_above_min():
    tr = TrustRegistry(default_trust=0.5, min_trust_to_adopt=0.2)
    assert tr.is_trusted("peer1") is True


def test_peer_rankings_sorted_descending():
    tr = TrustRegistry(default_trust=0.5, decay_alpha=0.5)
    tr.record_outcome("high", "t", True)
    tr.record_outcome("low", "t", False)
    rankings = tr.peer_rankings()
    assert rankings[0]["peer_id"] == "high"
    assert rankings[1]["peer_id"] == "low"


def test_trust_summary():
    tr = TrustRegistry()
    tr.record_outcome("p1", "t1", True)
    s = tr.summary()
    assert s["peer_count"] == 1
    assert s["total_outcomes"] == 1


# ---------------------------------------------------------------------------
# SocialLearning.export_templates
# ---------------------------------------------------------------------------


def test_export_filters_by_success_rate():
    lib = _seeded_library(tid="good", successes=4, failures=1)  # 80%
    lib.register_template(_tmpl(tid="bad"))
    lib.record_run(_record(tid="bad", success=False))  # 0%
    sl = SocialLearning(organism_id="A", library=lib)
    exchange = sl.export_templates(min_success_rate=0.6)
    ids = {t.template_id for t in exchange.templates}
    assert "good" in ids
    assert "bad" not in ids


def test_export_filters_by_min_runs():
    lib = _seeded_library(tid="t1", successes=1)
    sl = SocialLearning(organism_id="A", library=lib)
    exchange = sl.export_templates(min_runs=5)
    assert len(exchange.templates) == 0


def test_export_contains_organism_id():
    lib = _seeded_library()
    sl = SocialLearning(organism_id="org-A", library=lib)
    exchange = sl.export_templates()
    assert exchange.peer_id == "org-A"


def test_export_empty_library():
    sl = SocialLearning(organism_id="A", library=PatternLibrary())
    exchange = sl.export_templates()
    assert len(exchange.templates) == 0


# ---------------------------------------------------------------------------
# SocialLearning.import_from_peer
# ---------------------------------------------------------------------------


def test_import_adopts_templates():
    lib_a = _seeded_library(tid="shared", successes=3)
    sl_a = SocialLearning(organism_id="A", library=lib_a)
    exchange = sl_a.export_templates()

    lib_b = PatternLibrary()
    sl_b = SocialLearning(organism_id="B", library=lib_b)
    result = sl_b.import_from_peer(exchange)
    assert "shared" in result.adopted_template_ids
    assert lib_b.get_template("shared") is not None


def test_import_rejects_when_trust_too_low():
    lib_a = _seeded_library(tid="shared", successes=3)
    sl_a = SocialLearning(organism_id="A", library=lib_a)
    exchange = sl_a.export_templates()

    lib_b = PatternLibrary()
    tr = TrustRegistry(default_trust=0.05, min_trust_to_adopt=0.2)
    sl_b = SocialLearning(organism_id="B", library=lib_b, trust=tr)
    result = sl_b.import_from_peer(exchange)
    assert len(result.adopted_template_ids) == 0
    assert "shared" in result.rejected_template_ids


def test_import_with_trust_override():
    lib_a = _seeded_library(tid="shared", successes=3)
    sl_a = SocialLearning(organism_id="A", library=lib_a)
    exchange = sl_a.export_templates()

    lib_b = PatternLibrary()
    tr = TrustRegistry(default_trust=0.05, min_trust_to_adopt=0.2)
    sl_b = SocialLearning(organism_id="B", library=lib_b, trust=tr)
    result = sl_b.import_from_peer(exchange, trust_override=0.9)
    assert "shared" in result.adopted_template_ids


def test_import_tracks_provenance():
    lib_a = _seeded_library(tid="shared", successes=3)
    sl_a = SocialLearning(organism_id="A", library=lib_a)
    exchange = sl_a.export_templates()

    lib_b = PatternLibrary()
    sl_b = SocialLearning(organism_id="B", library=lib_b)
    sl_b.import_from_peer(exchange)
    assert sl_b.get_provenance("shared") == "A"


# ---------------------------------------------------------------------------
# record_adoption_outcome
# ---------------------------------------------------------------------------


def test_record_outcome_updates_trust():
    lib_a = _seeded_library(tid="shared", successes=3)
    sl_a = SocialLearning(organism_id="A", library=lib_a)
    exchange = sl_a.export_templates()

    lib_b = PatternLibrary()
    sl_b = SocialLearning(organism_id="B", library=lib_b)
    sl_b.import_from_peer(exchange)
    score = sl_b.record_adoption_outcome("shared", success=True)
    assert score is not None
    assert score > sl_b.trust.default_trust


def test_record_outcome_returns_none_for_local():
    lib = _seeded_library()
    sl = SocialLearning(organism_id="A", library=lib)
    assert sl.record_adoption_outcome("t1", success=True) is None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_two_organisms_share_templates():
    lib_a = _seeded_library(tid="template_a", successes=5)
    sl_a = SocialLearning(organism_id="A", library=lib_a)

    lib_b = PatternLibrary()
    sl_b = SocialLearning(organism_id="B", library=lib_b)

    exchange = sl_a.export_templates()
    sl_b.import_from_peer(exchange)

    assert lib_b.get_template("template_a") is not None
    assert sl_b.get_provenance("template_a") == "A"


def test_summary():
    lib = _seeded_library()
    sl = SocialLearning(organism_id="A", library=lib)
    s = sl.summary()
    assert s["organism_id"] == "A"
    assert "adopted_count" in s
