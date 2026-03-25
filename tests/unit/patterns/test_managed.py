"""Tests for managed_organism and consolidate convenience functions."""

from operon_ai import (
    ManagedOrganism,
    ManagedRunResult,
    MockProvider,
    Nucleus,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    SkillStage,
    TaskFingerprint,
    Telomere,
    managed_organism,
    consolidate,
)


def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=2, subtask_count=2, required_roles=("worker",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _tmpl(tid="t1"):
    return PatternTemplate(
        template_id=tid, name="test", topology="skill_organism",
        stage_specs=({"name": "s1", "role": "Worker"},),
        intervention_policy={}, fingerprint=_fp(),
    )


def _nuclei():
    return (
        Nucleus(provider=MockProvider(responses={})),
        Nucleus(provider=MockProvider(responses={})),
    )


def _seeded_library():
    lib = PatternLibrary()
    lib.register_template(_tmpl())
    lib.record_run(PatternRunRecord(
        record_id=lib.make_id(), template_id="t1",
        fingerprint=_fp(), success=True, latency_ms=100, tokens_used=500,
    ))
    return lib


# ---------------------------------------------------------------------------
# managed_organism with stages (direct path)
# ---------------------------------------------------------------------------


def test_managed_with_stages():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    assert isinstance(m, ManagedOrganism)
    result = m.run("hello")
    assert isinstance(result, ManagedRunResult)
    assert result.run_result.final_output == "hello"


def test_managed_with_watcher_false():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
        watcher=False,
    )
    result = m.run("test")
    assert result.watcher_summary is None


def test_managed_with_watcher_true():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    result = m.run("test")
    assert result.watcher_summary is not None
    assert "total_stages_observed" in result.watcher_summary


# ---------------------------------------------------------------------------
# managed_organism with library (adaptive path)
# ---------------------------------------------------------------------------


def test_managed_with_library():
    fast, deep = _nuclei()
    lib = _seeded_library()
    m = managed_organism(
        task="test",
        library=lib,
        fingerprint=_fp(),
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"s1": lambda t: "done"},
    )
    result = m.run("test")
    assert result.template_used is not None
    assert result.adaptive_result is not None


# ---------------------------------------------------------------------------
# Development integration
# ---------------------------------------------------------------------------


def test_managed_with_telomere():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
        telomere=Telomere(max_operations=100),
    )
    result = m.run("test")
    assert result.development_status is not None
    assert result.development_status.tick_count >= 1


# ---------------------------------------------------------------------------
# Social learning integration
# ---------------------------------------------------------------------------


def test_managed_with_social():
    fast, deep = _nuclei()
    lib = _seeded_library()
    m = managed_organism(
        task="test",
        library=lib,
        fingerprint=_fp(),
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"s1": lambda t: "done"},
        organism_id="org-A",
    )
    exchange = m.export_templates()
    assert exchange is not None
    assert exchange.peer_id == "org-A"


def test_managed_no_social_returns_none():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    assert m.export_templates() is None
    assert m.import_from_peer(None) is None  # type: ignore


# ---------------------------------------------------------------------------
# Consolidate
# ---------------------------------------------------------------------------


def test_managed_consolidate():
    fast, deep = _nuclei()
    lib = _seeded_library()
    m = managed_organism(
        task="test",
        library=lib,
        fingerprint=_fp(),
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"s1": lambda t: "done"},
    )
    m.run("test")
    cr = m.consolidate()
    assert cr is not None


def test_managed_consolidate_no_library():
    fast, deep = _nuclei()
    m = managed_organism(
        stages=[SkillStage(name="echo", role="Echo", handler=lambda t: t)],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    assert m.consolidate() is None


def test_consolidate_top_level():
    lib = _seeded_library()
    cr = consolidate(lib)
    assert cr is not None
    assert cr.duration_ms >= 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status():
    fast, deep = _nuclei()
    lib = _seeded_library()
    m = managed_organism(
        task="test",
        library=lib,
        fingerprint=_fp(),
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"s1": lambda t: "done"},
        organism_id="org-A",
    )
    m.run("test")
    s = m.status()
    assert "watcher" in s
    assert "library" in s
    assert "social" in s
