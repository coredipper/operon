from operon_ai import PatternLibrary, PatternRunRecord, PatternTemplate, TaskFingerprint


def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=3, subtask_count=2, required_roles=("writer",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _tmpl(tid="t1", name="tmpl", topology="skill_organism", fp=None, tags=()):
    return PatternTemplate(
        template_id=tid,
        name=name,
        topology=topology,
        stage_specs=({"name": "s1"},),
        intervention_policy={},
        fingerprint=fp or _fp(),
        tags=tags,
    )


def _record(tid="t1", success=True, **kw):
    defaults = dict(
        record_id=PatternLibrary.make_id(),
        template_id=tid,
        fingerprint=_fp(),
        success=success,
        latency_ms=100.0,
        tokens_used=500,
    )
    defaults.update(kw)
    return PatternRunRecord(**defaults)


# -- Template CRUD --------------------------------------------------------


def test_register_and_get_template():
    lib = PatternLibrary()
    t = _tmpl()
    lib.register_template(t)
    assert lib.get_template("t1") is t


def test_register_overwrites_existing_template():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="t1", name="old"))
    lib.register_template(_tmpl(tid="t1", name="new"))
    assert lib.get_template("t1").name == "new"


def test_get_template_returns_none_for_unknown_id():
    lib = PatternLibrary()
    assert lib.get_template("nonexistent") is None


def test_retrieve_templates_filters_by_topology():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="a", topology="reviewer_gate"))
    lib.register_template(_tmpl(tid="b", topology="skill_organism"))
    lib.register_template(_tmpl(tid="c", topology="reviewer_gate"))
    results = lib.retrieve_templates(topology="reviewer_gate")
    assert len(results) == 2
    assert all(t.topology == "reviewer_gate" for t in results)


def test_retrieve_templates_filters_by_tags():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="a", tags=("compliance", "finance")))
    lib.register_template(_tmpl(tid="b", tags=("finance",)))
    lib.register_template(_tmpl(tid="c", tags=("compliance",)))
    results = lib.retrieve_templates(tags=("compliance",))
    assert len(results) == 2
    ids = {t.template_id for t in results}
    assert ids == {"a", "c"}


# -- Run records ----------------------------------------------------------


def test_record_run_stores_record():
    lib = PatternLibrary()
    lib.record_run(_record(tid="t1", success=True))
    lib.record_run(_record(tid="t1", success=False))
    assert lib.summary()["record_count"] == 2


def test_success_rate_with_mixed_outcomes():
    lib = PatternLibrary()
    lib.record_run(_record(tid="t1", success=True))
    lib.record_run(_record(tid="t1", success=True))
    lib.record_run(_record(tid="t1", success=False))
    assert lib.success_rate("t1") == 2 / 3


def test_success_rate_returns_none_for_unrecorded_template():
    lib = PatternLibrary()
    assert lib.success_rate("nonexistent") is None


# -- Ranking --------------------------------------------------------------


def test_top_templates_for_exact_shape_match_scores_highest():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="seq", fp=_fp(task_shape="sequential")))
    lib.register_template(_tmpl(tid="par", fp=_fp(task_shape="parallel")))
    results = lib.top_templates_for(_fp(task_shape="sequential"))
    assert results[0][0].template_id == "seq"
    assert results[0][1] > results[1][1]


def test_top_templates_for_role_overlap_affects_score():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="match", fp=_fp(required_roles=("writer", "analyst"))))
    lib.register_template(_tmpl(tid="miss", fp=_fp(required_roles=("coder",))))
    results = lib.top_templates_for(_fp(required_roles=("writer", "analyst")))
    assert results[0][0].template_id == "match"


def test_top_templates_for_tool_count_proximity():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="close", fp=_fp(tool_count=4)))
    lib.register_template(_tmpl(tid="far", fp=_fp(tool_count=20)))
    results = lib.top_templates_for(_fp(tool_count=3))
    assert results[0][0].template_id == "close"


def test_top_templates_for_respects_limit():
    lib = PatternLibrary()
    for i in range(10):
        lib.register_template(_tmpl(tid=f"t{i}"))
    results = lib.top_templates_for(_fp(), limit=3)
    assert len(results) == 3


def test_top_templates_for_incorporates_success_rate():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="good", fp=_fp()))
    lib.register_template(_tmpl(tid="bad", fp=_fp()))
    # Give "good" a 100% success rate, "bad" 0%
    lib.record_run(_record(tid="good", success=True))
    lib.record_run(_record(tid="bad", success=False))
    results = lib.top_templates_for(_fp())
    scores = {r[0].template_id: r[1] for r in results}
    assert scores["good"] > scores["bad"]


def test_top_templates_for_empty_library_returns_empty():
    lib = PatternLibrary()
    results = lib.top_templates_for(_fp())
    assert results == []


# -- Summary --------------------------------------------------------------


def test_summary_contains_template_and_record_counts():
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="a", topology="reviewer_gate"))
    lib.register_template(_tmpl(tid="b", topology="skill_organism"))
    lib.record_run(_record(tid="a"))
    s = lib.summary()
    assert s["template_count"] == 2
    assert s["record_count"] == 1
    assert s["topologies"] == {"reviewer_gate": 1, "skill_organism": 1}


# -- make_id --------------------------------------------------------------


def test_make_id_returns_8_char_hex():
    mid = PatternLibrary.make_id()
    assert len(mid) == 8
    int(mid, 16)  # should not raise
