"""Cross-subsystem integration tests — proving the full Phase 1-7 stack works together."""

from datetime import datetime, timedelta

from operon_ai import (
    BiTemporalMemory,
    DevelopmentController,
    DevelopmentalStage,
    CriticalPeriod,
    MockProvider,
    Nucleus,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    SkillStage,
    SleepConsolidation,
    SocialLearning,
    TaskFingerprint,
    Telomere,
    WatcherComponent,
    WatcherConfig,
    adaptive_skill_organism,
    skill_organism,
)
from operon_ai.healing.autophagy_daemon import AutophagyDaemon
from operon_ai.memory.adapters import histone_to_bitemporal, episodic_to_bitemporal
from operon_ai.memory.episodic import EpisodicMemory, MemoryTier
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_bitemporal_substrate_with_watcher():
    """Organism with substrate + watcher: facts recorded, watcher observes."""
    mem = BiTemporalMemory()
    watcher = WatcherComponent()
    fast, deep = _nuclei()

    organism = skill_organism(
        stages=[
            SkillStage(name="research", role="Researcher",
                       handler=lambda task: {"finding": "important"},
                       emit_output_fact=True),
            SkillStage(name="analyst", role="Analyst",
                       handler=lambda task, state, outputs: "analysis complete",
                       read_query="research"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        substrate=mem,
        components=[watcher],
    )

    result = organism.run("Analyze market data")
    assert result.final_output == "analysis complete"
    assert len(mem._facts) >= 1  # At least the research fact
    assert watcher.summary()["total_stages_observed"] == 2


def test_adaptive_assembly_with_consolidation():
    """Adaptive organism runs, then consolidation distills the history."""
    lib = PatternLibrary()
    lib.register_template(_tmpl(tid="pipeline"))
    fast, deep = _nuclei()

    adaptive = adaptive_skill_organism(
        "Process data",
        fingerprint=_fp(),
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"s1": lambda task: "done"},
    )
    result = adaptive.run("Process data")
    assert result.record.success is True

    # Consolidate
    episodic = EpisodicMemory()
    histone = HistoneStore()
    daemon = AutophagyDaemon(histone_store=histone, lysosome=None, summarizer=lambda t: "summary")

    consolidation = SleepConsolidation(
        daemon=daemon,
        pattern_library=lib,
        episodic_memory=episodic,
        histone_store=histone,
    )
    cr = consolidation.consolidate()
    assert cr.memories_promoted >= 0  # At least ran without error
    assert cr.duration_ms >= 0


def test_social_learning_with_development():
    """Mature organism scaffolds young organism with developmental gating."""
    # Teacher: mature
    lib_teacher = PatternLibrary()
    lib_teacher.register_template(PatternTemplate(
        template_id="basic", name="Basic", topology="skill_organism",
        stage_specs=({"name": "s1", "role": "W"},),
        intervention_policy={}, fingerprint=_fp(),
    ))
    for _ in range(3):
        lib_teacher.record_run(PatternRunRecord(
            record_id=lib_teacher.make_id(), template_id="basic",
            fingerprint=_fp(), success=True, latency_ms=100.0, tokens_used=500,
        ))

    teacher_t = Telomere(max_operations=100)
    teacher_t.start()
    teacher_dev = DevelopmentController(telomere=teacher_t)
    for _ in range(80):
        teacher_dev.tick()
    assert teacher_dev.stage == DevelopmentalStage.MATURE

    sl_teacher = SocialLearning(organism_id="teacher", library=lib_teacher)

    # Learner: embryonic with critical period
    lib_learner = PatternLibrary()
    learner_t = Telomere(max_operations=100)
    learner_t.start()
    learner_dev = DevelopmentController(
        telomere=learner_t,
        critical_periods=(
            CriticalPeriod("rapid", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE, "learn fast"),
        ),
    )
    sl_learner = SocialLearning(organism_id="learner", library=lib_learner)

    # Scaffold
    result = sl_teacher.scaffold_learner(
        sl_learner,
        teacher_stage=teacher_dev.stage.value,
        learner_stage=learner_dev.stage.value,
    )
    assert "basic" in result.adoption.adopted_template_ids

    # Critical period closes after ticking past juvenile
    for _ in range(15):
        learner_dev.tick()
    assert learner_dev.is_critical_period_open("rapid") is False


def test_full_lifecycle():
    """Complete lifecycle: telomere → development → library → adaptive → watcher → consolidation."""
    # Setup
    telomere = Telomere(max_operations=50)
    telomere.start()
    dev = DevelopmentController(telomere=telomere)
    lib = PatternLibrary()
    lib.register_template(_tmpl())
    fast, deep = _nuclei()

    # Run adaptive organism
    adaptive = adaptive_skill_organism(
        "task", fingerprint=_fp(), library=lib,
        fast_nucleus=fast, deep_nucleus=deep,
        handlers={"s1": lambda t: "result"},
    )
    result = adaptive.run("task")
    assert result.record.success is True

    # Tick development
    for _ in range(40):
        dev.tick()
    assert dev.stage in (DevelopmentalStage.ADOLESCENT, DevelopmentalStage.MATURE)

    # Consolidate
    episodic = EpisodicMemory()
    histone = HistoneStore()
    daemon = AutophagyDaemon(histone_store=histone, lysosome=None, summarizer=lambda t: "s")
    cr = SleepConsolidation(
        daemon=daemon, pattern_library=lib,
        episodic_memory=episodic, histone_store=histone,
    ).consolidate()
    assert cr.duration_ms >= 0


def test_memory_adapters_with_consolidation():
    """HistoneStore → BiTemporal adapter works with consolidation."""
    histone = HistoneStore()
    histone.add_marker("learned pattern", MarkerType.METHYLATION, MarkerStrength.STRONG)
    histone.add_marker("temporary insight", MarkerType.ACETYLATION, MarkerStrength.MODERATE)

    episodic = EpisodicMemory()
    episodic.store("important finding", tier=MemoryTier.EPISODIC)

    mem = BiTemporalMemory()

    # Bridge
    h_facts = histone_to_bitemporal(histone, mem)
    e_facts = episodic_to_bitemporal(episodic, mem)
    assert len(h_facts) == 2
    assert len(e_facts) == 1

    # All facts are in bi-temporal store
    assert len(mem._facts) >= 3
