"""Managed organism — batteries-included factory for the full v0.19–0.23 stack.

Composes PatternLibrary, WatcherComponent, BiTemporalMemory, DevelopmentController,
SocialLearning, and SleepConsolidation behind a single factory function with
sensible defaults. Everything is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


def _lazy_imports():
    """Deferred imports to avoid circular dependency at module load time."""
    from ..memory.bitemporal import BiTemporalMemory
    from ..memory.episodic import EpisodicMemory
    from ..state.histone import HistoneStore
    from ..state.telomere import Telomere
    from ..state.development import DevelopmentController, DevelopmentConfig, CriticalPeriod
    from ..healing.autophagy_daemon import AutophagyDaemon
    from ..healing.consolidation import SleepConsolidation
    from ..coordination.social_learning import SocialLearning
    from .adaptive import adaptive_skill_organism
    from .organism import TelemetryProbe, skill_organism
    from .repository import PatternLibrary
    from .watcher import WatcherComponent, WatcherConfig
    return locals()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManagedRunResult:
    """Result of a managed organism run."""

    run_result: SkillRunResult
    adaptive_result: AdaptiveRunResult | None
    watcher_summary: dict[str, Any] | None
    development_status: DevelopmentStatus | None
    template_used: PatternTemplate | None


# ---------------------------------------------------------------------------
# ManagedOrganism
# ---------------------------------------------------------------------------


@dataclass
class ManagedOrganism:
    """Full-stack organism with run, consolidate, export, and scaffold.

    Every subsystem is optional. Methods return None when the relevant
    subsystem is not configured.
    """

    _organism: SkillOrganism | None = None
    _adaptive: AdaptiveSkillOrganism | None = None
    _watcher: WatcherComponent | None = None
    _telemetry: TelemetryProbe | None = None
    _library: PatternLibrary | None = None
    _development: DevelopmentController | None = None
    _social: SocialLearning | None = None
    _substrate: BiTemporalMemory | None = None

    def run(self, task: str) -> ManagedRunResult:
        """Execute the organism and return enriched results."""
        adaptive_result = None
        template_used = None

        if self._adaptive is not None:
            adaptive_result = self._adaptive.run(task)
            run_result = adaptive_result.run_result
            template_used = adaptive_result.template
        elif self._organism is not None:
            run_result = self._organism.run(task)
        else:
            raise ValueError("No organism or adaptive assembly configured")

        # Tick development if present
        if self._development is not None:
            self._development.tick()

        return ManagedRunResult(
            run_result=run_result,
            adaptive_result=adaptive_result,
            watcher_summary=self._watcher.summary() if self._watcher else None,
            development_status=self._development.get_status() if self._development else None,
            template_used=template_used,
        )

    def consolidate(self, context: str = "") -> ConsolidationResult | None:
        """Run sleep consolidation if library is available."""
        if self._library is None:
            return None
        return consolidate(
            self._library,
            bitemporal=self._substrate,
            context=context,
        )

    def export_templates(self, min_success_rate: float = 0.6) -> PeerExchange | None:
        """Export templates via social learning."""
        if self._social is None:
            return None
        return self._social.export_templates(min_success_rate=min_success_rate)

    def import_from_peer(self, exchange: PeerExchange) -> AdoptionResult | None:
        """Import templates from a peer."""
        if self._social is None:
            return None
        return self._social.import_from_peer(exchange)

    def scaffold(self, learner: ManagedOrganism) -> ScaffoldingResult | None:
        """Scaffold a younger organism's learning."""
        if self._social is None or learner._social is None:
            return None
        teacher_stage = self._development.stage.value if self._development else None
        learner_stage = learner._development.stage.value if learner._development else None
        return self._social.scaffold_learner(
            learner._social,
            teacher_stage=teacher_stage,
            learner_stage=learner_stage,
        )

    def status(self) -> dict[str, Any]:
        """Comprehensive status of all subsystems."""
        result: dict[str, Any] = {}
        if self._watcher:
            result["watcher"] = self._watcher.summary()
        if self._telemetry:
            result["telemetry"] = self._telemetry.summary()
        if self._library:
            result["library"] = self._library.summary()
        if self._development:
            result["development"] = self._development.get_statistics()
        if self._social:
            result["social"] = self._social.summary()
        if self._substrate:
            result["substrate_facts"] = len(self._substrate._facts)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def managed_organism(
    task: str | None = None,
    *,
    # Stage definition
    stages: list[SkillStage] | None = None,
    handlers: dict[str, Any] | None = None,
    fast_nucleus: Nucleus | None = None,
    deep_nucleus: Nucleus | None = None,
    # Template-based
    fingerprint: TaskFingerprint | None = None,
    library: PatternLibrary | None = None,
    # Watcher
    watcher: bool = True,
    watcher_config: WatcherConfig | None = None,
    # Substrate
    substrate: BiTemporalMemory | None = None,
    # Development
    telomere: Telomere | None = None,
    development_config: DevelopmentConfig | None = None,
    critical_periods: tuple[CriticalPeriod, ...] = (),
    # Social
    organism_id: str | None = None,
) -> ManagedOrganism:
    """Build a full-stack managed organism with sensible defaults.

    Two paths:
    - If ``library`` has templates: adaptive assembly (fingerprint → template → assemble)
    - If ``stages`` provided: direct skill_organism construction

    All subsystems are opt-in via their respective parameters.
    """
    _m = _lazy_imports()
    WatcherComponent = _m["WatcherComponent"]
    WatcherConfig_cls = _m["WatcherConfig"]
    TelemetryProbe = _m["TelemetryProbe"]
    DevelopmentController = _m["DevelopmentController"]
    DevelopmentConfig_cls = _m["DevelopmentConfig"]
    SocialLearning = _m["SocialLearning"]
    adaptive_skill_organism_fn = _m["adaptive_skill_organism"]
    skill_organism_fn = _m["skill_organism"]

    # Watcher + telemetry
    watcher_comp = WatcherComponent(config=watcher_config or WatcherConfig_cls()) if watcher else None
    telemetry_probe = TelemetryProbe() if watcher else None
    components: list = []
    if watcher_comp:
        components.append(watcher_comp)
    if telemetry_probe:
        components.append(telemetry_probe)

    # Development
    development = None
    if telomere is not None:
        telomere.start()
        development = DevelopmentController(
            telomere=telomere,
            config=development_config or DevelopmentConfig_cls(),
            critical_periods=critical_periods,
        )
        if watcher_comp:
            watcher_comp.development = development

    # Social learning
    social = None
    if organism_id is not None and library is not None:
        social = SocialLearning(organism_id=organism_id, library=library)

    # Build organism
    organism = None
    adaptive = None

    if library is not None and library.summary()["template_count"] > 0:
        # Adaptive path
        adaptive = adaptive_skill_organism_fn(
            task or "",
            fingerprint=fingerprint,
            library=library,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
            components=components,
            watcher_config=watcher_config,
            substrate=substrate,
            handlers=handlers,
        )
    elif stages is not None:
        # Direct path — patch stages with user-supplied handlers
        if handlers:
            patched: list = []
            for stage in stages:
                h = handlers.get(stage.name)
                if h is not None:
                    stage = replace(stage, handler=h)
                patched.append(stage)
            stages = patched
        organism = skill_organism_fn(
            stages=stages,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
            components=components,
            substrate=substrate,
        )
    else:
        raise ValueError("Provide either stages or a library with templates")

    return ManagedOrganism(
        _organism=organism,
        _adaptive=adaptive,
        _watcher=watcher_comp,
        _telemetry=telemetry_probe,
        _library=library,
        _development=development,
        _social=social,
        _substrate=substrate,
    )


# ---------------------------------------------------------------------------
# Convenience: top-level consolidate
# ---------------------------------------------------------------------------


def consolidate(
    library: PatternLibrary,
    *,
    episodic: EpisodicMemory | None = None,
    histone: HistoneStore | None = None,
    bitemporal: BiTemporalMemory | None = None,
    context: str = "",
    max_tokens: int = 8000,
) -> ConsolidationResult:
    """One-call sleep consolidation with sensible defaults."""
    _m = _lazy_imports()
    EpisodicMemory = _m["EpisodicMemory"]
    HistoneStore = _m["HistoneStore"]
    AutophagyDaemon = _m["AutophagyDaemon"]
    SleepConsolidation = _m["SleepConsolidation"]

    ep = episodic or EpisodicMemory()
    hs = histone or HistoneStore()
    daemon = AutophagyDaemon(
        histone_store=hs,
        lysosome=None,
        summarizer=lambda text: f"Consolidated: {text[:100]}...",
    )
    sc = SleepConsolidation(
        daemon=daemon,
        pattern_library=library,
        episodic_memory=ep,
        histone_store=hs,
        bitemporal_memory=bitemporal,
    )
    return sc.consolidate(context=context, max_tokens=max_tokens)
