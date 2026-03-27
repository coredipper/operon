"""Watcher component — runtime monitor with three-category signal taxonomy.

Classifies stage-level signals as epistemic (epiplexity / prediction error),
somatic (ATP / metabolic state), or species-specific (immune threats).
Can request interventions (retry, escalate, halt) via shared_state.

References:
  Dupoux, LeCun & Malik (arXiv:2603.15381) — System A/B/M taxonomy
  Hao et al. (arXiv:2603.15371, BIGMAS) — intervention count as convergence proxy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from .types import InterventionKind, WatcherIntervention, WATCHER_STATE_KEY

if TYPE_CHECKING:
    from ..health.epiplexity import EpiplexityMonitor
    from ..repository import TaskFingerprint
    from ..state.metabolism import ATP_Store
    from ..surveillance.immune_system import ImmuneSystem


# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------


class SignalCategory(Enum):
    """Three-category signal taxonomy (Dupoux/LeCun/Malik)."""

    EPISTEMIC = "epistemic"
    SOMATIC = "somatic"
    SPECIES_SPECIFIC = "species"


@dataclass(frozen=True)
class WatcherSignal:
    """A classified signal observation from the watcher."""

    category: SignalCategory
    source: str
    stage_name: str | None
    value: float  # normalized severity in [0, 1]
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatcherConfig:
    """Tunable thresholds for the watcher."""

    # Epistemic — epiplexity thresholds (lower = more stagnant)
    epiplexity_stagnant_threshold: float = 0.3
    epiplexity_critical_threshold: float = 0.15

    # Somatic — ATP fraction considered low
    atp_low_fraction: float = 0.1

    # Species-specific — immune threat levels that trigger intervention
    immune_threat_levels: tuple[str, ...] = ("confirmed", "critical")

    # Convergence / intervention budget
    max_intervention_rate: float = 0.5  # interventions / stages ratio
    max_retries_per_stage: int = 2

    # Curiosity — novelty threshold for escalation (Phase 6)
    curiosity_escalation_threshold: float = 0.6

    # shared_state key
    state_key: str = WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Experience pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceRecord:
    """A past intervention with its context, for cross-run learning."""

    fingerprint: TaskFingerprint | None
    stage_name: str
    signal_category: str
    signal_detail: dict[str, Any] = field(default_factory=dict)
    intervention_kind: str = ""
    intervention_reason: str = ""
    outcome_success: bool | None = None
    recorded_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# WatcherComponent
# ---------------------------------------------------------------------------


@dataclass
class WatcherComponent:
    """Runtime component that monitors stage execution and can intervene.

    Signal sources (all optional):
      - EpiplexityMonitor: epistemic health signals
      - ATP_Store: somatic / metabolic signals
      - ImmuneSystem: species-specific threat signals

    Implements the SkillRuntimeComponent protocol.
    """

    config: WatcherConfig = field(default_factory=WatcherConfig)
    epiplexity_monitor: EpiplexityMonitor | None = None
    budget: ATP_Store | None = None
    immune_system: ImmuneSystem | None = None
    immune_agent_id: str | None = None
    development: Any = None  # DevelopmentController (Phase 7)

    # Internal state (per-run — cleared on on_run_start)
    signals: list[WatcherSignal] = field(default_factory=list)
    interventions: list[WatcherIntervention] = field(default_factory=list)
    _stage_retry_counts: dict[str, int] = field(default_factory=dict)
    _total_stages: int = field(default=0, init=False)

    # Experience pool (persists across runs)
    experience_pool: list[ExperienceRecord] = field(default_factory=list)
    _current_fingerprint: TaskFingerprint | None = field(default=None, init=False)

    # -- SkillRuntimeComponent protocol ----------------------------------

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        """Reset per-run counters."""
        self.signals.clear()
        self.interventions.clear()
        self._stage_retry_counts.clear()
        self._total_stages = 0

    def on_stage_start(
        self,
        stage: Any,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Collect pre-stage somatic signals."""
        somatic = self._collect_somatic_signals(stage)
        somatic.extend(self._collect_developmental_signals(stage))
        self.signals.extend(somatic)

    def on_stage_result(
        self,
        stage: Any,
        result: Any,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Evaluate signals, decide intervention, write to shared_state."""
        self._total_stages += 1

        # Measure epiplexity once, share result across collectors
        ep_result = self._measure_epiplexity(stage, result)

        stage_signals: list[WatcherSignal] = []
        stage_signals.extend(self._collect_epistemic_signals(stage, result, ep_result))
        stage_signals.extend(self._collect_curiosity_signals(stage, result, ep_result))
        stage_signals.extend(self._collect_species_signals(stage, result))
        stage_signals.extend(self._collect_cognitive_mode_signals(stage, result))
        self.signals.extend(stage_signals)

        intervention = self._decide_intervention(stage, result, stage_signals)
        if intervention is not None:
            self.interventions.append(intervention)
            shared_state[self.config.state_key] = intervention

    def on_run_complete(
        self,
        result: Any,
        shared_state: dict[str, Any],
    ) -> None:
        """No-op; signals and interventions are already recorded."""

    # -- Signal collection -----------------------------------------------

    def _measure_epiplexity(self, stage: Any, result: Any) -> Any:
        """Measure epiplexity once per stage, returning result for reuse."""
        if self.epiplexity_monitor is None:
            return None
        try:
            output_str = str(getattr(result, "output", result))
            return self.epiplexity_monitor.measure(output_str)
        except Exception:
            return None

    def _collect_epistemic_signals(
        self,
        stage: Any,
        result: Any,
        ep_result: Any = None,
    ) -> list[WatcherSignal]:
        """Emit epistemic signals from pre-computed EpiplexityResult."""
        if ep_result is None:
            return []
        return [WatcherSignal(
            category=SignalCategory.EPISTEMIC,
            source="epiplexity",
            stage_name=getattr(stage, "name", None),
            value=ep_result.epiplexity,
            detail={
                "status": ep_result.status.value,
                "integral": ep_result.epiplexic_integral,
            },
        )]

    def _collect_curiosity_signals(
        self,
        stage: Any,
        result: Any,
        ep_result: Any = None,
    ) -> list[WatcherSignal]:
        """Emit curiosity signals when epiplexity status is EXPLORING."""
        if ep_result is None:
            return []
        if ep_result.status.value != "exploring":
            return []
        return [WatcherSignal(
            category=SignalCategory.EPISTEMIC,
            source="curiosity",
            stage_name=getattr(stage, "name", None),
            value=ep_result.embedding_novelty,
            detail={
                "status": "exploring",
                "embedding_novelty": ep_result.embedding_novelty,
                "epiplexity": ep_result.epiplexity,
            },
        )]

    def _collect_developmental_signals(self, stage: Any) -> list[WatcherSignal]:
        """Emit SOMATIC signals for developmental stage."""
        if self.development is None:
            return []
        try:
            from ..state.development import _STAGE_ORDER
            dev_stage = self.development.stage
            maturity = _STAGE_ORDER[dev_stage] / 3.0
            return [WatcherSignal(
                category=SignalCategory.SOMATIC,
                source="development",
                stage_name=getattr(stage, "name", None),
                value=maturity,
                detail={
                    "developmental_stage": dev_stage.value,
                    "learning_plasticity": self.development.learning_plasticity,
                    "open_periods": [p.name for p in self.development.open_critical_periods()],
                },
            )]
        except Exception:
            return []

    def _collect_somatic_signals(self, stage: Any) -> list[WatcherSignal]:
        """Read ATP_Store balance if available."""
        if self.budget is None:
            return []
        try:
            max_atp = getattr(self.budget, "max_atp", 0)
            current = getattr(self.budget, "atp", 0)
            if max_atp <= 0:
                return []
            fraction = current / max_atp
            return [WatcherSignal(
                category=SignalCategory.SOMATIC,
                source="atp_store",
                stage_name=getattr(stage, "name", None),
                value=1.0 - fraction,  # higher = worse (more depleted)
                detail={"atp": current, "max_atp": max_atp, "fraction": fraction},
            )]
        except Exception:
            return []

    def _collect_species_signals(
        self,
        stage: Any,
        result: Any,
    ) -> list[WatcherSignal]:
        """Call ImmuneSystem.inspect() if available."""
        if self.immune_system is None or self.immune_agent_id is None:
            return []
        try:
            response = self.immune_system.inspect(self.immune_agent_id)
            threat_value = {
                "none": 0.0,
                "suspicious": 0.3,
                "confirmed": 0.7,
                "critical": 1.0,
            }.get(response.threat_level.value, 0.0)
            return [WatcherSignal(
                category=SignalCategory.SPECIES_SPECIFIC,
                source="immune_system",
                stage_name=getattr(stage, "name", None),
                value=threat_value,
                detail={
                    "threat_level": response.threat_level.value,
                    "action": response.action.value,
                },
            )]
        except Exception:
            return []

    def _collect_cognitive_mode_signals(
        self,
        stage: Any,
        result: Any,
    ) -> list[WatcherSignal]:
        """Check if stage's cognitive mode aligns with model used."""
        from .types import CognitiveMode, resolve_cognitive_mode

        cognitive_mode = getattr(stage, "cognitive_mode", None)
        # Only emit signals for stages that have the cognitive_mode field
        if not hasattr(stage, "cognitive_mode"):
            return []
        resolved = resolve_cognitive_mode(stage)
        model_alias = getattr(result, "model_alias", "")
        mismatch = False
        if resolved == CognitiveMode.OBSERVATIONAL and model_alias == "deep":
            mismatch = True
        elif resolved == CognitiveMode.ACTION_ORIENTED and model_alias == "fast":
            mismatch = True

        return [WatcherSignal(
            category=SignalCategory.EPISTEMIC,
            source="cognitive_mode",
            stage_name=getattr(stage, "name", None),
            value=0.3 if mismatch else 0.0,
            detail={
                "cognitive_mode": resolved.value,
                "model_alias": model_alias,
                "mismatch": mismatch,
            },
        )]

    # -- Decision logic --------------------------------------------------

    def _decide_intervention(
        self,
        stage: Any,
        result: Any,
        signals: list[WatcherSignal],
    ) -> WatcherIntervention | None:
        """Decide whether to intervene based on collected signals.

        Priority order:
        1. Exceeded intervention rate → HALT (convergence failure)
        2. Critical immune threat → HALT
        3. Critical epiplexity → ESCALATE (or HALT if already deep)
        4. Stagnant epiplexity + fast model → ESCALATE
        5. Stage FAILURE + retries available → RETRY
        6. Otherwise → None
        """
        stage_name = getattr(stage, "name", "unknown")

        # 1. Convergence check — intervention rate exceeded
        if not self._check_convergence():
            return WatcherIntervention(
                kind=InterventionKind.HALT,
                stage_name=stage_name,
                reason=(
                    f"non-convergence: intervention rate "
                    f"{len(self.interventions)}/{self._total_stages} "
                    f"exceeds {self.config.max_intervention_rate}"
                ),
            )

        # Classify current signals
        epistemic_status = None
        immune_threat = None
        for sig in signals:
            if sig.category == SignalCategory.EPISTEMIC and sig.source == "epiplexity":
                epistemic_status = sig.detail.get("status")
            if sig.category == SignalCategory.SPECIES_SPECIFIC:
                immune_threat = sig.detail.get("threat_level")

        # 2. Critical immune threat → HALT
        if immune_threat in self.config.immune_threat_levels:
            return WatcherIntervention(
                kind=InterventionKind.HALT,
                stage_name=stage_name,
                reason=f"immune threat: {immune_threat}",
            )

        model_alias = getattr(result, "model_alias", "")

        # 3. Critical epiplexity → ESCALATE or HALT
        if epistemic_status == "critical":
            if model_alias == "deep":
                return WatcherIntervention(
                    kind=InterventionKind.HALT,
                    stage_name=stage_name,
                    reason="critical epiplexity on deep model — cannot escalate further",
                )
            return WatcherIntervention(
                kind=InterventionKind.ESCALATE,
                stage_name=stage_name,
                reason="critical epiplexity — escalating to deep model",
            )

        # 4. Stagnant + fast model → ESCALATE
        if epistemic_status == "stagnant" and model_alias == "fast":
            return WatcherIntervention(
                kind=InterventionKind.ESCALATE,
                stage_name=stage_name,
                reason="stagnant epiplexity on fast model — escalating",
            )

        # 4.5. Curiosity: high novelty on fast model → ESCALATE
        for sig in signals:
            if (
                sig.category == SignalCategory.EPISTEMIC
                and sig.source == "curiosity"
                and model_alias == "fast"
                and sig.value > self.config.curiosity_escalation_threshold
            ):
                return WatcherIntervention(
                    kind=InterventionKind.ESCALATE,
                    stage_name=stage_name,
                    reason=f"curiosity: high novelty ({sig.value:.2f}) on fast model — escalating",
                )

        # 5. Stage FAILURE + retries available → RETRY
        action_type = getattr(result, "action_type", "")
        if action_type == "FAILURE":
            retry_count = self._stage_retry_counts.get(stage_name, 0)
            if retry_count < self.config.max_retries_per_stage:
                self._stage_retry_counts[stage_name] = retry_count + 1
                return WatcherIntervention(
                    kind=InterventionKind.RETRY,
                    stage_name=stage_name,
                    reason=f"stage failure — retry {retry_count + 1}/{self.config.max_retries_per_stage}",
                )

        # 6. Experience-based recommendation (Phase 4)
        if self.experience_pool and self._current_fingerprint is not None:
            dominant_category = None
            for sig in signals:
                if sig.value > 0.3:
                    dominant_category = sig.category.value
                    break
            if dominant_category:
                recommended = self.recommend_intervention(
                    stage_name=stage_name,
                    signal_category=dominant_category,
                    fingerprint=self._current_fingerprint,
                )
                if recommended is not None:
                    return WatcherIntervention(
                        kind=recommended,
                        stage_name=stage_name,
                        reason=f"experience-based: {recommended.value} for {dominant_category}",
                    )

        return None

    def _check_convergence(self) -> bool:
        """Return True if intervention rate is within budget."""
        if self._total_stages == 0:
            return True
        rate = len(self.interventions) / self._total_stages
        return rate <= self.config.max_intervention_rate

    # -- Experience pool -------------------------------------------------

    def set_fingerprint(self, fingerprint: TaskFingerprint) -> None:
        """Set the current task fingerprint for experience-based recommendations."""
        self._current_fingerprint = fingerprint

    def record_experience(
        self,
        *,
        fingerprint: TaskFingerprint | None = None,
        stage_name: str,
        signal_category: str,
        signal_detail: dict[str, Any] | None = None,
        intervention_kind: str,
        intervention_reason: str,
        outcome_success: bool | None = None,
    ) -> ExperienceRecord:
        """Record an intervention outcome for future reference."""
        record = ExperienceRecord(
            fingerprint=fingerprint or self._current_fingerprint,
            stage_name=stage_name,
            signal_category=signal_category,
            signal_detail=signal_detail or {},
            intervention_kind=intervention_kind,
            intervention_reason=intervention_reason,
            outcome_success=outcome_success,
        )
        self.experience_pool.append(record)
        return record

    def retrieve_similar_experiences(
        self,
        *,
        stage_name: str | None = None,
        signal_category: str | None = None,
        fingerprint: TaskFingerprint | None = None,
        limit: int = 10,
    ) -> list[ExperienceRecord]:
        """Retrieve past experiences matching the given criteria."""
        matches = []
        for exp in self.experience_pool:
            if stage_name is not None and exp.stage_name != stage_name:
                continue
            if signal_category is not None and exp.signal_category != signal_category:
                continue
            if fingerprint is not None and exp.fingerprint is not None:
                if exp.fingerprint.task_shape != fingerprint.task_shape:
                    continue
            matches.append(exp)
        matches.sort(key=lambda e: e.recorded_at, reverse=True)
        return matches[:limit]

    def recommend_intervention(
        self,
        *,
        stage_name: str,
        signal_category: str,
        fingerprint: TaskFingerprint | None = None,
    ) -> InterventionKind | None:
        """Recommend an intervention based on past successful experiences."""
        similar = self.retrieve_similar_experiences(
            stage_name=stage_name,
            signal_category=signal_category,
            fingerprint=fingerprint,
        )
        if not similar:
            return None
        success_by_kind: dict[str, int] = {}
        for exp in similar:
            if exp.outcome_success:
                success_by_kind[exp.intervention_kind] = (
                    success_by_kind.get(exp.intervention_kind, 0) + 1
                )
        if not success_by_kind:
            return None
        best_kind = max(success_by_kind, key=lambda k: success_by_kind[k])
        return InterventionKind(best_kind)

    # -- Public API ------------------------------------------------------

    def mode_balance(self) -> dict[str, Any]:
        """Summarize cognitive mode distribution across observed stages."""
        observational = 0
        action_oriented = 0
        mismatches = 0
        for sig in self.signals:
            if sig.source == "cognitive_mode":
                cm = sig.detail.get("cognitive_mode")
                if cm == "observational":
                    observational += 1
                elif cm == "action_oriented":
                    action_oriented += 1
                if sig.detail.get("mismatch"):
                    mismatches += 1
        total = observational + action_oriented
        return {
            "observational": observational,
            "action_oriented": action_oriented,
            "balance_ratio": observational / total if total > 0 else 0.5,
            "mismatches": mismatches,
        }

    def summary(self) -> dict[str, Any]:
        """Return watcher statistics."""
        by_category: dict[str, int] = {}
        for sig in self.signals:
            key = sig.category.value
            by_category[key] = by_category.get(key, 0) + 1

        by_kind: dict[str, int] = {}
        for intv in self.interventions:
            key = intv.kind.value
            by_kind[key] = by_kind.get(key, 0) + 1

        return {
            "total_signals": len(self.signals),
            "signals_by_category": by_category,
            "total_interventions": len(self.interventions),
            "interventions_by_kind": by_kind,
            "total_stages_observed": self._total_stages,
            "convergent": self._check_convergence(),
        }
