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
from enum import Enum
from typing import TYPE_CHECKING, Any

from .types import InterventionKind, WatcherIntervention, WATCHER_STATE_KEY

if TYPE_CHECKING:
    from ..health.epiplexity import EpiplexityMonitor
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

    # shared_state key
    state_key: str = WATCHER_STATE_KEY


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

    # Internal state
    signals: list[WatcherSignal] = field(default_factory=list)
    interventions: list[WatcherIntervention] = field(default_factory=list)
    _stage_retry_counts: dict[str, int] = field(default_factory=dict)
    _total_stages: int = field(default=0, init=False)

    # -- SkillRuntimeComponent protocol ----------------------------------

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        """Reset per-run counters."""
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

        stage_signals: list[WatcherSignal] = []
        stage_signals.extend(self._collect_epistemic_signals(stage, result))
        stage_signals.extend(self._collect_species_signals(stage, result))
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

    def _collect_epistemic_signals(
        self,
        stage: Any,
        result: Any,
    ) -> list[WatcherSignal]:
        """Feed stage output to EpiplexityMonitor if available."""
        if self.epiplexity_monitor is None:
            return []
        try:
            output_str = str(getattr(result, "output", result))
            ep_result = self.epiplexity_monitor.measure(output_str)
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

        return None

    def _check_convergence(self) -> bool:
        """Return True if intervention rate is within budget."""
        if self._total_stages == 0:
            return True
        rate = len(self.interventions) / self._total_stages
        return rate <= self.config.max_intervention_rate

    # -- Public API ------------------------------------------------------

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
