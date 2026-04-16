"""Verifier component — rubric-based quality evaluation for stage outputs.

Biological Analogy: Adaptive Immune Response (B-cells)
- Innate immunity (WatcherComponent) detects generic anomalies via baseline
  deviations: novelty collapse, metabolic stress, behavioral fingerprints.
- Adaptive immunity (VerifierComponent) evaluates output against a specific
  rubric, like B-cells producing antibodies tailored to a specific antigen.
  The rubric is the "antigen template" — each task type gets a different
  quality assessment.

This completes the immune analogy:
  Innate  → fast, generic detection  (EpiplexityMonitor, ImmuneSystem)
  Adaptive → slower, specific evaluation (VerifierComponent)

The verifier emits WatcherSignals with source="verifier" that integrate
into the existing WatcherComponent decision logic.  When quality is below
threshold on a fast model, the watcher can ESCALATE to the deep model.

References:
  Rosset et al. (arXiv:2604.06240) — dual reward signals (process vs outcome)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import SkillStage, SkillStageResult, SkillRunResult
from .watcher import SignalCategory, WatcherSignal


# Type alias: rubric callable takes (output, stage_name) → quality 0.0-1.0
Rubric = Callable[[str, str], float]


@dataclass
class VerifierConfig:
    """Thresholds for quality-based intervention signals."""

    quality_low_threshold: float = 0.5
    """Quality below this emits a signal with value = 1 - quality."""


@dataclass
class VerifierComponent:
    """Rubric-based stage output quality evaluator.

    Evaluates each stage's output against a provided rubric function and
    emits WatcherSignals that the WatcherComponent can act on.  Signals
    use ``category=EPISTEMIC`` and ``source="verifier"`` so the watcher's
    decision logic can distinguish quality signals from novelty signals.

    Usage::

        def my_rubric(output: str, stage_name: str) -> float:
            # Return 0.0-1.0 quality score
            ...

        verifier = VerifierComponent(rubric=my_rubric)
        organism = skill_organism(..., components=[watcher, verifier])
    """

    rubric: Rubric | None = None
    config: VerifierConfig = field(default_factory=VerifierConfig)

    # Per-run state
    signals: list[WatcherSignal] = field(default_factory=list)
    quality_scores: list[tuple[str, float]] = field(default_factory=list)

    # -- SkillRuntimeComponent protocol ----------------------------------

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        """Reset per-run state."""
        self.signals.clear()
        self.quality_scores.clear()

    def on_stage_start(
        self,
        stage: SkillStage,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """No pre-stage action needed."""

    def on_stage_result(
        self,
        stage: SkillStage,
        result: SkillStageResult,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Evaluate stage output quality and emit signal."""
        if self.rubric is None:
            return

        output = result.output
        if not isinstance(output, str):
            output = str(output) if output is not None else ""

        stage_name = getattr(stage, "name", "unknown")

        try:
            quality = self.rubric(output, stage_name)
            quality = max(0.0, min(1.0, quality))
        except Exception:
            return

        self.quality_scores.append((stage_name, quality))

        # Emit signal: value = severity (higher = worse quality)
        severity = 1.0 - quality
        signal = WatcherSignal(
            category=SignalCategory.EPISTEMIC,
            source="verifier",
            stage_name=stage_name,
            value=severity,
            detail={
                "quality": quality,
                "threshold": self.config.quality_low_threshold,
                "below_threshold": quality < self.config.quality_low_threshold,
            },
        )
        self.signals.append(signal)

        # Write signal to shared_state for WatcherComponent to pick up
        shared_state.setdefault("_verifier_signals", []).append(signal)

    def on_run_complete(
        self,
        result: SkillRunResult,
        shared_state: dict[str, Any],
    ) -> None:
        """No post-run action needed."""

    # -- Public API -------------------------------------------------------

    def mean_quality(self) -> float:
        """Mean quality across all evaluated stages."""
        if not self.quality_scores:
            return 0.0
        return sum(q for _, q in self.quality_scores) / len(self.quality_scores)

    def certify_behavior(self, threshold: float = 0.8):
        """Produce a behavioral certificate from collected rubric scores.

        Returns a :class:`Certificate` asserting that mean rubric quality
        meets or exceeds *threshold*.  Only meaningful after a run has
        completed (i.e. ``quality_scores`` is non-empty).

        Returns ``None`` if no quality scores have been collected.
        """
        if not self.quality_scores:
            return None
        from ..core.certificate import Certificate, _verify_behavioral_quality

        scores = [q for _, q in self.quality_scores]
        n = len(scores)
        return Certificate(
            theorem="behavioral_quality",
            parameters={"scores": scores, "threshold": threshold},
            conclusion=f"Mean rubric quality >= {threshold} on {n} stages",
            source="VerifierComponent.certify_behavior",
            _verify_fn=_verify_behavioral_quality,
        )
