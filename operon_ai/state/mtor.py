"""
mTOR/AMPK Adaptive Scaling
===========================

Biological Analogy (KEGG hsa04152 — AMPK signaling pathway):
- AMPK senses the AMP:ATP ratio AND its rate of change
- When energy is abundant, mTOR is active (anabolic — growth)
- When energy depletes, AMPK activates (catabolic — conservation/autophagy)
- Hysteresis from Hill coefficient kinetics prevents oscillation at boundaries

This module reads ATP_Store state via composition (mTOR senses nutrients,
it does not store them) and recommends scaling decisions: worker counts
and feature gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from operon_ai.state.metabolism import ATP_Store, MetabolicState


class ScalingState(Enum):
    GROWTH = "growth"             # mTOR active, scale up
    MAINTENANCE = "maintenance"   # Steady state
    CONSERVATION = "conservation" # AMPK activating, scale down
    AUTOPHAGY = "autophagy"       # Severe depletion, minimal operation


@dataclass
class AMPKRatio:
    """Tracks AMP:ATP ratio and its rate of change.

    The biological insight: AMPK responds to the RATIO and its DERIVATIVE,
    not just the absolute ATP level.
    """

    current_ratio: float = 0.0
    previous_ratio: float = 0.0
    rate_of_change: float = 0.0

    def update(self, atp_level: int, max_atp: int) -> None:
        """Update ratio from current ATP state.

        AMP:ATP ratio is approximated as 1 - (atp/max_atp).
        When ATP is full, ratio is 0 (no AMP). When ATP is empty, ratio is 1 (all AMP).
        """
        self.previous_ratio = self.current_ratio
        self.current_ratio = 1.0 - (atp_level / max(1, max_atp))
        self.rate_of_change = self.current_ratio - self.previous_ratio


def _verify_no_oscillation(
    params: dict,
) -> tuple[bool, dict]:
    """Derivation replay for the mTOR no-oscillation guarantee."""
    g = params["growth_threshold"]
    c = params["conservation_threshold"]
    a = params["autophagy_threshold"]
    h = params["hysteresis"]
    gap_gc = (c - h) - (g + h)
    gap_ca = (a - h) - (c + h)
    holds = h > 0 and gap_gc > 0 and gap_ca > 0
    return holds, {
        "hysteresis": h,
        "gap_growth_conservation": gap_gc,
        "gap_conservation_autophagy": gap_ca,
    }


@dataclass
class MTORScaler:
    """Adaptive resource scaling inspired by mTOR/AMPK signaling.

    Reads ATP_Store state and recommends worker count and feature gating.
    Composition over inheritance — mTOR senses nutrients, it doesn't store them.

    Thresholds derived from AMPK pathway (KEGG hsa04152):
    - ratio < 0.3: GROWTH (anabolic, mTOR active)
    - 0.3 <= ratio < 0.7: MAINTENANCE
    - 0.7 <= ratio < 0.9: CONSERVATION (AMPK activating, catabolic)
    - ratio >= 0.9: AUTOPHAGY (ULK1-mediated)

    Hysteresis prevents oscillation at boundaries (from Hill coefficient kinetics).
    """

    atp_store: ATP_Store
    min_workers: int = 1
    max_workers: int = 8

    # AMPK pathway thresholds
    growth_threshold: float = 0.3
    conservation_threshold: float = 0.7
    autophagy_threshold: float = 0.9

    # Hysteresis margin — must cross threshold by this margin to change state
    hysteresis: float = 0.05

    # Rate sensitivity — how much rate of change matters
    rate_sensitivity: float = 0.1

    # Internal state
    ampk: AMPKRatio = field(default_factory=AMPKRatio)
    state: ScalingState = field(default=ScalingState.MAINTENANCE)
    _transitions: int = field(default=0, repr=False)

    def update(self) -> ScalingState:
        """Read current ATP state, compute AMPK ratio, determine scaling state."""
        self.ampk.update(self.atp_store.atp, self.atp_store.max_atp)
        new_state = self._compute_state()
        if new_state != self.state:
            self._transitions += 1
            self.state = new_state
        return self.state

    def recommended_workers(self) -> int:
        """Return recommended worker count based on current state."""
        fractions = {
            ScalingState.GROWTH: 1.0,
            ScalingState.MAINTENANCE: 0.6,
            ScalingState.CONSERVATION: 0.3,
            ScalingState.AUTOPHAGY: 0.0,  # min_workers only
        }
        fraction = fractions[self.state]
        workers = self.min_workers + int(
            fraction * (self.max_workers - self.min_workers)
        )
        return max(self.min_workers, min(self.max_workers, workers))

    def should_enable_feature(self, feature_cost: int) -> bool:
        """Gate expensive features based on current state."""
        if self.state == ScalingState.AUTOPHAGY:
            return False
        if self.state == ScalingState.CONSERVATION:
            return feature_cost <= 10  # Only cheap features
        return True  # GROWTH and MAINTENANCE allow all features

    def certify(self) -> "Certificate":
        """Return a certificate for the no-oscillation guarantee."""
        from ..core.certificate import Certificate

        return Certificate(
            theorem="no_oscillation",
            parameters={
                "growth_threshold": self.growth_threshold,
                "conservation_threshold": self.conservation_threshold,
                "autophagy_threshold": self.autophagy_threshold,
                "hysteresis": self.hysteresis,
            },
            conclusion="Adjacent-state transitions require crossing threshold +/- hysteresis; non-adjacent jumps bypass hysteresis",
            source="MTORScaler",
            _verify_fn=_verify_no_oscillation,
        )

    def _compute_state(self) -> ScalingState:
        """Core state computation with hysteresis and rate sensitivity.

        Effective ratio = AMPK ratio + rate_sensitivity * rate_of_change
        This makes the scaler respond to rapid depletion before hitting thresholds.
        """
        effective_ratio = (
            self.ampk.current_ratio
            + self.rate_sensitivity * self.ampk.rate_of_change
        )
        effective_ratio = max(0.0, min(1.0, effective_ratio))

        # Apply hysteresis — to leave current state, must cross threshold by margin
        h = self.hysteresis

        if self.state == ScalingState.GROWTH:
            if effective_ratio >= self.growth_threshold + h:
                if effective_ratio >= self.autophagy_threshold:
                    return ScalingState.AUTOPHAGY
                if effective_ratio >= self.conservation_threshold:
                    return ScalingState.CONSERVATION
                return ScalingState.MAINTENANCE
            return ScalingState.GROWTH

        elif self.state == ScalingState.MAINTENANCE:
            if effective_ratio < self.growth_threshold - h:
                return ScalingState.GROWTH
            if effective_ratio >= self.conservation_threshold + h:
                if effective_ratio >= self.autophagy_threshold:
                    return ScalingState.AUTOPHAGY
                return ScalingState.CONSERVATION
            return ScalingState.MAINTENANCE

        elif self.state == ScalingState.CONSERVATION:
            if effective_ratio < self.conservation_threshold - h:
                if effective_ratio < self.growth_threshold:
                    return ScalingState.GROWTH
                return ScalingState.MAINTENANCE
            if effective_ratio >= self.autophagy_threshold + h:
                return ScalingState.AUTOPHAGY
            return ScalingState.CONSERVATION

        else:  # AUTOPHAGY
            if effective_ratio < self.autophagy_threshold - h:
                if effective_ratio < self.growth_threshold:
                    return ScalingState.GROWTH
                if effective_ratio < self.conservation_threshold:
                    return ScalingState.MAINTENANCE
                return ScalingState.CONSERVATION
            return ScalingState.AUTOPHAGY

    def get_statistics(self) -> dict:
        """Return scaler statistics."""
        return {
            "state": self.state.value,
            "ampk_ratio": self.ampk.current_ratio,
            "rate_of_change": self.ampk.rate_of_change,
            "recommended_workers": self.recommended_workers(),
            "transitions": self._transitions,
        }
