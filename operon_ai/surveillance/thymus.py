"""Thymus - baseline training for surveillance."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import statistics

from .types import MHCPeptide


class SelectionResult(Enum):
    """Result of thymic selection."""
    POSITIVE = "positive"        # Successfully trained
    NEGATIVE = "negative"        # Would attack self (rejected)
    ANERGIC = "anergic"         # Too variable to be useful
    INSUFFICIENT_DATA = "insufficient"


@dataclass
class BaselineProfile:
    """
    Learned "self" pattern for an agent.

    Like the thymic repertoire that T-cells use to recognize
    self vs non-self.
    """

    agent_id: str

    # Statistical bounds (mean ± tolerance × std)
    output_length_bounds: tuple[float, float]
    response_time_bounds: tuple[float, float]
    confidence_bounds: tuple[float, float]

    # Maximums
    error_rate_max: float

    # Valid patterns
    valid_vocabulary_hashes: set[str]
    valid_structure_hashes: set[str]

    # Canary requirements
    canary_accuracy_min: float

    def check(self, peptide: MHCPeptide) -> list[str]:
        """
        Check peptide against baseline.

        Returns list of violation descriptions (empty = OK).
        """
        violations = []

        # Output length check
        low, high = self.output_length_bounds
        if not (low <= peptide.output_length_mean <= high):
            violations.append(
                f"output_length out of bounds: {peptide.output_length_mean:.1f} "
                f"not in [{low:.1f}, {high:.1f}]"
            )

        # Response time check
        low, high = self.response_time_bounds
        if not (low <= peptide.response_time_mean <= high):
            violations.append(
                f"response_time out of bounds: {peptide.response_time_mean:.3f} "
                f"not in [{low:.3f}, {high:.3f}]"
            )

        # Confidence check
        low, high = self.confidence_bounds
        if not (low <= peptide.confidence_mean <= high):
            violations.append(
                f"confidence out of bounds: {peptide.confidence_mean:.2f} "
                f"not in [{low:.2f}, {high:.2f}]"
            )

        # Error rate check
        if peptide.error_rate > self.error_rate_max:
            violations.append(
                f"error_rate too high: {peptide.error_rate:.2%} > {self.error_rate_max:.2%}"
            )

        # Vocabulary hash check
        if peptide.vocabulary_hash not in self.valid_vocabulary_hashes:
            violations.append(
                f"vocabulary_hash unknown: {peptide.vocabulary_hash}"
            )

        # Structure hash check
        if peptide.structure_hash not in self.valid_structure_hashes:
            violations.append(
                f"structure_hash unknown: {peptide.structure_hash}"
            )

        # Canary accuracy check
        if peptide.canary_accuracy is not None:
            if peptide.canary_accuracy < self.canary_accuracy_min:
                violations.append(
                    f"canary_accuracy too low: {peptide.canary_accuracy:.2%} "
                    f"< {self.canary_accuracy_min:.2%}"
                )

        return violations


@dataclass
class Thymus:
    """
    Trains surveillance baselines for agents.

    Biological parallel: The thymus where T-cells undergo
    positive and negative selection to learn self-tolerance.
    """

    min_training_samples: int = 10
    tolerance: float = 2.0  # Standard deviations for bounds
    variance_threshold: float = 0.5  # Max coefficient of variation before anergic

    profiles: dict[str, BaselineProfile] = field(default_factory=dict)

    def train(
        self,
        agent_id: str,
        samples: list[MHCPeptide],
    ) -> tuple[Optional[BaselineProfile], SelectionResult]:
        """
        Train baseline profile from sample peptides.

        Returns: (profile, selection_result)
        """
        if len(samples) < self.min_training_samples:
            return None, SelectionResult.INSUFFICIENT_DATA

        # Check for high variance (anergic)
        output_lengths = [s.output_length_mean for s in samples]
        if len(output_lengths) > 1:
            mean = statistics.mean(output_lengths)
            std = statistics.stdev(output_lengths)
            if mean > 0 and (std / mean) > self.variance_threshold:
                return None, SelectionResult.ANERGIC

        # Calculate bounds: mean ± tolerance × std
        def calc_bounds(values: list[float], stds: list[float]) -> tuple[float, float]:
            mean = statistics.mean(values)
            # Use max of actual std and mean of reported stds
            actual_std = statistics.stdev(values) if len(values) > 1 else 0
            reported_std = statistics.mean(stds) if stds else 0
            combined_std = max(actual_std, reported_std, 0.01)  # Avoid zero
            return (
                mean - self.tolerance * combined_std,
                mean + self.tolerance * combined_std,
            )

        output_length_bounds = calc_bounds(
            [s.output_length_mean for s in samples],
            [s.output_length_std for s in samples],
        )

        response_time_bounds = calc_bounds(
            [s.response_time_mean for s in samples],
            [s.response_time_std for s in samples],
        )

        confidence_bounds = calc_bounds(
            [s.confidence_mean for s in samples],
            [s.confidence_std for s in samples],
        )

        # Error rate: use max observed plus margin
        error_rates = [s.error_rate for s in samples]
        error_rate_max = max(error_rates) * 2  # Allow 2x observed max
        error_rate_max = max(error_rate_max, 0.05)  # Minimum tolerance

        # Collect hashes
        valid_vocabulary_hashes = {s.vocabulary_hash for s in samples}
        valid_structure_hashes = {s.structure_hash for s in samples}

        # Canary accuracy: 90% of minimum observed (or 0 if none)
        canary_accuracies = [s.canary_accuracy for s in samples if s.canary_accuracy is not None]
        canary_accuracy_min = min(canary_accuracies) * 0.9 if canary_accuracies else 0.0

        profile = BaselineProfile(
            agent_id=agent_id,
            output_length_bounds=output_length_bounds,
            response_time_bounds=response_time_bounds,
            confidence_bounds=confidence_bounds,
            error_rate_max=error_rate_max,
            valid_vocabulary_hashes=valid_vocabulary_hashes,
            valid_structure_hashes=valid_structure_hashes,
            canary_accuracy_min=canary_accuracy_min,
        )

        self.profiles[agent_id] = profile
        return profile, SelectionResult.POSITIVE

    def get_profile(self, agent_id: str) -> Optional[BaselineProfile]:
        """Retrieve trained profile for agent."""
        return self.profiles.get(agent_id)
