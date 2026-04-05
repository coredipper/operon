"""Naive alternatives to quorum sensing: three coordination strategies.

These represent the standard approaches an engineer would reach for
before considering a biologically-inspired signal accumulation model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndependentActors:
    """No coordination — each agent decides independently.

    If ANY agent's suspicion exceeds threshold, that agent acts alone.
    No consensus required.
    """

    threshold: float = 0.5

    def should_activate(self, suspicions: dict[str, float]) -> bool:
        """True if any agent exceeds threshold."""
        return any(s >= self.threshold for s in suspicions.values())

    def activating_agents(self, suspicions: dict[str, float]) -> set[str]:
        """Which agents would independently activate."""
        return {aid for aid, s in suspicions.items() if s >= self.threshold}


@dataclass
class CentralCoordinator:
    """One designated agent aggregates all signals and decides.

    Computes the mean suspicion across all agents; activates if the
    mean exceeds threshold.  Single point of failure by design.
    """

    threshold: float = 0.5

    def should_activate(self, suspicions: dict[str, float]) -> bool:
        """True if mean suspicion exceeds threshold."""
        if not suspicions:
            return False
        mean = sum(suspicions.values()) / len(suspicions)
        return mean >= self.threshold


@dataclass
class MajorityVote:
    """Agents vote ALERT/NORMAL based on own threshold; majority wins.

    Each agent independently decides ALERT (suspicion >= vote_threshold).
    Activation requires >= majority_fraction of agents to vote ALERT.
    """

    vote_threshold: float = 0.5
    majority_fraction: float = 0.5

    def should_activate(self, suspicions: dict[str, float]) -> bool:
        """True if enough agents vote ALERT."""
        if not suspicions:
            return False
        votes = sum(1 for s in suspicions.values() if s >= self.vote_threshold)
        return votes / len(suspicions) >= self.majority_fraction
