"""Integrated Immune System organelle."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import ThreatLevel, ResponseAction, MHCPeptide, Signal1, Signal2
from .display import MHCDisplay
from .thymus import Thymus, BaselineProfile, SelectionResult
from .tcell import TCell, ImmuneResponse
from .treg import RegulatoryTCell, ToleranceRecord, SuppressionResult
from .memory import ImmuneMemory, ThreatSignature


@dataclass
class ImmuneSystem:
    """
    Integrated surveillance system.

    Combines MHC display, thymus training, T-cell inspection,
    regulatory tolerance, and immune memory into a complete
    organelle for agent monitoring.
    """

    # Configuration
    min_training_samples: int = 10
    min_observations: int = 10
    window_size: int = 100

    # Components
    thymus: Thymus = field(default_factory=Thymus)
    treg: RegulatoryTCell = field(default_factory=RegulatoryTCell)
    memory: ImmuneMemory = field(default_factory=ImmuneMemory)

    # Per-agent components
    displays: dict[str, MHCDisplay] = field(default_factory=dict)
    tcells: dict[str, TCell] = field(default_factory=dict)
    profiles: dict[str, BaselineProfile] = field(default_factory=dict)

    def __post_init__(self):
        # Configure thymus
        self.thymus.min_training_samples = self.min_training_samples

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent for surveillance."""
        self.displays[agent_id] = MHCDisplay(
            agent_id=agent_id,
            window_size=self.window_size,
            min_observations=self.min_observations,
        )
        self.treg.register_agent(agent_id)

    def record_observation(
        self,
        agent_id: str,
        output: Optional[str],
        response_time: float,
        confidence: float,
        error: Optional[str] = None,
    ) -> None:
        """Record an observation for an agent."""
        if agent_id not in self.displays:
            raise ValueError(f"Agent {agent_id} not registered")

        self.displays[agent_id].record(
            output=output,
            response_time=response_time,
            confidence=confidence,
            error=error,
        )

    def record_canary_result(self, agent_id: str, passed: bool) -> None:
        """Record canary test result for an agent."""
        if agent_id not in self.displays:
            raise ValueError(f"Agent {agent_id} not registered")

        self.displays[agent_id].record_canary_result(passed)

    def train_agent(self, agent_id: str) -> SelectionResult:
        """Train baseline profile for agent from recorded observations."""
        if agent_id not in self.displays:
            raise ValueError(f"Agent {agent_id} not registered")

        display = self.displays[agent_id]

        # Generate peptides from observations
        samples = []
        # We need multiple peptide snapshots - simulate by generating one
        peptide = display.generate_peptide()
        if peptide is None:
            return SelectionResult.INSUFFICIENT_DATA

        # For training, we'll use the current peptide multiple times
        # (In production, you'd collect peptides over time)
        for _ in range(self.min_training_samples):
            samples.append(peptide)

        profile, result = self.thymus.train(agent_id, samples)

        if result == SelectionResult.POSITIVE and profile is not None:
            self.profiles[agent_id] = profile
            self.tcells[agent_id] = TCell(profile=profile)

        return result

    def inspect(self, agent_id: str) -> ImmuneResponse:
        """
        Inspect agent's current behavior.

        Flow: MHCDisplay -> Memory check -> TCell inspection -> Treg filtering
        """
        if agent_id not in self.tcells:
            raise ValueError(f"Agent {agent_id} not trained")

        display = self.displays[agent_id]
        tcell = self.tcells[agent_id]

        # Generate current peptide
        peptide = display.generate_peptide()
        if peptide is None:
            # Not enough observations - return clear
            return ImmuneResponse(
                agent_id=agent_id,
                threat_level=ThreatLevel.NONE,
                action=ResponseAction.IGNORE,
                signal1=Signal1.UNKNOWN,
                signal2=Signal2.NONE,
                violations=[],
            )

        # Check memory for known threats
        recalled = self.memory.recall_by_hashes(
            agent_id=agent_id,
            vocabulary_hash=peptide.vocabulary_hash,
            structure_hash=peptide.structure_hash,
        )

        if recalled is not None:
            # Known threat - fast response
            return ImmuneResponse(
                agent_id=agent_id,
                threat_level=recalled.threat_level,
                action=recalled.effective_response,
                signal1=Signal1.NON_SELF,
                signal2=Signal2.CROSS_VALIDATED,  # Memory serves as validation
                violations=["recalled from immune memory"],
            )

        # T-cell inspection
        response = tcell.inspect(peptide)

        # Treg filtering
        record = self.treg.get_record(agent_id)
        if record is not None:
            suppression = self.treg.evaluate(response, record)
            if suppression.suppressed:
                response = ImmuneResponse(
                    agent_id=response.agent_id,
                    threat_level=response.threat_level,
                    action=suppression.modified_action,
                    signal1=response.signal1,
                    signal2=response.signal2,
                    violations=response.violations,
                )

            # Update inspection record
            clean = response.threat_level == ThreatLevel.NONE
            record.record_inspection(clean=clean)

        # Store confirmed threats in memory
        if response.threat_level in [ThreatLevel.CONFIRMED, ThreatLevel.CRITICAL]:
            violation_types = tuple(
                v.split()[0] for v in response.violations
            )
            signature = ThreatSignature(
                agent_id=agent_id,
                vocabulary_hash=peptide.vocabulary_hash,
                structure_hash=peptide.structure_hash,
                violation_types=violation_types,
                threat_level=response.threat_level,
                effective_response=response.action,
            )
            self.memory.store(signature)

        return response

    def mark_agent_updated(self, agent_id: str) -> None:
        """Mark agent as recently updated (temporary tolerance)."""
        record = self.treg.get_record(agent_id)
        if record is not None:
            record.mark_updated()

    def flag_agent(self, agent_id: str, reason: str) -> None:
        """Manually flag an agent for Signal 2."""
        if agent_id in self.tcells:
            self.tcells[agent_id].flag_manually(reason)

    def health(self) -> dict:
        """Return system health status."""
        return {
            "registered_agents": len(self.displays),
            "trained_agents": len(self.tcells),
            "memory_stats": self.memory.stats(),
            "agents": {
                agent_id: {
                    "trained": agent_id in self.tcells,
                    "observations": len(self.displays[agent_id].observations),
                }
                for agent_id in self.displays
            },
        }
