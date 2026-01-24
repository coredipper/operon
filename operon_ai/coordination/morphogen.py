"""
Morphogen Gradients: Coordination Without Central Control
==========================================================

Biological Analogy:
In embryonic development, cells coordinate their behavior through morphogen
gradients—diffusible signaling molecules whose concentration varies spatially.
Cells read their local concentration and differentiate accordingly, enabling
pattern formation without a central controller.

In multi-agent systems, the morphogen maps to shared context variables that
influence agent behavior:
- Task Complexity Gradient: Agents in "high complexity" regions activate
  detailed reasoning; those in "low complexity" regions use fast heuristics.
- Confidence Gradient: Low confidence triggers Quorum Sensing; high confidence
  enables direct execution.
- Resource Gradient: Token budget remaining. Agents sense "metabolic scarcity"
  and adapt their strategies.
- Error Rate Gradient: Recent failure rate informs risk tolerance.

Implementation Pattern:
The gradient is represented as a JSON structure injected into each agent's
context by the orchestrator. Agents read their local concentration via a
standardized preamble in the system prompt.

References:
- Article Section 6.3.2: Morphogen Gradients - Coordination Without Central Control
"""
from dataclasses import dataclass, field
from typing import Protocol
from enum import Enum
from datetime import datetime
import json


class MorphogenType(Enum):
    """Types of morphogens for agent coordination."""
    COMPLEXITY = "complexity"      # Task difficulty (0.0 = simple, 1.0 = complex)
    CONFIDENCE = "confidence"      # Solution certainty (0.0 = uncertain, 1.0 = certain)
    BUDGET = "budget"              # Token budget remaining (absolute)
    ERROR_RATE = "error_rate"      # Recent failure rate (0.0 = no errors, 1.0 = all failing)
    URGENCY = "urgency"            # Time pressure (0.0 = relaxed, 1.0 = critical)
    RISK = "risk"                  # Operation risk level (0.0 = safe, 1.0 = dangerous)


@dataclass
class MorphogenValue:
    """A single morphogen concentration."""
    morphogen_type: MorphogenType
    value: float  # Current concentration
    threshold_low: float = 0.3  # Below this, "low" behavior
    threshold_high: float = 0.7  # Above this, "high" behavior
    description: str = ""

    @property
    def level(self) -> str:
        """Get qualitative level (low/medium/high)."""
        if self.value <= self.threshold_low:
            return "low"
        if self.value >= self.threshold_high:
            return "high"
        return "medium"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.morphogen_type.value,
            "value": round(self.value, 3),
            "level": self.level,
            "description": self.description,
        }


@dataclass
class MorphogenGradient:
    """
    A gradient of morphogen concentrations.

    Represents the environment state that agents sense and respond to.
    The orchestrator updates the gradient after each step, and agents
    condition their behavior on the current values.

    Example:
        >>> gradient = MorphogenGradient()
        >>> gradient.set(MorphogenType.COMPLEXITY, 0.8)
        >>> gradient.set(MorphogenType.CONFIDENCE, 0.3)
        >>> gradient.get_context_injection()
        'Current environment state: {"morphogens": {"complexity": 0.8, ...}}'
    """

    morphogens: dict[MorphogenType, MorphogenValue] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Initialize with default morphogens if empty
        if not self.morphogens:
            self._initialize_defaults()

    def _initialize_defaults(self):
        """Set up default morphogen concentrations."""
        defaults = [
            (MorphogenType.COMPLEXITY, 0.5, "Task difficulty level"),
            (MorphogenType.CONFIDENCE, 0.5, "Solution certainty"),
            (MorphogenType.BUDGET, 1.0, "Token budget ratio remaining"),
            (MorphogenType.ERROR_RATE, 0.0, "Recent error rate"),
            (MorphogenType.URGENCY, 0.3, "Time pressure"),
            (MorphogenType.RISK, 0.3, "Operation risk level"),
        ]
        for mtype, value, desc in defaults:
            self.morphogens[mtype] = MorphogenValue(
                morphogen_type=mtype,
                value=value,
                description=desc,
            )

    def set(
        self,
        morphogen_type: MorphogenType,
        value: float,
        description: str | None = None,
    ) -> None:
        """Set a morphogen concentration."""
        value = max(0.0, min(1.0, value))  # Clamp to [0, 1]

        if morphogen_type in self.morphogens:
            self.morphogens[morphogen_type].value = value
            if description:
                self.morphogens[morphogen_type].description = description
        else:
            self.morphogens[morphogen_type] = MorphogenValue(
                morphogen_type=morphogen_type,
                value=value,
                description=description or "",
            )

        self.last_updated = datetime.now()

    def get(self, morphogen_type: MorphogenType) -> float:
        """Get a morphogen concentration."""
        if morphogen_type in self.morphogens:
            return self.morphogens[morphogen_type].value
        return 0.5  # Default neutral

    def get_level(self, morphogen_type: MorphogenType) -> str:
        """Get qualitative level for a morphogen."""
        if morphogen_type in self.morphogens:
            return self.morphogens[morphogen_type].level
        return "medium"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            morphogen.value: self.morphogens[morphogen].value
            for morphogen in self.morphogens
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({"morphogens": self.to_dict()}, indent=2)

    def get_context_injection(self) -> str:
        """
        Get the context injection string for agent prompts.

        This is the standardized preamble that agents can parse to
        read their local morphogen concentrations.
        """
        return f"Current environment state: {self.to_json()}"

    def get_strategy_hints(self) -> list[str]:
        """
        Generate strategy hints based on current gradient.

        These can be injected into agent prompts to guide behavior.
        """
        hints = []

        # Complexity hints
        complexity = self.get(MorphogenType.COMPLEXITY)
        if complexity >= 0.7:
            hints.append("Use detailed, step-by-step reasoning for this complex task.")
        elif complexity <= 0.3:
            hints.append("Use fast heuristics for this straightforward task.")

        # Confidence hints
        confidence = self.get(MorphogenType.CONFIDENCE)
        if confidence <= 0.3:
            hints.append("Low confidence - consider consulting other sources or requesting verification.")
        elif confidence >= 0.8:
            hints.append("High confidence - proceed with direct execution.")

        # Budget hints
        budget = self.get(MorphogenType.BUDGET)
        if budget <= 0.2:
            hints.append("Token budget low - be concise and efficient.")
        elif budget <= 0.1:
            hints.append("CRITICAL: Token budget nearly exhausted. Summarize and conclude.")

        # Error rate hints
        error_rate = self.get(MorphogenType.ERROR_RATE)
        if error_rate >= 0.5:
            hints.append("High error rate - increase validation before proceeding.")

        # Risk hints
        risk = self.get(MorphogenType.RISK)
        if risk >= 0.7:
            hints.append("High-risk operation - require explicit confirmation before irreversible actions.")

        return hints


class AgentPhenotype(Protocol):
    """Protocol for agent phenotype adaptation."""

    def adapt_to_gradient(self, gradient: MorphogenGradient) -> None:
        """Adapt agent behavior based on morphogen gradient."""
        ...


@dataclass
class PhenotypeConfig:
    """Configuration for phenotype adaptation based on gradients."""

    # Temperature adjustments based on complexity
    low_complexity_temperature: float = 0.3
    high_complexity_temperature: float = 0.9

    # Token limits based on budget
    critical_budget_max_tokens: int = 100
    normal_max_tokens: int = 4096

    # Verification thresholds
    high_error_verification_threshold: float = 0.8
    normal_verification_threshold: float = 0.5

    def get_temperature(self, gradient: MorphogenGradient) -> float:
        """Calculate temperature based on complexity gradient."""
        complexity = gradient.get(MorphogenType.COMPLEXITY)
        return (
            self.low_complexity_temperature +
            (self.high_complexity_temperature - self.low_complexity_temperature) * complexity
        )

    def get_max_tokens(self, gradient: MorphogenGradient) -> int:
        """Calculate max tokens based on budget gradient."""
        budget = gradient.get(MorphogenType.BUDGET)
        if budget <= 0.1:
            return self.critical_budget_max_tokens
        return int(self.normal_max_tokens * budget)

    def get_verification_threshold(self, gradient: MorphogenGradient) -> float:
        """Calculate verification threshold based on error gradient."""
        error_rate = gradient.get(MorphogenType.ERROR_RATE)
        if error_rate >= 0.5:
            return self.high_error_verification_threshold
        return self.normal_verification_threshold


@dataclass
class GradientUpdate:
    """A single update to the gradient."""
    morphogen_type: MorphogenType
    delta: float  # Change in value
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GradientOrchestrator:
    """
    Orchestrator that manages morphogen gradients across agent steps.

    Updates gradients after each step based on agent performance,
    enabling adaptive coordination without explicit communication.

    Example:
        >>> orchestrator = GradientOrchestrator()
        >>> orchestrator.report_step_result(success=True, tokens_used=500, total_budget=1000)
        >>> context = orchestrator.get_agent_context()
        >>> # Context now includes updated gradient state
    """

    gradient: MorphogenGradient = field(default_factory=MorphogenGradient)
    phenotype_config: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    update_history: list[GradientUpdate] = field(default_factory=list)

    # Decay rates for automatic gradient adjustment
    error_rate_decay: float = 0.1  # Error rate decays over time
    confidence_boost_on_success: float = 0.1
    confidence_drop_on_failure: float = 0.2

    silent: bool = False

    def report_step_result(
        self,
        success: bool,
        tokens_used: int = 0,
        total_budget: int = 0,
        complexity_estimate: float | None = None,
    ) -> None:
        """
        Report the result of an agent step.

        Updates gradients based on outcome.
        """
        updates: list[GradientUpdate] = []

        # Update confidence
        current_confidence = self.gradient.get(MorphogenType.CONFIDENCE)
        if success:
            new_confidence = min(1.0, current_confidence + self.confidence_boost_on_success)
            updates.append(GradientUpdate(
                MorphogenType.CONFIDENCE,
                self.confidence_boost_on_success,
                "Step succeeded",
            ))
        else:
            new_confidence = max(0.0, current_confidence - self.confidence_drop_on_failure)
            updates.append(GradientUpdate(
                MorphogenType.CONFIDENCE,
                -self.confidence_drop_on_failure,
                "Step failed",
            ))
        self.gradient.set(MorphogenType.CONFIDENCE, new_confidence)

        # Update error rate
        current_error = self.gradient.get(MorphogenType.ERROR_RATE)
        if success:
            new_error = max(0.0, current_error - self.error_rate_decay)
        else:
            new_error = min(1.0, current_error + 0.3)
        if new_error != current_error:
            updates.append(GradientUpdate(
                MorphogenType.ERROR_RATE,
                new_error - current_error,
                "Error rate adjustment",
            ))
            self.gradient.set(MorphogenType.ERROR_RATE, new_error)

        # Update budget
        if total_budget > 0:
            remaining_ratio = max(0.0, (total_budget - tokens_used) / total_budget)
            self.gradient.set(MorphogenType.BUDGET, remaining_ratio)
            updates.append(GradientUpdate(
                MorphogenType.BUDGET,
                remaining_ratio - self.gradient.get(MorphogenType.BUDGET),
                f"Budget: {tokens_used}/{total_budget} used",
            ))

        # Update complexity if estimated
        if complexity_estimate is not None:
            self.gradient.set(MorphogenType.COMPLEXITY, complexity_estimate)
            updates.append(GradientUpdate(
                MorphogenType.COMPLEXITY,
                complexity_estimate - self.gradient.get(MorphogenType.COMPLEXITY),
                f"Complexity estimate: {complexity_estimate:.2f}",
            ))

        self.update_history.extend(updates)

        if not self.silent and updates:
            print(
                f"🧬 [Morphogen] Updated: "
                + ", ".join(f"{u.morphogen_type.value}={self.gradient.get(u.morphogen_type):.2f}" for u in updates)
            )

    def set_urgency(self, urgency: float, reason: str = "Manual adjustment") -> None:
        """Set urgency level."""
        self.gradient.set(MorphogenType.URGENCY, urgency)
        self.update_history.append(GradientUpdate(
            MorphogenType.URGENCY, urgency, reason
        ))

    def set_risk(self, risk: float, reason: str = "Manual adjustment") -> None:
        """Set risk level."""
        self.gradient.set(MorphogenType.RISK, risk)
        self.update_history.append(GradientUpdate(
            MorphogenType.RISK, risk, reason
        ))

    def get_agent_context(self) -> str:
        """
        Get the full context injection for an agent.

        Includes gradient state and strategy hints.
        """
        lines = [
            self.gradient.get_context_injection(),
            "",
            "Strategy hints based on current environment:",
        ]
        for hint in self.gradient.get_strategy_hints():
            lines.append(f"  - {hint}")

        return "\n".join(lines)

    def get_phenotype_params(self) -> dict:
        """
        Get phenotype parameters based on current gradient.

        Returns dict with temperature, max_tokens, etc.
        """
        return {
            "temperature": self.phenotype_config.get_temperature(self.gradient),
            "max_tokens": self.phenotype_config.get_max_tokens(self.gradient),
            "verification_threshold": self.phenotype_config.get_verification_threshold(self.gradient),
        }

    def should_recruit_help(self) -> bool:
        """Check if gradient suggests recruiting additional agents (Quorum)."""
        confidence = self.gradient.get(MorphogenType.CONFIDENCE)
        error_rate = self.gradient.get(MorphogenType.ERROR_RATE)
        complexity = self.gradient.get(MorphogenType.COMPLEXITY)

        # Low confidence + high error/complexity = recruit help
        return confidence <= 0.3 and (error_rate >= 0.5 or complexity >= 0.7)

    def should_reduce_capabilities(self) -> bool:
        """Check if gradient suggests reducing capabilities (resource scarcity)."""
        budget = self.gradient.get(MorphogenType.BUDGET)
        return budget <= 0.2

    def stats(self) -> dict:
        """Get orchestrator statistics."""
        return {
            "current_gradient": self.gradient.to_dict(),
            "total_updates": len(self.update_history),
            "should_recruit_help": self.should_recruit_help(),
            "should_reduce_capabilities": self.should_reduce_capabilities(),
            "phenotype_params": self.get_phenotype_params(),
        }
