"""
Chaperone Loop: Structural Self-Healing through Error Feedback
==============================================================

Biological Analogy:
- GroEL/GroES: Chaperone proteins that cage misfolded proteins and provide
  a protected environment for refolding. The cage doesn't just reject failures;
  it gives the protein multiple chances to fold correctly.
- Unfolded Protein Response (UPR): A cellular stress pathway activated when
  misfolded proteins accumulate, triggering repair mechanisms.

The Chaperone Loop extends the basic Chaperone (which validates outputs) into
a feedback loop where validation errors are passed back to the generator. This
enables context-aware correction rather than blind retry.

The key insight: The error trace itself is information. "TypeError: 'one hundred'
is not a valid float" tells the generator exactly what went wrong, enabling
targeted repair rather than random regeneration.
"""

from dataclasses import dataclass, field
from typing import Type, TypeVar, Callable, Protocol
from enum import Enum

from pydantic import BaseModel

from ..organelles.chaperone import (
    Chaperone,
    EnhancedFoldedProtein,
)

T = TypeVar("T", bound=BaseModel)


class HealingOutcome(Enum):
    """Outcome of the chaperone healing loop."""

    VALID_FIRST_TRY = "valid_first_try"  # Folded correctly on first attempt
    HEALED = "healed"  # Required refolding but eventually succeeded
    DEGRADED = "degraded"  # All retries exhausted, marked for degradation


@dataclass
class RefoldingAttempt:
    """Record of a single refolding attempt."""

    attempt_number: int
    raw_output: str
    error_trace: str | None
    success: bool
    confidence: float


@dataclass
class HealingResult:
    """
    Result of the chaperone healing loop.

    Contains the folded protein (if successful), healing outcome,
    and full history of refolding attempts.
    """

    outcome: HealingOutcome
    folded: EnhancedFoldedProtein | None
    attempts: list[RefoldingAttempt] = field(default_factory=list)
    final_confidence: float = 0.0

    # Degradation marker (biological: ubiquitin tag)
    ubiquitin_tagged: bool = False

    @property
    def valid(self) -> bool:
        """Check if healing produced a valid result."""
        return self.outcome in (HealingOutcome.VALID_FIRST_TRY, HealingOutcome.HEALED)

    @property
    def structure(self):
        """Get the folded structure if valid."""
        return self.folded.structure if self.folded and self.folded.valid else None


class Generator(Protocol):
    """Protocol for output generators that support error feedback."""

    def __call__(self, prompt: str, error_context: str | None = None) -> str:
        """
        Generate output for a prompt.

        Args:
            prompt: The original prompt/request
            error_context: Optional error trace from previous failed attempt

        Returns:
            Raw output string (e.g., JSON)
        """
        ...


@dataclass
class ChaperoneLoop:
    """
    Feedback loop for structural self-healing.

    Wraps a generator and chaperone to create a healing loop where
    validation errors are fed back to the generator for context-aware
    refolding.

    Biological parallel: GroEL/GroES chaperone cage that gives misfolded
    proteins multiple chances to refold correctly. The error (misfolding)
    becomes input to the repair process.

    Example:
        >>> def my_generator(prompt: str, error: str | None = None) -> str:
        ...     if error and "not a valid float" in error:
        ...         return '{"price": 100.0}'  # Fix the type error
        ...     return '{"price": "one hundred"}'  # Initial bad output
        ...
        >>> loop = ChaperoneLoop(
        ...     generator=my_generator,
        ...     chaperone=Chaperone(),
        ...     schema=PriceQuote,
        ...     max_retries=3,
        ... )
        >>> result = loop.heal("Generate a price quote")
        >>> result.outcome
        HealingOutcome.HEALED
    """

    generator: Callable[[str, str | None], str]
    chaperone: Chaperone
    schema: Type[BaseModel]
    max_retries: int = 3
    confidence_decay: float = 0.1  # Reduce confidence each retry
    silent: bool = False

    def heal(self, prompt: str) -> HealingResult:
        """
        Run the healing loop.

        1. Generate output
        2. Attempt to fold
        3. If folding fails, feed error back to generator
        4. Repeat until success or max_retries exhausted

        Args:
            prompt: The prompt to send to the generator

        Returns:
            HealingResult with outcome, folded protein, and attempt history
        """
        attempts: list[RefoldingAttempt] = []
        error_context: str | None = None
        base_confidence = 1.0

        for attempt_num in range(self.max_retries + 1):
            # Calculate confidence for this attempt (decays with each retry)
            current_confidence = max(0.0, base_confidence - (attempt_num * self.confidence_decay))

            # Generate output (with error context if this is a retry)
            raw_output = self.generator(prompt, error_context)

            # Attempt to fold
            folded = self.chaperone.fold_enhanced(raw_output, self.schema)

            if folded.valid:
                # Success!
                outcome = (
                    HealingOutcome.VALID_FIRST_TRY
                    if attempt_num == 0
                    else HealingOutcome.HEALED
                )

                attempts.append(
                    RefoldingAttempt(
                        attempt_number=attempt_num,
                        raw_output=raw_output,
                        error_trace=None,
                        success=True,
                        confidence=current_confidence,
                    )
                )

                # Adjust folded confidence based on retry count
                folded.confidence = min(folded.confidence, current_confidence)

                if not self.silent:
                    if outcome == HealingOutcome.HEALED:
                        print(
                            f"🧬 [ChaperoneLoop] Healed after {attempt_num + 1} attempts "
                            f"(confidence: {folded.confidence:.2f})"
                        )

                return HealingResult(
                    outcome=outcome,
                    folded=folded,
                    attempts=attempts,
                    final_confidence=folded.confidence,
                    ubiquitin_tagged=False,
                )

            # Folding failed - record attempt and prepare error context for retry
            error_trace = folded.error_trace or "Unknown folding error"
            attempts.append(
                RefoldingAttempt(
                    attempt_number=attempt_num,
                    raw_output=raw_output,
                    error_trace=error_trace,
                    success=False,
                    confidence=0.0,
                )
            )

            if not self.silent and attempt_num < self.max_retries:
                print(
                    f"🧬 [ChaperoneLoop] Misfolded output detected (attempt {attempt_num + 1}). "
                    f"Feeding error back for refolding..."
                )

            # Prepare error context for next attempt
            error_context = self._format_error_context(error_trace, raw_output)

        # All retries exhausted - tag for degradation
        if not self.silent:
            print(
                f"🧬 [ChaperoneLoop] UBIQUITIN_TAG - All {self.max_retries + 1} attempts failed. "
                f"Marking for degradation."
            )

        return HealingResult(
            outcome=HealingOutcome.DEGRADED,
            folded=None,
            attempts=attempts,
            final_confidence=0.0,
            ubiquitin_tagged=True,
        )

    def _format_error_context(self, error_trace: str, raw_output: str) -> str:
        """
        Format error context to feed back to the generator.

        This is the "information signal" that enables context-aware repair.
        """
        return (
            f"Previous output was invalid. Error: {error_trace}\n"
            f"Your output was: {raw_output[:200]}{'...' if len(raw_output) > 200 else ''}\n"
            f"Please correct the output to match the expected schema."
        )


def create_mock_healing_generator(
    initial_output: str,
    healed_output: str,
    heal_on_error_containing: str,
) -> Callable[[str, str | None], str]:
    """
    Create a mock generator for testing that heals when it sees a specific error.

    Args:
        initial_output: The (potentially invalid) initial output
        healed_output: The corrected output to return when error is detected
        heal_on_error_containing: Substring to look for in error context

    Returns:
        A generator function suitable for ChaperoneLoop
    """

    def generator(prompt: str, error_context: str | None = None) -> str:  # noqa: ARG001
        _ = prompt  # Unused in mock - real generator would use this
        if error_context and heal_on_error_containing in error_context:
            return healed_output
        return initial_output

    return generator
