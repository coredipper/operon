#!/usr/bin/env python3
"""
Example 39: Chaperone Healing Loop (Structural Self-Repair)
===========================================================

Demonstrates the biological Chaperone Loop pattern where validation
failures are fed back to the generator for context-aware refolding.

Key concepts:
- Feedback-driven repair: Error traces guide regeneration
- Confidence decay: Each retry reduces output confidence
- Ubiquitin tagging: Mark unfixable outputs for degradation

Biological parallel:
- GroEL/GroES: Isolation chamber giving proteins a second chance to fold
- Unfolded Protein Response: Stress pathway when folding repeatedly fails

The key insight: The error trace itself is information. "TypeError: 'one hundred'
is not a valid float" tells the generator exactly what went wrong, enabling
targeted repair rather than blind retry.

Usage:
    python examples/39_chaperone_healing_loop.py
    python examples/39_chaperone_healing_loop.py --test
"""

import sys

from pydantic import BaseModel, Field

from operon_ai import Chaperone
from operon_ai.healing import (
    ChaperoneLoop,
    HealingOutcome,
    create_mock_healing_generator,
)


# =============================================================================
# Schema Definition
# =============================================================================

class PriceQuote(BaseModel):
    """Schema for price quotes with strict numeric types."""

    product: str
    price: float = Field(ge=0, description="Price must be a non-negative number")
    currency: str = Field(default="USD", pattern="^[A-Z]{3}$")


class InventoryItem(BaseModel):
    """Schema for inventory items."""

    sku: str
    name: str
    quantity: int = Field(ge=0)
    in_stock: bool


# =============================================================================
# Demo: Successful Healing
# =============================================================================

def demo_successful_healing():
    """
    Demo where the generator learns from the error and heals.

    The mock generator initially outputs {"price": "one hundred"} (invalid),
    but when it sees the error trace containing "not a valid float", it
    corrects to {"price": 100.0}.
    """
    print("=" * 60)
    print("Demo 1: Successful Healing (Structural Repair)")
    print("=" * 60)

    # Create a mock generator that heals when it sees type error
    generator = create_mock_healing_generator(
        initial_output='{"product": "Widget", "price": "one hundred", "currency": "USD"}',
        healed_output='{"product": "Widget", "price": 100.0, "currency": "USD"}',
        heal_on_error_containing="validation error",
    )

    # Create the healing loop
    loop = ChaperoneLoop(
        generator=generator,
        chaperone=Chaperone(),
        schema=PriceQuote,
        max_retries=3,
        confidence_decay=0.15,
    )

    # Run the healing
    print("\nInitial generator output is malformed (price as string)...")
    result = loop.heal("Generate a price quote for Widget")

    print(f"\nHealing Result:")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Valid: {result.valid}")
    print(f"  Confidence: {result.final_confidence:.2f}")
    print(f"  Attempts: {len(result.attempts)}")

    if result.valid:
        print(f"  Structure: {result.structure}")

    print("\nAttempt History:")
    for attempt in result.attempts:
        status = "✓" if attempt.success else "✗"
        error = f" - {attempt.error_trace[:50]}..." if attempt.error_trace else ""
        print(f"  [{status}] Attempt {attempt.attempt_number + 1}: conf={attempt.confidence:.2f}{error}")

    return result


# =============================================================================
# Demo: Exhausted Retries (Ubiquitin Tagging)
# =============================================================================

def demo_exhausted_retries():
    """
    Demo where the generator cannot be healed and is marked for degradation.

    The mock generator always outputs the same malformed data, never learning
    from the error feedback. After max_retries, it's ubiquitin-tagged.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Exhausted Retries (Ubiquitin Tagging for Degradation)")
    print("=" * 60)

    # Create a stubborn generator that never heals
    def stubborn_generator(prompt: str, error_context: str | None = None) -> str:
        _ = prompt, error_context  # Always ignores feedback
        return '{"product": "Gadget", "price": "expensive", "currency": "USD"}'

    loop = ChaperoneLoop(
        generator=stubborn_generator,
        chaperone=Chaperone(),
        schema=PriceQuote,
        max_retries=2,  # Fewer retries for demo
    )

    print("\nGenerator always outputs invalid data (price='expensive')...")
    result = loop.heal("Generate a price quote for Gadget")

    print(f"\nHealing Result:")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Valid: {result.valid}")
    print(f"  Ubiquitin Tagged: {result.ubiquitin_tagged}")
    print(f"  Attempts: {len(result.attempts)}")

    print("\nAttempt History:")
    for attempt in result.attempts:
        status = "✓" if attempt.success else "✗"
        error = f" - {attempt.error_trace[:40]}..." if attempt.error_trace else ""
        print(f"  [{status}] Attempt {attempt.attempt_number + 1}{error}")

    if result.ubiquitin_tagged:
        print("\n⚠️  Output marked for degradation (ubiquitin tagged)")
        print("    This signals downstream systems to handle gracefully")

    return result


# =============================================================================
# Demo: First-Try Success
# =============================================================================

def demo_first_try_success():
    """
    Demo where the generator produces valid output on the first try.

    No healing is needed - this is the ideal case.
    """
    print("\n" + "=" * 60)
    print("Demo 3: First-Try Success (No Healing Needed)")
    print("=" * 60)

    def perfect_generator(prompt: str, error_context: str | None = None) -> str:
        _ = prompt, error_context
        return '{"sku": "ABC-123", "name": "Premium Widget", "quantity": 50, "in_stock": true}'

    loop = ChaperoneLoop(
        generator=perfect_generator,
        chaperone=Chaperone(),
        schema=InventoryItem,
        max_retries=3,
    )

    print("\nGenerator produces valid output on first try...")
    result = loop.heal("Get inventory for Premium Widget")

    print(f"\nHealing Result:")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Valid: {result.valid}")
    print(f"  Confidence: {result.final_confidence:.2f}")
    print(f"  Attempts: {len(result.attempts)}")
    print(f"  Structure: {result.structure}")

    return result


# =============================================================================
# Demo: Gradual Healing
# =============================================================================

def demo_gradual_healing():
    """
    Demo where healing takes multiple attempts.

    The generator gets progressively better, fixing one issue at a time.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Gradual Healing (Multiple Corrections)")
    print("=" * 60)

    # Simulate a generator that improves gradually
    attempt_counter = {"count": 0}

    def improving_generator(prompt: str, error_context: str | None = None) -> str:
        _ = prompt
        attempt_counter["count"] += 1
        attempt = attempt_counter["count"]

        if attempt == 1:
            # First: completely wrong
            return '{"product": "Gizmo", "price": "free", "currency": "dollars"}'
        elif attempt == 2:
            # Second: fixed price, currency still wrong
            return '{"product": "Gizmo", "price": 29.99, "currency": "dollars"}'
        else:
            # Third: all correct
            return '{"product": "Gizmo", "price": 29.99, "currency": "USD"}'

    loop = ChaperoneLoop(
        generator=improving_generator,
        chaperone=Chaperone(),
        schema=PriceQuote,
        max_retries=5,
        confidence_decay=0.1,
    )

    print("\nGenerator gradually improves with each feedback cycle...")
    result = loop.heal("Generate a price quote for Gizmo")

    print(f"\nHealing Result:")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Final Confidence: {result.final_confidence:.2f}")
    print(f"  Total Attempts: {len(result.attempts)}")

    print("\nAttempt History (showing gradual improvement):")
    for attempt in result.attempts:
        status = "✓" if attempt.success else "✗"
        output_preview = attempt.raw_output[:50].replace("\n", " ")
        print(f"  [{status}] Attempt {attempt.attempt_number + 1}: {output_preview}...")

    return result


# =============================================================================
# Smoke Test
# =============================================================================

def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke test...")

    # Test 1: Successful healing
    generator = create_mock_healing_generator(
        initial_output='{"product": "X", "price": "bad", "currency": "USD"}',
        healed_output='{"product": "X", "price": 10.0, "currency": "USD"}',
        heal_on_error_containing="validation",
    )
    loop = ChaperoneLoop(
        generator=generator,
        chaperone=Chaperone(silent=True),
        schema=PriceQuote,
        max_retries=3,
        silent=True,
    )
    result = loop.heal("test")
    assert result.outcome == HealingOutcome.HEALED, f"Expected HEALED, got {result.outcome}"
    assert result.valid, "Expected valid result"
    assert result.structure is not None, "Expected structure"
    assert result.structure.price == 10.0, f"Expected price=10.0, got {result.structure.price}"

    # Test 2: Degradation
    def bad_gen(p, e=None):
        _ = p, e
        return '{"invalid": true}'

    loop2 = ChaperoneLoop(
        generator=bad_gen,
        chaperone=Chaperone(silent=True),
        schema=PriceQuote,
        max_retries=1,
        silent=True,
    )
    result2 = loop2.heal("test")
    assert result2.outcome == HealingOutcome.DEGRADED, f"Expected DEGRADED, got {result2.outcome}"
    assert result2.ubiquitin_tagged, "Expected ubiquitin tag"

    # Test 3: First-try success
    def good_gen(p, e=None):
        _ = p, e
        return '{"product": "Y", "price": 5.0, "currency": "EUR"}'

    loop3 = ChaperoneLoop(
        generator=good_gen,
        chaperone=Chaperone(silent=True),
        schema=PriceQuote,
        max_retries=3,
        silent=True,
    )
    result3 = loop3.heal("test")
    assert result3.outcome == HealingOutcome.VALID_FIRST_TRY, f"Expected VALID_FIRST_TRY, got {result3.outcome}"
    assert len(result3.attempts) == 1, "Expected single attempt"

    print("✓ All smoke tests passed!")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 39: Chaperone Healing Loop")
    print("Structural Self-Repair through Error Feedback")
    print("=" * 60)

    demo_successful_healing()
    demo_exhausted_retries()
    demo_first_try_success()
    demo_gradual_healing()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Chaperone Loop demonstrates structural healing:

1. VALID_FIRST_TRY: Perfect output, no healing needed
2. HEALED: Required refolding but eventually succeeded
3. DEGRADED: All retries failed, marked for degradation

Key biological parallel: GroEL/GroES chaperone proteins don't just
reject misfolded proteins - they give them a protected environment
to try again. The error becomes information for repair.

Key software insight: Instead of blind retry, we feed the error trace
back to the generator. "TypeError: 'one hundred' is not float" enables
targeted correction, not random regeneration.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
