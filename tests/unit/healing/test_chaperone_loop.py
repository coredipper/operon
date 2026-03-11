"""Tests for chaperone-loop healing feedback."""

from pydantic import BaseModel, Field

from operon_ai.healing import (
    ChaperoneLoop,
    HealingOutcome,
    create_mock_healing_generator,
)
from operon_ai.organelles.chaperone import Chaperone


class PriceQuote(BaseModel):
    """Minimal schema for chaperone-loop tests."""

    product: str
    price: float = Field(ge=0)
    currency: str = Field(default="USD", pattern="^[A-Z]{3}$")


def test_fold_enhanced_error_trace_includes_strategy_details():
    """Enhanced folding should preserve the underlying validation failures."""

    chaperone = Chaperone(silent=True)
    result = chaperone.fold_enhanced(
        '{"product": "X", "price": "bad", "currency": "USD"}',
        PriceQuote,
    )

    assert not result.valid
    assert result.error_trace is not None
    assert "Validation:" in result.error_trace
    assert "strict:" in result.error_trace


def test_mock_healing_generator_matches_error_feedback_case_insensitively():
    """The example helper should heal when validation feedback is present."""

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

    assert result.outcome == HealingOutcome.HEALED
    assert result.valid
    assert result.structure is not None
    assert result.structure.price == 10.0
