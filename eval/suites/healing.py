from __future__ import annotations

from dataclasses import dataclass
import json
import random

from pydantic import BaseModel

from operon_ai.healing.chaperone_loop import ChaperoneLoop, HealingOutcome
from operon_ai.organelles.chaperone import Chaperone
from eval.utils import Counter


@dataclass
class HealingConfig:
    trials: int = 200
    max_retries: int = 3
    fix_prob_with_error: float = 0.85
    fix_prob_blind: float = 0.25


class PriceQuote(BaseModel):
    price: float
    qty: int
    currency: str


def _make_generator(rng: random.Random, fix_prob: float, honor_error: bool):
    def generator(_prompt: str, error_context: str | None = None) -> str:
        if honor_error and error_context and rng.random() < fix_prob:
            payload = {
                "price": round(rng.random() * 100, 2),
                "qty": rng.randint(1, 5),
                "currency": rng.choice(["USD", "EUR", "JPY"]),
            }
            return json.dumps(payload)
        if (not honor_error) and rng.random() < fix_prob:
            payload = {
                "price": round(rng.random() * 100, 2),
                "qty": rng.randint(1, 5),
                "currency": rng.choice(["USD", "EUR", "JPY"]),
            }
            return json.dumps(payload)
        return '{"price": "one hundred", "qty": "ten", "currency": 123}'

    return generator


def _run_loop(rng: random.Random, config: HealingConfig, honor_error: bool, fix_prob: float) -> dict:
    loop = ChaperoneLoop(
        generator=_make_generator(rng, fix_prob, honor_error),
        chaperone=Chaperone(silent=True),
        schema=PriceQuote,
        max_retries=config.max_retries,
        silent=True,
    )

    success = Counter("success")
    healed = Counter("healed")

    for _ in range(config.trials):
        result = loop.heal("Generate a price quote")
        success.record(result.valid)
        healed.record(result.outcome == HealingOutcome.HEALED)

    return {
        "success": success.as_dict(),
        "healed": healed.as_dict(),
    }


def run_healing(config: HealingConfig, rng: random.Random) -> dict:
    with_error = _run_loop(rng, config, honor_error=True, fix_prob=config.fix_prob_with_error)
    blind = _run_loop(rng, config, honor_error=False, fix_prob=config.fix_prob_blind)

    return {
        "trials": config.trials,
        "with_error_context": with_error,
        "blind_retry": blind,
    }
