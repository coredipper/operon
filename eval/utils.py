from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import random
import string


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def random_word(rng: random.Random, min_len: int = 3, max_len: int = 8) -> str:
    length = rng.randint(min_len, max_len)
    letters = [rng.choice(string.ascii_lowercase) for _ in range(length)]
    return "".join(letters)


def random_words(rng: random.Random, count: int) -> list[str]:
    return [random_word(rng) for _ in range(count)]


@dataclass
class Counter:
    name: str
    total: int = 0
    success: int = 0

    def record(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.success += 1

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total

    def as_dict(self) -> dict:
        low, high = wilson_interval(self.success, self.total)
        return {
            "success": self.success,
            "total": self.total,
            "rate": self.rate,
            "wilson_95": [low, high],
        }
