"""Pathway-grounded scenario parameters for EpiplexityMonitor benchmark.

Derives test scenarios from the Free Energy Principle and the biology
of trophic factor withdrawal (neuronal atrophy).

The key biological insight: stagnation isn't just "repeating yourself."
It's high uncertainty (perplexity) combined with low novelty.  An agent
can repeat similar outputs while making progress (converging), or produce
diverse outputs while stuck (exploring without learning).

The Bayesian two-signal approach should distinguish these cases;
a single-signal cosine detector can't.

References:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Trophic factor withdrawal induces neuronal death via atrophy, not necrosis
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from eval.benchmarks.scenarios import ScenarioStep


@dataclass
class TrophicParams:
    """Parameters derived from trophic factor withdrawal biology.

    Trophic withdrawal signature: novelty decays gradually while
    perplexity stays high.  The neuron is still "trying" (uncertain)
    but not getting anywhere (no new information).
    """

    novelty_decay_rate: float = 0.05  # 5% novelty decrease per step
    perplexity_maintenance: float = 0.7  # Perplexity stays high
    initial_novelty: float = 0.8
    critical_phase_onset: int = 20  # Steps before atrophy is pathological
    sequence_length: int = 50


def generate_loop(rng: random.Random, length: int = 50) -> list[ScenarioStep]:
    """Simple loop: exact same message repeated.

    Ground truth: stagnant after first few steps.
    Both biological and naive detectors should catch this easily.
    """
    msg = f"result_{rng.randint(0, 9999)}"
    steps = [ScenarioStep(message=msg, ground_truth_status="healthy")]
    for i in range(1, length):
        steps.append(ScenarioStep(
            message=msg,
            ground_truth_status="stagnant" if i > 3 else "healthy",
        ))
    return steps


def generate_convergence(rng: random.Random, length: int = 50) -> list[ScenarioStep]:
    """Healthy convergence: outputs become similar AND perplexity drops.

    Ground truth: converging (NOT stagnant).  A naive detector that only
    looks at output similarity will false-alarm here.
    """
    base = f"solution_{rng.randint(0, 9999)}"
    steps = []
    for i in range(length):
        # Messages become increasingly similar (converging to solution)
        noise_chars = rng.randint(0, max(0, length - i))
        noise = "".join(rng.choices("abcdef", k=noise_chars))
        msg = f"{base}_{noise}" if noise else base
        # Perplexity drops as agent becomes confident
        perp = max(0.1, 1.0 - (i / length) * 0.9)
        steps.append(ScenarioStep(
            message=msg,
            perplexity=perp,
            ground_truth_status="converging",
        ))
    return steps


def generate_exploration(rng: random.Random, length: int = 50) -> list[ScenarioStep]:
    """Healthy exploration: high novelty, agent making progress.

    Ground truth: exploring (healthy).
    """
    steps = []
    for i in range(length):
        msg = f"approach_{rng.randint(0, 999999)}_{i}"
        steps.append(ScenarioStep(
            message=msg,
            perplexity=rng.uniform(0.5, 1.0),
            ground_truth_status="exploring",
        ))
    return steps


def generate_trophic_withdrawal(
    rng: random.Random,
    params: TrophicParams | None = None,
) -> list[ScenarioStep]:
    """Trophic factor withdrawal: the biological signature of neuronal atrophy.

    Novelty decays gradually while perplexity remains high.  The agent
    is stuck but still uncertain — it produces outputs that are
    superficially different (different words) but semantically the same
    (controlled by gradually reducing token variation).

    THIS IS THE KEY TEST: EpiplexityMonitor should detect stagnation
    (high perplexity + low novelty).  RepetitionCounter may miss it
    because the outputs aren't exact copies, or may false-alarm on
    convergence if threshold is lowered to compensate.
    """
    if params is None:
        params = TrophicParams()

    base_tokens = [f"tok_{rng.randint(0, 50)}" for _ in range(8)]
    steps = []

    for i in range(params.sequence_length):
        # Novelty decays: fewer tokens change per step
        novelty = max(0.05, params.initial_novelty * (1 - params.novelty_decay_rate) ** i)
        n_changing = max(0, int(len(base_tokens) * novelty))

        # Build message: mix stable base with some changing tokens
        msg_tokens = list(base_tokens)
        for j in rng.sample(range(len(msg_tokens)), min(n_changing, len(msg_tokens))):
            msg_tokens[j] = f"var_{rng.randint(0, 999)}"
        msg = " ".join(msg_tokens)

        # Perplexity stays high (the agent is uncertain)
        perp = params.perplexity_maintenance + rng.uniform(-0.1, 0.1)

        # Ground truth: healthy early, stagnant after critical phase onset
        if i < params.critical_phase_onset:
            gt = "healthy"
        else:
            gt = "stagnant"

        steps.append(ScenarioStep(
            message=msg,
            perplexity=max(0.1, perp),
            ground_truth_status=gt,
        ))

    return steps


def generate_false_stagnation(rng: random.Random, length: int = 50) -> list[ScenarioStep]:
    """False stagnation: low novelty + low perplexity = convergence.

    This looks like stagnation to a naive detector (outputs are similar)
    but is actually healthy convergence (low perplexity = agent is confident).

    Ground truth: converging (NOT stagnant).
    A naive cosine-similarity detector WILL false-alarm here.
    EpiplexityMonitor should correctly identify CONVERGING (both signals low).
    """
    base = f"final_answer_{rng.randint(0, 9999)}"
    steps = []
    for i in range(length):
        # Very similar outputs
        minor_variation = rng.choice([".", "", " ", "  "])
        msg = f"{base}{minor_variation}"
        # Low perplexity — agent is confident
        perp = 0.1 + rng.uniform(0, 0.15)
        steps.append(ScenarioStep(
            message=msg,
            perplexity=perp,
            ground_truth_status="converging",
        ))
    return steps


# Registry for scenario generation
SCENARIOS = {
    "loop": generate_loop,
    "convergence": generate_convergence,
    "exploration": generate_exploration,
    "trophic_withdrawal": generate_trophic_withdrawal,
    "false_stagnation": generate_false_stagnation,
}
