from __future__ import annotations

from dataclasses import dataclass
import json
import random

from operon_ai.surveillance import ImmuneSystem
from operon_ai.surveillance.types import ThreatLevel
from eval.utils import Counter, random_words, clamp


@dataclass
class ImmuneConfig:
    agents: int = 40
    compromised: int = 8
    train_observations: int = 30
    eval_observations: int = 20
    min_training_samples: int = 10
    min_observations: int = 10
    window_size: int = 50
    baseline_confidence: float = 0.9
    baseline_confidence_std: float = 0.05
    baseline_time: float = 0.25
    baseline_time_std: float = 0.05
    compromised_confidence: float = 0.55
    compromised_time: float = 0.65
    canary_fail_rate: float = 0.4


def _make_output(rng: random.Random, vocab: list[str], structure: str) -> str:
    words = rng.choices(vocab, k=rng.randint(8, 16))
    if structure == "json":
        payload = {"text": " ".join(words), "value": rng.randint(0, 10)}
        return json.dumps(payload)
    if structure == "bullet":
        return "- " + "\n- ".join(words)
    return " ".join(words)


def _sample_confidence(rng: random.Random, mean: float, std: float) -> float:
    return clamp(rng.gauss(mean, std), 0.0, 1.0)


def _sample_time(rng: random.Random, mean: float, std: float) -> float:
    return max(0.001, rng.gauss(mean, std))


def run_immune(config: ImmuneConfig, rng: random.Random) -> dict:
    immune = ImmuneSystem(
        min_training_samples=config.min_training_samples,
        min_observations=config.min_observations,
        window_size=config.window_size,
    )

    agent_ids = [f"agent_{i}" for i in range(config.agents)]
    compromised_set = set(rng.sample(agent_ids, k=config.compromised))

    base_vocab = random_words(rng, 40)
    compromised_vocab = random_words(rng, 40)

    # Training phase
    for agent_id in agent_ids:
        immune.register_agent(agent_id)
        for _ in range(config.train_observations):
            output = _make_output(rng, base_vocab, structure="plain")
            confidence = _sample_confidence(rng, config.baseline_confidence, config.baseline_confidence_std)
            response_time = _sample_time(rng, config.baseline_time, config.baseline_time_std)
            immune.record_observation(
                agent_id=agent_id,
                output=output,
                response_time=response_time,
                confidence=confidence,
                error=None,
            )
            immune.record_canary_result(agent_id, passed=True)
        immune.train_agent(agent_id)

    # Evaluation phase
    detected_compromised = 0
    false_positive = 0

    for agent_id in agent_ids:
        is_compromised = agent_id in compromised_set
        detected = False

        for _ in range(config.eval_observations):
            if is_compromised:
                output = _make_output(rng, compromised_vocab, structure="json")
                confidence = _sample_confidence(rng, config.compromised_confidence, config.baseline_confidence_std)
                response_time = _sample_time(rng, config.compromised_time, config.baseline_time_std)
                immune.record_canary_result(agent_id, passed=rng.random() > config.canary_fail_rate)
            else:
                output = _make_output(rng, base_vocab, structure="plain")
                confidence = _sample_confidence(rng, config.baseline_confidence, config.baseline_confidence_std)
                response_time = _sample_time(rng, config.baseline_time, config.baseline_time_std)
                immune.record_canary_result(agent_id, passed=True)

            immune.record_observation(
                agent_id=agent_id,
                output=output,
                response_time=response_time,
                confidence=confidence,
                error=None,
            )

            response = immune.inspect(agent_id)
            if response.threat_level in (ThreatLevel.CONFIRMED, ThreatLevel.CRITICAL):
                detected = True

        if is_compromised:
            if detected:
                detected_compromised += 1
        else:
            if detected:
                false_positive += 1

    sensitivity = Counter("sensitivity", total=config.compromised, success=detected_compromised)
    false_positive_rate = Counter("false_positive", total=config.agents - config.compromised, success=false_positive)

    return {
        "agents": config.agents,
        "compromised": config.compromised,
        "sensitivity": sensitivity.as_dict(),
        "false_positive_rate": false_positive_rate.as_dict(),
    }
