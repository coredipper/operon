"""Pathway-grounded scenario parameters for quorum sensing benchmark.

Derives test scenarios from KEGG map02024 (quorum sensing pathway).
Parameters are normalized from V. fischeri LuxI/LuxR system kinetics.

Key biological constants used:
- AHL production rate: proportional to suspicion (LuxI synthase kinetics)
- AHL decay half-life: ~5 time units (lactonase degradation, 4-30 min in bacteria)
- Activation threshold: ~10 nM equivalent (LuxR binding constant), scaled to agent units
- Population scaling: log(N) (QS threshold depends on cell density)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class QSPathwayParams:
    """Parameters derived from KEGG map02024."""

    ahl_production_rate: float = 0.1  # Signal per unit suspicion per step
    ahl_decay_half_life: float = 5.0  # From lactonase degradation rates
    threshold_base: float = 10.0  # From LuxR activation concentration
    population_scaling: str = "log"  # Sublinear with population


@dataclass
class AgentProfile:
    """Profile for one agent in a quorum sensing scenario."""

    agent_id: str
    compromised: bool
    base_suspicion: float  # Baseline suspicion output


@dataclass
class QSScenarioConfig:
    """Configuration for a quorum sensing scenario."""

    n_agents: int
    n_compromised: int
    time_steps: int
    noise_std: float = 0.15
    agents: list[AgentProfile] = field(default_factory=list)


def make_agents(
    n_agents: int,
    n_compromised: int,
    rng: random.Random,
    compromised_suspicion: float = 0.8,
    normal_suspicion: float = 0.15,
) -> list[AgentProfile]:
    """Create agent profiles with compromised subset."""
    agents = []
    ids = [f"agent_{i}" for i in range(n_agents)]
    compromised_set = set(rng.sample(ids, k=min(n_compromised, n_agents)))
    for aid in ids:
        is_comp = aid in compromised_set
        agents.append(AgentProfile(
            agent_id=aid,
            compromised=is_comp,
            base_suspicion=compromised_suspicion if is_comp else normal_suspicion,
        ))
    return agents


def generate_static_compromise(
    n_agents: int,
    n_compromised: int,
    time_steps: int,
    noise_std: float,
    rng: random.Random,
) -> list[dict]:
    """Static scenario: compromised agents emit high suspicion throughout.

    All agents are present from t=0.  Compromised agents consistently
    emit high suspicion; normal agents emit low.  Tests basic detection.
    """
    agents = make_agents(n_agents, n_compromised, rng)
    steps = []
    for t in range(time_steps):
        suspicions = {}
        for a in agents:
            noise = rng.gauss(0, noise_std)
            suspicions[a.agent_id] = max(0.0, min(1.0, a.base_suspicion + noise))
        steps.append({
            "time": float(t),
            "suspicions": suspicions,
            "ground_truth_compromised": n_compromised > 0,
        })
    return steps


def generate_gradual_infiltration(
    n_agents: int,
    n_compromised: int,
    time_steps: int,
    noise_std: float,
    rng: random.Random,
) -> list[dict]:
    """Gradual infiltration: compromised agents appear over time.

    Tests whether signal decay naturally handles the temporal aspect.
    Old "clean" signals from before infiltration should decay, making
    the system responsive to the new threat.
    """
    agents = make_agents(n_agents, n_compromised, rng)
    compromised = [a for a in agents if a.compromised]
    activation_times = {
        a.agent_id: rng.randint(time_steps // 4, 3 * time_steps // 4)
        for a in compromised
    }

    steps = []
    for t in range(time_steps):
        suspicions = {}
        for a in agents:
            if a.compromised and t < activation_times.get(a.agent_id, 0):
                # Not yet compromised
                base = 0.15
            else:
                base = a.base_suspicion
            noise = rng.gauss(0, noise_std)
            suspicions[a.agent_id] = max(0.0, min(1.0, base + noise))

        active_compromised = sum(
            1 for a in compromised
            if t >= activation_times.get(a.agent_id, 0)
        )
        steps.append({
            "time": float(t),
            "suspicions": suspicions,
            "ground_truth_compromised": active_compromised > 0,
            "active_compromised": active_compromised,
        })
    return steps


def generate_noisy_environment(
    n_agents: int,
    time_steps: int,
    noise_std: float,
    rng: random.Random,
) -> list[dict]:
    """No actual compromise — all agents are noisy.

    Tests false positive rate: does the system alert on noise alone?
    Signal accumulation should average out noise better than voting.
    """
    agents = make_agents(n_agents, 0, rng)
    steps = []
    for t in range(time_steps):
        suspicions = {}
        for a in agents:
            noise = rng.gauss(0, noise_std)
            suspicions[a.agent_id] = max(0.0, min(1.0, a.base_suspicion + noise))
        steps.append({
            "time": float(t),
            "suspicions": suspicions,
            "ground_truth_compromised": False,
        })
    return steps


SCENARIOS = {
    "static": generate_static_compromise,
    "gradual_infiltration": generate_gradual_infiltration,
    "noisy": generate_noisy_environment,
}
