"""Synthetic signal harness for paper 6 (Theorem 3 empirical test).

A minimal, tunable-difficulty task used to compare evaluator shapes in
GEPA: a certificate-based binary evaluator vs. a scalar-reward evaluator
vs. a scalar-with-evidence active control.  The task is intentionally
*synthetic* because the experiment measures the **evaluator-to-optimizer
pipeline**, not realism.  A real agent would introduce retrieval noise,
API flakiness, and agent-complexity variance that would confound the
signal we are trying to isolate.

The candidate prompt encodes a single numeric knob — ``policy_throttle``
in [0, 1] — via a well-known parser.  The harness simulates an agent
that emits per-window means drawn from a Gaussian centered on the
throttle value.  If ``max(window_means) > threshold``, the
``behavioral_stability_windowed`` theorem fails; the LM's job during
reflective evolution is to lower the throttle value in the candidate
text.

The task is deliberately degenerate in content (no actual domain
behavior) but non-trivial in *signal*: the LM must read feedback, map
it to a numeric edit, and propose a coherent replacement prompt.  That
is exactly the channel the Theorem 3 conjecture is about.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Candidate parsing — the LM's only lever
# ---------------------------------------------------------------------------

_THROTTLE_PATTERN = re.compile(
    r"policy_throttle\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

_DEFAULT_THROTTLE = 1.0
"""Starting throttle value — fails the theorem.  The LM must reduce it."""


def parse_throttle(candidate_text: str) -> float:
    """Extract the policy_throttle value from a candidate prompt.

    Returns ``_DEFAULT_THROTTLE`` if the prompt is missing the marker
    (unparseable prompts get the worst-case value rather than raising,
    so GEPA's reflection loop can still make progress from any garbage
    seed mutation the LM produces).

    Values are clipped to [0, 1] so an LM that emits ``policy_throttle=5``
    behaves identically to ``policy_throttle=1`` — this makes the
    failure-to-convergence a one-way function of *direction*, not
    magnitude.

    If the candidate text contains *multiple* ``policy_throttle``
    assignments (a common LM failure mode where a revised value is
    appended without removing the old one), the **last** assignment
    wins.  This matches the author's intent — editing by append is
    syntactically messy but semantically a refinement — and prevents
    stale first-assignment values from silently scoring valid
    mutations as failures (Roborev #864 H).
    """
    matches = _THROTTLE_PATTERN.findall(candidate_text or "")
    if not matches:
        return _DEFAULT_THROTTLE
    try:
        value = float(matches[-1])  # last assignment wins
    except (TypeError, ValueError):
        return _DEFAULT_THROTTLE
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Task difficulty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """Knobs for calibrating task difficulty.

    threshold:
        ``behavioral_stability_windowed`` stability threshold.  A
        candidate passes iff every window mean is at most ``threshold``.
    noise_sigma:
        Standard deviation of per-observation Gaussian noise.  Larger
        values push window means further from the candidate's throttle
        and therefore make the task harder.
    windows_per_rollout:
        Number of non-overlapping windows the harness computes per
        rollout.  More windows = more chances for one to exceed the
        threshold on a borderline candidate.
    observations_per_window:
        Observations averaged into each window mean.  Reduces noise.
    target_throttle:
        Throttle value at which a perfectly-reduced candidate would
        sit.  The conjecture is about *reaching* this neighborhood, not
        the exact value.
    """

    threshold: float = 0.5
    noise_sigma: float = 0.08
    windows_per_rollout: int = 4
    observations_per_window: int = 8
    target_throttle: float = 0.35


# ---------------------------------------------------------------------------
# Rollout + theorem parameters
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Record of a single harness rollout.  Consumed by reflection feedback."""

    data_inst_id: int
    candidate_text: str
    parsed_throttle: float
    window_means: tuple[float, ...]
    violating_windows: tuple[int, ...]
    threshold: float


def _seed_for(data_inst_id: int, run_seed: int) -> int:
    """Deterministic per-data-instance seed for full reproducibility."""
    payload = f"{run_seed}:{data_inst_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Take 8 bytes -> int, fits within Python's random.seed range.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def run_rollout(
    candidate_text: str,
    data_inst_id: int,
    *,
    config: TaskConfig | None = None,
    run_seed: int = 0,
) -> tuple[str, Trajectory, dict[str, Any]]:
    """Execute one rollout; return ``(output, trajectory, theorem_parameters)``.

    This signature matches the contract :class:`OperonCertificateAdapter`
    expects from its harness callback (see
    ``operon_ai/convergence/gepa_adapter.py``).
    """
    cfg = config or TaskConfig()
    throttle = parse_throttle(candidate_text)

    rng = random.Random(_seed_for(data_inst_id, run_seed))
    window_means: list[float] = []
    for _ in range(cfg.windows_per_rollout):
        observations = [
            rng.gauss(throttle, cfg.noise_sigma)
            for _ in range(cfg.observations_per_window)
        ]
        window_means.append(sum(observations) / len(observations))

    violating = tuple(
        idx for idx, mean in enumerate(window_means) if mean > cfg.threshold
    )

    trajectory = Trajectory(
        data_inst_id=data_inst_id,
        candidate_text=candidate_text,
        parsed_throttle=throttle,
        window_means=tuple(window_means),
        violating_windows=violating,
        threshold=cfg.threshold,
    )

    # ``behavioral_stability_windowed`` parameters — see
    # operon_ai/core/certificate.py:261
    parameters = {
        "signal_values": tuple(window_means),
        "threshold": cfg.threshold,
    }

    output = f"rollout(throttle={throttle:.3f}, max_mean={max(window_means):.3f})"
    return output, trajectory, parameters


# ---------------------------------------------------------------------------
# Data loader — trivial, just a sequence of integer ids
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    """Trivial dataset of integer rollout ids.

    GEPA's DataInst protocol is untyped; for this experiment we don't
    need any payload beyond a per-instance identifier that seeds the
    RNG inside ``run_rollout``.
    """

    size: int
    offset: int = 0

    def __iter__(self):
        return iter(range(self.offset, self.offset + self.size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        if not 0 <= idx < self.size:
            raise IndexError(idx)
        return self.offset + idx


# ---------------------------------------------------------------------------
# Seed candidate — the starting prompt every arm begins from
# ---------------------------------------------------------------------------


SEED_CANDIDATE_TEXT = """You operate a synthetic throttled source.

Emit observations drawn around the configured policy value.

policy_throttle = 1.0
""".strip()
"""Starting prompt — contains the policy_throttle marker at 1.0 (fails)."""


SEED_COMPONENT_NAME = "policy_prompt"
"""Name of the single mutable component in the GEPA candidate dict."""


def seed_candidate() -> dict[str, str]:
    """Return the starting candidate mapping: ``{policy_prompt: "..."}``."""
    return {SEED_COMPONENT_NAME: SEED_CANDIDATE_TEXT}


# ---------------------------------------------------------------------------
# Shared evidence block (Roborev #855)
# ---------------------------------------------------------------------------


def render_window_evidence(trajectory: Trajectory) -> str:
    """Render the per-window evidence + repair line common to both arms.

    Both cert-binary and scalar-evidence emit the *same* evidence text so
    the only remaining prompt-framing differences between them are:

    - cert-binary prepends a ``Theorem: <name> [FAILED|HOLDS]`` line,
    - scalar-evidence prepends a ``Score: <value>`` line.

    This function is the single source of truth for the shared block.
    Tests compare arm renderings after stripping the prepended line,
    guaranteeing any future drift fails the content-matched invariant
    immediately.
    """
    lines = ["Evidence:"]
    for idx, mean in enumerate(trajectory.window_means):
        marker = " [VIOLATING]" if idx in trajectory.violating_windows else ""
        lines.append(f"  - window {idx}: mean={mean:.3f}{marker}")
    if trajectory.violating_windows:
        lines.append(
            f"Adjust candidate so every window mean is at most {trajectory.threshold}."
        )
    return "\n".join(lines)
