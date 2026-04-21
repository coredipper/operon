"""Synthetic signal harness for paper 6 (Theorem 3 empirical test).

A minimal, tunable-difficulty task used to compare evaluator shapes in
GEPA: a certificate-based binary evaluator vs. a scalar-reward evaluator
vs. a scalar-with-evidence active control.  The task is intentionally
*synthetic* because the experiment measures the **evaluator-to-optimizer
pipeline**, not realism.  A real agent would introduce retrieval noise,
API flakiness, and agent-complexity variance that would confound the
signal we are trying to isolate.

Candidate shape (Roborev #872 root-cause fix)
---------------------------------------------

The candidate is a two-component dict:

- ``policy_prompt``: prose the LM is free to evolve.  The harness
  **never** parses this component.  It exists so the reflection LM has
  a scratch-space for reasoning and framing.
- ``policy_throttle``: a numeric-only component whose value is the
  throttle the harness uses.  The harness reads it directly via
  :func:`parse_throttle` — no regex, no anchoring, no structural
  validation beyond ``float()``.

Rationale
---------

Earlier revisions (see Roborev #864, #867, #868, #869, #870, #871,
#872) tried to parse the throttle out of the prose prompt by pattern
matching.  Each pattern-based rule (``re.search`` first-match,
line-start anchoring, fenced code blocks, ``CONFIG:`` prefixes, the
exactly-one constraint) was defeated by a slightly different LM
output shape: stale first match, later line-start example, fenced
example, quoted ``CONFIG:`` example, multi-line split ``CONFIG:``.
The class of bug is structural: *semantic* validation (real config
vs. quoted example) cannot be performed reliably on LM-generated
free-form text.

The root-cause fix is to take the throttle out of the prose entirely
and make it its own GEPA component.  Reading
``candidate["policy_throttle"]`` has no text-level failure modes; it
either is a parseable number or it isn't.  This closes the entire
spoofing bug class and trivializes validation to a single ``float()``
call with a clip.

The task is deliberately degenerate in content (no actual domain
behavior) but non-trivial in *signal*: the LM still reads obligation
text (or scalar rewards) from the reflection feedback and must
propose an updated numeric component value.  That is exactly the
channel the Theorem 3 conjecture is about.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Candidate shape — two GEPA components
# ---------------------------------------------------------------------------

PROMPT_COMPONENT_NAME = "policy_prompt"
"""Name of the prose component (LM scratch-space; harness never parses it)."""

THROTTLE_COMPONENT_NAME = "policy_throttle"
"""Name of the numeric component read directly by the harness."""

CANDIDATE_COMPONENTS: tuple[str, ...] = (
    PROMPT_COMPONENT_NAME,
    THROTTLE_COMPONENT_NAME,
)
"""Public tuple of GEPA components paper-6 candidates expose."""

# Backward-compatible alias retained so callers that only ever touched
# the prose component (pre-#872) continue to import it.  New code
# should use ``PROMPT_COMPONENT_NAME``.
SEED_COMPONENT_NAME = PROMPT_COMPONENT_NAME

_DEFAULT_THROTTLE = 1.0
"""Starting throttle value — fails the theorem.  The LM must reduce it."""


# ---------------------------------------------------------------------------
# Read throttle from the numeric component directly.
# No regex.  No structural validation of prose.  No spoofing class.
# ---------------------------------------------------------------------------


def parse_throttle(candidate: Mapping[str, str]) -> float:
    """Read ``policy_throttle`` from a GEPA candidate dict.

    The candidate is expected to contain a ``policy_throttle`` component
    whose value is a numeric string (e.g. ``"0.3"``, ``"1.0"``).  The
    function does exactly three things:

    1. Extract the string.
    2. ``float()`` it.  A non-numeric / empty value falls back to
       ``_DEFAULT_THROTTLE`` so GEPA's reflection feedback sees a clear
       "that value isn't a number" signal.
    3. Clip to ``[0, 1]``.

    No regex.  No line-start anchoring.  No prose-vs-example
    disambiguation.  The spoofing bug class that produced Roborev #864,
    #867, #868, #869, #870, #871, and #872 is structurally absent
    because there is no text-level parsing surface.

    For safety with legacy callers that still pass the prose string,
    passing a ``str`` rather than a dict raises ``TypeError`` — this
    is a breaking contract change, documented in the module docstring.
    """
    if not isinstance(candidate, Mapping):
        raise TypeError(
            "parse_throttle now takes a candidate dict "
            "({'policy_prompt': ..., 'policy_throttle': ...}) rather than "
            "a prose string.  The prose prompt is no longer parsed — see "
            "the module docstring for the post-#872 contract."
        )
    raw = candidate.get(THROTTLE_COMPONENT_NAME, "")
    if not isinstance(raw, str):
        return _DEFAULT_THROTTLE
    raw = raw.strip()
    if not raw:
        return _DEFAULT_THROTTLE
    try:
        value = float(raw)
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
    candidate: Mapping[str, str]
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
    candidate: Mapping[str, str],
    data_inst_id: int,
    *,
    config: TaskConfig | None = None,
    run_seed: int = 0,
) -> tuple[str, Trajectory, dict[str, Any]]:
    """Execute one rollout; return ``(output, trajectory, theorem_parameters)``.

    This signature matches the contract :class:`OperonCertificateAdapter`
    expects from its harness callback (see
    ``operon_ai/convergence/gepa_adapter.py``).  The harness reads
    ``candidate["policy_throttle"]`` directly — the prose component is
    not parsed.
    """
    cfg = config or TaskConfig()
    throttle = parse_throttle(candidate)

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
        candidate=dict(candidate),
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
# Seed candidate — the starting candidate every arm begins from
# ---------------------------------------------------------------------------


SEED_PROMPT_TEXT = """You operate a synthetic throttled source.

Emit observations drawn around the configured policy value.  Lower
values are safer; higher values violate the stability threshold.

You will be asked to revise the ``policy_throttle`` component.  That
component is a bare number; respond with a single float.
""".strip()
"""Starting prose prompt.  Harness never parses this.  Exists so the
reflection LM has a scratch-space for reasoning."""


SEED_CANDIDATE: Mapping[str, str] = {
    PROMPT_COMPONENT_NAME: SEED_PROMPT_TEXT,
    THROTTLE_COMPONENT_NAME: str(_DEFAULT_THROTTLE),
}
"""Starting candidate — throttle at 1.0 (fails the theorem)."""


# Backward-compatible alias; pre-#872 code referred to the prose body
# directly via ``SEED_CANDIDATE_TEXT``.  New code should read the
# component out of ``SEED_CANDIDATE`` by name.
SEED_CANDIDATE_TEXT = SEED_PROMPT_TEXT


def seed_candidate() -> dict[str, str]:
    """Return a fresh copy of the starting candidate dict."""
    return dict(SEED_CANDIDATE)


def candidate_dict_with_throttle(value: float | str) -> dict[str, str]:
    """Build a well-formed candidate dict with the given throttle value.

    Canonical builder for tests and synthetic candidates under the
    post-#872 two-component candidate contract.  Callers pass the
    numeric value they want the harness to read; the prose component
    defaults to the seed prompt.
    """
    return {
        PROMPT_COMPONENT_NAME: SEED_PROMPT_TEXT,
        THROTTLE_COMPONENT_NAME: str(value),
    }


# Backward-compatible alias.  Pre-#872 callers used
# ``candidate_text_with_throttle`` to build a prose string.  The shape
# is now a dict; we keep the old name pointing at the new builder to
# minimise downstream churn.  New code should use
# ``candidate_dict_with_throttle``.
candidate_text_with_throttle = candidate_dict_with_throttle


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
