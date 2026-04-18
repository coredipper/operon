"""Tests for SWE-bench Phase 2 format-correction retry.

Phase C wedge: when the sanitizer rejects a model's diff, optionally
re-prompt the model once with the specific rejection reason and try
again. The retry is gated by a CLI flag (``--retry-on-reject``) so
v0.34.5-era callers see zero behavior change.

These tests cover:

* ``_sanitize_for_submission`` invokes a retry callback iff the first
  sanitization rejects AND a callback is provided.
* The callback receives the sanitizer's reason code and the original
  failed output, so the retry prompt can be targeted.
* A successful retry returns the retry's sanitized patch (not the
  original failed one).
* A failing retry returns ``""`` — no infinite loops, no fallback to
  the original.
* Retry count is capped at :data:`_FORMAT_RETRY_MAX`.
* No callback = old behavior (direct ``sanitize_with_reason`` call
  with no retry attempt).
* ``_build_retry_prompt`` embeds the reason string so the model can
  act on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.swebench_phase2 import (  # noqa: E402
    _FORMAT_RETRY_MAX,
    _build_retry_prompt,
    _sanitize_for_submission,
)


VALID_DIFF = (
    "--- a/django/foo.py\n"
    "+++ b/django/foo.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-old\n"
    "+new\n"
)

# A diff the sanitizer will reject for placeholder hunks.
PLACEHOLDER_DIFF = (
    "--- a/django/foo.py\n"
    "+++ b/django/foo.py\n"
    "@@ -XXX,1 +XXX,1 @@\n"
    "-old\n"
    "+new\n"
)


def test_format_retry_max_default_is_one():
    """One retry is the diagnostic default: doubles cost per failure,
    more retries diminish in returns, easy to bump via the constant."""
    assert _FORMAT_RETRY_MAX == 1


def test_sanitize_for_submission_no_callback_is_old_behavior():
    """With no callback, a reject yields ``patch=""`` like v0.34.5.
    retry_attempted must be False so the artifact reflects "never
    tried" rather than conflating with "tried and failed". Review
    #757 expanded the return shape from ``str`` to ``SanitizeOutcome``."""
    outcome = _sanitize_for_submission(
        PLACEHOLDER_DIFF, "django/django", tree_paths=None,
    )
    assert outcome.patch == ""
    assert outcome.reason == "placeholder_hunk"
    assert outcome.retry_attempted is False
    assert outcome.retry_recovered is False


def test_sanitize_for_submission_callback_fires_on_reject():
    """When the sanitizer rejects, the callback is invoked with the
    reason code and the original failed output, and the returned
    outcome records retry_attempted=True even on recovery success."""
    seen: dict = {}

    def cb(reason: str, failed_output: str) -> str:
        seen["reason"] = reason
        seen["failed_output"] = failed_output
        return VALID_DIFF  # pretend the retry succeeded

    outcome = _sanitize_for_submission(
        PLACEHOLDER_DIFF, "django/django", tree_paths=None,
        retry_callback=cb,
    )

    assert seen["reason"] == "placeholder_hunk"
    assert seen["failed_output"] == PLACEHOLDER_DIFF
    assert outcome.patch, "successful retry should yield a non-empty patch"
    assert "--- a/django/foo.py" in outcome.patch
    assert outcome.retry_attempted is True
    assert outcome.retry_recovered is True


def test_sanitize_for_submission_callback_not_fired_on_success():
    """If the first sanitization already succeeds, the retry callback
    must NOT be invoked (wasted budget, misleading side effects), and
    retry_attempted must be False in the outcome."""
    calls: list = []

    def cb(reason: str, failed_output: str) -> str:
        calls.append(reason)
        return ""

    outcome = _sanitize_for_submission(
        VALID_DIFF, "django/django", tree_paths=None,
        retry_callback=cb,
    )

    assert outcome.patch, "valid diff should pass through unchanged"
    assert outcome.reason == ""
    assert outcome.retry_attempted is False
    assert outcome.retry_recovered is False
    assert calls == [], f"callback must not fire on success; got {calls}"


def test_sanitize_for_submission_retry_also_rejects_returns_empty():
    """If the retry also produces garbage, the outcome has patch=""
    but retry_attempted=True so the artifact can tell the difference
    between "tried and failed" and "never tried"."""
    def cb(reason: str, failed_output: str) -> str:
        # The retry produced the SAME rejected output.
        return PLACEHOLDER_DIFF

    outcome = _sanitize_for_submission(
        PLACEHOLDER_DIFF, "django/django", tree_paths=None,
        retry_callback=cb,
    )

    assert outcome.patch == ""
    assert outcome.retry_attempted is True
    assert outcome.retry_recovered is False
    assert outcome.reason, "reason must be populated so the retry prompt had a target"


def test_sanitize_for_submission_retry_cap_enforced():
    """Retry count must be capped at _FORMAT_RETRY_MAX so a chronically
    failing model can't trigger unbounded retry calls."""
    call_count = [0]

    def cb(reason: str, failed_output: str) -> str:
        call_count[0] += 1
        return PLACEHOLDER_DIFF  # always fails

    _sanitize_for_submission(
        PLACEHOLDER_DIFF, "django/django", tree_paths=None,
        retry_callback=cb,
    )

    assert call_count[0] == _FORMAT_RETRY_MAX, (
        f"callback fired {call_count[0]}x; expected {_FORMAT_RETRY_MAX}"
    )


def test_build_retry_prompt_includes_reason_and_failed_output():
    """The retry prompt must contain both the reason code (so the
    model knows what specifically went wrong) and the original failed
    output (so it can correct rather than regenerate from scratch)."""
    failed = "--- a/foo.py\n+++ b/foo.py\n@@ -XXX,1 +XXX,1 @@\n-x\n+y\n"
    prompt = _build_retry_prompt(
        original_prompt="Fix this bug.",
        reason="placeholder_hunk",
        failed_output=failed,
    )

    assert "placeholder_hunk" in prompt
    assert failed in prompt
    assert "real integer line numbers" in prompt.lower() or (
        "real line numbers" in prompt.lower()
    )
    assert "single fenced diff block" in prompt.lower()


def test_build_retry_prompt_distinguishes_reason_codes():
    """The prompt must be reason-aware: a path_not_found retry should
    mention paths; a placeholder_hunk retry should mention line
    numbers. Without this, the retry degrades to a generic re-ask."""
    path_prompt = _build_retry_prompt(
        "Fix this bug.", "path_not_found", "--- a/nope.py\n+++ b/nope.py\n",
    )
    placeholder_prompt = _build_retry_prompt(
        "Fix this bug.", "placeholder_hunk",
        "@@ -XXX,1 +XXX,1 @@\n-x\n+y\n",
    )

    assert path_prompt != placeholder_prompt
    # Reason-specific guidance should be in each.
    assert "path" in path_prompt.lower()
    assert (
        "line number" in placeholder_prompt.lower()
        or "integer" in placeholder_prompt.lower()
    )
