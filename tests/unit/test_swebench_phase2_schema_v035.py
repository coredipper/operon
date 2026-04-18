"""Tests for Phase C artifact schema extensions.

Review #757: the v0.35.0 Phase C artifact claims retry behavior and
reason codes in the paper but the schema doesn't persist either. These
tests lock in the schema extension so the artifact actually carries
the evidence the paper makes claims from.

Run-level fields:
- ``grounding``: bool — whether --grounding was active
- ``retry_on_reject``: bool — whether --retry-on-reject was active

Per-result (per instance × condition) fields:
- ``sanitize_reason``: str — reason code from sanitize_with_reason
  on the final (possibly retried) attempt. "" on success.
- ``retry_attempted``: bool — whether a retry was invoked at all
- ``retry_recovered``: bool — whether the retry produced a usable patch
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.swebench_phase2 import Prediction, build_artifact  # noqa: E402


def test_prediction_has_retry_metadata_fields():
    """Prediction carries the three retry-related fields so the per-
    result writer can serialize them."""
    p = Prediction(
        instance_id="x", repo="o/r", condition="baseline",
        raw_output="", model_patch="",
        latency_ms=0.0, extract_ok=False,
    )
    # New fields must exist and default cleanly.
    assert hasattr(p, "sanitize_reason")
    assert hasattr(p, "retry_attempted")
    assert hasattr(p, "retry_recovered")
    assert p.sanitize_reason == ""
    assert p.retry_attempted is False
    assert p.retry_recovered is False


def test_build_artifact_top_level_includes_retry_and_grounding_flags():
    """The artifact envelope must carry grounding + retry_on_reject so
    a reader can tell which pipeline produced these numbers without
    needing the original CLI invocation."""
    artifact = build_artifact(
        model="x",
        model_identity={"tag": "x"},
        post_run_check={"status": "match"},
        run_id="r",
        n_instances=0,
        offset=0,
        conditions=[],
        timestamp="t",
        skip_harness=False,
        results=[],
        summary={},
        grounding=True,
        retry_on_reject=True,
    )
    assert artifact["grounding"] is True
    assert artifact["retry_on_reject"] is True


def test_build_artifact_grounding_and_retry_default_to_false():
    """When the caller omits the flags (old v0.34.x callers that
    preceded the retry feature), the artifact still has the fields
    with safe defaults so a reader doesn't have to guess."""
    artifact = build_artifact(
        model="x",
        model_identity={"tag": "x"},
        post_run_check={"status": "match"},
        run_id="r",
        n_instances=0,
        offset=0,
        conditions=[],
        timestamp="t",
        skip_harness=False,
        results=[],
        summary={},
    )
    assert artifact["grounding"] is False
    assert artifact["retry_on_reject"] is False
