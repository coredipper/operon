"""Tests for SWE-bench Phase 2 model-identity resolution.

Roborev #697 required that the shape of `model_identity` in the committed
results artifact match what `_resolve_model_identity()` produces on a
fresh run. These tests lock the schema and the failure contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.swebench_phase2 import (  # noqa: E402
    POST_RUN_CHECK_STATUSES,
    ModelIdentityError,
    _IDENTITY_REQUIRED,
    _parse_ollama_show,
    _resolve_model_identity,
    _rewrite_envelope,
    build_artifact,
)


def _writer_top_level_keys() -> set:
    """The writer's canonical top-level key set, derived from build_artifact().

    Dummy values here — we only care about the shape, not the data.
    Deriving from ``build_artifact`` instead of a hand-maintained
    constant makes it impossible for the writer and test to drift.
    """
    shape = build_artifact(
        model="x",
        model_identity={"tag": "x"},
        post_run_check={"status": "match"},
        run_id="x",
        n_instances=0,
        offset=0,
        conditions=[],
        timestamp="x",
        skip_harness=False,
        results=[],
        summary={},
    )
    return set(shape.keys())


OLLAMA_SHOW_SAMPLE = """  Model
    architecture        gemma4
    parameters          8.0B
    context length      131072
    embedding length    2560
    quantization        Q4_K_M
    requires            0.20.0

  Capabilities
    completion
    vision

  Parameters
    temperature    1
"""


def test_parse_ollama_show_extracts_published_fields():
    parsed = _parse_ollama_show(OLLAMA_SHOW_SAMPLE)
    assert parsed == {
        "architecture": "gemma4",
        "parameters": "8.0B",
        "quantization": "Q4_K_M",
    }


def test_parse_ollama_show_ignores_non_model_sections():
    # Values in the Parameters section (e.g. temperature) must not leak.
    parsed = _parse_ollama_show(OLLAMA_SHOW_SAMPLE)
    assert "temperature" not in parsed


def test_resolve_identity_returns_full_schema_on_success():
    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["ollama", "list"]:
            stdout = (
                "NAME            ID              SIZE      MODIFIED\n"
                "gemma4:latest   c6eb396dbd59    9.6 GB    10 days ago\n"
            )
        elif cmd[:3] == ["ollama", "show", "--modelfile"]:
            stdout = (
                "FROM /Users/x/.ollama/models/blobs/"
                "sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9"
                "aba5385242cb6e09a\n"
            )
        elif cmd[:3] == ["ollama", "show", "gemma4:latest"]:
            stdout = OLLAMA_SHOW_SAMPLE
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return SimpleNamespace(returncode=0, stdout=stdout)

    with patch("eval.swebench_phase2.subprocess.run", side_effect=fake_run):
        identity = _resolve_model_identity("gemma4:latest")

    # Every published field must be present — the committed artifact and
    # the paper both reference these keys.
    for key in _IDENTITY_REQUIRED:
        assert key in identity, f"missing required key: {key}"
    assert identity["tag"] == "gemma4:latest"
    assert identity["digest"] == "c6eb396dbd59"
    assert identity["blob_sha256"].startswith("4c27e0f5")
    assert identity["architecture"] == "gemma4"
    assert identity["parameters"] == "8.0B"
    assert identity["quantization"] == "Q4_K_M"


def test_resolve_identity_raises_when_ollama_missing():
    def fail_run(_cmd, **_kwargs):
        raise FileNotFoundError("ollama")

    with patch("eval.swebench_phase2.subprocess.run", side_effect=fail_run):
        with pytest.raises(ModelIdentityError):
            _resolve_model_identity("gemma4:latest")


def test_resolve_identity_raises_when_digest_not_in_list():
    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["ollama", "list"]:
            stdout = "NAME  ID  SIZE  MODIFIED\n"
        elif cmd[:3] == ["ollama", "show", "--modelfile"]:
            stdout = "FROM /x/sha256-deadbeef\n"
        else:
            stdout = OLLAMA_SHOW_SAMPLE
        return SimpleNamespace(returncode=0, stdout=stdout)

    with patch("eval.swebench_phase2.subprocess.run", side_effect=fake_run):
        with pytest.raises(ModelIdentityError, match="digest"):
            _resolve_model_identity("gemma4:latest")


def test_committed_artifact_schema_matches_writer():
    """The checked-in results file must match what the writer emits.

    The expected key set is derived from ``build_artifact`` itself, not
    from a hand-maintained constant. Exact equality is enforced: any
    drift — writer grows a key, writer shrinks a key, or the artifact
    ages out of sync — must be resolved explicitly (e.g. by running
    ``python -m eval.swebench_phase2 --rewrite-envelope PATH`` against
    the committed file). See reviews #700, #702, #704.
    """
    import json

    artifact = json.loads(
        (Path(__file__).resolve().parents[2]
         / "eval/results/swebench_phase2.json").read_text()
    )

    # Nested identity keys.
    identity = artifact["model_identity"]
    for key in _IDENTITY_REQUIRED:
        assert key in identity, (
            f"committed artifact is missing required identity key {key!r}; "
            "regenerate eval/results/swebench_phase2.json via "
            "`python -m eval.swebench_phase2 --rewrite-envelope "
            "eval/results/swebench_phase2.json`"
        )

    expected = _writer_top_level_keys()
    actual = set(artifact.keys())
    assert actual == expected, (
        f"writer/artifact top-level drift: "
        f"missing={expected - actual} extra={actual - expected}; "
        "regenerate via `python -m eval.swebench_phase2 "
        "--rewrite-envelope eval/results/swebench_phase2.json` or "
        "update build_artifact() in eval/swebench_phase2.py"
    )

    check = artifact["model_identity_post_run_check"]
    assert isinstance(check, dict), (
        "model_identity_post_run_check must be a dict"
    )
    assert check.get("status") in POST_RUN_CHECK_STATUSES, (
        f"unexpected status {check.get('status')!r}; "
        f"the live writer only emits {sorted(POST_RUN_CHECK_STATUSES)}"
    )
    if check["status"] == "mismatch":
        for key in ("digest_now", "blob_sha256_now"):
            assert key in check, f"mismatch status requires {key!r}"
    elif check["status"] == "error":
        assert "error" in check, "error status requires 'error' message"


def test_rewrite_envelope_refuses_explicit_model_mismatch(tmp_path):
    """If --model is supplied explicitly and disagrees with the artifact,
    --rewrite-envelope must refuse rather than verify against the wrong tag.

    Catches the review #707 regression: previously, args.model (defaulting
    to gemma4:latest) was used for verification even when the artifact
    recorded a different model, which could silently emit a false
    match/mismatch for a model that never ran.
    """
    artifact = build_artifact(
        model="llama3.1:8b",  # artifact is for a different model
        model_identity={
            "tag": "llama3.1:8b",
            "digest": "abc123",
            "blob_sha256": "def",
            "architecture": "llama",
            "parameters": "8.0B",
            "quantization": "Q4_K_M",
            "source": "test",
        },
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
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))

    # Caller passed --model explicitly with a DIFFERENT tag — must refuse.
    with pytest.raises(SystemExit) as exc_info:
        _rewrite_envelope(p, "gemma4:latest", cli_model_was_default=False)
    assert exc_info.value.code == 1

    # Artifact must not have been overwritten.
    after = json.loads(p.read_text())
    assert after == artifact


def test_rewrite_envelope_uses_artifact_model_when_default(tmp_path):
    """When --model takes its default, rewrite verifies against the
    artifact's recorded model, not the CLI default.

    This is the "rewrite a historical artifact" path. The artifact
    records what actually ran; the rewrite must trust it.
    """
    artifact = build_artifact(
        model="llama3.1:8b",
        model_identity={
            "tag": "llama3.1:8b",
            "digest": "abc123",
            "blob_sha256": "deadbeef",
            "architecture": "llama",
            "parameters": "8.0B",
            "quantization": "Q4_K_M",
            "source": "test",
        },
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
    p = tmp_path / "a.json"
    p.write_text(json.dumps(artifact))

    seen_tags = []

    def fake_resolve(tag):
        seen_tags.append(tag)
        return {
            "tag": tag,
            "digest": "abc123",
            "blob_sha256": "deadbeef",
            "architecture": "llama",
            "parameters": "8.0B",
            "quantization": "Q4_K_M",
            "source": "test",
        }

    # cli_model_was_default=True means user did NOT set --model.
    # The CLI tag passed in is the default ("gemma4:latest"), but the
    # rewrite should ignore it and verify against the artifact's "llama3.1:8b".
    with patch(
        "eval.swebench_phase2._resolve_model_identity",
        side_effect=fake_resolve,
    ):
        _rewrite_envelope(p, "gemma4:latest", cli_model_was_default=True)

    assert seen_tags == ["llama3.1:8b"], (
        f"verification tag must come from the artifact, not the CLI default; "
        f"got {seen_tags}"
    )
    after = json.loads(p.read_text())
    assert after["model"] == "llama3.1:8b"
    assert after["model_identity_post_run_check"]["status"] == "match"


def test_classify_prediction_error_reason_wins_over_empty_patch():
    """When the harness reports empty_patch, an error_reason still wins.
    This is the original review #747 reclassification."""
    from eval.swebench_phase2 import classify_prediction  # noqa: PLC0415
    assert classify_prediction("TimeoutError", "empty_patch") == "runtime_error"


def test_classify_prediction_error_reason_wins_over_not_evaluated():
    """The review #748 case: --skip-harness or a missing harness report
    leaves harness_status = not_evaluated, but the model still failed
    to return. error_reason must win so the failure stays visible."""
    from eval.swebench_phase2 import classify_prediction  # noqa: PLC0415
    assert classify_prediction("TimeoutError", "not_evaluated") == "runtime_error"


def test_classify_prediction_error_reason_wins_over_harness_error():
    """Even a harness-reported 'error' loses to error_reason — if the
    model call raised, no patch ever reached the harness, so a
    harness 'error' would be a phantom from a stale state."""
    from eval.swebench_phase2 import classify_prediction  # noqa: PLC0415
    assert classify_prediction("TimeoutError", "error") == "runtime_error"


def test_classify_prediction_no_error_reason_passes_harness_through():
    """When error_reason is None, the harness status is authoritative."""
    from eval.swebench_phase2 import classify_prediction  # noqa: PLC0415
    assert classify_prediction(None, "resolved") == "resolved"
    assert classify_prediction(None, "unresolved") == "unresolved"
    assert classify_prediction(None, "empty_patch") == "empty_patch"
    assert classify_prediction(None, "error") == "error"
    assert classify_prediction(None, "not_evaluated") == "not_evaluated"


def test_classify_prediction_no_error_reason_empty_harness_defaults_to_not_evaluated():
    """Defensive: an empty/missing harness_status string defaults to
    not_evaluated rather than propagating the empty value."""
    from eval.swebench_phase2 import classify_prediction  # noqa: PLC0415
    assert classify_prediction(None, "") == "not_evaluated"


def test_eval_runtime_error_status_distinct_from_empty_patch():
    """Review #747: model-call exceptions must record as ``runtime_error``
    not ``empty_patch``, so the artifact distinguishes "model failed to
    return" from "model returned but sanitizer rejected".
    """
    from eval.swebench_phase2 import (  # noqa: PLC0415
        EVAL_EMPTY, EVAL_RUNTIME_ERROR,
    )
    assert EVAL_RUNTIME_ERROR == "runtime_error"
    assert EVAL_RUNTIME_ERROR != EVAL_EMPTY


def test_prediction_error_reason_default_is_none():
    """Sanity: a Prediction without an explicit error_reason should be
    None, signalling "model returned text, classification is downstream
    of the harness". Review #747.
    """
    from eval.swebench_phase2 import Prediction  # noqa: PLC0415
    p = Prediction(
        instance_id="x", repo="o/r", condition="baseline",
        raw_output="some text", model_patch="--- a/x.py\n",
        latency_ms=100.0, extract_ok=True,
    )
    assert p.error_reason is None


def test_prediction_error_reason_marks_runtime_failure():
    """When constructed with error_reason set (the exception path in
    main()), the Prediction carries the failure tag forward."""
    from eval.swebench_phase2 import Prediction  # noqa: PLC0415
    p = Prediction(
        instance_id="x", repo="o/r", condition="baseline",
        raw_output="ERROR: timeout", model_patch="",
        latency_ms=0.0, extract_ok=False,
        error_reason="TimeoutError: API request timed out",
    )
    assert p.error_reason is not None
    assert "Timeout" in p.error_reason


def test_committed_artifact_distinguishes_runtime_errors():
    """The committed artifact must carry the runtime_error reclassification.

    The 2026-04-17 grounded rerun had two baseline timeouts
    (astropy-12907 and astropy-14995). After review #747 they should
    appear as ``eval_status=runtime_error`` with a non-null
    ``error_reason``, NOT ``empty_patch`` lumped with sanitizer-rejected
    outputs. Review #748 added the count + identity pins below so a
    silent regeneration that lost the reclassification would fail.
    """
    artifact = json.loads(
        (Path(__file__).resolve().parents[2]
         / "eval/results/swebench_phase2.json").read_text()
    )
    runtime_errors = [
        r for r in artifact["results"]
        if r.get("eval_status") == "runtime_error"
    ]

    # Pin: exactly the two known baseline timeouts must be classified
    # as runtime_error. If a future rerun produces zero or different
    # runtime errors, this test should fail loudly so the artifact
    # gets re-examined rather than silently shipping wrong numbers.
    assert len(runtime_errors) == 2, (
        f"expected exactly 2 runtime_error rows in the committed "
        f"artifact (the two 2026-04-17 baseline API timeouts); got "
        f"{len(runtime_errors)}"
    )
    expected_ids = {"astropy__astropy-12907", "astropy__astropy-14995"}
    actual_ids = {r["instance_id"] for r in runtime_errors}
    assert actual_ids == expected_ids, (
        f"runtime_error rows must be the two baseline timeouts "
        f"({expected_ids}); got {actual_ids}"
    )
    for r in runtime_errors:
        assert r["condition"] == "baseline", (
            f"the known runtime errors are baseline-only; got "
            f"{r['instance_id']}/{r['condition']}"
        )
        assert r.get("error_reason"), (
            f"runtime_error row {r['instance_id']}/{r['condition']} "
            f"must carry an error_reason; got {r.get('error_reason')!r}"
        )
        assert r["latency_ms"] == 0.0, (
            f"runtime_error rows have latency_ms=0.0 by construction"
        )

    # Summary must reflect the same count.
    assert artifact["summary"]["baseline"]["n_runtime_errors"] == 2
    assert artifact["summary"]["baseline"]["status_counts"]["runtime_error"] == 2
    assert artifact["summary"]["organism"]["n_runtime_errors"] == 0
    assert artifact["summary"]["langgraph"]["n_runtime_errors"] == 0


def test_committed_artifact_summary_includes_runtime_error_count():
    """The summary must expose ``n_runtime_errors`` and ``n_completed``
    so downstream readers can recompute honest mean-latency and reason
    about model-failure vs sanitizer-rejection ratios."""
    artifact = json.loads(
        (Path(__file__).resolve().parents[2]
         / "eval/results/swebench_phase2.json").read_text()
    )
    for cond, s in artifact["summary"].items():
        assert "n_runtime_errors" in s, (
            f"summary[{cond}] missing n_runtime_errors (review #747)"
        )
        assert "n_completed" in s, (
            f"summary[{cond}] missing n_completed (review #747)"
        )
        # status_counts now includes the runtime_error key.
        assert "runtime_error" in s["status_counts"], (
            f"summary[{cond}].status_counts missing 'runtime_error' key"
        )


def test_build_artifact_has_stable_shape():
    """build_artifact is the single source of truth for the envelope.

    A snapshot test so changes to the shape surface as an explicit test
    failure rather than a silent drift. If you edit build_artifact(),
    also update this expected set.
    """
    assert _writer_top_level_keys() == {
        "model",
        "model_identity",
        "model_identity_post_run_check",
        "dataset",
        "run_id",
        "n_instances",
        "offset",
        "conditions",
        "timestamp",
        "skip_harness",
        "results",
        "summary",
    }


def test_rewrite_envelope_writes_to_output_path_when_provided(tmp_path):
    """Review #755: when --rewrite-envelope is combined with --output
    <DIFFERENT_PATH>, the rewritten artifact must be written to the
    output path, NOT to the input path. Without this, --output is
    silently ignored in rewrite mode and callers can accidentally
    overwrite the artifact they were trying to preserve.
    """
    artifact = build_artifact(
        model="gemma4:latest",
        model_identity={
            "tag": "gemma4:latest",
            "digest": "abc123",
            "blob_sha256": "deadbeef",
            "architecture": "gemma4",
            "parameters": "8.0B",
            "quantization": "Q4_K_M",
            "source": "test",
        },
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
    src = tmp_path / "source_artifact.json"
    dst = tmp_path / "rewritten_artifact.json"
    src.write_text(json.dumps(artifact))

    def fake_resolve(tag):
        return {
            "tag": tag, "digest": "abc123", "blob_sha256": "deadbeef",
            "architecture": "gemma4", "parameters": "8.0B",
            "quantization": "Q4_K_M", "source": "test",
        }

    with patch(
        "eval.swebench_phase2._resolve_model_identity",
        side_effect=fake_resolve,
    ):
        _rewrite_envelope(
            src, "gemma4:latest", cli_model_was_default=True,
            output_path=dst,
        )

    # Source artifact must be untouched — that's the whole point of
    # using --output instead of the default in-place rewrite.
    assert src.exists(), "source artifact must still exist"
    src_content = json.loads(src.read_text())
    assert src_content == artifact, (
        "source artifact must not be mutated when --output is supplied"
    )

    # Destination exists with the rewritten envelope.
    assert dst.exists(), "rewrite must write the new path"
    dst_content = json.loads(dst.read_text())
    assert dst_content["model"] == "gemma4:latest"
    assert dst_content["model_identity_post_run_check"]["status"] == "match"


def test_rewrite_envelope_defaults_to_input_path_when_output_omitted(tmp_path):
    """Backward-compat: without --output, rewrite stays in-place so
    existing callers behave exactly as before review #755.
    """
    artifact = build_artifact(
        model="gemma4:latest",
        model_identity={
            "tag": "gemma4:latest",
            "digest": "abc123",
            "blob_sha256": "deadbeef",
            "architecture": "gemma4",
            "parameters": "8.0B",
            "quantization": "Q4_K_M",
            "source": "test",
        },
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
    p = tmp_path / "in_place.json"
    p.write_text(json.dumps(artifact))

    def fake_resolve(tag):
        return {
            "tag": tag, "digest": "abc123", "blob_sha256": "deadbeef",
            "architecture": "gemma4", "parameters": "8.0B",
            "quantization": "Q4_K_M", "source": "test",
        }

    with patch(
        "eval.swebench_phase2._resolve_model_identity",
        side_effect=fake_resolve,
    ):
        _rewrite_envelope(p, "gemma4:latest", cli_model_was_default=True)

    # In-place rewrite still works — path exists, envelope updated.
    assert p.exists()
    content = json.loads(p.read_text())
    assert content["model"] == "gemma4:latest"
    assert content["model_identity_post_run_check"]["status"] == "match"
