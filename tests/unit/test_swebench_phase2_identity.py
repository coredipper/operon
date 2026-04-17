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
