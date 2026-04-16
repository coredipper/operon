"""Tests for SWE-bench Phase 2 model-identity resolution.

Roborev #697 required that the shape of `model_identity` in the committed
results artifact match what `_resolve_model_identity()` produces on a
fresh run. These tests lock the schema and the failure contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.swebench_phase2 import (  # noqa: E402
    ARTIFACT_TOP_LEVEL_KEYS,
    POST_RUN_CHECK_STATUSES,
    ModelIdentityError,
    _IDENTITY_REQUIRED,
    _parse_ollama_show,
    _resolve_model_identity,
)


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


def test_committed_artifact_schema_matches_resolver():
    """The checked-in results file must match the writer's contract.

    Locks (a) the nested identity key set, and (b) the exact top-level
    artifact key set. Exact equality is used because drift in either
    direction — a new writer field not in the artifact, or an artifact
    field no longer emitted — is a contract mismatch and must be
    addressed explicitly (see reviews #700, #702).
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
            "regenerate eval/results/swebench_phase2.json"
        )

    # Exact top-level key equality. If the writer grows/shrinks keys,
    # this test must be updated alongside the artifact, making drift
    # impossible to introduce silently.
    actual = set(artifact.keys())
    assert actual == set(ARTIFACT_TOP_LEVEL_KEYS), (
        f"top-level keys drift: "
        f"missing={set(ARTIFACT_TOP_LEVEL_KEYS) - actual} "
        f"extra={actual - set(ARTIFACT_TOP_LEVEL_KEYS)}; "
        "regenerate eval/results/swebench_phase2.json or update "
        "ARTIFACT_TOP_LEVEL_KEYS in eval/swebench_phase2.py"
    )

    check = artifact["model_identity_post_run_check"]
    assert isinstance(check, dict), (
        "model_identity_post_run_check must be a dict"
    )
    assert check.get("status") in POST_RUN_CHECK_STATUSES, (
        f"unexpected status {check.get('status')!r}; "
        f"expected one of {sorted(POST_RUN_CHECK_STATUSES)}"
    )
    if check["status"] == "mismatch":
        for key in ("digest_now", "blob_sha256_now"):
            assert key in check, f"mismatch status requires {key!r}"
    elif check["status"] == "error":
        assert "error" in check, "error status requires 'error' message"
    elif check["status"] == "not_performed":
        # This status is only valid for artifacts predating the writer
        # contract; a note is required so readers see why it is there.
        assert "note" in check, (
            "not_performed status requires a 'note' explaining why the "
            "post-run check was not executed"
        )
