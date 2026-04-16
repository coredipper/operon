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


_POST_RUN_CHECK_VALID_STATUSES = {"match", "mismatch", "error"}


def test_committed_artifact_schema_matches_resolver():
    """The checked-in results file must match the resolver's output schema.

    If someone changes the resolver, they must regenerate the artifact
    (or vice versa). This test fails if the two drift. It also covers
    sibling fields that the writer always emits so a missing top-level
    key cannot slip through (see review #700).
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

    # Top-level keys that the writer in eval/swebench_phase2.py always
    # emits alongside the summary. Extending this list is a deliberate
    # contract change and must include a matching artifact regeneration.
    required_top_level = {
        "model",
        "model_identity",
        "model_identity_post_run_check",
        "dataset",
        "run_id",
        "summary",
        "results",
    }
    missing = required_top_level - artifact.keys()
    assert not missing, (
        f"committed artifact is missing top-level keys {missing}; "
        "regenerate eval/results/swebench_phase2.json or update the writer"
    )

    check = artifact["model_identity_post_run_check"]
    assert isinstance(check, dict), (
        "model_identity_post_run_check must be a dict "
        "{status: match|mismatch|error, ...}"
    )
    assert check.get("status") in _POST_RUN_CHECK_VALID_STATUSES, (
        f"unexpected model_identity_post_run_check.status {check.get('status')!r}; "
        f"expected one of {_POST_RUN_CHECK_VALID_STATUSES}"
    )
    if check["status"] == "mismatch":
        for key in ("digest_now", "blob_sha256_now"):
            assert key in check, (
                f"mismatch status requires {key!r}"
            )
    elif check["status"] == "error":
        assert "error" in check, "error status requires 'error' message"
