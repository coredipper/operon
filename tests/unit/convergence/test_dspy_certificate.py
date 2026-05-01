"""Tests for the DSPy compile-pinned-inputs provenance binding (cheap T2).

Covers the verify function, the public factory helper, and the
``Certificate.from_dspy_compile`` classmethod entry point. No DSPy
import — the binding is wire-format only by design (§2 lines 105–106).
"""

from __future__ import annotations

import json

import pytest

from operon_ai.convergence.dspy_certificate import (
    make_dspy_compile_certificate,
)
from operon_ai.core.certificate import (
    Certificate,
    certificate_to_dict,
)


# 16-char lowercase hex strings — operon's truncated-SHA256 convention.
_HASH_A = "a" * 16
_HASH_B = "b" * 16
_HASH_C = "c" * 16
_HASH_D = "d" * 16


# ---------------------------------------------------------------------------
# Construction — both entry points
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_classmethod_returns_certificate(self) -> None:
        cert = Certificate.from_dspy_compile(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        assert isinstance(cert, Certificate)
        assert cert.theorem == "dspy_compile_pinned_inputs"

    def test_helper_returns_certificate(self) -> None:
        cert = make_dspy_compile_certificate(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        assert isinstance(cert, Certificate)
        assert cert.theorem == "dspy_compile_pinned_inputs"

    def test_default_source_label(self) -> None:
        cert = Certificate.from_dspy_compile(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        assert cert.source == "Certificate.from_dspy_compile"

    def test_source_override(self) -> None:
        cert = make_dspy_compile_certificate(
            _HASH_A, _HASH_B, _HASH_C, _HASH_D, source="my_pipeline"
        )
        assert cert.source == "my_pipeline"

    def test_conclusion_keys_on_trace_hash(self) -> None:
        cert = Certificate.from_dspy_compile(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        assert _HASH_D in cert.conclusion


# ---------------------------------------------------------------------------
# verify() — pass cases
# ---------------------------------------------------------------------------


class TestVerifyPasses:
    def test_all_well_formed_holds(self) -> None:
        cert = Certificate.from_dspy_compile(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        v = cert.verify()
        assert v.holds is True
        assert list(v.evidence["missing"]) == []
        assert list(v.evidence["malformed"]) == []
        assert set(v.evidence["present_keys"]) == {
            "program_hash",
            "trainset_hash",
            "metric_hash",
            "trace_hash",
        }

    def test_minimum_length_8_holds(self) -> None:
        """Length-8 hex is the operon truncation floor — should still hold."""
        h8 = "abcdef01"
        cert = Certificate.from_dspy_compile(h8, h8, h8, h8)
        assert cert.verify().holds is True

    def test_distinct_hashes_held_separately(self) -> None:
        """Each parameter is verified independently — not a global hash check."""
        cert = Certificate.from_dspy_compile(
            "deadbeef" * 2, "1234abcd" * 2, "ffeeddcc" * 2, "00112233" * 2
        )
        assert cert.verify().holds is True


# ---------------------------------------------------------------------------
# verify() — fail cases
# ---------------------------------------------------------------------------


class TestVerifyFails:
    def test_too_short_hash_fails(self) -> None:
        cert = make_dspy_compile_certificate(
            "abc", _HASH_B, _HASH_C, _HASH_D
        )
        v = cert.verify()
        assert v.holds is False
        assert "program_hash" in list(v.evidence["malformed"])

    def test_non_hex_chars_fail(self) -> None:
        cert = make_dspy_compile_certificate(
            "z" * 16, _HASH_B, _HASH_C, _HASH_D
        )
        v = cert.verify()
        assert v.holds is False
        assert "program_hash" in list(v.evidence["malformed"])

    def test_uppercase_hex_fails(self) -> None:
        """Operon's convention is lowercase hexdigest — uppercase is malformed."""
        cert = make_dspy_compile_certificate(
            "A" * 16, _HASH_B, _HASH_C, _HASH_D
        )
        v = cert.verify()
        assert v.holds is False
        assert "program_hash" in list(v.evidence["malformed"])

    def test_empty_string_fails(self) -> None:
        cert = make_dspy_compile_certificate("", _HASH_B, _HASH_C, _HASH_D)
        v = cert.verify()
        assert v.holds is False
        assert "program_hash" in list(v.evidence["malformed"])

    def test_non_string_value_fails(self) -> None:
        """Direct theorem construction with a non-string leaks past the helper.

        The helper only takes str args, but a downstream consumer could
        construct from_theorem directly with a numeric value.
        """
        cert = Certificate.from_theorem(
            theorem="dspy_compile_pinned_inputs",
            parameters={
                "program_hash": 12345,
                "trainset_hash": _HASH_B,
                "metric_hash": _HASH_C,
                "trace_hash": _HASH_D,
            },
            conclusion="manual",
            source="manual",
        )
        v = cert.verify()
        assert v.holds is False
        assert "program_hash" in list(v.evidence["malformed"])

    def test_missing_key_fails(self) -> None:
        """Construct via from_theorem with a missing key; verify reports it."""
        cert = Certificate.from_theorem(
            theorem="dspy_compile_pinned_inputs",
            parameters={
                "program_hash": _HASH_A,
                "trainset_hash": _HASH_B,
                # metric_hash deliberately missing
                "trace_hash": _HASH_D,
            },
            conclusion="manual",
            source="manual",
        )
        v = cert.verify()
        assert v.holds is False
        assert "metric_hash" in list(v.evidence["missing"])


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_certificate_to_dict_is_jsonable(self) -> None:
        cert = Certificate.from_dspy_compile(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
        d = certificate_to_dict(cert)
        # Should serialize cleanly — no exceptions, dict round-trips.
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["theorem"] == "dspy_compile_pinned_inputs"
        assert round_tripped["parameters"]["trace_hash"] == _HASH_D
        assert round_tripped["source"] == "Certificate.from_dspy_compile"


# ---------------------------------------------------------------------------
# Theorem registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_theorem_is_resolvable(self) -> None:
        """The theorem name should resolve via Certificate.from_theorem too."""
        cert = Certificate.from_theorem(
            theorem="dspy_compile_pinned_inputs",
            parameters={
                "program_hash": _HASH_A,
                "trainset_hash": _HASH_B,
                "metric_hash": _HASH_C,
                "trace_hash": _HASH_D,
            },
            conclusion="resolved-via-from_theorem",
            source="test",
        )
        assert cert.verify().holds is True

    def test_unknown_theorem_still_raises(self) -> None:
        """Sanity: registry isn't accidentally accepting arbitrary names."""
        with pytest.raises(KeyError):
            Certificate.from_theorem(
                theorem="dspy_compile_pinned_inputs_typo",
                parameters={"program_hash": _HASH_A},
                conclusion="x",
                source="test",
            )
