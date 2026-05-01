"""Tests for the agentflow evolve-pinned-inputs provenance binding (cheap L2).

Covers the verify function, the public factory helper, and the
``Certificate.from_agentflow_compile`` classmethod entry point. No
agentflow import — agentflow is a Python-CLI framework; the binding
is wire-format only by design (§8.3 + §2 lines 105–106).
"""

from __future__ import annotations

import json

import pytest

from operon_ai.convergence.agentflow_certificate import (
    make_agentflow_compile_certificate,
)
from operon_ai.core.certificate import (
    Certificate,
    certificate_to_dict,
)


_HASH_A = "a" * 16
_HASH_B = "b" * 16
_HASH_C = "c" * 16


class TestConstruction:
    def test_classmethod_returns_certificate(self) -> None:
        cert = Certificate.from_agentflow_compile(_HASH_A, _HASH_B, _HASH_C)
        assert isinstance(cert, Certificate)
        assert cert.theorem == "agentflow_evolve_pinned_inputs"

    def test_helper_returns_certificate(self) -> None:
        cert = make_agentflow_compile_certificate(_HASH_A, _HASH_B, _HASH_C)
        assert isinstance(cert, Certificate)
        assert cert.theorem == "agentflow_evolve_pinned_inputs"

    def test_classmethod_default_source_label(self) -> None:
        cert = Certificate.from_agentflow_compile(_HASH_A, _HASH_B, _HASH_C)
        assert cert.source == "Certificate.from_agentflow_compile"

    def test_helper_default_source_label(self) -> None:
        """Helper-direct callers get the helper's name, not the classmethod's
        (Roborev #922)."""
        cert = make_agentflow_compile_certificate(_HASH_A, _HASH_B, _HASH_C)
        assert cert.source == "make_agentflow_compile_certificate"

    def test_source_override(self) -> None:
        cert = make_agentflow_compile_certificate(
            _HASH_A, _HASH_B, _HASH_C, source="my_pipeline"
        )
        assert cert.source == "my_pipeline"

    def test_conclusion_keys_on_tuned_agent_hash(self) -> None:
        cert = Certificate.from_agentflow_compile(_HASH_A, _HASH_B, _HASH_C)
        assert _HASH_C in cert.conclusion


class TestVerifyPasses:
    def test_all_well_formed_holds(self) -> None:
        cert = Certificate.from_agentflow_compile(_HASH_A, _HASH_B, _HASH_C)
        v = cert.verify()
        assert v.holds is True
        assert list(v.evidence["missing"]) == []
        assert list(v.evidence["malformed"]) == []
        assert set(v.evidence["present_keys"]) == {
            "graph_hash",
            "traces_hash",
            "tuned_agent_hash",
        }

    def test_minimum_length_8_holds(self) -> None:
        h8 = "abcdef01"
        cert = Certificate.from_agentflow_compile(h8, h8, h8)
        assert cert.verify().holds is True


class TestVerifyFails:
    def test_too_short_hash_fails(self) -> None:
        cert = make_agentflow_compile_certificate("abc", _HASH_B, _HASH_C)
        v = cert.verify()
        assert v.holds is False
        assert "graph_hash" in list(v.evidence["malformed"])

    def test_non_hex_chars_fail(self) -> None:
        cert = make_agentflow_compile_certificate(
            "z" * 16, _HASH_B, _HASH_C
        )
        v = cert.verify()
        assert v.holds is False
        assert "graph_hash" in list(v.evidence["malformed"])

    def test_uppercase_hex_fails(self) -> None:
        cert = make_agentflow_compile_certificate(
            "A" * 16, _HASH_B, _HASH_C
        )
        v = cert.verify()
        assert v.holds is False
        assert "graph_hash" in list(v.evidence["malformed"])

    def test_empty_string_fails(self) -> None:
        cert = make_agentflow_compile_certificate("", _HASH_B, _HASH_C)
        v = cert.verify()
        assert v.holds is False
        assert "graph_hash" in list(v.evidence["malformed"])

    def test_missing_key_fails(self) -> None:
        cert = Certificate.from_theorem(
            theorem="agentflow_evolve_pinned_inputs",
            parameters={
                "graph_hash": _HASH_A,
                # traces_hash deliberately missing
                "tuned_agent_hash": _HASH_C,
            },
            conclusion="manual",
            source="manual",
        )
        v = cert.verify()
        assert v.holds is False
        assert "traces_hash" in list(v.evidence["missing"])


class TestRoundTrip:
    def test_certificate_to_dict_is_jsonable(self) -> None:
        cert = Certificate.from_agentflow_compile(_HASH_A, _HASH_B, _HASH_C)
        d = certificate_to_dict(cert)
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["theorem"] == "agentflow_evolve_pinned_inputs"
        assert round_tripped["parameters"]["tuned_agent_hash"] == _HASH_C
        assert round_tripped["source"] == "Certificate.from_agentflow_compile"


class TestRegistration:
    def test_theorem_is_resolvable(self) -> None:
        cert = Certificate.from_theorem(
            theorem="agentflow_evolve_pinned_inputs",
            parameters={
                "graph_hash": _HASH_A,
                "traces_hash": _HASH_B,
                "tuned_agent_hash": _HASH_C,
            },
            conclusion="resolved-via-from_theorem",
            source="test",
        )
        assert cert.verify().holds is True

    def test_unknown_theorem_still_raises(self) -> None:
        with pytest.raises(KeyError):
            Certificate.from_theorem(
                theorem="agentflow_evolve_pinned_inputs_typo",
                parameters={"graph_hash": _HASH_A},
                conclusion="x",
                source="test",
            )
