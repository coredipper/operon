"""Tests for the A2A certificate codec.

Covers:

- Round-trip ``cert -> part -> cert`` preserves ``verify()`` (Theorem 1)
- Unknown theorem handling: strict raises, safe returns None (Theorem 4)
- Part schema detection (``is_certificate_part``)
- AgentCard skill entry shape
- Deep-thaw of nested verification evidence (JSON serializability)
- Type-validation at the codec boundary (malformed payloads)
"""

from __future__ import annotations

import json

import pytest

from operon_ai.convergence.a2a_certificate import (
    A2A_CERTIFICATE_MIME_TYPE,
    A2A_CERTIFICATE_SCHEMA,
    InvalidCertificatePartError,
    UnknownTheoremError,
    agent_card_skill_for_theorem,
    certificate_from_a2a_part,
    certificate_to_a2a_part,
    is_certificate_part,
    safe_certificate_from_a2a_part,
)
from operon_ai.core.certificate import Certificate, register_verify_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def holding_cert() -> Certificate:
    return Certificate.from_theorem(
        theorem="behavioral_quality",
        parameters={"scores": [0.9, 0.95, 0.92], "threshold": 0.7},
        conclusion="Mean quality above threshold on a 3-sample evidence snapshot",
        source="test_a2a_certificate",
    )


@pytest.fixture
def failing_cert() -> Certificate:
    return Certificate.from_theorem(
        theorem="behavioral_quality",
        parameters={"scores": [0.1, 0.2, 0.3], "threshold": 0.7},
        conclusion="Mean quality below threshold on a 3-sample evidence snapshot",
        source="test_a2a_certificate",
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class TestCertificateToA2APart:
    """Tests for certificate_to_a2a_part."""

    def test_produces_data_part(self, holding_cert: Certificate) -> None:
        part = certificate_to_a2a_part(holding_cert)
        assert part["kind"] == "data"
        assert isinstance(part["data"], dict)

    def test_payload_carries_schema_marker(self, holding_cert: Certificate) -> None:
        part = certificate_to_a2a_part(holding_cert)
        assert part["data"]["schema"] == A2A_CERTIFICATE_SCHEMA
        assert part["metadata"]["schema"] == A2A_CERTIFICATE_SCHEMA
        assert part["metadata"]["mimeType"] == A2A_CERTIFICATE_MIME_TYPE

    def test_payload_carries_core_fields(self, holding_cert: Certificate) -> None:
        part = certificate_to_a2a_part(holding_cert)
        data = part["data"]
        assert data["theorem"] == "behavioral_quality"
        assert data["conclusion"] == holding_cert.conclusion
        assert data["source"] == holding_cert.source
        assert "parameters" in data

    def test_verify_true_embeds_verification(self, holding_cert: Certificate) -> None:
        part = certificate_to_a2a_part(holding_cert, verify=True)
        verification = part["data"]["verification"]
        assert verification is not None
        assert verification["holds"] is True
        assert "evidence" in verification

    def test_verify_false_leaves_verification_null(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert, verify=False)
        assert part["data"]["verification"] is None

    def test_verifier_version_stored_in_metadata(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert, verifier_version="0.36.3")
        assert part["metadata"]["verifierVersion"] == "0.36.3"

    def test_failing_cert_marks_holds_false(self, failing_cert: Certificate) -> None:
        part = certificate_to_a2a_part(failing_cert)
        assert part["data"]["verification"]["holds"] is False

    def test_nested_evidence_json_serializes(self) -> None:
        """Regression: nested evidence must deep-thaw to plain Python types.

        ``Certificate.verify`` freezes evidence recursively (MappingProxyType,
        tuples, frozensets).  A shallow ``dict(...)`` copy leaves the nested
        values frozen, which breaks ``json.dumps``.  Ensure the encoded Part
        is end-to-end JSON-serializable even for nested evidence.
        """
        def _nested_evidence_verify(params):
            del params
            return True, {
                "nested": {"inner_tuple": (1, 2, 3), "inner_set": frozenset({"a"})},
                "list": [{"k": 1}, {"k": 2}],
            }

        register_verify_fn("nested_evidence_test_theorem", _nested_evidence_verify)
        cert = Certificate.from_theorem(
            theorem="nested_evidence_test_theorem",
            parameters={"dummy": 1},
            conclusion="nested evidence regression",
            source="test_a2a_certificate",
        )
        part = certificate_to_a2a_part(cert)
        # The critical assertion: the entire Part round-trips through JSON.
        encoded = json.dumps(part)
        decoded = json.loads(encoded)
        evidence = decoded["data"]["verification"]["evidence"]
        # Nested container types are plain Python after thawing.
        assert isinstance(evidence["nested"], dict)
        assert evidence["nested"]["inner_tuple"] == [1, 2, 3]
        assert evidence["nested"]["inner_set"] == ["a"]
        assert evidence["list"] == [{"k": 1}, {"k": 2}]


# ---------------------------------------------------------------------------
# Decoding / round-trip (Theorem 1)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for cert -> part -> cert preservation."""

    def test_round_trip_preserves_theorem_name(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert)
        decoded = certificate_from_a2a_part(part)
        assert decoded.theorem == holding_cert.theorem

    def test_round_trip_preserves_verify_result(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert)
        decoded = certificate_from_a2a_part(part)
        assert decoded.verify().holds == holding_cert.verify().holds

    def test_round_trip_preserves_parameters_equivalence(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert)
        decoded = certificate_from_a2a_part(part)
        assert dict(decoded.parameters) == dict(holding_cert.parameters)

    def test_round_trip_preserves_conclusion_and_source(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert)
        decoded = certificate_from_a2a_part(part)
        assert decoded.conclusion == holding_cert.conclusion
        assert decoded.source == holding_cert.source


# ---------------------------------------------------------------------------
# Graceful degradation (Theorem 4)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for unknown-theorem handling."""

    def test_unknown_theorem_strict_raises(self) -> None:
        part = {
            "kind": "data",
            "data": {
                "schema": A2A_CERTIFICATE_SCHEMA,
                "theorem": "never_registered_in_a_real_run",
                "parameters": {},
                "conclusion": "synthetic",
                "source": "synthetic",
            },
            "metadata": {"schema": A2A_CERTIFICATE_SCHEMA},
        }
        with pytest.raises(UnknownTheoremError):
            certificate_from_a2a_part(part)

    def test_unknown_theorem_safe_returns_none(self) -> None:
        part = {
            "kind": "data",
            "data": {
                "schema": A2A_CERTIFICATE_SCHEMA,
                "theorem": "never_registered_in_a_real_run",
                "parameters": {},
                "conclusion": "synthetic",
                "source": "synthetic",
            },
            "metadata": {"schema": A2A_CERTIFICATE_SCHEMA},
        }
        assert safe_certificate_from_a2a_part(part) is None

    def test_unknown_theorem_error_is_keyerror_subclass(self) -> None:
        """UnknownTheoremError preserves KeyError for downstream handlers."""
        assert issubclass(UnknownTheoremError, KeyError)


# ---------------------------------------------------------------------------
# is_certificate_part
# ---------------------------------------------------------------------------


class TestIsCertificatePart:
    """Tests for Part-schema detection."""

    def test_encoded_certificate_is_detected(
        self, holding_cert: Certificate
    ) -> None:
        part = certificate_to_a2a_part(holding_cert)
        assert is_certificate_part(part) is True

    def test_text_part_is_not_a_certificate(self) -> None:
        assert is_certificate_part({"kind": "text", "text": "hello"}) is False

    def test_unrelated_data_part_is_not_a_certificate(self) -> None:
        assert (
            is_certificate_part(
                {"kind": "data", "data": {"schema": "other.thing.v1"}}
            )
            is False
        )

    def test_non_dict_input_is_not_a_certificate(self) -> None:
        assert is_certificate_part("not a part") is False
        assert is_certificate_part(None) is False
        assert is_certificate_part([{"kind": "data"}]) is False

    def test_metadata_only_schema_marker_is_detected(self) -> None:
        """Support receivers that strip payload schema but preserve metadata."""
        part = {
            "kind": "data",
            "data": {
                "theorem": "behavioral_quality",
                "parameters": {},
                "conclusion": "",
                "source": "",
            },
            "metadata": {"schema": A2A_CERTIFICATE_SCHEMA},
        }
        assert is_certificate_part(part) is True


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------


class TestMalformedPart:
    """Tests for error reporting on malformed certificate Parts."""

    def test_not_a_cert_part_raises(self) -> None:
        with pytest.raises(InvalidCertificatePartError):
            certificate_from_a2a_part(
                {"kind": "text", "text": "not a cert"}
            )

    def test_missing_theorem_field_raises(self) -> None:
        part = {
            "kind": "data",
            "data": {
                "schema": A2A_CERTIFICATE_SCHEMA,
                # theorem intentionally omitted
                "parameters": {},
                "conclusion": "",
                "source": "",
            },
        }
        with pytest.raises(InvalidCertificatePartError, match="theorem"):
            certificate_from_a2a_part(part)

    @pytest.mark.parametrize(
        "bad_field, bad_value",
        [
            ("theorem", 42),
            ("parameters", "not a dict"),
            ("conclusion", 123),
            ("source", ["list", "not", "str"]),
        ],
    )
    def test_wrong_type_field_raises_codec_error(
        self, bad_field: str, bad_value: object
    ) -> None:
        """Malformed payloads surface as InvalidCertificatePartError at the boundary."""
        payload = {
            "schema": A2A_CERTIFICATE_SCHEMA,
            "theorem": "behavioral_quality",
            "parameters": {"scores": [0.9], "threshold": 0.5},
            "conclusion": "ok",
            "source": "test",
        }
        payload[bad_field] = bad_value  # type: ignore[assignment]
        part = {"kind": "data", "data": payload}
        with pytest.raises(InvalidCertificatePartError, match=bad_field):
            certificate_from_a2a_part(part)


# ---------------------------------------------------------------------------
# AgentCard skill generation
# ---------------------------------------------------------------------------


class TestAgentCardSkill:
    """Tests for agent_card_skill_for_theorem."""

    def test_emit_role_declares_output_mode_only(self) -> None:
        skill = agent_card_skill_for_theorem("behavioral_quality", role="emit")
        assert skill["outputModes"] == [A2A_CERTIFICATE_MIME_TYPE]
        assert skill["inputModes"] == []
        assert "emit" in skill["id"]

    def test_verify_role_declares_input_mode_only(self) -> None:
        skill = agent_card_skill_for_theorem("behavioral_quality", role="verify")
        assert skill["inputModes"] == [A2A_CERTIFICATE_MIME_TYPE]
        assert skill["outputModes"] == []

    def test_both_role_declares_both_modes(self) -> None:
        skill = agent_card_skill_for_theorem("behavioral_quality", role="both")
        assert skill["inputModes"] == [A2A_CERTIFICATE_MIME_TYPE]
        assert skill["outputModes"] == [A2A_CERTIFICATE_MIME_TYPE]

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="emit/verify/both"):
            agent_card_skill_for_theorem("behavioral_quality", role="invalid")

    def test_skill_tags_include_theorem_name(self) -> None:
        skill = agent_card_skill_for_theorem("behavioral_quality", role="emit")
        assert "behavioral_quality" in skill["tags"]
        assert "operon" in skill["tags"]
        assert "certificate" in skill["tags"]

    def test_custom_description_is_preserved(self) -> None:
        skill = agent_card_skill_for_theorem(
            "behavioral_quality",
            role="emit",
            description="Custom description for my service.",
        )
        assert skill["description"] == "Custom description for my service."
