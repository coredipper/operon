"""Tests for certificate preservation through convergence compilers."""

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.swarms_compiler import organism_to_swarms
from operon_ai.convergence.ralph_compiler import organism_to_ralph
from operon_ai.convergence.scion_compiler import organism_to_scion
from operon_ai.convergence.deerflow_compiler import organism_to_deerflow
from operon_ai.core.certificate import (
    certificate_from_dict,
    certificate_to_dict,
    verify_compiled,
)
from operon_ai.providers.mock import MockProvider


def _make_organism(budget: int = 1000):
    provider = MockProvider()
    nucleus = Nucleus(provider=provider)
    return skill_organism(
        stages=[
            SkillStage(name="reader", role="reader", instructions="Read input"),
            SkillStage(name="writer", role="writer", instructions="Write output"),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
        budget=ATP_Store(budget=budget),
    )


class TestCollectCertificates:
    def test_budget_certificate_collected(self):
        org = _make_organism()
        certs = org.collect_certificates()
        assert len(certs) >= 1
        assert any(c.theorem == "priority_gating" for c in certs)

    def test_zero_budget_still_collected(self):
        org = _make_organism(budget=0)
        certs = org.collect_certificates()
        assert len(certs) >= 1
        # Certificate exists but verify will fail
        result = certs[0].verify()
        assert result.holds is False


class TestCertificateSerialization:
    def test_round_trip(self):
        org = _make_organism()
        cert = org.collect_certificates()[0]
        d = certificate_to_dict(cert)
        restored = certificate_from_dict(d)
        assert restored is not None
        assert restored.theorem == cert.theorem
        assert dict(restored.parameters) == dict(cert.parameters)
        result = restored.verify()
        assert result.holds is True

    def test_unknown_theorem_returns_none(self):
        d = {
            "theorem": "unknown_theorem",
            "parameters": {},
            "conclusion": "",
            "source": "",
        }
        assert certificate_from_dict(d) is None


class TestSwarmsCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_swarms(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled(self):
        compiled = organism_to_swarms(_make_organism())
        results = verify_compiled(compiled)
        assert len(results) >= 1
        assert all(r.holds for r in results)


class TestRalphCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_ralph(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled(self):
        compiled = organism_to_ralph(_make_organism())
        results = verify_compiled(compiled)
        assert all(r.holds for r in results)


class TestScionCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_scion(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled(self):
        compiled = organism_to_scion(_make_organism())
        results = verify_compiled(compiled)
        assert all(r.holds for r in results)


class TestDeerflowCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_deerflow(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled(self):
        compiled = organism_to_deerflow(_make_organism())
        results = verify_compiled(compiled)
        assert all(r.holds for r in results)


class TestVerifyCompiledEdgeCases:
    def test_no_certificates_key(self):
        assert verify_compiled({}) == []

    def test_empty_certificates(self):
        assert verify_compiled({"certificates": []}) == []

    def test_failing_certificate_survives_compilation(self):
        compiled = organism_to_swarms(_make_organism(budget=0))
        results = verify_compiled(compiled)
        assert len(results) >= 1
        assert any(not r.holds for r in results)
