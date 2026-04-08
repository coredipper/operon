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
        assert restored.theorem == cert.theorem
        assert dict(restored.parameters) == dict(cert.parameters)
        result = restored.verify()
        assert result.holds is True

    def test_unknown_theorem_raises(self):
        d = {
            "theorem": "unknown_theorem",
            "parameters": {},
            "conclusion": "",
            "source": "",
        }
        try:
            certificate_from_dict(d)
            assert False, "Should raise KeyError"
        except KeyError:
            pass


class TestSwarmsCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_swarms(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled_all_verified(self):
        compiled = organism_to_swarms(_make_organism())
        results = verify_compiled(compiled)
        assert len(results) == len(compiled["certificates"])
        assert all(r.holds for r in results)


class TestRalphCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_ralph(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled_all_verified(self):
        compiled = organism_to_ralph(_make_organism())
        results = verify_compiled(compiled)
        assert len(results) == len(compiled["certificates"])
        assert all(r.holds for r in results)


class TestScionCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_scion(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled_all_verified(self):
        compiled = organism_to_scion(_make_organism())
        results = verify_compiled(compiled)
        assert len(results) == len(compiled["certificates"])
        assert all(r.holds for r in results)


class TestDeerflowCertificates:
    def test_certificates_in_output(self):
        compiled = organism_to_deerflow(_make_organism())
        assert "certificates" in compiled
        assert len(compiled["certificates"]) >= 1

    def test_verify_compiled_all_verified(self):
        compiled = organism_to_deerflow(_make_organism())
        results = verify_compiled(compiled)
        assert len(results) == len(compiled["certificates"])
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

    def test_lazy_import_resolves_missing_theorem(self):
        """Deserializing a theorem not yet in the registry triggers lazy import."""
        from operon_ai.core.certificate import _VERIFY_REGISTRY

        # Remove QS verifier if present (simulate partial registry)
        saved = _VERIFY_REGISTRY.pop("no_false_activation", None)
        try:
            d = {
                "theorem": "no_false_activation",
                "parameters": {"N": 10, "s": 0.15, "h": 5.0, "dt": 1.0, "safety_margin": 2.0},
                "conclusion": "test",
                "source": "test",
            }
            # Should succeed via lazy import
            restored = certificate_from_dict(d)
            assert restored.theorem == "no_false_activation"
            result = restored.verify()
            assert result.holds is True
        finally:
            # Restore registry state
            if saved is not None:
                _VERIFY_REGISTRY["no_false_activation"] = saved

    def test_qs_certificate_round_trip(self):
        """QuorumSensing certificate survives serialize → deserialize."""
        from operon_ai.coordination.quorum_sensing import QuorumSensingBio
        qs = QuorumSensingBio(population_size=10)
        qs.calibrate()
        cert = qs.certify()
        d = certificate_to_dict(cert)
        restored = certificate_from_dict(d)
        assert restored.theorem == "no_false_activation"
        result = restored.verify()
        assert result.holds is True

    def test_mtor_certificate_round_trip(self):
        """MTORScaler certificate survives serialize → deserialize."""
        from operon_ai.state.mtor import MTORScaler
        scaler = MTORScaler(atp_store=ATP_Store(budget=1000))
        cert = scaler.certify()
        d = certificate_to_dict(cert)
        restored = certificate_from_dict(d)
        assert restored.theorem == "no_oscillation"
        result = restored.verify()
        assert result.holds is True
