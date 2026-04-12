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

    def _lazy_import_test(self, theorem, params):
        """Helper: clear registry entry, deserialize, verify lazy resolution."""
        from operon_ai.core.certificate import _VERIFY_REGISTRY

        had_key = theorem in _VERIFY_REGISTRY
        saved = _VERIFY_REGISTRY.pop(theorem, None)
        try:
            d = {"theorem": theorem, "parameters": params, "conclusion": "t", "source": "t"}
            restored = certificate_from_dict(d)
            assert restored.theorem == theorem
            result = restored.verify()
            assert result.holds is True
        finally:
            if had_key and saved is not None:
                _VERIFY_REGISTRY[theorem] = saved
            elif not had_key:
                _VERIFY_REGISTRY.pop(theorem, None)

    def test_lazy_import_qs(self):
        self._lazy_import_test("no_false_activation", {
            "N": 10, "s": 0.15, "h": 5.0, "dt": 1.0, "safety_margin": 2.0,
        })

    def test_lazy_import_mtor(self):
        self._lazy_import_test("no_oscillation", {
            "growth_threshold": 0.3, "conservation_threshold": 0.7,
            "autophagy_threshold": 0.9, "hysteresis": 0.05,
        })

    def test_lazy_import_atp(self):
        self._lazy_import_test("priority_gating", {
            "budget": 1000, "priority_threshold_starving": 5,
            "priority_threshold_dormant": 10,
        })

    def test_missing_target_module_returns_keyerror(self):
        """ModuleNotFoundError for the target module → KeyError."""
        from unittest.mock import patch
        from operon_ai.core.certificate import _VERIFY_REGISTRY, _THEOREM_FN_PATHS

        # Temporarily add a fake theorem pointing to a nonexistent module
        _THEOREM_FN_PATHS["_test_missing"] = ("nonexistent.module", "fn")
        try:
            d = {"theorem": "_test_missing", "parameters": {}, "conclusion": "", "source": ""}
            try:
                certificate_from_dict(d)
                assert False, "Should raise KeyError"
            except KeyError:
                pass
        finally:
            del _THEOREM_FN_PATHS["_test_missing"]
            _VERIFY_REGISTRY.pop("_test_missing", None)

    def test_transitive_import_error_propagates(self):
        """ModuleNotFoundError from a dependency re-raises, not swallowed."""
        from unittest.mock import patch
        from operon_ai.core.certificate import _VERIFY_REGISTRY

        saved = _VERIFY_REGISTRY.pop("no_false_activation", None)
        try:
            def bad_import(name):
                raise ModuleNotFoundError(name="some_dependency")

            with patch("importlib.import_module", side_effect=bad_import):
                d = {
                    "theorem": "no_false_activation",
                    "parameters": {"N": 10, "s": 0.15, "h": 5.0, "dt": 1.0, "safety_margin": 2.0},
                    "conclusion": "", "source": "",
                }
                try:
                    certificate_from_dict(d)
                    assert False, "Should propagate ModuleNotFoundError"
                except ModuleNotFoundError as e:
                    assert e.name == "some_dependency"
        finally:
            if saved is not None:
                _VERIFY_REGISTRY["no_false_activation"] = saved

    def test_near_prefix_import_error_propagates(self):
        """ModuleNotFoundError with near-prefix name (no dot boundary) re-raises."""
        from unittest.mock import patch
        from operon_ai.core.certificate import _VERIFY_REGISTRY

        saved = _VERIFY_REGISTRY.pop("no_oscillation", None)
        try:
            # "operon_ai.state.mto" is a prefix of "operon_ai.state.mtor"
            # but not at a dot boundary — should NOT be treated as "target missing"
            def bad_import(name):
                raise ModuleNotFoundError(name="operon_ai.state.mto")

            with patch("importlib.import_module", side_effect=bad_import):
                d = {
                    "theorem": "no_oscillation",
                    "parameters": {"growth_threshold": 0.3, "conservation_threshold": 0.7,
                                   "autophagy_threshold": 0.9, "hysteresis": 0.05},
                    "conclusion": "", "source": "",
                }
                try:
                    certificate_from_dict(d)
                    assert False, "Should propagate ModuleNotFoundError"
                except ModuleNotFoundError as e:
                    assert e.name == "operon_ai.state.mto"
        finally:
            if saved is not None:
                _VERIFY_REGISTRY["no_oscillation"] = saved

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


class TestCertificatePreservationMeasurement:
    """Quantitative measurement of certificate preservation across all 4 compilers.

    Compiles organisms with multiple certificates, counts preserved and
    verified fractions per compiler.
    """

    def _make_multi_cert_organism(self):
        """Build organism with ATP + QuorumSensing + mTOR certificates.

        Attaches QS and mTOR to the organism's components list so that
        collect_certificates() includes all three certificate sources.
        """
        from operon_ai.coordination.quorum_sensing import QuorumSensingBio
        from operon_ai.state.mtor import MTORScaler

        provider = MockProvider()
        nucleus = Nucleus(provider=provider)
        budget = ATP_Store(budget=1000)

        qs = QuorumSensingBio(population_size=10)
        qs.calibrate()
        mtor = MTORScaler(atp_store=budget)

        org = skill_organism(
            stages=[
                SkillStage(name="reader", role="reader", instructions="Read"),
                SkillStage(name="analyzer", role="analyzer", instructions="Analyze"),
                SkillStage(name="writer", role="writer", instructions="Write"),
            ],
            fast_nucleus=nucleus,
            deep_nucleus=nucleus,
            budget=budget,
        )

        # Attach certifiable components so collect_certificates() finds them.
        # QS and mTOR have certify() but don't implement the full
        # SkillRuntimeComponent protocol — that's fine, collect_certificates()
        # only checks for certify().
        org.components = (*org.components, qs, mtor)  # type: ignore[assignment]

        return org, []

    @staticmethod
    def _cert_identity(d: dict) -> tuple:
        """Normalize a certificate dict to a hashable identity tuple."""
        params = d.get("parameters", {})
        # Sort nested dict to ensure stable comparison
        param_key = tuple(sorted(params.items())) if isinstance(params, dict) else ()
        return (d.get("theorem"), param_key, d.get("source"))

    def _measure_preservation(self, compiler_fn, org, extra_certs, **kwargs):
        """Compile, count preserved and verified certificates."""
        compiled = compiler_fn(org, **kwargs)
        compiled_certs = compiled.get("certificates", [])

        # Source certificates — serialize to dicts for identity comparison
        source_certs = org.collect_certificates() + extra_certs
        source_dicts = [certificate_to_dict(c) for c in source_certs]
        source_count = len(source_certs)

        # Full identity preservation (theorem + parameters + source)
        source_ids = {self._cert_identity(d) for d in source_dicts}
        compiled_ids = {self._cert_identity(d) for d in compiled_certs}
        preserved_ids = source_ids & compiled_ids

        # Theorem-name preservation (weaker check, kept for diagnostics)
        source_theorems = {c.theorem for c in source_certs}
        compiled_theorems = {c.get("theorem") for c in compiled_certs}
        preserved_theorems = source_theorems & compiled_theorems

        # Count verified (still hold after serialization)
        verified = 0
        for cd in compiled_certs:
            try:
                restored = certificate_from_dict(cd)
                result = restored.verify()
                if result.holds:
                    verified += 1
            except Exception:
                pass

        return {
            "source_count": source_count,
            "compiled_count": len(compiled_certs),
            "preserved_theorems": len(preserved_theorems),
            "preserved_identities": len(preserved_ids),
            "verified_count": verified,
        }

    def test_deerflow_preservation(self):
        org, extra = self._make_multi_cert_organism()
        result = self._measure_preservation(organism_to_deerflow, org, extra)
        assert result["compiled_count"] >= 3, f"Expected >=3 certs, got {result['compiled_count']}"
        assert result["preserved_identities"] == result["source_count"], "Not all source certificates survived (full identity check)"
        assert result["verified_count"] == result["compiled_count"]

    def test_swarms_preservation(self):
        org, extra = self._make_multi_cert_organism()
        result = self._measure_preservation(organism_to_swarms, org, extra)
        assert result["compiled_count"] >= 3, f"Expected >=3 certs, got {result['compiled_count']}"
        assert result["preserved_identities"] == result["source_count"], "Not all source certificates survived (full identity check)"
        assert result["verified_count"] == result["compiled_count"]

    def test_ralph_preservation(self):
        org, extra = self._make_multi_cert_organism()
        result = self._measure_preservation(organism_to_ralph, org, extra)
        assert result["compiled_count"] >= 3, f"Expected >=3 certs, got {result['compiled_count']}"
        assert result["preserved_identities"] == result["source_count"], "Not all source certificates survived (full identity check)"
        assert result["verified_count"] == result["compiled_count"]

    def test_scion_preservation(self):
        org, extra = self._make_multi_cert_organism()
        result = self._measure_preservation(organism_to_scion, org, extra)
        assert result["compiled_count"] >= 3, f"Expected >=3 certs, got {result['compiled_count']}"
        assert result["preserved_identities"] == result["source_count"], "Not all source certificates survived (full identity check)"
        assert result["verified_count"] == result["compiled_count"]

    def test_all_compilers_100_percent_verification(self):
        """All 4 compilers must achieve 100% certificate verification rate."""
        org, extra = self._make_multi_cert_organism()
        compilers = {
            "deerflow": organism_to_deerflow,
            "swarms": organism_to_swarms,
            "ralph": organism_to_ralph,
            "scion": organism_to_scion,
        }
        for name, compiler_fn in compilers.items():
            result = self._measure_preservation(compiler_fn, org, extra)
            assert result["preserved_identities"] == result["source_count"], (
                f"{name}: {result['preserved_identities']}/{result['source_count']} "
                f"certificate identities preserved (expected 100%)"
            )
            assert result["verified_count"] == result["compiled_count"], (
                f"{name}: {result['verified_count']}/{result['compiled_count']} "
                f"certificates verified (expected 100%)"
            )
