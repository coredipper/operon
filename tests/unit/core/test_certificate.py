"""Tests for the categorical certificate framework."""

from operon_ai.core.certificate import Certificate, CertificateVerification
from operon_ai.coordination.quorum_sensing import QuorumSensingBio
from operon_ai.state.metabolism import ATP_Store
from operon_ai.state.mtor import MTORScaler


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_verify_returns_verification(self):
        cert = Certificate(
            theorem="test",
            parameters={"x": 1},
            conclusion="x is positive",
            source="test",
            _verify_fn=lambda p: (p["x"] > 0, {"x": p["x"]}),
        )
        result = cert.verify()
        assert isinstance(result, CertificateVerification)
        assert result.holds is True
        assert result.evidence == {"x": 1}
        assert result.certificate is cert

    def test_verify_failing_certificate(self):
        cert = Certificate(
            theorem="test",
            parameters={"x": -1},
            conclusion="x is positive",
            source="test",
            _verify_fn=lambda p: (p["x"] > 0, {"x": p["x"]}),
        )
        result = cert.verify()
        assert result.holds is False

    def test_frozen(self):
        cert = Certificate(
            theorem="test",
            parameters={},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        try:
            cert.theorem = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_parameters_immutable(self):
        orig = {"x": 1}
        cert = Certificate(
            theorem="test",
            parameters=orig,
            conclusion="",
            source="",
            _verify_fn=lambda p: (p["x"] > 0, dict(p)),
        )
        # Mutating the original dict doesn't affect the certificate
        orig["x"] = -999
        result = cert.verify()
        assert result.holds is True
        # Can't mutate parameters directly
        try:
            cert.parameters["x"] = -1
            assert False, "Should be immutable"
        except TypeError:
            pass

    def test_evidence_immutable(self):
        cert = Certificate(
            theorem="test",
            parameters={"x": 1},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {"y": 2}),
        )
        result = cert.verify()
        try:
            result.evidence["y"] = -1
            assert False, "Should be immutable"
        except TypeError:
            pass

    def test_nested_parameters_immutable(self):
        cert = Certificate(
            theorem="test",
            parameters={"config": {"threshold": 0.5}, "items": [1, 2]},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        # Nested dict is frozen
        try:
            cert.parameters["config"]["threshold"] = 999
            assert False, "Nested dict should be immutable"
        except TypeError:
            pass
        # List converted to tuple
        assert isinstance(cert.parameters["items"], tuple)

    def test_tuple_contents_frozen(self):
        cert = Certificate(
            theorem="test",
            parameters={"pair": ([1, 2], {"a": 3})},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {"out": ([4],)}),
        )
        # List inside tuple is frozen to tuple
        assert isinstance(cert.parameters["pair"][0], tuple)
        # Dict inside tuple is frozen
        try:
            cert.parameters["pair"][1]["a"] = 999
            assert False, "Dict inside tuple should be immutable"
        except TypeError:
            pass
        # Evidence tuple contents also frozen
        result = cert.verify()
        assert isinstance(result.evidence["out"][0], tuple)

    def test_single_field_namedtuple(self):
        from collections import namedtuple
        Single = namedtuple("Single", ["x"])
        cert = Certificate(
            theorem="test",
            parameters={"data": Single(x={"k": "v"})},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        assert type(cert.parameters["data"]).__name__ == "Single"
        assert cert.parameters["data"].x is not None
        try:
            cert.parameters["data"].x["k"] = "mutated"
            assert False, "Single-field namedtuple should be frozen"
        except TypeError:
            pass

    def test_namedtuple_preserved_and_frozen(self):
        from collections import namedtuple
        Pair = namedtuple("Pair", ["items", "meta"])
        cert = Certificate(
            theorem="test",
            parameters={"data": Pair(items=[1, 2], meta={"k": "v"})},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        # Type preserved
        assert type(cert.parameters["data"]).__name__ == "Pair"
        assert hasattr(cert.parameters["data"], "items")
        # Contents frozen
        assert isinstance(cert.parameters["data"].items, tuple)
        try:
            cert.parameters["data"].meta["k"] = "mutated"
            assert False, "Namedtuple dict field should be immutable"
        except TypeError:
            pass

    def test_custom_tuple_subclass_preserved(self):
        class MyTuple(tuple):
            """Tuple subclass with single-iterable constructor."""
            def __new__(cls, iterable):
                return super().__new__(cls, iterable)

        cert = Certificate(
            theorem="test",
            parameters={"data": MyTuple([[1, 2], {"k": "v"}])},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        # Type preserved
        assert type(cert.parameters["data"]).__name__ == "MyTuple"
        # Contents frozen
        assert isinstance(cert.parameters["data"][0], tuple)
        try:
            cert.parameters["data"][1]["k"] = "mutated"
            assert False, "Dict inside custom tuple should be immutable"
        except TypeError:
            pass

    def test_single_element_tuple_subclass(self):
        class MyTuple(tuple):
            def __new__(cls, iterable):
                return super().__new__(cls, iterable)

        cert = Certificate(
            theorem="test",
            parameters={"data": MyTuple([{"k": "v"}])},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        assert len(cert.parameters["data"]) == 1
        assert type(cert.parameters["data"]).__name__ == "MyTuple"
        try:
            cert.parameters["data"][0]["k"] = "mutated"
            assert False, "Should be immutable"
        except TypeError:
            pass

    def test_positional_tuple_subclass_preserved(self):
        class Pair(tuple):
            def __new__(cls, left, right):
                return super().__new__(cls, (left, right))

        cert = Certificate(
            theorem="test",
            parameters={"data": Pair({"a": 1}, [2, 3])},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        assert type(cert.parameters["data"]).__name__ == "Pair"
        assert len(cert.parameters["data"]) == 2
        # Contents frozen
        try:
            cert.parameters["data"][0]["a"] = 999
            assert False, "Should be immutable"
        except TypeError:
            pass
        assert isinstance(cert.parameters["data"][1], tuple)

    def test_single_field_positional_tuple_subclass(self):
        class Box(tuple):
            def __new__(cls, item):
                return super().__new__(cls, (item,))

        cert = Certificate(
            theorem="test",
            parameters={"data": Box({"k": "v"})},
            conclusion="",
            source="",
            _verify_fn=lambda p: (True, {}),
        )
        # Type and shape preserved
        assert type(cert.parameters["data"]).__name__ == "Box"
        assert len(cert.parameters["data"]) == 1
        # Content frozen
        try:
            cert.parameters["data"][0]["k"] = "mutated"
            assert False, "Should be immutable"
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# QuorumSensingBio certificate
# ---------------------------------------------------------------------------


class TestQSCertificate:
    def test_certify_after_calibrate(self):
        qs = QuorumSensingBio(population_size=10)
        qs.calibrate()
        cert = qs.certify()
        assert cert.theorem == "no_false_activation"
        assert cert.source == "QuorumSensingBio.calibrate"

    def test_certify_without_calibrate_raises(self):
        qs = QuorumSensingBio(population_size=10)
        try:
            qs.certify()
            assert False, "Should raise"
        except ValueError:
            pass

    def test_verify_holds(self):
        qs = QuorumSensingBio(population_size=10)
        qs.calibrate()
        result = qs.certify().verify()
        assert result.holds is True
        assert result.evidence["ratio"] < 1.0

    def test_verify_holds_across_population_sizes(self):
        for n in [5, 10, 20, 50, 100]:
            qs = QuorumSensingBio(population_size=n)
            qs.calibrate()
            result = qs.certify().verify()
            assert result.holds is True, f"Failed for N={n}"

    def test_ratio_equals_inverse_safety_margin(self):
        qs = QuorumSensingBio(population_size=10, safety_margin=4.0)
        qs.calibrate()
        result = qs.certify().verify()
        assert abs(result.evidence["ratio"] - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# MTORScaler certificate
# ---------------------------------------------------------------------------


class TestMTORCertificate:
    def test_certify(self):
        atp = ATP_Store(budget=1000)
        scaler = MTORScaler(atp_store=atp)
        cert = scaler.certify()
        assert cert.theorem == "no_oscillation"

    def test_verify_default_thresholds(self):
        atp = ATP_Store(budget=1000)
        scaler = MTORScaler(atp_store=atp)
        result = scaler.certify().verify()
        assert result.holds is True
        assert result.evidence["gap_growth_conservation"] > 0
        assert result.evidence["gap_conservation_autophagy"] > 0

    def test_verify_fails_with_overlapping_bands(self):
        atp = ATP_Store(budget=1000)
        scaler = MTORScaler(
            atp_store=atp,
            growth_threshold=0.3,
            conservation_threshold=0.35,
            autophagy_threshold=0.9,
            hysteresis=0.1,
        )
        result = scaler.certify().verify()
        assert result.holds is False

    def test_verify_fails_with_zero_hysteresis(self):
        atp = ATP_Store(budget=1000)
        scaler = MTORScaler(atp_store=atp, hysteresis=0.0)
        result = scaler.certify().verify()
        assert result.holds is False


# ---------------------------------------------------------------------------
# ATP_Store certificate
# ---------------------------------------------------------------------------


class TestATPCertificate:
    def test_certify(self):
        atp = ATP_Store(budget=1000)
        cert = atp.certify()
        assert cert.theorem == "priority_gating"

    def test_verify_holds(self):
        atp = ATP_Store(budget=1000)
        result = atp.certify().verify()
        assert result.holds is True

    def test_verify_fails_with_zero_budget(self):
        atp = ATP_Store(budget=0)
        result = atp.certify().verify()
        assert result.holds is False


# ---------------------------------------------------------------------------
# Behavioral certificates
# ---------------------------------------------------------------------------


class TestBehavioralQuality:
    def test_verify_holds(self):
        from operon_ai.core.certificate import _verify_behavioral_quality

        cert = Certificate(
            theorem="behavioral_quality",
            parameters={"scores": [0.9, 0.85, 0.95], "threshold": 0.8},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_quality,
        )
        result = cert.verify()
        assert result.holds is True
        assert result.evidence["mean"] == 0.9
        assert result.evidence["min"] == 0.85
        assert result.evidence["n"] == 3

    def test_verify_fails_below_threshold(self):
        from operon_ai.core.certificate import _verify_behavioral_quality

        cert = Certificate(
            theorem="behavioral_quality",
            parameters={"scores": [0.5, 0.4, 0.6], "threshold": 0.8},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_quality,
        )
        result = cert.verify()
        assert result.holds is False

    def test_empty_scores_fails(self):
        from operon_ai.core.certificate import _verify_behavioral_quality

        cert = Certificate(
            theorem="behavioral_quality",
            parameters={"scores": [], "threshold": 0.8},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_quality,
        )
        result = cert.verify()
        assert result.holds is False


class TestBehavioralStability:
    def test_verify_holds(self):
        from operon_ai.core.certificate import _verify_behavioral_stability

        cert = Certificate(
            theorem="behavioral_stability",
            parameters={
                "signal_values": [0.1, 0.2, 0.15],
                "threshold": 0.5,
                "category": "epistemic",
            },
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability,
        )
        result = cert.verify()
        assert result.holds is True
        assert result.evidence["mean"] == 0.15
        assert result.evidence["max"] == 0.2

    def test_verify_fails_above_threshold(self):
        from operon_ai.core.certificate import _verify_behavioral_stability

        cert = Certificate(
            theorem="behavioral_stability",
            parameters={
                "signal_values": [0.8, 0.9, 0.7],
                "threshold": 0.5,
                "category": "epistemic",
            },
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability,
        )
        result = cert.verify()
        assert result.holds is False


class TestBehavioralStabilityWindowed:
    """Sibling of TestBehavioralStability for rolling-window detectors.

    The windowed verifier stores one mean per violating rolling window
    (not a flattened history) and checks ``max(signal_values) <= threshold``.
    Used by ``operon-langgraph-gates`` and ``operon-openhands-gates`` to
    keep cert replay faithful to per-window detection semantics.
    """

    def test_verify_holds_when_all_windows_within_bound(self):
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        cert = Certificate(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.1, 0.2, 0.15), "threshold": 0.5},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability_windowed,
        )
        result = cert.verify()
        assert result.holds is True
        assert result.evidence["max"] == 0.2
        assert result.evidence["mean"] == 0.15
        assert result.evidence["n"] == 3

    def test_verify_fails_when_any_window_above_threshold(self):
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        cert = Certificate(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.3, 0.6, 0.4), "threshold": 0.5},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability_windowed,
        )
        result = cert.verify()
        assert result.holds is False
        assert result.evidence["max"] == 0.6

    def test_verify_treats_threshold_equality_as_stable(self):
        """Boundary: ``<=`` mirrors detection's strict ``<``. When a
        window mean is exactly at the stability threshold, the detector
        treats the complementary integral as stable (``integral >= τ``),
        so the verifier must agree."""
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        for window_mean, expected_holds in [
            (0.799, True),   # below threshold — stable
            (0.800, True),   # exactly at threshold — stable (inclusive)
            (0.801, False),  # above threshold — unstable
        ]:
            cert = Certificate(
                theorem="behavioral_stability_windowed",
                parameters={"signal_values": (window_mean,), "threshold": 0.8},
                conclusion="test",
                source="test",
                _verify_fn=_verify_behavioral_stability_windowed,
            )
            assert cert.verify().holds is expected_holds, (
                f"expected holds={expected_holds} for window_mean={window_mean}"
            )

    def test_verify_rejects_overlapping_window_flat_mean_blind_spot(self):
        """Overlapping rolling windows weight interior samples more than a
        flat mean does. With ``window=2, critical_duration=2`` and
        severities ``[0.61, 1.0, 0.61]``, both window means are ``0.805``
        (detection fires against stability threshold ``0.8``), but a
        flat mean over the union is only ``0.74`` and would incorrectly
        say stability held. The windowed verifier takes per-window means
        directly, so it returns holds=False as expected.
        """
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        cert = Certificate(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.805, 0.805), "threshold": 0.8},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability_windowed,
        )
        result = cert.verify()
        assert result.holds is False
        assert result.evidence["max"] == 0.805

    def test_verify_rejects_empty_evidence_as_malformed(self):
        """Empty ``signal_values`` is not vacuous stability — emitters
        that can legitimately produce a windowed cert always carry at
        least one violating window. Treat empty as malformed."""
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        cert = Certificate(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (), "threshold": 0.8},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability_windowed,
        )
        result = cert.verify()
        assert result.holds is False
        assert result.evidence["reason"] == "empty_evidence"
        assert result.evidence["n"] == 0

    def test_theorem_resolves_without_prior_import(self):
        """The registry entry must let deserialized certs round-trip
        without any sibling package being imported. This is the core
        reason for upstreaming the verifier — sibling packages no longer
        need to call ``register_verify_fn`` as an import-time side effect.
        """
        from operon_ai.core.certificate import (
            _resolve_verify_fn,
            _verify_behavioral_stability_windowed,
        )

        resolved = _resolve_verify_fn("behavioral_stability_windowed")
        assert resolved is _verify_behavioral_stability_windowed


class TestPublicTheoremResolution:
    """Public API surface for theorem resolution (since 0.36.1).

    ``resolve_verify_fn`` and ``Certificate.from_theorem`` let downstream
    packages emit + verify certificates without coupling to any
    underscore-prefixed internal symbol. Reviewer-requested; prevents
    a class of private-API-coupling findings in sibling packages.
    """

    def test_resolve_verify_fn_returns_registered_callable(self):
        from operon_ai.core.certificate import resolve_verify_fn

        fn = resolve_verify_fn("behavioral_stability_windowed")
        assert fn is not None
        assert callable(fn)

    def test_resolve_verify_fn_returns_none_for_unknown_theorem(self):
        from operon_ai.core.certificate import resolve_verify_fn

        assert resolve_verify_fn("not_a_real_theorem_xyz") is None

    def test_resolve_verify_fn_is_public_alias_of_internal(self):
        """The public name should be a direct alias so the two stay in
        sync automatically. Downstream packages can import either, but
        the public name is preferred."""
        from operon_ai.core.certificate import (
            _resolve_verify_fn,
            resolve_verify_fn,
        )

        assert resolve_verify_fn is _resolve_verify_fn

    def test_from_theorem_builds_verifiable_certificate(self):
        cert = Certificate.from_theorem(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.1, 0.15), "threshold": 0.5},
            conclusion="test",
            source="test",
        )
        assert cert.theorem == "behavioral_stability_windowed"
        result = cert.verify()
        assert result.holds is True
        assert result.evidence["n"] == 2

    def test_from_theorem_detects_instability_like_direct_construction(self):
        cert = Certificate.from_theorem(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.805, 0.805), "threshold": 0.8},
            conclusion="test",
            source="test",
        )
        assert cert.verify().holds is False

    def test_from_theorem_raises_for_unknown_theorem(self):
        import pytest

        with pytest.raises(KeyError, match="not_a_real_theorem"):
            Certificate.from_theorem(
                theorem="not_a_real_theorem",
                parameters={"x": 1},
                conclusion="test",
                source="test",
            )

    def test_from_theorem_error_lists_both_registries_even_on_cold_process(self):
        """Regression for roborev #791 Low.

        The ``KeyError`` raised when a theorem is unknown must enumerate
        theorems from BOTH ``_VERIFY_REGISTRY`` (dynamic) and
        ``_THEOREM_FN_PATHS`` (static built-ins). On a cold process where
        no theorem has been resolved yet, ``_VERIFY_REGISTRY`` is empty
        but several built-ins (e.g. ``behavioral_stability``,
        ``behavioral_stability_windowed``) are available through the path
        map. The error message must surface those to the user, not say
        ``Known theorems: []``.
        """
        import pytest

        # Use a subprocess to guarantee a cold ``_VERIFY_REGISTRY`` —
        # earlier tests in this session will have populated it by now
        # via theorem resolutions.
        import subprocess
        import sys

        probe = (
            "from operon_ai.core.certificate import Certificate\n"
            "try:\n"
            "    Certificate.from_theorem(\n"
            "        theorem='definitely_not_a_real_theorem',\n"
            "        parameters={'x': 1},\n"
            "        conclusion='t',\n"
            "        source='t',\n"
            "    )\n"
            "except KeyError as e:\n"
            "    msg = str(e)\n"
            "else:\n"
            "    raise SystemExit('expected KeyError')\n"
            "\n"
            "# The error must name at least the built-in behavioral theorems\n"
            "# that live in _THEOREM_FN_PATHS, proving we didn't restrict\n"
            "# the listing to _VERIFY_REGISTRY only.\n"
            "assert 'behavioral_stability' in msg, f'missing built-in: {msg!r}'\n"
            "assert 'behavioral_stability_windowed' in msg, (\n"
            "    f'missing built-in: {msg!r}'\n"
            ")\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert result.stdout.strip().endswith("ok")

    def test_from_theorem_round_trip_matches_direct_construction(self):
        """Certs built via ``from_theorem`` must have the same verify
        behavior as certs built via the public constructor — the factory
        is sugar, not a behavior change."""
        from operon_ai.core.certificate import _verify_behavioral_stability_windowed

        factory_cert = Certificate.from_theorem(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.3,), "threshold": 0.5},
            conclusion="test",
            source="test",
        )
        direct_cert = Certificate(
            theorem="behavioral_stability_windowed",
            parameters={"signal_values": (0.3,), "threshold": 0.5},
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_stability_windowed,
        )
        f_result = factory_cert.verify()
        d_result = direct_cert.verify()
        assert f_result.holds == d_result.holds
        assert f_result.evidence == d_result.evidence

    def test_certificate_from_dict_error_lists_both_registries_cold(self):
        """Regression for roborev #794 Low.

        ``certificate_from_dict`` (the deserialization path) must use the
        same union-of-registries diagnostic as ``Certificate.from_theorem``.
        The earlier #791 fix only updated the classmethod; this test pins
        that ``certificate_from_dict`` got the same treatment, and does
        so on a cold-process path — which is the exact scenario that
        matters (deserializing a cert somewhere other than where it was
        produced).
        """
        import subprocess
        import sys

        probe = (
            "from operon_ai.core.certificate import certificate_from_dict\n"
            "try:\n"
            "    certificate_from_dict({\n"
            "        'theorem': 'definitely_not_a_real_theorem',\n"
            "        'parameters': {'x': 1},\n"
            "        'conclusion': 't',\n"
            "        'source': 't',\n"
            "    })\n"
            "except KeyError as e:\n"
            "    msg = str(e)\n"
            "else:\n"
            "    raise SystemExit('expected KeyError')\n"
            "\n"
            "assert 'behavioral_stability' in msg, f'missing built-in: {msg!r}'\n"
            "assert 'behavioral_stability_windowed' in msg, (\n"
            "    f'missing built-in: {msg!r}'\n"
            ")\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert result.stdout.strip().endswith("ok")


class TestBehavioralNoAnomaly:
    def test_verify_holds(self):
        from operon_ai.core.certificate import _verify_behavioral_no_anomaly

        cert = Certificate(
            theorem="behavioral_no_anomaly",
            parameters={
                "threat_levels": ["none", "none", "suspicious"],
                "max_allowed": "suspicious",
            },
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_no_anomaly,
        )
        result = cert.verify()
        assert result.holds is True
        assert result.evidence["max_threat"] == "suspicious"

    def test_verify_fails_with_confirmed(self):
        from operon_ai.core.certificate import _verify_behavioral_no_anomaly

        cert = Certificate(
            theorem="behavioral_no_anomaly",
            parameters={
                "threat_levels": ["none", "confirmed"],
                "max_allowed": "suspicious",
            },
            conclusion="test",
            source="test",
            _verify_fn=_verify_behavioral_no_anomaly,
        )
        result = cert.verify()
        assert result.holds is False
