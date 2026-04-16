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
