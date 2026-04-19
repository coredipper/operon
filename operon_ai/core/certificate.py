"""Categorical certificate framework for structural and behavioral guarantees.

A :class:`Certificate` captures a guarantee as a self-verifiable record.
It stores the theorem being claimed, the parameters that make it hold,
and a verification function that re-derives the conclusion from the
parameters (derivation replay).

**Structural certificates** (configuration → algebraic check):

- **QuorumSensingBio.certify()** — no-false-activation under normal traffic
- **MTORScaler.certify()** — no-oscillation via hysteresis dead bands
- **ATP_Store.certify()** — priority gating serves critical operations

**Behavioral certificates** (evidence snapshot → statistical check):

- **VerifierComponent.certify_behavior()** — mean rubric quality ≥ threshold
- **WatcherComponent.certify_behavior()** — signal stability (mean < threshold)

Example::

    qs = QuorumSensingBio(population_size=10)
    qs.calibrate()
    cert = qs.certify()
    result = cert.verify()
    assert result.holds
    print(result.evidence)  # {"c_ss": 1.5, "threshold": 3.0, "ratio": 0.5}
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze mutable containers."""
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        frozen = tuple(_deep_freeze(v) for v in value)
        # Reconstruct subclasses to preserve type
        if type(value) is not tuple:
            if hasattr(type(value), "_fields"):
                # namedtuple: variadic constructor
                try:
                    return type(value)(*frozen)
                except (TypeError, ValueError):
                    pass
            else:
                # Other tuple subclasses: try iterable, then variadic.
                # Validate content matches to catch constructor misinterpretation.
                for factory in (lambda: type(value)(frozen), lambda: type(value)(*frozen)):
                    try:
                        result = factory()
                        if tuple(result) == frozen:
                            return result
                    except (TypeError, ValueError):
                        continue
        return frozen
    return value


def _freeze(d: dict[str, Any]) -> MappingProxyType:
    """Return a deeply frozen read-only view of *d*."""
    return _deep_freeze(d)


@dataclass(frozen=True)
class Certificate:
    """A self-verifiable structural guarantee.

    Attributes:
        theorem: Short identifier for the guarantee
            (e.g. ``"no_false_activation"``).
        parameters: The values the guarantee depends on (immutable).
        conclusion: Human-readable statement of what is guaranteed.
        source: Where the certificate was produced
            (e.g. ``"QuorumSensingBio.calibrate"``).
    """

    theorem: str
    parameters: Mapping[str, Any]
    conclusion: str
    source: str
    _verify_fn: Callable[[Mapping[str, Any]], tuple[bool, dict[str, Any]]]

    def __init__(
        self,
        theorem: str,
        parameters: dict[str, Any] | Mapping[str, Any],
        conclusion: str,
        source: str,
        _verify_fn: Callable,
    ) -> None:
        object.__setattr__(self, "theorem", theorem)
        object.__setattr__(self, "parameters", _freeze(dict(parameters)))
        object.__setattr__(self, "conclusion", conclusion)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_verify_fn", _verify_fn)

    def verify(self) -> CertificateVerification:
        """Re-derive the guarantee from current parameters.

        Returns a :class:`CertificateVerification` with the derivation
        result and all intermediate evidence values.
        """
        holds, evidence = self._verify_fn(self.parameters)
        return CertificateVerification(
            certificate=self,
            holds=holds,
            evidence=_freeze(evidence),
        )

    @classmethod
    def from_theorem(
        cls,
        theorem: str,
        parameters: dict[str, Any] | Mapping[str, Any],
        conclusion: str,
        source: str,
    ) -> "Certificate":
        """Construct a certificate by resolving its verify function from
        the theorem registry.

        This is the preferred entry point for downstream packages that
        emit certificates — it removes the need to import and bind a
        verify function directly, which would couple them to a specific
        internal symbol. Raises :class:`KeyError` if the theorem is not
        registered (via ``_THEOREM_FN_PATHS`` or :func:`register_verify_fn`).

        Example::

            cert = Certificate.from_theorem(
                theorem="behavioral_stability_windowed",
                parameters={"signal_values": (0.5,), "threshold": 0.8},
                conclusion="Stagnation detected after 21 measurements",
                source="my_gate_package",
            )
            assert cert.verify().holds is True
        """
        fn = _resolve_verify_fn(theorem)
        if fn is None:
            raise KeyError(
                f"No verify function registered for theorem {theorem!r}. "
                f"Known theorems: "
                f"{sorted(set(_VERIFY_REGISTRY) | set(_THEOREM_FN_PATHS))}"
            )
        return cls(
            theorem=theorem,
            parameters=parameters,
            conclusion=conclusion,
            source=source,
            _verify_fn=fn,
        )


@dataclass(frozen=True)
class CertificateVerification:
    """Result of verifying a :class:`Certificate`.

    Attributes:
        certificate: The certificate that was verified.
        holds: Whether the guarantee currently holds.
        evidence: Intermediate values from the derivation replay
            (e.g. steady-state concentration, threshold, ratio).
    """

    certificate: Certificate
    holds: bool
    evidence: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Verification function registry (enables serialization/deserialization)
# ---------------------------------------------------------------------------

_VERIFY_REGISTRY: dict[str, Callable] = {}


def register_verify_fn(theorem: str, fn: Callable) -> None:
    """Register a verification function for a theorem name."""
    _VERIFY_REGISTRY[theorem] = fn


def _thaw(value: Any) -> Any:
    """Recursively convert frozen containers to JSON-serializable types."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(v) for v in sorted(value, key=repr)]
    return value


def certificate_to_dict(cert: Certificate) -> dict[str, Any]:
    """Serialize a certificate to a JSON-compatible dict."""
    return {
        "theorem": cert.theorem,
        "parameters": _thaw(cert.parameters),
        "conclusion": cert.conclusion,
        "source": cert.source,
    }


_THEOREM_FN_PATHS: dict[str, tuple[str, str]] = {
    # Structural certificates
    "no_false_activation": ("operon_ai.coordination.quorum_sensing", "_verify_no_false_activation"),
    "no_oscillation": ("operon_ai.state.mtor", "_verify_no_oscillation"),
    "priority_gating": ("operon_ai.state.metabolism", "_verify_priority_gating"),
    "state_integrity_verified": ("operon_ai.state.dna_repair", "_verify_state_integrity"),
    # Behavioral certificates
    "behavioral_quality": ("operon_ai.core.certificate", "_verify_behavioral_quality"),
    "behavioral_stability": ("operon_ai.core.certificate", "_verify_behavioral_stability"),
    "behavioral_stability_windowed": (
        "operon_ai.core.certificate",
        "_verify_behavioral_stability_windowed",
    ),
    "behavioral_no_anomaly": ("operon_ai.core.certificate", "_verify_behavioral_no_anomaly"),
}


# ---------------------------------------------------------------------------
# Behavioral verification functions
# ---------------------------------------------------------------------------


def _verify_behavioral_quality(params: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Replay: mean rubric quality ≥ threshold over frozen score evidence."""
    scores = list(params["scores"])
    threshold = params["threshold"]
    if not scores:
        return False, {"mean": 0.0, "min": 0.0, "n": 0}
    mean_q = sum(scores) / len(scores)
    return mean_q >= threshold, {
        "mean": round(mean_q, 4),
        "min": round(min(scores), 4),
        "n": len(scores),
    }


def _verify_behavioral_stability(params: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Replay: mean signal severity < threshold (no stagnation/collapse)."""
    values = list(params["signal_values"])
    threshold = params["threshold"]
    if not values:
        return False, {"mean": 0.0, "max": 0.0, "n": 0}
    mean_v = sum(values) / len(values)
    return mean_v < threshold, {
        "mean": round(mean_v, 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def _verify_behavioral_stability_windowed(
    params: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Replay: every windowed-signal mean is within the stability bound.

    Stable ⟺ ``max(signal_values) <= threshold``.

    Sibling of :func:`_verify_behavioral_stability` for rolling-window
    detectors. ``signal_values`` is a sequence of per-window means — one
    entry per violating window — and the predicate is the max-based
    inclusive bound rather than a flattened-history mean. This preserves
    the per-window structure that a rolling-window detector operates on;
    a flat mean over the union of overlapping windows weights interior
    samples more heavily than any individual window's mean and can
    disagree with detection.

    ``threshold`` is the stability threshold in the signal domain (e.g.
    ``1 - τ_detect`` when the detector's τ is in the complementary domain).
    The ``<=`` (not ``<``) mirrors a strict ``< τ_detect`` detection
    predicate: ``integral < τ`` ⟺ ``stable ⟺ integral >= τ`` ⟺
    ``mean(signal) <= 1 - τ``, which is inclusive at the boundary.

    Empty ``signal_values`` is treated as malformed evidence (``holds=False``
    with ``reason="empty_evidence"``). A stagnation certificate without
    per-window evidence is a contradiction in terms; emitters that can
    legitimately produce one already enforce ``len(signal_values) >= 1``
    at construction time.
    """
    values = list(params["signal_values"])
    threshold = params["threshold"]
    if not values:
        return False, {"max": 0.0, "mean": 0.0, "n": 0, "reason": "empty_evidence"}
    max_v = max(values)
    mean_v = sum(values) / len(values)
    return max_v <= threshold, {
        "max": round(max_v, 4),
        "mean": round(mean_v, 4),
        "n": len(values),
    }


def _verify_behavioral_no_anomaly(params: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Replay: no confirmed/critical threats in inspection evidence."""
    levels = list(params["threat_levels"])
    max_allowed = params["max_allowed"]
    order = {"none": 0, "suspicious": 1, "confirmed": 2, "critical": 3}
    if not levels:
        return False, {"max_threat": "none", "n": 0}
    max_seen = max(order.get(str(level), 0) for level in levels)
    holds = max_seen <= order.get(max_allowed, 1)
    # Reverse-map for evidence
    reverse = {v: k for k, v in order.items()}
    return holds, {
        "max_threat": reverse.get(max_seen, "unknown"),
        "n": len(levels),
    }


def _resolve_verify_fn(theorem: str) -> Callable | None:
    """Look up a verify function by theorem name.

    Checks the dynamic registry first, then falls back to the built-in
    function path map for lazy resolution.
    """
    fn = _VERIFY_REGISTRY.get(theorem)
    if fn is not None:
        return fn
    path = _THEOREM_FN_PATHS.get(theorem)
    if path is None:
        return None
    module_name, fn_name = path
    import importlib
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and (e.name == module_name or module_name.startswith(f"{e.name}.")):
            return None
        raise
    fn = getattr(module, fn_name, None)
    if fn is not None:
        _VERIFY_REGISTRY[theorem] = fn
    return fn


# Public alias of ``_resolve_verify_fn`` for downstream packages that need to
# resolve a verify function by theorem name. Exposes the lookup as part of the
# public API surface so consumers are not coupled to the underscore-prefixed
# internal name. Preferred over direct use of ``_resolve_verify_fn``.
#
# Direct alias (not a wrapper) so the two stay in sync automatically and
# ``resolve_verify_fn is _resolve_verify_fn`` holds.
resolve_verify_fn = _resolve_verify_fn


def certificate_from_dict(d: dict[str, Any]) -> Certificate:
    """Deserialize a certificate from a dict.

    Raises ``KeyError`` if the theorem's verify function is not registered.
    """
    theorem = d["theorem"]
    fn = _resolve_verify_fn(theorem)
    if fn is None:
        raise KeyError(
            f"No verify function registered for theorem {theorem!r}. "
            f"Known theorems: "
            f"{sorted(set(_VERIFY_REGISTRY) | set(_THEOREM_FN_PATHS))}"
        )
    return Certificate(
        theorem=theorem,
        parameters=d["parameters"],
        conclusion=d["conclusion"],
        source=d["source"],
        _verify_fn=fn,
    )


def verify_compiled(compiled: dict[str, Any]) -> list[CertificateVerification]:
    """Verify all certificates in a compiled organism dict.

    Raises ``KeyError`` if any certificate's theorem is not registered.
    """
    results = []
    for cert_dict in compiled.get("certificates", []):
        cert = certificate_from_dict(cert_dict)
        results.append(cert.verify())
    return results
