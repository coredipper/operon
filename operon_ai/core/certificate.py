"""Categorical certificate framework for structural guarantees.

A :class:`Certificate` captures a structural guarantee as a self-verifiable
record.  It stores the theorem being claimed, the parameters that make it
hold, and a verification function that re-derives the conclusion from the
parameters (derivation replay).

Three mechanisms currently produce certificates:

- **QuorumSensingBio.certify()** — no-false-activation under normal traffic
- **MTORScaler.certify()** — no-oscillation via hysteresis dead bands
- **ATP_Store.certify()** — priority gating serves critical operations

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


def certificate_to_dict(cert: Certificate) -> dict[str, Any]:
    """Serialize a certificate to a JSON-compatible dict."""
    return {
        "theorem": cert.theorem,
        "parameters": dict(cert.parameters),
        "conclusion": cert.conclusion,
        "source": cert.source,
    }


def certificate_from_dict(d: dict[str, Any]) -> Certificate | None:
    """Deserialize a certificate from a dict.

    Returns None if the theorem's verify function is not registered.
    """
    theorem = d["theorem"]
    fn = _VERIFY_REGISTRY.get(theorem)
    if fn is None:
        return None
    return Certificate(
        theorem=theorem,
        parameters=d["parameters"],
        conclusion=d["conclusion"],
        source=d["source"],
        _verify_fn=fn,
    )


def verify_compiled(compiled: dict[str, Any]) -> list[CertificateVerification]:
    """Verify all certificates in a compiled organism dict.

    Returns a list of verification results. Certificates whose theorem
    is not in the registry are skipped.
    """
    results = []
    for cert_dict in compiled.get("certificates", []):
        cert = certificate_from_dict(cert_dict)
        if cert is not None:
            results.append(cert.verify())
    return results
