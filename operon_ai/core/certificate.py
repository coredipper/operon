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
from typing import Any, Callable


@dataclass(frozen=True)
class Certificate:
    """A self-verifiable structural guarantee.

    Attributes:
        theorem: Short identifier for the guarantee
            (e.g. ``"no_false_activation"``).
        parameters: The values the guarantee depends on.
        conclusion: Human-readable statement of what is guaranteed.
        source: Where the certificate was produced
            (e.g. ``"QuorumSensingBio.calibrate"``).
    """

    theorem: str
    parameters: dict[str, Any]
    conclusion: str
    source: str
    _verify_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]]

    def verify(self) -> CertificateVerification:
        """Re-derive the guarantee from current parameters.

        Returns a :class:`CertificateVerification` with the derivation
        result and all intermediate evidence values.
        """
        holds, evidence = self._verify_fn(self.parameters)
        return CertificateVerification(
            certificate=self,
            holds=holds,
            evidence=evidence,
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
    evidence: dict[str, Any]
