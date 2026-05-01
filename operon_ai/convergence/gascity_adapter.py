"""Gas City adapter — turns Operon certificates into gascity audit-trail records.

This module implements a gate-style adapter (duck-typed against
``gastownhall/gascity`` event payloads) so Operon certificates can be
attached at gascity's ``hooks/``, ``dispatch/``, and ``mail/`` integration
points. No ``gascity`` import is required — the adapter consumes dataclass
shapes that mirror gascity's JSON event payloads.

Design (see ``docs/site/external-frameworks.md`` §8.1):

- Gas City exposes lifecycle hooks (``SessionStart``, ``PreToolUse``,
  ``UserPromptSubmit``, ``Stop``), a dispatch layer for pre-nudge structural
  checks, and a mail layer for message-boundary certificates. Each is a
  natural attach point for a gate.
- Certificates produced here flow into gascity's Beads/Dolt audit trail
  via :func:`verification_to_dolt_envelope`, which renders a flat
  JSON-serializable dict suitable for Dolt insertion.
- The adapter takes theorem names as constructor parameters (mirroring
  ``gepa_adapter.OperonCertificateAdapter``). Callers pass either
  operon-ai built-in theorem names or constants imported from
  ``operon-langgraph-gates`` (``STAGNATION_THEOREM``,
  ``INTEGRITY_THEOREM``) — the adapter does not bind to any upstream
  string literal.

The adapter never imports ``gascity`` or ``operon_langgraph_gates``;
downstream users install either optionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.certificate import (
    Certificate,
    CertificateVerification,
    certificate_to_dict,
    resolve_verify_fn,
)


# ---------------------------------------------------------------------------
# Structural mirrors of gascity event payloads.
# ---------------------------------------------------------------------------


@dataclass
class HookEvent:
    """Mirror of a gascity ``hooks/`` lifecycle event payload.

    Fields cover the four documented hook kinds (``SessionStart``,
    ``PreToolUse``, ``UserPromptSubmit``, ``Stop``). ``payload`` carries
    the hook-kind-specific body without further typing — the harness
    callback is responsible for extracting theorem parameters from it.
    """

    session_id: str
    hook_kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass
class DispatchEvent:
    """Mirror of a gascity ``dispatch/`` pre-nudge event payload.

    Used for structural checks before a Bead nudge fires (e.g. integrity
    of dispatched message routing). ``nudge_kind`` distinguishes
    formula/molecule/wait dispatch.
    """

    session_id: str
    nudge_kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass
class MailEvent:
    """Mirror of a gascity ``mail/`` message-boundary event payload.

    Used for inter-agent message certificates — gates fire when an
    agent's outbound message crosses a Pack boundary.
    """

    sender: str
    recipient: str
    subject: str
    body: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


HarnessHook = Callable[[HookEvent], Mapping[str, Any]]
HarnessDispatch = Callable[[DispatchEvent], Mapping[str, Any]]
HarnessMail = Callable[[MailEvent], Mapping[str, Any]]


@dataclass
class GascityCertificateAdapter:
    """Adapter that emits Operon certificates from gascity events.

    Parameters
    ----------
    theorem:
        Name of a theorem registered in ``operon_ai.core.certificate``'s
        registry — either a built-in (``"behavioral_stability_windowed"``,
        ``"state_integrity_verified"``, …) or a name registered
        dynamically by an installed package (e.g. ``operon-langgraph-gates``
        registers ``"langgraph_state_integrity"`` at import time).
        Resolved eagerly at construction; raises :class:`KeyError` if
        unknown.
    harness_hook, harness_dispatch, harness_mail:
        Optional callbacks that extract theorem parameters from each
        event kind. If a harness is ``None``, the corresponding
        ``evaluate_*`` method raises :class:`RuntimeError`. Callers that
        only attach to a subset of gascity's gate points only need to
        supply harnesses for the ones they use.
    conclusion_template:
        Format string for the certificate's ``conclusion``. Receives
        ``{theorem}`` and ``{attach_point}`` as keyword fields.
    source:
        Value used for the certificate's ``source`` field. Default:
        ``"gascity_adapter"``.
    """

    theorem: str
    harness_hook: HarnessHook | None = None
    harness_dispatch: HarnessDispatch | None = None
    harness_mail: HarnessMail | None = None
    conclusion_template: str = "{theorem} on gascity {attach_point}"
    source: str = "gascity_adapter"

    def __post_init__(self) -> None:
        if resolve_verify_fn(self.theorem) is None:
            raise KeyError(
                f"Theorem {self.theorem!r} is not registered. "
                "Register via operon_ai.core.certificate.register_verify_fn "
                "or use a theorem name from the built-in registry."
            )

    def evaluate_hook(self, event: HookEvent) -> CertificateVerification:
        """Run the configured theorem against a hook payload."""
        if self.harness_hook is None:
            raise RuntimeError(
                "evaluate_hook called but harness_hook was not supplied at "
                "adapter construction."
            )
        return self._evaluate(self.harness_hook(event), attach_point="hook")

    def evaluate_dispatch(self, event: DispatchEvent) -> CertificateVerification:
        """Run the configured theorem against a pre-nudge dispatch event."""
        if self.harness_dispatch is None:
            raise RuntimeError(
                "evaluate_dispatch called but harness_dispatch was not "
                "supplied at adapter construction."
            )
        return self._evaluate(
            self.harness_dispatch(event), attach_point="dispatch"
        )

    def evaluate_mail(self, event: MailEvent) -> CertificateVerification:
        """Run the configured theorem against an inter-agent mail event."""
        if self.harness_mail is None:
            raise RuntimeError(
                "evaluate_mail called but harness_mail was not supplied at "
                "adapter construction."
            )
        return self._evaluate(self.harness_mail(event), attach_point="mail")

    def _evaluate(
        self,
        parameters: Mapping[str, Any],
        *,
        attach_point: str,
    ) -> CertificateVerification:
        cert = Certificate.from_theorem(
            theorem=self.theorem,
            parameters=parameters,
            conclusion=self.conclusion_template.format(
                theorem=self.theorem, attach_point=attach_point
            ),
            source=self.source,
        )
        return cert.verify()


# ---------------------------------------------------------------------------
# Dolt audit-trail serializer
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert frozen containers to plain JSON-serializable types.

    ``Certificate.verify()`` returns evidence as a frozen ``MappingProxyType``
    (with nested tuples in place of lists) for immutability; Dolt and
    ``json.dumps`` want plain dicts and lists.
    """
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(v) for v in sorted(value, key=repr)]
    return value


def verification_to_dolt_envelope(
    verification: CertificateVerification,
    *,
    attach_point: str,
) -> dict[str, Any]:
    """Render a verification as a flat dict for the gascity Beads/Dolt audit trail.

    Keys are JSON-friendly primitives only — the resulting dict can be
    passed straight to ``json.dumps()`` and inserted into a Dolt table
    without further conversion. ``attach_point`` is one of ``"hook"``,
    ``"dispatch"``, or ``"mail"`` and identifies which gate fired.
    """
    cert_dict = certificate_to_dict(verification.certificate)
    return {
        "theorem": cert_dict["theorem"],
        "holds": bool(verification.holds),
        "conclusion": cert_dict["conclusion"],
        "source": cert_dict["source"],
        "attach_point": attach_point,
        "parameters": cert_dict["parameters"],
        "evidence": _to_jsonable(verification.evidence),
    }
