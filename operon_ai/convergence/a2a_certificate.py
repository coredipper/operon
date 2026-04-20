"""A2A certificate codec — Operon certificates as A2A ``Part`` payloads.

The `A2A protocol <https://github.com/a2aproject/A2A>`__ is a JSON-RPC 2.0
interoperability layer for cross-vendor agent communication.  Agents
exchange ``Message`` objects whose ``parts`` are polymorphic
(``TextPart``, ``FilePart``, ``DataPart``).

This module defines a canonical ``DataPart`` shape for carrying Operon
certificates between vendor-opaque agents (see
``docs/site/external-frameworks.md`` §3.3, Theorem 4 — Graceful
degradation):

.. code-block:: json

    {
      "kind": "data",
      "data": {
        "schema": "operon.cert.v1",
        "theorem": "<str>",
        "parameters": { ... },
        "conclusion": "<str>",
        "source": "<str>",
        "verification": {
          "holds": <bool>,
          "evidence": { ... }
        } | null
      },
      "metadata": {
        "schema": "operon.cert.v1",
        "mimeType": "application/vnd.operon.cert+json"
      }
    }

Design constraints:

- **No ``a2a`` import.**  The codec produces plain dicts that match A2A's
  wire-level JSON, so callers may use any A2A SDK (Python, Go, JS, etc.).
- **Graceful degradation.**  :func:`certificate_from_a2a_part` raises on
  malformed parts, but :func:`safe_certificate_from_a2a_part` returns
  ``None`` for unknown theorems so receivers can forward-without-verify.
- **Registry-aware.**  Round-trip uses :func:`resolve_verify_fn` from
  either the dynamic or static theorem registry (fix 4d2a7cb ensures
  both are consulted).
"""

from __future__ import annotations

from typing import Any

from ..core.certificate import (
    Certificate,
    certificate_from_dict,
    certificate_to_dict,
    resolve_verify_fn,
)

# ---------------------------------------------------------------------------
# Constants (wire-level identifiers)
# ---------------------------------------------------------------------------

A2A_CERTIFICATE_SCHEMA = "operon.cert.v1"
A2A_CERTIFICATE_MIME_TYPE = "application/vnd.operon.cert+json"


class UnknownTheoremError(KeyError):
    """Raised when a Part references a theorem not in either registry.

    Subclass of :class:`KeyError` so existing ``except KeyError`` clauses
    in downstream code continue to work; callers that want graceful
    degradation should use :func:`safe_certificate_from_a2a_part` instead.
    """


class InvalidCertificatePartError(ValueError):
    """Raised when a Part claims ``schema == operon.cert.v1`` but is malformed."""


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def certificate_to_a2a_part(
    cert: Certificate,
    *,
    verify: bool = True,
    verifier_version: str | None = None,
) -> dict[str, Any]:
    """Encode an Operon certificate as an A2A ``DataPart`` dict.

    Parameters
    ----------
    cert:
        The certificate to encode.
    verify:
        If True (default), call :meth:`Certificate.verify` and embed
        ``verification`` in the payload.  If False, the ``verification``
        field is ``None`` — useful when the sender does not currently hold
        the verify function and is merely forwarding.
    verifier_version:
        Optional string identifying the verifier build (e.g. package
        version).  Stored in the Part metadata for audit.

    Returns
    -------
    dict[str, Any]
        A plain dict matching A2A's ``DataPart`` schema.  Safe to include
        as an element of an A2A ``Message.parts`` list.
    """
    payload: dict[str, Any] = dict(certificate_to_dict(cert))
    payload["schema"] = A2A_CERTIFICATE_SCHEMA

    if verify:
        verification = cert.verify()
        payload["verification"] = {
            "holds": verification.holds,
            "evidence": dict(verification.evidence),
        }
    else:
        payload["verification"] = None

    metadata: dict[str, Any] = {
        "schema": A2A_CERTIFICATE_SCHEMA,
        "mimeType": A2A_CERTIFICATE_MIME_TYPE,
    }
    if verifier_version is not None:
        metadata["verifierVersion"] = verifier_version

    return {
        "kind": "data",
        "data": payload,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def is_certificate_part(part: Any) -> bool:
    """Return True iff the Part claims to carry an Operon certificate.

    Accepts ``Any`` rather than ``dict`` because callers often receive
    parts from untyped JSON-RPC payloads.  Uses the canonical schema
    marker in either the payload (``data.schema``) or the part metadata
    (``metadata.schema``).  A part that matches this predicate but is
    malformed will still raise from :func:`certificate_from_a2a_part` —
    callers that need graceful handling should wrap the call or use
    :func:`safe_certificate_from_a2a_part`.
    """
    if not isinstance(part, dict):
        return False
    if part.get("kind") != "data":
        return False
    data = part.get("data")
    if isinstance(data, dict) and data.get("schema") == A2A_CERTIFICATE_SCHEMA:
        return True
    metadata = part.get("metadata")
    if (
        isinstance(metadata, dict)
        and metadata.get("schema") == A2A_CERTIFICATE_SCHEMA
    ):
        return True
    return False


def certificate_from_a2a_part(part: dict[str, Any]) -> Certificate:
    """Decode an A2A Part back into a :class:`Certificate`.

    Raises
    ------
    InvalidCertificatePartError
        If ``part`` does not carry ``operon.cert.v1`` or is missing required
        fields.
    UnknownTheoremError
        If the theorem named in the Part is not registered.  Use
        :func:`safe_certificate_from_a2a_part` to receive ``None`` instead
        (Theorem 4 — Graceful degradation).
    """
    if not is_certificate_part(part):
        raise InvalidCertificatePartError(
            "Part does not carry an operon.cert.v1 payload"
        )

    data = part.get("data")
    if not isinstance(data, dict):
        raise InvalidCertificatePartError("Part 'data' field is not a dict")

    for key in ("theorem", "parameters", "conclusion", "source"):
        if key not in data:
            raise InvalidCertificatePartError(
                f"Certificate Part missing required field: {key!r}"
            )

    if resolve_verify_fn(data["theorem"]) is None:
        raise UnknownTheoremError(data["theorem"])

    return certificate_from_dict({
        "theorem": data["theorem"],
        "parameters": data["parameters"],
        "conclusion": data["conclusion"],
        "source": data["source"],
    })


def safe_certificate_from_a2a_part(
    part: dict[str, Any],
) -> Certificate | None:
    """Like :func:`certificate_from_a2a_part` but returns ``None`` for unknown theorems.

    This is the receiver-side primitive for Theorem 4: an A2A node that
    does not have a given theorem's verifier registered can still forward
    the Part unchanged without raising.
    """
    try:
        return certificate_from_a2a_part(part)
    except UnknownTheoremError:
        return None


# ---------------------------------------------------------------------------
# AgentCard skill declaration
# ---------------------------------------------------------------------------


def agent_card_skill_for_theorem(
    theorem: str,
    *,
    role: str = "emit",
    description: str | None = None,
) -> dict[str, Any]:
    """Build an A2A AgentCard ``skill`` entry declaring certificate support.

    The generated skill entry follows A2A's ``AgentSkill`` shape and
    advertises that the agent can produce and/or verify a particular
    Operon theorem's certificates.

    Parameters
    ----------
    theorem:
        Theorem name from Operon's registry.  Used as the skill ``id`` suffix.
    role:
        Either ``"emit"`` (agent produces certificates), ``"verify"``
        (agent can re-verify incoming certificates), or ``"both"``.
    description:
        Optional human-readable description.  Auto-generated from the role
        and theorem if omitted.

    Returns
    -------
    dict[str, Any]
        An AgentSkill entry.  Append to ``AgentCard.skills``.
    """
    if role not in {"emit", "verify", "both"}:
        raise ValueError(f"role must be one of emit/verify/both, got {role!r}")

    if description is None:
        verb = {
            "emit": "Emit",
            "verify": "Verify",
            "both": "Emit and verify",
        }[role]
        description = f"{verb} Operon certificates for theorem {theorem!r}."

    return {
        "id": f"operon.cert.{role}.{theorem}",
        "name": f"Operon certificate ({role}): {theorem}",
        "description": description,
        "tags": ["operon", "certificate", theorem, role],
        "inputModes": [A2A_CERTIFICATE_MIME_TYPE]
        if role in {"verify", "both"}
        else [],
        "outputModes": [A2A_CERTIFICATE_MIME_TYPE]
        if role in {"emit", "both"}
        else [],
    }
