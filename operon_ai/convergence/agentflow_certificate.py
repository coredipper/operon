"""agentflow ``evolve`` compile-time provenance binding (cheap-variant L2 hook).

This module ships the cheap-variant L2 certificate hook described in
``docs/site/external-frameworks.md`` §8.3, mirroring the §2 cheap-variant
T2 DSPy binding (see ``dspy_certificate.py``).

§8.3 names the L2 angle as "the more interesting angle" of the agentflow
wedge: ``agentflow evolve`` compiles successful traces into tuned agent
versions written to ``.agentflow/tuned_agents/``. That compile boundary
is structurally analogous to DSPy's — uncompiled topology + traces in,
compiled artifact out — so the same provenance pattern applies. agentflow
exposes no explicit metric (selection of "successful" traces is implicit
in the evolve pipeline), so this binding records three hashes instead
of four:

- ``graph_hash`` — structure hash of the uncompiled ``Graph()`` definition
- ``traces_hash`` — content hash of the successful-traces input file
- ``tuned_agent_hash`` — content hash of the resulting tuned agent

A certificate bound here is a *provenance marker*: it asserts the three
pinned inputs/outputs were recorded at evolve time. The reproducibility
witness is downstream — re-running ``agentflow evolve`` with identical
input traces should produce a tuned agent whose hash matches the
recorded ``tuned_agent_hash``.

The factory accepts pre-computed hashes; agentflow itself is not
imported (it ships only as a CLI; no Python public API at the time of
writing per §8.3). Callers compute hashes from their evolve inputs and
outputs using whatever canonical serialization they prefer; operon-ai's
convention is ``hashlib.sha256(...).hexdigest()`` truncated.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.certificate import Certificate, register_verify_fn

_THEOREM_NAME = "agentflow_evolve_pinned_inputs"

_REQUIRED_KEYS: tuple[str, ...] = (
    "graph_hash",
    "traces_hash",
    "tuned_agent_hash",
)

_HEX_CHARS = frozenset("0123456789abcdef")
_MIN_HASH_LEN = 8


def _is_well_formed_hash(value: Any) -> bool:
    """Hex string of length >= _MIN_HASH_LEN, lowercase only.

    Mirrors the helper in ``dspy_certificate.py`` — kept local to avoid
    cross-coupling between sibling cert modules. If a third compile-cert
    binding lands, factor into a shared private helper.
    """
    if not isinstance(value, str):
        return False
    if len(value) < _MIN_HASH_LEN:
        return False
    return all(ch in _HEX_CHARS for ch in value)


def _verify_agentflow_evolve_pinned_inputs(
    params: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify all three agentflow evolve-boundary hashes are well-formed.

    Holds iff every key in ``_REQUIRED_KEYS`` is present and maps to a
    non-empty lowercase-hex string of length ≥ 8. Evidence surfaces
    ``present_keys``, ``missing``, and ``malformed`` for audit replay.
    """
    present = [k for k in _REQUIRED_KEYS if k in params]
    missing = [k for k in _REQUIRED_KEYS if k not in params]
    malformed = [
        k for k in present if not _is_well_formed_hash(params[k])
    ]
    holds = not missing and not malformed
    return holds, {
        "present_keys": present,
        "missing": missing,
        "malformed": malformed,
    }


register_verify_fn(_THEOREM_NAME, _verify_agentflow_evolve_pinned_inputs)


def make_agentflow_compile_certificate(
    graph_hash: str,
    traces_hash: str,
    tuned_agent_hash: str,
    *,
    source: str = "Certificate.from_agentflow_compile",
) -> Certificate:
    """Build an agentflow evolve-pinned-inputs provenance certificate.

    Parameters are the three hashes that pin the evolve compile
    boundary: structure hash of the uncompiled ``Graph()``, content hash
    of the input traces, and content hash of the tuned agent output.

    The certificate's ``conclusion`` is keyed on ``tuned_agent_hash``
    since that is the field that uniquely identifies the compiled
    artifact in the provenance record.
    """
    return Certificate.from_theorem(
        theorem=_THEOREM_NAME,
        parameters={
            "graph_hash": graph_hash,
            "traces_hash": traces_hash,
            "tuned_agent_hash": tuned_agent_hash,
        },
        conclusion=(
            f"agentflow evolve artifact pinned by tuned_agent_hash "
            f"{tuned_agent_hash!r}"
        ),
        source=source,
    )
