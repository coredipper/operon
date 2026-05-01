"""DSPy compile-time provenance binding (cheap variant of Theorem 2).

This module ships the cheap variant of Theorem 2 from
``docs/site/external-frameworks.md`` §2 (lines 46–106): a registered
theorem ``dspy_compile_pinned_inputs`` whose verifier checks that the
four input/output hashes recorded at the DSPy compile boundary are
present and well-formed. It does *not* re-execute ``dspy.compile``;
that is the heavy variant ``dspy_compile_reproducible``, which §2
explicitly defers because re-running compilation costs seconds to
minutes per ``verify()`` call and is therefore unsuited to routine
verification.

§2 (lines 105–106) names this cheap variant as the "recommended first
step when the full reproducibility theorem arrives." A certificate
bound here is a *provenance marker*: it asserts that the four pinned
inputs (uncompiled program structure, training-set content, metric
source, and trace hash) were recorded at compile time. Anyone who
later re-runs ``dspy.compile(π_0, D_train, m, seed)`` and obtains a
matching ``trace_hash`` has confirmed the reproducibility witness;
mismatch is the audit signal §2 calls out (model drift, sampling
noise, trainset drift).

The factory accepts pre-computed hashes — DSPy is not imported, and
operon-ai stays framework-neutral. Callers compute the four hashes
from their DSPy artifacts using whatever canonical serialization they
prefer (operon-ai's convention is ``hashlib.sha256(...).hexdigest()``
truncated; see ``operon_ai/convergence/memory_bridge.py`` and
``operon_ai/topology/loops.py`` for examples).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.certificate import Certificate, register_verify_fn

_THEOREM_NAME = "dspy_compile_pinned_inputs"

_REQUIRED_KEYS: tuple[str, ...] = (
    "program_hash",
    "trainset_hash",
    "metric_hash",
    "trace_hash",
)

_HEX_CHARS = frozenset("0123456789abcdef")
_MIN_HASH_LEN = 8


def _is_well_formed_hash(value: Any) -> bool:
    """Hex string of length >= _MIN_HASH_LEN, lowercase only.

    Operon's truncation convention is sha256(...).hexdigest()[:N] with
    N typically 8/12/16. Eight is the floor — anything shorter is too
    weak to count as a provenance marker.
    """
    if not isinstance(value, str):
        return False
    if len(value) < _MIN_HASH_LEN:
        return False
    return all(ch in _HEX_CHARS for ch in value)


def _verify_dspy_compile_pinned_inputs(
    params: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Verify all four DSPy compile-boundary hashes are recorded and well-formed.

    Holds iff every key in ``_REQUIRED_KEYS`` is present in ``params``
    and maps to a non-empty lowercase-hex string of length ≥ 8.
    Evidence surfaces ``present_keys``, ``missing``, and ``malformed``
    so a failing certificate's audit context is preserved.
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


# Register at module import time so the static-path lookup in
# certificate._THEOREM_FN_PATHS is mirrored by the dynamic registry —
# whichever resolver fires first finds a function. Same belt-and-
# suspenders pattern operon-langgraph-gates uses for langgraph_state_integrity
# (see operon_langgraph_gates/integrity.py:80).
register_verify_fn(_THEOREM_NAME, _verify_dspy_compile_pinned_inputs)


def make_dspy_compile_certificate(
    program_hash: str,
    trainset_hash: str,
    metric_hash: str,
    trace_hash: str,
    *,
    source: str = "Certificate.from_dspy_compile",
) -> Certificate:
    """Build a DSPy compile-pinned-inputs provenance certificate.

    Parameters are the four hashes named in §2 of the external-frameworks
    memo: structure hash of the uncompiled program ``π₀``, content hash of
    the training set ``D_train``, source-level hash of the metric ``m``,
    and the trace hash ``h = hash(traces(π, D_train))``.

    The certificate's ``conclusion`` is keyed on ``trace_hash`` since that
    is the field that uniquely identifies the compiled artifact in the
    provenance record.
    """
    return Certificate.from_theorem(
        theorem=_THEOREM_NAME,
        parameters={
            "program_hash": program_hash,
            "trainset_hash": trainset_hash,
            "metric_hash": metric_hash,
            "trace_hash": trace_hash,
        },
        conclusion=(
            f"DSPy compile artifact pinned by trace_hash {trace_hash!r}"
        ),
        source=source,
    )
