"""agentflow L1 adapter — emit Operon certificates from agentflow runtime events.

This module ships the L1 runtime-side companion to
:mod:`operon_ai.convergence.agentflow_certificate` (the L2 compile-time
hook for ``agentflow evolve`` shipped 2026-05-01). It mirrors
:class:`operon_ai.convergence.gascity_adapter.GascityCertificateAdapter`
shape — a dataclass adapter with optional harnesses per event kind, eager
theorem-name validation in ``__post_init__``, and
``CertificateVerification`` returned from each ``evaluate_*`` method.

Design (see ``docs/site/external-frameworks.md`` §8.3):

agentflow (``berabuddies/agentflow``) has no hook or middleware surface —
its primitives are a ``Graph()`` context manager, named agent nodes, and
the ``>> / fanout / merge / on_failure`` operators. The natural attach
points for gates are therefore *runtime-observable boundaries* rather
than explicit hooks:

- :class:`NodeEvent` — pre/post execution of a single named node
  (a place to check `behavioral_stability_windowed`,
  `state_integrity_verified`, or any per-step invariant).
- :class:`EdgeEvent` — transition between two named nodes
  (a place to check routing or message-shape invariants).
- :class:`EvolveEvent` — the ``agentflow evolve`` compile boundary
  (mirrors the parameter shape of ``agentflow_evolve_pinned_inputs``;
  a runtime-side rendering of the same provenance facts the L2 hook
  records).

Callers construct events themselves — the adapter does not import
``agentflow``. A typical integration wraps ``Graph.run()`` in a façade
that emits these events to user-supplied harnesses; that wiring is
deployment-specific and intentionally not bundled here.

Certificates produced flow into the same flat Dolt-friendly JSON envelope
gascity uses, via
:func:`operon_ai.convergence.gascity_adapter.verification_to_dolt_envelope`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from ..core.certificate import (
    Certificate,
    CertificateVerification,
    resolve_verify_fn,
)


# ---------------------------------------------------------------------------
# Event mirrors — runtime-observable boundaries in agentflow's Graph.run().
# ---------------------------------------------------------------------------


@dataclass
class NodeEvent:
    """Pre- or post-execution event for a single agentflow node.

    The harness callback receives this and returns a parameter dict for
    the configured theorem. ``phase`` distinguishes pre-entry checks
    (e.g. invariant guards) from post-exit checks (e.g. stability over
    a window of completed steps).
    """

    session_id: str
    node_name: str
    phase: Literal["pre", "post"]
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass
class EdgeEvent:
    """Transition event between two named agentflow nodes.

    Fired when control leaves ``from_node`` and is about to enter
    ``to_node`` — the natural place for routing checks, message-shape
    invariants, or pre-nudge structural validation.
    """

    session_id: str
    from_node: str
    to_node: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass
class EvolveEvent:
    """Compile-boundary event for ``agentflow evolve``.

    Field shape mirrors the parameters of the L2 theorem
    ``agentflow_evolve_pinned_inputs`` so the same provenance facts the
    L2 hook records at compile time can be observed at runtime. Useful
    when an adapter is configured with that theorem and a harness that
    forwards the three hashes verbatim.
    """

    session_id: str
    graph_hash: str
    traces_hash: str
    tuned_agent_hash: str
    timestamp: str | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


HarnessNode = Callable[[NodeEvent], Mapping[str, Any]]
HarnessEdge = Callable[[EdgeEvent], Mapping[str, Any]]
HarnessEvolve = Callable[[EvolveEvent], Mapping[str, Any]]


@dataclass
class AgentflowCertificateAdapter:
    """Adapter that emits Operon certificates from agentflow runtime events.

    Parameters
    ----------
    theorem:
        Name of a theorem registered in ``operon_ai.core.certificate``'s
        registry — either a built-in (``"behavioral_stability_windowed"``,
        ``"state_integrity_verified"``, ``"agentflow_evolve_pinned_inputs"``,
        …) or a name registered dynamically by an installed package
        (e.g. ``operon-langgraph-gates`` registers
        ``"langgraph_state_integrity"`` at import time). Resolved eagerly
        at construction; raises :class:`KeyError` if unknown.
    harness_node, harness_edge, harness_evolve:
        Optional callbacks that extract theorem parameters from each
        event kind. If a harness is ``None``, the corresponding
        ``evaluate_*`` method raises :class:`RuntimeError`. Callers that
        only attach to a subset of agentflow's boundaries only need to
        supply harnesses for the ones they use.
    conclusion_template:
        Format string for the certificate's ``conclusion``. Receives
        ``{theorem}`` and ``{attach_point}`` as keyword fields.
    source:
        Value used for the certificate's ``source`` field. Default:
        ``"agentflow_adapter"``.
    """

    theorem: str
    harness_node: HarnessNode | None = None
    harness_edge: HarnessEdge | None = None
    harness_evolve: HarnessEvolve | None = None
    conclusion_template: str = "{theorem} on agentflow {attach_point}"
    source: str = "agentflow_adapter"

    def __post_init__(self) -> None:
        if resolve_verify_fn(self.theorem) is None:
            raise KeyError(
                f"Theorem {self.theorem!r} is not registered. "
                "Register via operon_ai.core.certificate.register_verify_fn "
                "or use a theorem name from the built-in registry."
            )

    def evaluate_node(self, event: NodeEvent) -> CertificateVerification:
        """Run the configured theorem against a node pre/post event."""
        if self.harness_node is None:
            raise RuntimeError(
                "evaluate_node called but harness_node was not supplied at "
                "adapter construction."
            )
        return self._evaluate(self.harness_node(event), attach_point="node")

    def evaluate_edge(self, event: EdgeEvent) -> CertificateVerification:
        """Run the configured theorem against an inter-node transition."""
        if self.harness_edge is None:
            raise RuntimeError(
                "evaluate_edge called but harness_edge was not supplied at "
                "adapter construction."
            )
        return self._evaluate(self.harness_edge(event), attach_point="edge")

    def evaluate_evolve(
        self, event: EvolveEvent
    ) -> CertificateVerification:
        """Run the configured theorem against the ``evolve`` compile boundary."""
        if self.harness_evolve is None:
            raise RuntimeError(
                "evaluate_evolve called but harness_evolve was not supplied "
                "at adapter construction."
            )
        return self._evaluate(
            self.harness_evolve(event), attach_point="evolve"
        )

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
