"""Multi-channel priming context for stage handlers.

Extends SubstrateView with additional channels from AnimaWorks' 6-channel
priming model, keeping backward compatibility with existing stage handler
signatures that expect a SubstrateView.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .types import SubstrateView


@dataclass(frozen=True)
class PrimingView(SubstrateView):
    """Multi-channel context envelope for stage handlers.

    Extends SubstrateView with additional channels from AnimaWorks' 6-channel
    priming model. Backward-compatible: isinstance(view, SubstrateView) passes.
    """

    recent_outputs: tuple[dict[str, Any], ...] = ()
    telemetry: tuple[Any, ...] = ()
    experience: tuple[Any, ...] = ()
    developmental_status: Any = None  # DevelopmentStatus | None
    trust_context: dict[str, float] = field(default_factory=dict)


def build_priming_view(
    substrate_view: SubstrateView,
    *,
    recent_outputs: tuple[dict[str, Any], ...] = (),
    telemetry: tuple[Any, ...] = (),
    experience: tuple[Any, ...] = (),
    developmental_status: Any = None,
    trust_context: dict[str, float] | None = None,
) -> PrimingView:
    """Promote a SubstrateView to a PrimingView with additional channels.

    Snapshots mutable inputs (trust_context, recent_outputs) to prevent
    post-construction mutation of the "read-only" context envelope.
    """
    # Snapshot trust_context to prevent caller mutation leaking in.
    frozen_trust = dict(trust_context) if trust_context is not None else {}
    # Snapshot recent_outputs dicts.
    frozen_outputs = tuple(dict(d) for d in recent_outputs)
    return PrimingView(
        facts=substrate_view.facts,
        query=substrate_view.query,
        record_time=substrate_view.record_time,
        recent_outputs=frozen_outputs,
        telemetry=telemetry,
        experience=experience,
        developmental_status=developmental_status,
        trust_context=frozen_trust,
    )
