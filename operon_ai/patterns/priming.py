"""Multi-channel priming context for stage handlers.

Extends SubstrateView with additional channels from AnimaWorks' 6-channel
priming model, keeping backward compatibility with existing stage handler
signatures that expect a SubstrateView.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Promote a SubstrateView to a PrimingView with additional channels."""
    return PrimingView(
        facts=substrate_view.facts,
        query=substrate_view.query,
        record_time=substrate_view.record_time,
        recent_outputs=recent_outputs,
        telemetry=telemetry,
        experience=experience,
        developmental_status=developmental_status,
        trust_context=trust_context if trust_context is not None else {},
    )
