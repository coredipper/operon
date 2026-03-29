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

    recent_outputs: tuple[MappingProxyType, ...] = ()
    telemetry: tuple[Any, ...] = ()
    experience: tuple[Any, ...] = ()
    developmental_status: Any = None  # DevelopmentStatus | None
    trust_context: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Freeze all mutable mapping inputs on construction."""
        def _freeze_tuple(t: tuple) -> tuple:
            return tuple(
                MappingProxyType(dict(d)) if isinstance(d, dict) and not isinstance(d, MappingProxyType) else d
                for d in t
            )
        if isinstance(self.trust_context, dict) and not isinstance(self.trust_context, MappingProxyType):
            object.__setattr__(self, "trust_context", MappingProxyType(dict(self.trust_context)))
        for field_name in ("recent_outputs", "telemetry", "experience"):
            val = getattr(self, field_name)
            if val and any(isinstance(d, dict) and not isinstance(d, MappingProxyType) for d in val):
                object.__setattr__(self, field_name, _freeze_tuple(val))


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
    # Freeze mutable inputs into immutable containers.
    frozen_trust = MappingProxyType(dict(trust_context)) if trust_context is not None else MappingProxyType({})
    frozen_outputs = tuple(MappingProxyType(dict(d)) for d in recent_outputs)
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
