"""Distributed watcher — transport-agnostic signal publishing and consumption.

Enables the same WatcherComponent convergence detection logic to work
across single-process, multi-process, and multi-machine deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Protocol


class WatcherTransport(Protocol):
    """Transport protocol for watcher signals and interventions."""

    def publish_signal(self, signal: dict[str, Any]) -> None: ...
    def publish_intervention(self, intervention: dict[str, Any]) -> None: ...
    def subscribe_signals(self, callback: Callable[[dict[str, Any]], None]) -> None: ...
    def subscribe_interventions(self, callback: Callable[[dict[str, Any]], None]) -> None: ...


@dataclass
class InMemoryTransport:
    """In-process transport for single-process deployments."""

    _signal_subscribers: list[Callable] = field(default_factory=list)
    _intervention_subscribers: list[Callable] = field(default_factory=list)
    _signal_log: list[dict[str, Any]] = field(default_factory=list)
    _intervention_log: list[dict[str, Any]] = field(default_factory=list)

    def publish_signal(self, signal: dict[str, Any]) -> None:
        self._signal_log.append(signal)
        for cb in self._signal_subscribers:
            cb(signal)

    def publish_intervention(self, intervention: dict[str, Any]) -> None:
        self._intervention_log.append(intervention)
        for cb in self._intervention_subscribers:
            cb(intervention)

    def subscribe_signals(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._signal_subscribers.append(callback)

    def subscribe_interventions(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._intervention_subscribers.append(callback)


@dataclass
class HttpTransport:
    """HTTP callback transport for webhook-based deployments.

    Produces request payload dicts that the caller sends via their HTTP
    client of choice.  Does NOT make actual HTTP calls.
    """

    signal_url: str = ""
    intervention_url: str = ""
    _pending_requests: list[dict[str, Any]] = field(default_factory=list)
    _signal_subscribers: list[Callable] = field(default_factory=list)
    _intervention_subscribers: list[Callable] = field(default_factory=list)

    def publish_signal(self, signal: dict[str, Any]) -> None:
        self._pending_requests.append({
            "method": "POST",
            "url": self.signal_url,
            "body": signal,
        })

    def publish_intervention(self, intervention: dict[str, Any]) -> None:
        self._pending_requests.append({
            "method": "POST",
            "url": self.intervention_url,
            "body": intervention,
        })

    def get_pending_requests(self) -> list[dict[str, Any]]:
        """Return and clear pending HTTP requests."""
        pending = list(self._pending_requests)
        self._pending_requests.clear()
        return pending

    def subscribe_signals(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._signal_subscribers.append(callback)

    def subscribe_interventions(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._intervention_subscribers.append(callback)


@dataclass
class DistributedWatcher:
    """Watcher that publishes signals/interventions over a transport layer.

    Wraps a WatcherComponent's output and distributes it across processes.
    """

    transport: Any  # WatcherTransport
    organism_id: str = "default"
    _signals_published: int = field(default=0, init=False)
    _interventions_published: int = field(default=0, init=False)

    def publish_stage_result(
        self,
        stage_name: str,
        signals: list[dict[str, Any]],
        intervention: dict[str, Any] | None = None,
    ) -> None:
        """Publish a stage result with its signals and optional intervention."""
        for sig in signals:
            # Caller payload first, framework fields last (authoritative).
            msg = dict(sig)
            msg["organism_id"] = self.organism_id
            msg["stage_name"] = stage_name
            msg["timestamp"] = datetime.now(UTC).isoformat()
            self.transport.publish_signal(msg)
            self._signals_published += 1

        if intervention is not None:
            msg = dict(intervention)
            msg["organism_id"] = self.organism_id
            msg["stage_name"] = stage_name
            msg["timestamp"] = datetime.now(UTC).isoformat()
            self.transport.publish_intervention(msg)
            self._interventions_published += 1

    def publish_heartbeat(self, status: dict[str, Any]) -> None:
        """Publish a heartbeat status check."""
        msg = dict(status)
        msg["organism_id"] = self.organism_id
        msg["type"] = "heartbeat"
        msg["timestamp"] = datetime.now(UTC).isoformat()
        self.transport.publish_signal(msg)
        self._signals_published += 1

    def summary(self) -> dict[str, Any]:
        """Return transport-level summary."""
        return {
            "organism_id": self.organism_id,
            "signals_published": self._signals_published,
            "interventions_published": self._interventions_published,
        }
