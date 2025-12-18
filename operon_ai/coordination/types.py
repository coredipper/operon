"""Core types for the coordination system."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class Phase(Enum):
    """Cell cycle phases for operation coordination."""
    G0 = "g0"  # Quiescent - not active
    G1 = "g1"  # Gap 1 - acquiring resources
    S = "s"    # Synthesis - executing operation
    G2 = "g2"  # Gap 2 - validating results
    M = "m"    # Mitosis - committing/finalizing

    def next(self) -> "Phase":
        """Get next phase in cycle."""
        cycle = [Phase.G0, Phase.G1, Phase.S, Phase.G2, Phase.M]
        idx = cycle.index(self)
        return cycle[(idx + 1) % len(cycle)]


class CheckpointResult(Enum):
    """Result of checkpoint evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    WAITING = "waiting"
    TIMEOUT = "timeout"


class LockResult(Enum):
    """Result of lock acquisition attempt."""
    ACQUIRED = "acquired"
    BLOCKED = "blocked"
    REENTRANT = "reentrant"
    PREEMPTED = "preempted"
    TIMEOUT = "timeout"


@dataclass
class ResourceLock:
    """
    Lockable resource with ownership and priority.

    Biological parallel: Resource competition like
    ATP or ribosome availability in cells.
    """

    resource_id: str
    owner: Optional[str] = None
    owner_priority: int = 0
    hold_count: int = 0
    acquired_at: Optional[datetime] = None
    allow_preemption: bool = False

    # Waiting list: [(owner_id, priority), ...] sorted by priority desc
    waiting_list: list[tuple[str, int]] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        """Check if lock is available."""
        return self.owner is None

    @property
    def hold_duration(self) -> Optional[timedelta]:
        """Get how long lock has been held."""
        if self.acquired_at is None:
            return None
        return datetime.utcnow() - self.acquired_at

    def try_acquire(self, owner: str, priority: int = 0) -> LockResult:
        """
        Attempt to acquire the lock.

        Returns:
        - ACQUIRED: Lock was free, now owned
        - BLOCKED: Lock is held by another, added to wait list
        - REENTRANT: Same owner, increment count
        - PREEMPTED: Higher priority took lock from lower
        """
        # Reentrant case - same owner
        if self.owner == owner:
            self.hold_count += 1
            return LockResult.REENTRANT

        # Available
        if self.is_available:
            self.owner = owner
            self.owner_priority = priority
            self.hold_count = 1
            self.acquired_at = datetime.utcnow()
            return LockResult.ACQUIRED

        # Preemption case
        if self.allow_preemption and priority > self.owner_priority:
            # Preempt current owner
            old_owner = self.owner
            old_priority = self.owner_priority

            # Add old owner to waiting list
            self._add_to_waiting(old_owner, old_priority)

            # New owner takes lock
            self.owner = owner
            self.owner_priority = priority
            self.hold_count = 1
            self.acquired_at = datetime.utcnow()
            return LockResult.PREEMPTED

        # Blocked - add to waiting list
        self._add_to_waiting(owner, priority)
        return LockResult.BLOCKED

    def _add_to_waiting(self, owner: str, priority: int) -> None:
        """Add to waiting list, sorted by priority (highest first)."""
        # Remove if already waiting
        self.waiting_list = [(o, p) for o, p in self.waiting_list if o != owner]
        self.waiting_list.append((owner, priority))
        self.waiting_list.sort(key=lambda x: x[1], reverse=True)

    def release(self, owner: str) -> bool:
        """
        Release the lock.

        Returns True if released, False if wrong owner.
        """
        if self.owner != owner:
            return False

        self.hold_count -= 1

        if self.hold_count <= 0:
            self.owner = None
            self.owner_priority = 0
            self.hold_count = 0
            self.acquired_at = None

            # Optionally hand off to highest priority waiter
            # (Not done here - caller should manage this)

        return True

    def pop_next_waiter(self) -> Optional[tuple[str, int]]:
        """Get and remove highest priority waiter."""
        if not self.waiting_list:
            return None
        return self.waiting_list.pop(0)


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""
    agents: list[str]
    resources: list[str]
    cycle: list[tuple[str, str, str]]  # [(waiter, blocking, resource), ...]


@dataclass
class DependencyGraph:
    """
    Tracks wait-for relationships for deadlock detection.

    Uses DFS to detect cycles (deadlocks).
    """

    # edges: waiter -> [(blocking, resource), ...]
    edges: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def add_dependency(self, waiter: str, blocking: str, resource: str) -> None:
        """Add a wait-for dependency."""
        if waiter not in self.edges:
            self.edges[waiter] = []

        # Avoid duplicates
        dep = (blocking, resource)
        if dep not in self.edges[waiter]:
            self.edges[waiter].append(dep)

    def remove_dependency(self, waiter: str, blocking: str) -> None:
        """Remove a dependency."""
        if waiter in self.edges:
            self.edges[waiter] = [
                (b, r) for b, r in self.edges[waiter] if b != blocking
            ]
            if not self.edges[waiter]:
                del self.edges[waiter]

    def remove_all_for_agent(self, agent: str) -> None:
        """Remove all dependencies involving an agent."""
        # Remove as waiter
        if agent in self.edges:
            del self.edges[agent]

        # Remove as blocking
        for waiter in list(self.edges.keys()):
            self.edges[waiter] = [
                (b, r) for b, r in self.edges[waiter] if b != agent
            ]
            if not self.edges[waiter]:
                del self.edges[waiter]

    def detect_cycle(self) -> Optional[DeadlockInfo]:
        """
        Detect if there's a cycle (deadlock).

        Returns DeadlockInfo if cycle found, None otherwise.
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> Optional[list]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            if node in self.edges:
                for blocking, resource in self.edges[node]:
                    if blocking not in visited:
                        result = dfs(blocking)
                        if result is not None:
                            return result
                    elif blocking in rec_stack:
                        # Found cycle - extract it
                        cycle_start = path.index(blocking)
                        return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in list(self.edges.keys()):
            if node not in visited:
                cycle = dfs(node)
                if cycle is not None:
                    # Build DeadlockInfo
                    agents = cycle
                    resources = []
                    cycle_edges = []

                    for i, agent in enumerate(cycle):
                        next_agent = cycle[(i + 1) % len(cycle)]
                        if agent in self.edges:
                            for blocking, resource in self.edges[agent]:
                                if blocking == next_agent:
                                    resources.append(resource)
                                    cycle_edges.append((agent, blocking, resource))
                                    break

                    return DeadlockInfo(
                        agents=agents,
                        resources=resources,
                        cycle=cycle_edges,
                    )

        return None

    def get_blocking_chain(self, agent: str) -> list[str]:
        """Get the chain of agents blocking this agent."""
        chain = [agent]
        current = agent
        visited = {agent}

        while current in self.edges and self.edges[current]:
            blocking = self.edges[current][0][0]  # First blocking agent
            if blocking in visited:
                break  # Cycle detected
            chain.append(blocking)
            visited.add(blocking)
            current = blocking

        return chain

    def clear(self) -> None:
        """Clear all dependencies."""
        self.edges.clear()
