"""
Optic-Based Wiring: Lens, Prism, Traversal
============================================

Paper §3.4: Wire-level optics beyond basic lenses.

Biological Analogy:
- Lens  = constitutive expression — always active, passes through
- Prism = receptor specificity — only responds to matching ligand type
- Traversal = polymerase processivity — walks along a sequence,
  applying a transformation to each element

Current wiring uses lens-like PortType checks (type + integrity match)
but has no mechanism for:
- Conditional routing by DataType  (prism)
- Collection processing on wires   (traversal)

This module adds three optic types plus composition, all backward
compatible with existing wiring — a wire without an optic behaves
exactly as before.

References:
- Article Section 3.4: Wire-Level Optics - Beyond Lenses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .types import DataType, IntegrityLabel


class OpticError(ValueError):
    """Raised when an optic rejects data it cannot transmit."""


@runtime_checkable
class Optic(Protocol):
    """
    Protocol for wire-level optics.

    An optic decides whether data can flow through a wire
    (can_transmit) and optionally transforms it (transmit).
    """

    @property
    def name(self) -> str: ...

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool: ...

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Concrete optics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LensOptic:
    """
    Pass-through optic (current behavior made explicit).

    Always transmits, never transforms.  Equivalent to a wire
    with no optic — useful when you want to be explicit about
    the optic choice in a diagram that mixes optic types.
    """

    @property
    def name(self) -> str:
        return "lens"

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool:
        return True

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any:
        return value


@dataclass(frozen=True)
class PrismOptic:
    """
    Conditional routing optic — only transmits matching DataTypes.

    Rejects data whose DataType is not in the accepted set.
    When used with fan-out wiring, different prisms on different
    wires route data to the correct destination based on type.

    Example:
        >>> prism = PrismOptic(accept=frozenset({DataType.JSON}))
        >>> prism.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)
        True
        >>> prism.can_transmit(DataType.ERROR, IntegrityLabel.VALIDATED)
        False
    """

    accept: frozenset[DataType]

    @property
    def name(self) -> str:
        types = ", ".join(sorted(dt.value for dt in self.accept))
        return f"prism({types})"

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool:
        return data_type in self.accept

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any:
        if data_type not in self.accept:
            raise OpticError(
                f"PrismOptic rejects {data_type.value}; "
                f"accepted: {[dt.value for dt in self.accept]}"
            )
        return value


@dataclass(frozen=True)
class TraversalOptic:
    """
    Collection-processing optic — maps a transform over list elements.

    If no transform is provided, acts as a pass-through.
    Works on both lists and single values (single values are
    treated as one-element collections).

    Example:
        >>> t = TraversalOptic(transform=str.upper)
        >>> t.transmit(["hello", "world"], DataType.JSON, IntegrityLabel.VALIDATED)
        ['HELLO', 'WORLD']
    """

    transform: Callable[[Any], Any] | None = None

    @property
    def name(self) -> str:
        return "traversal"

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool:
        return True

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any:
        if self.transform is None:
            return value
        if isinstance(value, list):
            return [self.transform(item) for item in value]
        return self.transform(value)


@dataclass
class BudgetOptic:
    """
    Cost-aware optic — blocks transmission if cumulative cost exceeds budget.

    Tracks cumulative wire cost across transmissions. When the running
    total exceeds max_cost, further transmissions are blocked.

    Biological Analogy: ATP budget cap on a signaling pathway — once
    the cell has spent its energy budget on a pathway, further signal
    transduction is inhibited.

    Example:
        >>> budget = BudgetOptic(max_cost=10)
        >>> budget.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)
        True
        >>> budget.add_cost(8)
        >>> budget.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)
        True
        >>> budget.add_cost(5)
        >>> budget.can_transmit(DataType.JSON, IntegrityLabel.VALIDATED)
        False
    """

    max_cost: int
    _spent: int = 0

    @property
    def name(self) -> str:
        return f"budget({self._spent}/{self.max_cost})"

    @property
    def remaining(self) -> int:
        return max(0, self.max_cost - self._spent)

    def add_cost(self, cost: int) -> None:
        """Record cost spent through this optic."""
        self._spent += cost

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool:
        return self._spent < self.max_cost

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any:
        if self._spent >= self.max_cost:
            raise OpticError(
                f"BudgetOptic exhausted: spent {self._spent} of {self.max_cost}"
            )
        return value


@dataclass(frozen=True)
class ComposedOptic:
    """
    Sequential composition of optics.

    All optics must accept (can_transmit) for the composed optic
    to accept.  Transforms are applied left-to-right.

    Example:
        >>> composed = ComposedOptic(optics=(
        ...     PrismOptic(accept=frozenset({DataType.JSON})),
        ...     TraversalOptic(transform=str.upper),
        ... ))
    """

    optics: tuple[Optic, ...]

    @property
    def name(self) -> str:
        if not self.optics:
            return "composed()"
        return "composed(" + " | ".join(o.name for o in self.optics) + ")"

    def can_transmit(self, data_type: DataType, integrity: IntegrityLabel) -> bool:
        return all(o.can_transmit(data_type, integrity) for o in self.optics)

    def transmit(
        self, value: Any, data_type: DataType, integrity: IntegrityLabel
    ) -> Any:
        result = value
        for optic in self.optics:
            result = optic.transmit(result, data_type, integrity)
        return result
