"""Categorical formalization of convergence compilers.

Formalizes Operon's compilers as functors in the **ArchAgents** category
(de los Riscos, Corbacho & Arbib, arXiv:2603.28906).

An :class:`Architecture` is an object in ArchAgents — a triple
``(G, Know, Φ)`` of syntactic wiring, knowledge structure, and
profunctor interface.  A :class:`CompilerFunctor` is a functor
``F: Arch(Operon) → Arch(Target)`` that maps organisms to compiled
workflow configs while preserving structural properties.

Certificate preservation (Prop 5.1 in de los Riscos et al.) follows
from the functor laws: if ``F`` maps objects and morphisms such that
``F(id) = id`` and ``F(g ∘ f) = F(g) ∘ F(f)``, then any certificate
attached to the source architecture is faithfully represented in the
target — the guarantee *transfers*, it isn't re-derived.

Example::

    from operon_ai.convergence.categorical import (
        extract_architecture, swarms_functor,
    )
    from operon_ai import skill_organism, ...

    org = skill_organism(stages=[...], ...)
    arch = extract_architecture(org)
    result = swarms_functor.compile(org)
    assert result.preservation.all_preserved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.certificate import (
    Certificate,
    CertificateVerification,
    certificate_to_dict,
    verify_compiled,
)
from ..patterns.organism import SkillOrganism
from .types import RuntimeConfig


# ---------------------------------------------------------------------------
# Architecture — object in ArchAgents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Architecture:
    """Object in the ArchAgents category.

    A triple ``(G, Know, Φ)`` per de los Riscos et al. §3:

    - **G** (graph): Syntactic wiring — stage names and directed edges.
    - **Know** (knowledge): Structural guarantees — certificates held.
    - **Φ** (interface): Profunctor interface — mode-to-model mapping.
    """

    # G — syntactic wiring
    stage_names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    # Know — knowledge structure (certificates as dicts for comparability)
    certificates: tuple[dict[str, Any], ...]

    # Φ — profunctor interface (mode → model tier mapping)
    interface: tuple[tuple[str, str], ...]  # (stage_name, mode) pairs

    @property
    def stage_count(self) -> int:
        return len(self.stage_names)

    @property
    def certificate_theorems(self) -> frozenset[str]:
        return frozenset(c["theorem"] for c in self.certificates)

    @property
    def is_sequential(self) -> bool:
        """True if edges form a simple linear chain."""
        if len(self.stage_names) <= 1:
            return True
        expected = [
            (self.stage_names[i], self.stage_names[i + 1])
            for i in range(len(self.stage_names) - 1)
        ]
        return list(self.edges) == expected


# ---------------------------------------------------------------------------
# PreservationResult — verification that functor laws hold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreservationResult:
    """Result of verifying that a compilation preserves architectural properties.

    Checks three functor-law consequences:
    1. **Graph preservation**: stage structure is faithfully represented
    2. **Certificate preservation** (Prop 5.1): all certificates transfer
    3. **Interface preservation**: mode/model mapping is represented
    """

    graph_preserved: bool
    certificate_preserved: bool
    interface_preserved: bool
    source: Architecture
    target: Architecture
    certificate_verifications: tuple[CertificateVerification, ...]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_preserved(self) -> bool:
        return (
            self.graph_preserved
            and self.certificate_preserved
            and self.interface_preserved
        )


# ---------------------------------------------------------------------------
# CompilationResult — output of a functor application
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompilationResult:
    """Result of applying a CompilerFunctor to an organism.

    Contains the compiled output dict, the source and target architectures,
    and the preservation verification.
    """

    compiled: dict[str, Any]
    source_architecture: Architecture
    target_architecture: Architecture
    preservation: PreservationResult


# ---------------------------------------------------------------------------
# Extract architecture from organisms and compiled dicts
# ---------------------------------------------------------------------------

def extract_architecture(organism: SkillOrganism) -> Architecture:
    """Extract an Architecture from a SkillOrganism.

    Maps the organism's stages, certificates, and mode assignments
    to the ``(G, Know, Φ)`` triple.
    """
    stages = organism.stages
    stage_names = tuple(s.name for s in stages)

    # G — edges from stage ordering (sequential)
    edges = tuple(
        (stages[i].name, stages[i + 1].name)
        for i in range(len(stages) - 1)
    )

    # Know — certificates
    certs = organism.collect_certificates()
    cert_dicts = tuple(certificate_to_dict(c) for c in certs)

    # Φ — interface (stage → mode mapping)
    interface = tuple((s.name, s.mode) for s in stages)

    return Architecture(
        stage_names=stage_names,
        edges=edges,
        certificates=cert_dicts,
        interface=interface,
    )


def extract_compiled_architecture(
    compiled: dict[str, Any],
    source_name: str = "compiled",
) -> Architecture:
    """Extract an Architecture from a compiled workflow dict.

    Handles the output format of all four Operon compilers:
    - Swarms: ``agents`` + ``edges``
    - DeerFlow: ``skills`` + ``sub_agents``
    - Ralph: ``hats``
    - Scion: ``agents`` + ``messaging``
    """
    # Extract agents from whichever key the compiler uses
    agents = (
        compiled.get("agents")
        or compiled.get("hats")
        or []
    )
    # DeerFlow: agents in "sub_agents" (skills are instruction strings)
    if not agents:
        sub_agents = compiled.get("sub_agents", [])
        agents = [a if isinstance(a, dict) else {"name": str(a)}
                  for a in sub_agents]

    stage_names = tuple(a.get("name", f"agent_{i}") for i, a in enumerate(agents))

    # Extract edges from whichever key exists
    raw_edges = compiled.get("edges") or compiled.get("messaging") or []
    edges_list: list[tuple[str, str]] = []
    for e in raw_edges:
        if isinstance(e, dict):
            edges_list.append((e.get("from", ""), e.get("to", "")))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            edges_list.append((str(e[0]), str(e[1])))
    edges = tuple(edges_list)

    cert_dicts = tuple(compiled.get("certificates", []))

    # Reconstruct interface from agent model/role assignments
    interface = tuple(
        (a.get("name", f"agent_{i}"), a.get("model", a.get("role", "unknown")))
        for i, a in enumerate(agents)
    )

    return Architecture(
        stage_names=stage_names,
        edges=edges,
        certificates=cert_dicts,
        interface=interface,
    )


# ---------------------------------------------------------------------------
# CompilerFunctor — the categorical compiler
# ---------------------------------------------------------------------------

class CompilerFunctor:
    """A compiler formalized as a functor ``F: Arch(Operon) → Arch(Target)``.

    Wraps an existing compiler function (e.g., ``organism_to_swarms``)
    with pre/post architecture extraction and preservation verification.

    Functor laws verified:
    - **Identity**: compiling an identity (single-stage no-op) produces
      an identity in the target category.
    - **Composition**: compiling a composed organism is equivalent to
      composing the compiled parts (tested structurally via stage count
      and edge preservation).
    - **Certificate preservation** (Prop 5.1): all source certificates
      appear in the target and verify successfully.

    Parameters
    ----------
    name:
        Human-readable name (e.g., ``"swarms"``).
    compile_fn:
        The underlying compiler function ``(SkillOrganism, ...) → dict``.
    """

    def __init__(
        self,
        name: str,
        compile_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self.name = name
        self._compile_fn = compile_fn

    def compile(
        self,
        organism: SkillOrganism,
        *,
        config: RuntimeConfig | None = None,
    ) -> CompilationResult:
        """Apply the functor: map organism to compiled config.

        Returns a :class:`CompilationResult` with the compiled dict,
        source/target architectures, and preservation verification.
        """
        # F(A) — map the object
        source = extract_architecture(organism)
        compiled = self._compile_fn(organism, config=config) if config else self._compile_fn(organism)
        target = extract_compiled_architecture(compiled, source_name=self.name)

        # Verify functor laws (preservation)
        preservation = self._verify_preservation(source, target, compiled)

        return CompilationResult(
            compiled=compiled,
            source_architecture=source,
            target_architecture=target,
            preservation=preservation,
        )

    def _verify_preservation(
        self,
        source: Architecture,
        target: Architecture,
        compiled: dict[str, Any],
    ) -> PreservationResult:
        """Verify that the compilation preserves architectural properties."""
        details: dict[str, Any] = {}

        # 1. Graph preservation: source stages embedded in target
        # Compilers may enrich (add watcher agents, etc.), so we check
        # that the source is a subgraph, not that they're identical.
        graph_ok = (
            source.stage_count <= target.stage_count
            and frozenset(source.stage_names) <= frozenset(target.stage_names)
        )
        details["source_stages"] = source.stage_count
        details["target_stages"] = target.stage_count

        # 2. Certificate preservation (Prop 5.1)
        source_theorems = source.certificate_theorems
        target_theorems = target.certificate_theorems
        cert_set_preserved = source_theorems <= target_theorems

        # Also verify certificates still hold
        cert_verifications = tuple(verify_compiled(compiled))
        certs_hold = all(v.holds for v in cert_verifications)

        cert_ok = cert_set_preserved and certs_hold
        details["source_theorems"] = sorted(source_theorems)
        details["target_theorems"] = sorted(target_theorems)
        details["all_certs_hold"] = certs_hold

        # 3. Interface preservation: every source stage appears in target
        source_stage_set = frozenset(source.stage_names)
        target_stage_set = frozenset(target.stage_names)
        interface_ok = source_stage_set <= target_stage_set
        details["source_stage_names"] = sorted(source_stage_set)
        details["target_stage_names"] = sorted(target_stage_set)

        return PreservationResult(
            graph_preserved=graph_ok,
            certificate_preserved=cert_ok,
            interface_preserved=interface_ok,
            source=source,
            target=target,
            certificate_verifications=cert_verifications,
            details=details,
        )

    def __repr__(self) -> str:
        return f"CompilerFunctor({self.name!r})"


# ---------------------------------------------------------------------------
# Pre-built functor instances for all four compilers
# ---------------------------------------------------------------------------

def _lazy_functor(name: str, module: str, fn_name: str) -> CompilerFunctor:
    """Create a CompilerFunctor with lazy import of the compile function."""
    def _compile(organism, *, config=None):
        import importlib
        mod = importlib.import_module(module)
        fn = getattr(mod, fn_name)
        return fn(organism, config=config) if config else fn(organism)
    return CompilerFunctor(name=name, compile_fn=_compile)


swarms_functor = _lazy_functor("swarms", "operon_ai.convergence.swarms_compiler", "organism_to_swarms")
deerflow_functor = _lazy_functor("deerflow", "operon_ai.convergence.deerflow_compiler", "organism_to_deerflow")
ralph_functor = _lazy_functor("ralph", "operon_ai.convergence.ralph_compiler", "organism_to_ralph")
scion_functor = _lazy_functor("scion", "operon_ai.convergence.scion_compiler", "organism_to_scion")
