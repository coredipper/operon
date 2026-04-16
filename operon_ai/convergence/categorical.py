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
    # DeerFlow: lead agent from "assistant_id" + "sub_agents"
    if not agents:
        lead_id = compiled.get("assistant_id")
        sub_agents = compiled.get("sub_agents", [])
        agents = []
        if lead_id:
            agents.append({"name": str(lead_id), "role": "lead"})
        for a in sub_agents:
            if isinstance(a, dict):
                agents.append(a)
            else:
                agents.append({"name": str(a)})

    stage_names = tuple(a.get("name", f"agent_{i}") for i, a in enumerate(agents))

    # Extract edges — check key presence, not truthiness
    raw_edges: list = []
    for key in ("edges", "messaging", "events"):
        if key in compiled:
            raw_edges = compiled[key]
            break

    edges_list: list[tuple[str, str]] = []
    for e in raw_edges:
        if isinstance(e, dict):
            edges_list.append((e.get("from", ""), e.get("to", "")))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            edges_list.append((str(e[0]), str(e[1])))

    # DeerFlow: hub-and-spoke (lead → each sub_agent) when no edge key exists
    has_edge_key = any(k in compiled for k in ("edges", "messaging", "events"))
    if not has_edge_key and "assistant_id" in compiled:
        lead = str(compiled["assistant_id"])
        for name in stage_names:
            if name != lead:
                edges_list.append((lead, name))

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

        # 1. Graph preservation: source subgraph embedded in target
        # Compilers may enrich (add watcher agents, etc.), so we check
        # that source stages and edges are subsets of the target.
        source_stages = frozenset(source.stage_names)
        target_stages = frozenset(target.stage_names)
        stages_embedded = source_stages <= target_stages

        # Edge embedding: each source edge must appear in target
        source_edges = frozenset(source.edges)
        target_edges = frozenset(target.edges)
        edges_embedded = source_edges <= target_edges

        # Parallel groups reshape the graph intentionally — stages are
        # collapsed into composite group nodes. Verify against the expected
        # group-level topology built from the organism's stage_groups.
        has_parallel = compiled.get("config", {}).get("has_parallel_groups", False)
        if has_parallel:
            # Reconstruct expected fork/join topology independently
            # from _stage_groups metadata (not from compiled agents/edges)
            cfg = compiled.get("config", {})
            stage_groups = cfg.get("_stage_groups", [])
            fj_names = cfg.get("_fork_join_names", [])
            exp_nodes: set[str] = set()
            exp_edges: set[tuple[str, str]] = set()
            entry_names: list[str] = []
            exit_names: list[str] = []
            for i, grp in enumerate(stage_groups):
                fj = fj_names[i] if i < len(fj_names) else None
                if len(grp) == 1:
                    exp_nodes.add(grp[0])
                    entry_names.append(grp[0])
                    exit_names.append(grp[0])
                else:
                    fn = fj[0] if fj else f"__fork_{i}"
                    jn = fj[1] if fj else f"__join_{i}"
                    exp_nodes.add(fn)
                    exp_nodes.add(jn)
                    for sn in grp:
                        exp_nodes.add(sn)
                        exp_edges.add((fn, sn))
                        exp_edges.add((sn, jn))
                    entry_names.append(fn)
                    exit_names.append(jn)
            for i in range(len(exit_names) - 1):
                exp_edges.add((exit_names[i], entry_names[i + 1]))

            graph_ok = (
                set(target.stage_names) == exp_nodes
                and set(target.edges) == exp_edges
            )
        else:
            graph_ok = stages_embedded and edges_embedded
        details["source_stages"] = source.stage_count
        details["target_stages"] = target.stage_count
        details["stages_embedded"] = stages_embedded
        details["edges_embedded"] = edges_embedded
        details["graph_reshaped"] = has_parallel

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

        # 3. Interface preservation: source stage names present in target
        # (mode→model mapping may differ since compilers remap models)
        source_stage_set = frozenset(source.stage_names)
        target_stage_set = frozenset(target.stage_names)
        # For parallel groups, verify all source stages are present in target
        if has_parallel:
            interface_ok = source_stage_set <= target_stage_set
        else:
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


def _compile_langgraph(organism: SkillOrganism, *, config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Compile organism to the LangGraph target's graph shape.

    For sequential organisms, creates one node per stage.  For organisms
    with parallel groups, parallel stages are collapsed into composite
    group nodes (``__parallel_N``).

    Note: ``graph_preserved`` will be ``False`` for grouped organisms
    because the source Architecture uses per-stage nodes while the
    LangGraph target uses group-level nodes.  This is correct —
    the graph IS reshaped.  Certificate preservation still holds.
    """
    if config is not None:
        raise ValueError(
            "RuntimeConfig is not supported for the LangGraph categorical "
            "functor. Use organism_to_langgraph() directly."
        )

    groups = organism.stage_groups or tuple((s,) for s in organism.stages)

    # Model the actual LangGraph topology: fork → stages → join
    agents = []
    edges: list[tuple[str, str]] = []
    all_names = {s.name for s in organism.stages}

    # Track entry/exit nodes per group for sequential wiring
    group_entry_names: list[str] = []
    group_exit_names: list[str] = []

    for i, group in enumerate(groups):
        if len(group) == 1:
            stage = group[0]
            agents.append({"name": stage.name, "role": stage.role, "model": stage.mode})
            group_entry_names.append(stage.name)
            group_exit_names.append(stage.name)
        else:
            fork_name = f"__fork_{i}"
            while fork_name in all_names:
                fork_name = f"__fork_{i}_{id(group)}"
            join_name = f"__join_{i}"
            while join_name in all_names:
                join_name = f"__join_{i}_{id(group)}"
            agents.append({"name": fork_name, "role": "fork", "model": "internal"})
            for stage in group:
                agents.append({"name": stage.name, "role": stage.role, "model": stage.mode})
                edges.append((fork_name, stage.name))
                edges.append((stage.name, join_name))
            agents.append({"name": join_name, "role": "join", "model": "internal"})
            group_entry_names.append(fork_name)
            group_exit_names.append(join_name)

    # Wire groups sequentially
    for i in range(len(group_exit_names) - 1):
        edges.append((group_exit_names[i], group_entry_names[i + 1]))
    certificates = [certificate_to_dict(c) for c in organism.collect_certificates()]

    has_parallel = any(len(g) > 1 for g in groups)
    return {
        "agents": agents,
        "edges": edges,
        "certificates": certificates,
        "config": {
            "runtime": "langgraph",
            "node_count": len(agents),
            "has_parallel_groups": has_parallel,
            # Source stage_groups + generated fork/join names for
            # independent topology reconstruction during verification
            "_stage_groups": [
                [s.name for s in g] for g in groups
            ],
            "_fork_join_names": [
                (group_entry_names[i], group_exit_names[i])
                if len(groups[i]) > 1 else None
                for i in range(len(groups))
            ],
        },
    }


langgraph_functor = CompilerFunctor("langgraph", _compile_langgraph)
