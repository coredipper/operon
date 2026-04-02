"""Configuration specifications for the C6 convergence evaluation harness.

Defines 7 configurations spanning 4 frameworks, with and without Operon
structural guidance.  Each configuration specifies which compiler and
adapter to use, and whether Operon's risk-reduction guidance is active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigurationSpec:
    """A compilation/analysis configuration for evaluation."""

    config_id: str
    name: str
    framework: str
    structural_guidance: bool
    compiler_fn: str | None
    adapter_fn: str | None
    tags: tuple[str, ...]


def get_configurations() -> list[ConfigurationSpec]:
    """Return all 7 evaluation configurations."""
    return [
        ConfigurationSpec(
            config_id="swarms_baseline",
            name="Swarms Baseline",
            framework="swarms",
            structural_guidance=False,
            compiler_fn="organism_to_swarms",
            adapter_fn="parse_swarm_topology",
            tags=("swarms", "baseline"),
        ),
        ConfigurationSpec(
            config_id="swarms_operon",
            name="Swarms + Operon Guidance",
            framework="swarms",
            structural_guidance=True,
            compiler_fn="organism_to_swarms",
            adapter_fn="parse_swarm_topology",
            tags=("swarms", "guided"),
        ),
        ConfigurationSpec(
            config_id="deerflow_baseline",
            name="DeerFlow Baseline",
            framework="deerflow",
            structural_guidance=False,
            compiler_fn="organism_to_deerflow",
            adapter_fn="parse_deerflow_session",
            tags=("deerflow", "baseline"),
        ),
        ConfigurationSpec(
            config_id="ralph_baseline",
            name="Ralph Baseline",
            framework="ralph",
            structural_guidance=False,
            compiler_fn="organism_to_ralph",
            adapter_fn="parse_ralph_config",
            tags=("ralph", "baseline"),
        ),
        ConfigurationSpec(
            config_id="scion_baseline",
            name="Scion Baseline",
            framework="scion",
            structural_guidance=False,
            compiler_fn="organism_to_scion",
            adapter_fn=None,
            tags=("scion", "baseline"),
        ),
        ConfigurationSpec(
            config_id="scion_operon",
            name="Scion + Operon Guidance",
            framework="scion",
            structural_guidance=True,
            compiler_fn="organism_to_scion",
            adapter_fn=None,
            tags=("scion", "guided"),
        ),
        ConfigurationSpec(
            config_id="operon_adaptive",
            name="Operon Adaptive (no external framework)",
            framework="operon",
            structural_guidance=True,
            compiler_fn=None,
            adapter_fn=None,
            tags=("operon", "adaptive", "guided"),
        ),
    ]
