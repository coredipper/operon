# Release Notes

This page tracks the recent direction of the project.

## v0.19.1

Focus:

- bi-temporal memory integration with SkillOrganism
- three-layer context model (topology, ephemeral, bi-temporal)
- HuggingFace Space for bi-temporal memory explorer

New:

- `SubstrateView` — frozen read-only envelope for substrate queries
- `SkillStage` fields: `read_query`, `fact_extractor`, `emit_output_fact`, `fact_tags`
- `SkillOrganism.substrate` — optional `BiTemporalMemory` for auditable shared facts
- `examples/71_bitemporal_skill_organism.py` — enterprise workflow with substrate
- [Bi-Temporal Memory Space](https://huggingface.co/spaces/coredipper/operon-bitemporal)
- Article updates: three-layer context model (§6), substrate integration (§8)

## v0.19.0

Focus:

- bi-temporal memory (valid time vs record time)
- append-only correction semantics
- belief-state reconstruction for compliance auditing
- article updates: temporal databases, temporal coalgebra, temporal epistemics

New:

- `BiTemporalMemory`, `BiTemporalFact`, `BiTemporalQuery`, `FactSnapshot`, `CorrectionResult`
- `examples/69_bitemporal_memory.py` — core API demo
- `examples/70_bitemporal_compliance_audit.py` — enterprise audit scenario
- [Bi-Temporal Memory docs](../bitemporal-memory/)

## v0.18

Focus:

- thinner front door
- pattern-first API
- provider-bound skill organisms
- attachable telemetry

Related writing:

- [Medium: Operon v0.18](https://medium.com/@coredipper/operon-v0-18-the-point-where-it-started-feeling-usable-b284d7b7317f)
- [Blog: Operon v0.18](https://banu.be/blog/operon-v018-pattern-first-skills/)

## v0.17

Focus:

- epistemic topology
- architecture-level analysis
- practical comparison to Kim et al.

Related writing:

- [Blog: Operon v0.17](https://banu.be/blog/operon-v017-epistemic-topology/)
