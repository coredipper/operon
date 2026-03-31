"""
Example 106 -- Workflow Generation Interface (Phase C7)
========================================================

Demonstrates the WorkflowGenerator protocol: create a HeuristicGenerator,
generate a template from natural language, register it in a PatternLibrary,
and call refine().

Usage:
    python examples/106_workflow_generation_interface.py
"""

from operon_ai.convergence.workflow_generation import (
    HeuristicGenerator,
    WorkflowGenerator,
    generate_and_register,
)
from operon_ai.patterns.repository import PatternLibrary

# ---------------------------------------------------------------------------
# 1. Create HeuristicGenerator and verify Protocol conformance
# ---------------------------------------------------------------------------

gen = HeuristicGenerator()
print("=== Protocol conformance ===")
print(f"  isinstance(gen, WorkflowGenerator): {isinstance(gen, WorkflowGenerator)}")
assert isinstance(gen, WorkflowGenerator)

# ---------------------------------------------------------------------------
# 2. Generate template from natural language
# ---------------------------------------------------------------------------

task = "Summarize a PDF document, extract key findings, and produce a slide deck."
template = gen.generate(task)

print(f"\n=== Generated template ===")
print(f"  template_id: {template.template_id}")
print(f"  name:        {template.name}")
print(f"  topology:    {template.topology}")
print(f"  stages:      {len(template.stage_specs)}")
for spec in template.stage_specs:
    print(f"    - {spec['name']} ({spec['mode']})")
print(f"  fingerprint: shape={template.fingerprint.task_shape}, "
      f"subtasks={template.fingerprint.subtask_count}, "
      f"roles={template.fingerprint.required_roles}")

assert len(template.stage_specs) > 0, "Template must have stages"
assert template.topology == "skill_organism"

# ---------------------------------------------------------------------------
# 3. Register in PatternLibrary via generate_and_register()
# ---------------------------------------------------------------------------

library = PatternLibrary()
registered = generate_and_register(gen, task, library)

print(f"\n=== Library after generate_and_register ===")
summary = library.summary()
print(f"  template_count: {summary['template_count']}")
print(f"  topologies:     {summary['topologies']}")
assert summary["template_count"] == 1
assert library.get_template(registered.template_id) is not None

# ---------------------------------------------------------------------------
# 4. Verify library contains the template
# ---------------------------------------------------------------------------

retrieved = library.retrieve_templates(topology="skill_organism")
print(f"\n=== retrieve_templates(topology='skill_organism') ===")
print(f"  found: {len(retrieved)} template(s)")
assert len(retrieved) == 1
assert retrieved[0].template_id == registered.template_id

# ---------------------------------------------------------------------------
# 5. Call refine() -- reference impl returns unchanged
# ---------------------------------------------------------------------------

refined = gen.refine(template, feedback="Add an error-handling step.")
print(f"\n=== refine() ===")
print(f"  same object: {refined is template}")
assert refined is template, "HeuristicGenerator.refine() should return template unchanged"

print("\n--- all assertions passed ---")
