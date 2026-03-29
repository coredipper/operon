"""
Example 92 — Memory Bridge
=============================

Bridges external memory formats (AnimaWorks, DeerFlow) into Operon's
BiTemporalMemory for auditable fact tracking.

Usage:
    python examples/92_memory_bridge.py
"""

from operon_ai import BiTemporalMemory
from operon_ai.convergence import bridge_animaworks_memory, bridge_deerflow_memory

mem = BiTemporalMemory()

# AnimaWorks memories
animaworks_entries = [
    {"id": "mem_001", "type": "episodic", "content": "User prefers Python", "timestamp": "2026-03-01T10:00:00", "source_agent": "assistant"},
    {"id": "mem_002", "type": "semantic", "content": "Project uses FastAPI", "timestamp": "2026-03-01T11:00:00", "source_agent": "researcher"},
]
aw_facts = bridge_animaworks_memory(animaworks_entries, mem)
print(f"Bridged {len(aw_facts)} AnimaWorks facts")

# DeerFlow session + vector store
session = [
    {"role": "user", "content": "Find AI scaling papers", "timestamp": "2026-03-01T12:00:00"},
    {"role": "assistant", "content": "Found 3 relevant papers", "timestamp": "2026-03-01T12:01:00"},
]
vectors = [
    {"id": "vec_001", "content": "Scaling laws for neural LMs", "metadata": {"source": "arxiv"}, "inserted_at": "2026-03-01T12:02:00"},
]
df_facts = bridge_deerflow_memory(session, vectors, mem)
print(f"Bridged {len(df_facts)} DeerFlow facts")

# Query the unified store
all_facts = mem.retrieve_known_at(at=mem._facts[-1].recorded_from)
print(f"\nTotal facts in bi-temporal store: {len(all_facts)}")
for f in all_facts[:5]:
    print(f"  [{f.source}] {f.subject}: {f.value}")

# --test
assert len(aw_facts) == 2
assert len(df_facts) == 3  # 2 session + 1 vector
assert len(all_facts) == 5
print("\n--- all assertions passed ---")
