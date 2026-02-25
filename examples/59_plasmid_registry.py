"""
Example 59: Plasmid Registry — Horizontal Gene Transfer
========================================================

Demonstrates dynamic tool acquisition from a searchable registry,
implementing Paper §6.2, Eq. 12: Agent_new = Agent_old ⊗ ToolSchema.

Biological Analogy:
- Plasmid = small circular DNA encoding a useful gene (tool)
- Registry = environmental pool of plasmids
- Acquisition = bacterial conjugation / transformation
- Capability gating = restriction enzymes (only compatible plasmids accepted)
- Release = plasmid curing

Key points:
1. Tools are no longer static at construction time
2. Agents can dynamically discover and acquire new capabilities
3. Capability gating prevents privilege escalation
4. Released tools are cleanly removed

References:
- Article Section 6.2: Horizontal Gene Transfer
"""

from operon_ai.core.types import Capability
from operon_ai.organelles.mitochondria import Mitochondria
from operon_ai.organelles.plasmid import Plasmid, PlasmidRegistry


def main():
    try:
        print("=" * 60)
        print("Plasmid Registry — Horizontal Gene Transfer")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Create a Plasmid Registry
        # =================================================================
        print("\n--- Section 1: Plasmid Registry ---")

        registry = PlasmidRegistry()

        registry.register(Plasmid(
            name="reverse",
            description="Reverse a string",
            func=lambda s: s[::-1],
            tags=frozenset({"text", "utility"}),
        ))

        registry.register(Plasmid(
            name="word_count",
            description="Count words in text",
            func=lambda s: len(s.split()),
            tags=frozenset({"text", "analysis"}),
        ))

        registry.register(Plasmid(
            name="fetch_url",
            description="Fetch a URL (simulated)",
            func=lambda url: f"<html>Content from {url}</html>",
            tags=frozenset({"network", "io"}),
            required_capabilities=frozenset({Capability.NET}),
        ))

        print(f"  Registry size: {len(registry)} plasmids")
        for item in registry.list_available():
            caps = item["required_capabilities"] or ["none"]
            print(f"    {item['name']}: tags={item['tags']}, caps={caps}")

        # =================================================================
        # SECTION 2: Search the Registry
        # =================================================================
        print("\n--- Section 2: Search ---")

        text_tools = registry.search("text", tags={"text"})
        print(f"  Text tools: {[p.name for p in text_tools]}")

        net_tools = registry.search("fetch")
        print(f"  Network tools: {[p.name for p in net_tools]}")

        # =================================================================
        # SECTION 3: Dynamic Acquisition
        # =================================================================
        print("\n--- Section 3: Dynamic Acquisition ---")

        mito = Mitochondria(
            allowed_capabilities={Capability.READ_FS},
            silent=True,
        )
        print(f"  Initial tools: {list(mito.tools.keys())}")

        # Acquire a tool with no capability requirements
        result = mito.acquire("reverse", registry)
        print(f"  Acquire 'reverse': success={result.success}")

        result = mito.acquire("word_count", registry)
        print(f"  Acquire 'word_count': success={result.success}")

        print(f"  Tools after acquisition: {list(mito.tools.keys())}")

        # =================================================================
        # SECTION 4: Capability Gating
        # =================================================================
        print("\n--- Section 4: Capability Gating ---")

        # fetch_url requires NET, but mito only allows READ_FS
        result = mito.acquire("fetch_url", registry)
        print(f"  Acquire 'fetch_url': success={result.success}")
        print(f"  Reason: {result.error}")

        # Create a mito with NET capability
        net_mito = Mitochondria(
            allowed_capabilities={Capability.NET},
            silent=True,
        )
        result = net_mito.acquire("fetch_url", registry)
        print(f"  Acquire with NET cap: success={result.success}")

        # =================================================================
        # SECTION 5: Execute Acquired Tools
        # =================================================================
        print("\n--- Section 5: Execute Acquired Tools ---")

        r = mito.metabolize('reverse("hello world")')
        print(f"  reverse('hello world') = {r.atp.value}")

        r = mito.metabolize('word_count("the quick brown fox")')
        print(f"  word_count('the quick brown fox') = {r.atp.value}")

        # =================================================================
        # SECTION 6: Release (Plasmid Curing)
        # =================================================================
        print("\n--- Section 6: Plasmid Release ---")

        print(f"  Tools before release: {list(mito.tools.keys())}")
        mito.release("reverse")
        print(f"  Tools after releasing 'reverse': {list(mito.tools.keys())}")

        # Verify tool is gone
        r = mito.metabolize('reverse("test")')
        print(f"  Execute after release: success={r.success}")
        print(f"  Error: {r.error}")

        # Re-acquire is possible
        result = mito.acquire("reverse", registry)
        print(f"  Re-acquire 'reverse': success={result.success}")
        print(f"  Tools: {list(mito.tools.keys())}")

        # =================================================================
        # SECTION 7: Schema Export
        # =================================================================
        print("\n--- Section 7: Schema Export ---")

        schemas = mito.export_tool_schemas()
        for s in schemas:
            print(f"  {s.name}: {s.description}")

        print("\n" + "=" * 60)
        print("DONE — Plasmid Registry demonstrated successfully")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
