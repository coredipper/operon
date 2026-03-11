"""Tests for Plasmid Registry / Horizontal Gene Transfer (Paper §6.2).

Dynamic tool acquisition from a searchable registry with capability gating.
"""

import pytest

from operon_ai.core.types import Capability
from operon_ai.organelles.mitochondria import Mitochondria, SimpleTool
from operon_ai.organelles.plasmid import (
    AcquisitionResult,
    Plasmid,
    PlasmidError,
    PlasmidRegistry,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_plasmid(
    name: str = "reverse",
    tags: frozenset[str] | None = None,
    caps: frozenset[Capability] | None = None,
) -> Plasmid:
    return Plasmid(
        name=name,
        description=f"A {name} tool",
        func=lambda s: s[::-1],
        tags=tags or frozenset({"text", "utility"}),
        required_capabilities=caps or frozenset(),
    )


# ── Plasmid ──────────────────────────────────────────────────────

class TestPlasmid:
    def test_creation(self):
        p = _make_plasmid()
        assert p.name == "reverse"
        assert p.version == "1.0.0"
        assert "text" in p.tags

    def test_frozen(self):
        p = _make_plasmid()
        with pytest.raises(AttributeError):
            p.name = "other"  # type: ignore[misc]

    def test_to_tool(self):
        p = _make_plasmid()
        tool = p.to_tool()
        assert isinstance(tool, SimpleTool)
        assert tool.name == "reverse"
        assert tool.execute("abc") == "cba"

    def test_to_tool_capabilities(self):
        p = _make_plasmid(caps=frozenset({Capability.NET}))
        tool = p.to_tool()
        assert Capability.NET in tool.required_capabilities


# ── PlasmidRegistry ──────────────────────────────────────────────

class TestPlasmidRegistry:
    def test_register_and_get(self):
        registry = PlasmidRegistry()
        p = _make_plasmid()
        registry.register(p)
        assert registry.get("reverse") is p

    def test_duplicate_raises(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())
        with pytest.raises(PlasmidError, match="already registered"):
            registry.register(_make_plasmid())

    def test_unknown_raises(self):
        registry = PlasmidRegistry()
        with pytest.raises(PlasmidError, match="Unknown plasmid"):
            registry.get("nonexistent")

    def test_unregister(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())
        registry.unregister("reverse")
        assert "reverse" not in registry

    def test_unregister_unknown(self):
        registry = PlasmidRegistry()
        with pytest.raises(PlasmidError, match="Unknown plasmid"):
            registry.unregister("nonexistent")

    def test_list_available(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("alpha"))
        registry.register(_make_plasmid("beta"))
        items = registry.list_available()
        assert len(items) == 2
        names = {i["name"] for i in items}
        assert names == {"alpha", "beta"}

    def test_search_by_name(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("reverse"))
        registry.register(_make_plasmid("uppercase"))
        results = registry.search("reverse")
        assert len(results) == 1
        assert results[0].name == "reverse"

    def test_search_by_description(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("alpha"))
        results = registry.search("alpha tool")
        assert len(results) == 1

    def test_search_by_tags(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("a", tags=frozenset({"math"})))
        registry.register(_make_plasmid("b", tags=frozenset({"text"})))
        results = registry.search("A", tags={"math"})
        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_by_tags_only(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("a", tags=frozenset({"math"})))
        registry.register(_make_plasmid("b", tags=frozenset({"text"})))
        results = registry.search(tags={"math"})
        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_case_insensitive(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid("REVERSE"))
        results = registry.search("reverse")
        assert len(results) == 1

    def test_search_no_results(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())
        results = registry.search("nonexistent")
        assert len(results) == 0

    def test_len(self):
        registry = PlasmidRegistry()
        assert len(registry) == 0
        registry.register(_make_plasmid())
        assert len(registry) == 1

    def test_contains(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())
        assert "reverse" in registry
        assert "unknown" not in registry


# ── Horizontal Gene Transfer (Mitochondria integration) ──────────

class TestHGT:
    def test_acquire_success(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        result = mito.acquire("reverse", registry)
        assert result.success
        assert result.plasmid_name == "reverse"
        assert "reverse" in mito.tools

    def test_acquire_and_execute(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)

        r = mito.metabolize('reverse("hello")')
        assert r.success
        assert r.atp is not None
        assert r.atp.value == "olleh"

    def test_acquire_unknown_plasmid(self):
        registry = PlasmidRegistry()
        mito = Mitochondria(silent=True)
        result = mito.acquire("nonexistent", registry)
        assert not result.success
        assert "Unknown plasmid" in (result.error or "")

    def test_acquire_insufficient_capabilities(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid(caps=frozenset({Capability.NET})))

        mito = Mitochondria(
            silent=True,
            allowed_capabilities={Capability.READ_FS},  # No NET
        )
        result = mito.acquire("reverse", registry)
        assert not result.success
        assert "Insufficient capabilities" in (result.error or "")

    def test_acquire_sufficient_capabilities(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid(caps=frozenset({Capability.NET})))

        mito = Mitochondria(
            silent=True,
            allowed_capabilities={Capability.NET, Capability.READ_FS},
        )
        result = mito.acquire("reverse", registry)
        assert result.success

    def test_acquire_duplicate_fails(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)
        result = mito.acquire("reverse", registry)
        assert not result.success
        assert "already present" in (result.error or "")

    def test_acquire_unrestricted_capabilities(self):
        """When allowed_capabilities is None, any plasmid can be acquired."""
        registry = PlasmidRegistry()
        registry.register(_make_plasmid(caps=frozenset({Capability.NET, Capability.EXEC_CODE})))

        mito = Mitochondria(silent=True)  # No capability restriction
        result = mito.acquire("reverse", registry)
        assert result.success

    def test_release(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)
        mito.release("reverse")
        assert "reverse" not in mito.tools

    def test_release_nonexistent(self):
        mito = Mitochondria(silent=True)
        with pytest.raises(ValueError, match="not present"):
            mito.release("nonexistent")

    def test_release_and_reacquire(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)
        mito.release("reverse")
        result = mito.acquire("reverse", registry)
        assert result.success
        assert "reverse" in mito.tools

    def test_export_schemas_includes_acquired(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)

        schemas = mito.export_tool_schemas()
        names = [s.name for s in schemas]
        assert "reverse" in names

    def test_release_removes_from_schemas(self):
        registry = PlasmidRegistry()
        registry.register(_make_plasmid())

        mito = Mitochondria(silent=True)
        mito.acquire("reverse", registry)
        mito.release("reverse")

        schemas = mito.export_tool_schemas()
        names = [s.name for s in schemas]
        assert "reverse" not in names
