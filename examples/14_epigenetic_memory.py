"""
Example 14: Epigenetic Memory System
====================================

This example demonstrates the HistoneStore with:
- Multiple marker types (methylation, acetylation, phosphorylation)
- Marker decay and expiration
- Context-based memory recall
- Memory inheritance to child agents

Biological Analogy:
Epigenetic modifications (like histone methylation) don't change DNA
but affect how genes are expressed. Similarly, our histone store
maintains memories that influence agent behavior without changing
the core configuration.
"""

from operon_ai.state import (
    HistoneStore,
    MarkerType,
    MarkerStrength,
)


def demonstrate_marker_types():
    """Demonstrate different marker types."""
    print("\n" + "="*60)
    print("1. MARKER TYPES AND PERSISTENCE")
    print("="*60)

    store = HistoneStore(silent=True)

    # Add different types of markers
    print("\n--- Adding Different Marker Types ---")

    # Methylation: Permanent, strong memory
    hash1 = store.methylate(
        "Never run DELETE without WHERE clause - critical safety rule",
        strength=MarkerStrength.STRONG,
        tags=["safety", "sql", "critical"]
    )
    print(f"  Methylation: SQL safety rule (permanent) -> {hash1}")

    # Acetylation: Temporary activation
    hash2 = store.acetylate(
        "User prefers verbose output with detailed explanations",
        decay_hours=168,  # 1 week
        strength=MarkerStrength.MODERATE,
        tags=["preference", "output"]
    )
    print(f"  Acetylation: User preference (1 week) -> {hash2}")

    # Phosphorylation: Transient signal
    hash3 = store.phosphorylate(
        "Currently working on machine learning project",
        decay_hours=24,  # 1 day
        tags=["context", "project"]
    )
    print(f"  Phosphorylation: Current context (1 day) -> {hash3}")

    # Ubiquitination: Tagged for processing
    hash4 = store.ubiquitinate(
        "Temporary cache entry for API response",
        tags=["cache", "temporary"]
    )
    print(f"  Ubiquitination: Cache entry (1 hour) -> {hash4}")

    return store


def demonstrate_context_recall():
    """Demonstrate context-based memory recall."""
    print("\n" + "="*60)
    print("2. CONTEXT-BASED MEMORY RECALL")
    print("="*60)

    store = HistoneStore(silent=True)

    # Add various memories
    memories = [
        ("Always validate user input before database queries", ["security", "database"]),
        ("Use parameterized queries to prevent SQL injection", ["security", "sql"]),
        ("Log all authentication attempts for audit", ["security", "logging"]),
        ("User prefers Python over JavaScript", ["preference", "language"]),
        ("Current project uses PostgreSQL database", ["context", "database"]),
        ("Team follows PEP8 style guidelines", ["style", "python"]),
    ]

    print("\n--- Adding Memories ---")
    for lesson, tags in memories:
        store.methylate(lesson, tags=tags)
        print(f"  Added: {lesson[:40]}... (tags: {', '.join(tags)})")

    # Recall by query
    print("\n--- Recall by Query: 'database security' ---")
    result = store.retrieve_context(query="database security")
    print(f"  Found {len(result.markers)} relevant markers:")
    for marker in result.markers[:3]:
        print(f"    - {marker.content[:50]}...")
    print(f"\n  Formatted Context:\n{result.formatted_context}")

    # Recall by tags
    print("\n--- Recall by Tags: ['security'] ---")
    result = store.retrieve_context(tags=["security"])
    print(f"  Found {len(result.markers)} markers with security tag")

    # Recall by marker type
    print("\n--- Recall Only Methylation (permanent) Markers ---")
    result = store.retrieve_context(marker_types=[MarkerType.METHYLATION])
    print(f"  Found {len(result.markers)} permanent markers")

    return store


def demonstrate_marker_strength():
    """Demonstrate marker strength levels."""
    print("\n" + "="*60)
    print("3. MARKER STRENGTH LEVELS")
    print("="*60)

    store = HistoneStore(silent=True)

    # Add markers with different strengths
    print("\n--- Adding Markers with Different Strengths ---")

    store.methylate("Critical: Never expose API keys",
                   strength=MarkerStrength.PERMANENT, tags=["critical"])
    print("  PERMANENT: API key protection")

    store.methylate("Important: Always use HTTPS",
                   strength=MarkerStrength.STRONG, tags=["important"])
    print("  STRONG: HTTPS requirement")

    store.acetylate("Moderate: User likes dark mode",
                   strength=MarkerStrength.MODERATE, tags=["preference"])
    print("  MODERATE: Dark mode preference")

    store.phosphorylate("Weak: Temporary debugging note", tags=["temp"])
    print("  WEAK: Debug note")

    # Recall with minimum strength filter
    print("\n--- Recall with Minimum STRONG Strength ---")
    result = store.retrieve_context(min_strength=MarkerStrength.STRONG)
    print(f"  Found {len(result.markers)} markers at STRONG or above")
    for marker in result.markers:
        print(f"    [{marker.strength.name}] {marker.content[:40]}...")

    return store


def demonstrate_memory_inheritance():
    """Demonstrate memory inheritance to child agents."""
    print("\n" + "="*60)
    print("4. MEMORY INHERITANCE")
    print("="*60)

    # Parent agent's memory
    parent = HistoneStore(silent=True)

    # Add various memories to parent
    parent.methylate(
        "Core value: Always prioritize user safety",
        strength=MarkerStrength.PERMANENT,
        tags=["values", "core"]
    )
    print("  Parent: Added core value (PERMANENT)")

    parent.methylate(
        "Security: Validate all external inputs",
        strength=MarkerStrength.STRONG,
        tags=["security"]
    )
    print("  Parent: Added security rule (STRONG)")

    parent.acetylate(
        "User prefers technical explanations",
        strength=MarkerStrength.MODERATE,
        tags=["preference"]
    )
    print("  Parent: Added user preference (MODERATE, temporary)")

    parent.phosphorylate(
        "Currently analyzing code review request",
        tags=["context"]
    )
    print("  Parent: Added current context (WEAK, transient)")

    # Show parent stats
    parent_stats = parent.get_statistics()
    print(f"\n--- Parent Statistics ---")
    print(f"  Total markers: {parent_stats['total_markers']}")
    print(f"  By type: {parent_stats['by_type']}")

    # Create child and inherit strong markers
    print("\n--- Creating Child Agent ---")
    child = HistoneStore(silent=True)

    # Inherit only strong+ markers from parent
    parent.inherit_to(child, min_strength=MarkerStrength.STRONG)

    child_stats = child.get_statistics()
    print(f"\n--- Child Statistics (after inheritance) ---")
    print(f"  Total markers: {child_stats['total_markers']}")
    print(f"  Inherited strong+ markers only")

    # Verify what was inherited
    print("\n--- Child's Inherited Memories ---")
    child_result = child.retrieve_context()
    for marker in child_result.markers:
        print(f"  [{marker.strength.name}] {marker.content[:40]}...")

    return parent, child


def demonstrate_statistics():
    """Demonstrate memory statistics and reporting."""
    print("\n" + "="*60)
    print("5. MEMORY STATISTICS")
    print("="*60)

    store = HistoneStore(silent=True)

    # Add diverse memories
    for i in range(5):
        store.methylate(f"Permanent rule #{i}: Important guideline",
                       tags=["permanent", f"rule_{i}"])

    for i in range(3):
        store.acetylate(f"Temporary preference #{i}: User setting",
                       tags=["temporary", f"pref_{i}"])

    for i in range(2):
        store.phosphorylate(f"Signal #{i}: Current context",
                           tags=["signal", f"ctx_{i}"])

    # Perform some recalls (affects access counts)
    store.retrieve_context(query="rule")
    store.retrieve_context(tags=["permanent"])
    store.retrieve_context()

    # Get statistics
    stats = store.get_statistics()

    print(f"\n Memory Statistics:")
    print(f"  Total Markers: {stats['total_markers']}")
    print(f"  Active Markers: {stats['active_markers']}")
    print(f"  Total Added: {stats['total_added']}")
    print(f"  Total Expired: {stats['total_expired']}")
    print(f"  Total Recalls: {stats['total_retrievals']}")

    print(f"\n  By Type:")
    for marker_type, count in stats['by_type'].items():
        print(f"    {marker_type}: {count}")

    print(f"\n  By Strength:")
    for strength, count in stats['by_strength'].items():
        print(f"    {strength}: {count}")

    return store


def demonstrate_export_import():
    """Demonstrate exporting and importing memories."""
    print("\n" + "="*60)
    print("6. EXPORT AND IMPORT MEMORIES")
    print("="*60)

    # Create source store with memories
    source = HistoneStore(silent=True)
    source.methylate("Rule 1: Safety first", tags=["safety"])
    source.methylate("Rule 2: Validate inputs", tags=["validation"])
    source.acetylate("Preference: Verbose mode", tags=["preference"])

    print(f"  Source store has {source.get_statistics()['total_markers']} markers")

    # Export markers
    print("\n--- Exporting Markers ---")
    exported = source.export_markers()
    print(f"  Exported {len(exported)} markers")
    for exp in exported:
        print(f"    - {exp['content'][:30]}... ({exp['marker_type']})")

    # Import into new store
    print("\n--- Importing to New Store ---")
    destination = HistoneStore(silent=True)
    destination.import_markers(exported)

    dest_stats = destination.get_statistics()
    print(f"  Destination now has {dest_stats['total_markers']} markers")

    # Verify import
    result = destination.retrieve_context()
    print(f"\n--- Imported Memories ---")
    for marker in result.markers:
        print(f"  - {marker.content[:40]}...")

    return source, destination


def main():
    """Run all epigenetic memory demonstrations."""
    print("="*60)
    print("EPIGENETIC MEMORY SYSTEM DEMONSTRATION")
    print("Advanced Histone-Based Memory Management")
    print("="*60)

    demonstrate_marker_types()
    demonstrate_context_recall()
    demonstrate_marker_strength()
    demonstrate_memory_inheritance()
    demonstrate_statistics()
    demonstrate_export_import()

    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
