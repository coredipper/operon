"""
Example 11: Lysosome Waste Management
=====================================

Demonstrates the Lysosome's cleanup and recycling capabilities:

1. **Waste Collection**: Queue different types of cellular waste
2. **Digestion**: Process and break down waste items
3. **Recycling**: Extract useful information from failures
4. **Autophagy**: Self-cleaning of expired items
5. **Toxic Disposal**: Secure handling of sensitive data

Biological Analogy:
- Autophagy: Breaking down old/damaged cellular components
- Phagocytosis: Digesting external material brought into the cell
- Exocytosis: Expelling waste products
- pH regulation: Maintaining optimal digestion conditions
- Enzyme compartmentalization: Isolated breakdown processes

The Lysosome is the garbage collector and janitor of the cellular system,
preventing memory leaks and extracting value from failed operations.
"""

from datetime import datetime, timedelta
from operon_ai import (
    Lysosome,
    Waste,
    WasteType,
    DigestResult,
)


def main():
    print("=" * 60)
    print("Lysosome Waste Management - Cleanup & Recycling Demo")
    print("=" * 60)

    # =================================================================
    # SECTION 1: Basic Waste Ingestion
    # =================================================================
    print("\n--- 1. WASTE INGESTION ---")
    print("Collecting different types of cellular waste...\n")

    lysosome = Lysosome(silent=True)

    # Ingest various waste types
    waste_items = [
        Waste(
            waste_type=WasteType.FAILED_OPERATION,
            content={"error_type": "TimeoutError", "error_message": "Request timed out"},
            source="api_client"
        ),
        Waste(
            waste_type=WasteType.MISFOLDED_PROTEIN,
            content={"raw_input": '{"invalid json', "error": "Unexpected EOF"},
            source="chaperone"
        ),
        Waste(
            waste_type=WasteType.EXPIRED_CACHE,
            content={"key": "user_session_123", "expired_at": "2024-01-01"},
            source="cache_manager"
        ),
        Waste(
            waste_type=WasteType.ORPHANED_RESOURCE,
            content={"resource_id": "conn_456", "type": "database_connection"},
            source="connection_pool"
        ),
    ]

    for waste in waste_items:
        lysosome.ingest(waste)
        print(f"  Ingested: {waste.waste_type.value} from {waste.source}")

    # Check queue status
    status = lysosome.get_queue_status()
    print(f"\n  Queue status:")
    print(f"    Size: {status['size']}/{status['capacity']}")
    print(f"    Utilization: {status['utilization']:.1%}")
    print(f"    By type: {status['by_type']}")

    # =================================================================
    # SECTION 2: Convenient Ingestion Methods
    # =================================================================
    print("\n--- 2. CONVENIENT METHODS ---")
    print("Shortcuts for common waste types...\n")

    # Ingest an error directly
    try:
        raise ValueError("Something went wrong!")
    except ValueError as e:
        lysosome.ingest_error(
            error=e,
            source="demo_function",
            context={"input": "test_data", "step": 3}
        )
        print(f"  Ingested error: ValueError")

    # Ingest sensitive data for secure disposal
    sensitive_data = {
        "api_key": "sk-secret-123456",
        "password": "hunter2",
        "ssn": "123-45-6789"
    }
    lysosome.ingest_sensitive(sensitive_data, source="user_input")
    print(f"  Ingested sensitive data for secure disposal")

    status = lysosome.get_queue_status()
    print(f"\n  Queue now has {status['size']} items")

    # =================================================================
    # SECTION 3: Digestion Process
    # =================================================================
    print("\n--- 3. DIGESTION ---")
    print("Breaking down waste and extracting recyclables...\n")

    # Digest all queued waste
    result = lysosome.digest()

    print(f"  Digestion result:")
    print(f"    Success: {result.success}")
    print(f"    Items disposed: {result.disposed}")
    print(f"    Errors: {len(result.errors)}")

    # Check what was recycled
    recycled = lysosome.get_recycled()
    print(f"\n  Recycled materials:")
    for key, value in recycled.items():
        print(f"    {key}: {str(value)[:50]}...")

    # =================================================================
    # SECTION 4: Recycling Bin
    # =================================================================
    print("\n--- 4. RECYCLING BIN ---")
    print("Extracted useful information from waste...\n")

    # Add more waste with extractable info
    lysosome.ingest(Waste(
        waste_type=WasteType.FAILED_OPERATION,
        content={
            "error_type": "ConnectionError",
            "error_message": "Database unreachable",
            "context": {"host": "db.example.com", "port": 5432}
        },
        source="db_client"
    ))

    lysosome.ingest(Waste(
        waste_type=WasteType.MISFOLDED_PROTEIN,
        content={
            "raw_input": "malformed user data here...",
            "error": "Validation failed"
        },
        source="api_handler"
    ))

    # Digest and check recycling
    lysosome.digest()

    recycled = lysosome.get_recycled()
    print(f"  Total recycled items: {len(recycled)}")

    # Get specific recycled items
    last_input = lysosome.get_recycled("last_failed_input")
    if last_input:
        print(f"  Last failed input: {last_input}")

    last_context = lysosome.get_recycled("last_failure_context")
    if last_context:
        print(f"  Last failure context: {last_context}")

    # =================================================================
    # SECTION 5: Toxic Waste Handling
    # =================================================================
    print("\n--- 5. TOXIC WASTE HANDLING ---")
    print("Secure disposal of sensitive data...\n")

    toxic_log = []

    def on_toxic(waste: Waste):
        """Callback for toxic waste disposal."""
        toxic_log.append({
            "source": waste.source,
            "timestamp": datetime.now().isoformat(),
            "content_type": type(waste.content).__name__
        })

    secure_lysosome = Lysosome(on_toxic=on_toxic, silent=True)

    # Ingest multiple sensitive items
    secure_lysosome.ingest_sensitive({"credit_card": "4111-1111-1111-1111"}, "payment")
    secure_lysosome.ingest_sensitive({"token": "bearer_xyz123"}, "auth")
    secure_lysosome.ingest_sensitive({"private_key": "-----BEGIN RSA..."}, "crypto")

    # Digest - toxic callback is triggered
    secure_lysosome.digest()

    print(f"  Toxic waste disposed: {len(toxic_log)} items")
    for entry in toxic_log:
        print(f"    - From {entry['source']} at {entry['timestamp'][:19]}")

    # Note: Toxic waste is NOT recycled for security
    toxic_recycled = secure_lysosome.get_recycled()
    print(f"  Recycled from toxic: {len(toxic_recycled)} items (should be 0)")

    # =================================================================
    # SECTION 6: Autophagy (Self-Cleaning)
    # =================================================================
    print("\n--- 6. AUTOPHAGY ---")
    print("Self-cleaning of expired waste...\n")

    # Create lysosome with short retention
    temp_lysosome = Lysosome(retention_hours=0.0001, silent=True)  # Very short retention

    # Add some waste
    for i in range(5):
        temp_lysosome.ingest(Waste(
            waste_type=WasteType.EXPIRED_CACHE,
            content={"item": i},
            source="test"
        ))

    print(f"  Queue before autophagy: {temp_lysosome.get_queue_status()['size']}")

    # Wait a tiny bit and run autophagy
    import time
    time.sleep(0.01)

    removed = temp_lysosome.autophagy()
    print(f"  Items removed by autophagy: {removed}")
    print(f"  Queue after autophagy: {temp_lysosome.get_queue_status()['size']}")

    # =================================================================
    # SECTION 7: Auto-Digest and Emergency Digest
    # =================================================================
    print("\n--- 7. AUTO-DIGEST ---")
    print("Automatic digestion when queue fills up...\n")

    # Create lysosome with low thresholds
    small_lysosome = Lysosome(
        max_queue_size=10,
        auto_digest_threshold=5,
        silent=True
    )

    print(f"  Queue capacity: 10, auto-digest at: 5")

    # Fill it up
    for i in range(8):
        small_lysosome.ingest(Waste(
            waste_type=WasteType.EXPIRED_CACHE,
            content={"item": i},
            source="bulk_test"
        ))

    # Auto-digest should have triggered
    print(f"  After 8 ingestions, queue size: {small_lysosome.get_queue_status()['size']}")
    print("  (Auto-digest reduced the queue!)")

    # =================================================================
    # SECTION 8: Custom Digesters
    # =================================================================
    print("\n--- 8. CUSTOM DIGESTERS ---")
    print("Domain-specific waste processing...\n")

    metrics = {"processed": 0, "data_recovered": 0}

    def custom_failed_op_digester(waste: Waste) -> dict:
        """Custom digester that extracts more info from failed operations."""
        metrics["processed"] += 1
        content = waste.content

        recycled = {}
        if isinstance(content, dict):
            # Extract all error info
            if "error_type" in content:
                recycled["last_error_type"] = content["error_type"]
            if "error_message" in content:
                recycled["last_error_message"] = content["error_message"]
            if "context" in content:
                recycled["last_error_context"] = content["context"]
                metrics["data_recovered"] += 1

        return recycled

    custom_lysosome = Lysosome(
        digesters={
            WasteType.FAILED_OPERATION: custom_failed_op_digester
        },
        silent=True
    )

    # Process some failures
    errors = [
        {"error_type": "NetworkError", "error_message": "Connection lost", "context": {"retry": 3}},
        {"error_type": "ParseError", "error_message": "Invalid JSON", "context": {"line": 42}},
        {"error_type": "AuthError", "error_message": "Token expired", "context": {"user": "alice"}},
    ]

    for err in errors:
        custom_lysosome.ingest(Waste(
            waste_type=WasteType.FAILED_OPERATION,
            content=err,
            source="custom_test"
        ))

    custom_lysosome.digest()

    print(f"  Custom digester processed: {metrics['processed']} items")
    print(f"  Context data recovered: {metrics['data_recovered']} items")

    recycled = custom_lysosome.get_recycled()
    print(f"\n  Recycled info:")
    print(f"    Last error type: {recycled.get('last_error_type')}")
    print(f"    Last error msg: {recycled.get('last_error_message')}")
    print(f"    Last context: {recycled.get('last_error_context')}")

    # =================================================================
    # SECTION 9: Statistics
    # =================================================================
    print("\n--- 9. STATISTICS ---")
    stats = lysosome.get_statistics()
    print(f"  Queue size: {stats['queue_size']}")
    print(f"  Total ingested: {stats['total_ingested']}")
    print(f"  Total digested: {stats['total_digested']}")
    print(f"  Total recycled: {stats['total_recycled']}")
    print(f"  Recycling bin size: {stats['recycling_bin_size']}")
    print(f"\n  By waste type:")
    for wtype, count in stats['by_type'].items():
        if count > 0:
            print(f"    {wtype}: {count}")

    # =================================================================
    # SECTION 10: Cleanup Best Practices
    # =================================================================
    print("\n--- 10. BEST PRACTICES ---")
    print("Recommended usage patterns...\n")

    print("  1. Always dispose sensitive data via ingest_sensitive()")
    print("  2. Use autophagy() periodically to prevent memory leaks")
    print("  3. Check recycled materials for debugging insights")
    print("  4. Register toxic callbacks for audit trails")
    print("  5. Use custom digesters for domain-specific extraction")
    print("  6. Monitor statistics to detect unusual patterns")

    # Clear recycling bin when done
    lysosome.clear_recycling_bin()
    print(f"\n  Recycling bin cleared.")

    print("\n" + "=" * 60)
    print("Lysosome demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
