"""
Example 8: Multi-Pathway Mitochondria
=====================================

Demonstrates the enhanced Mitochondria's safe computation capabilities:

1. **Glycolysis**: Fast arithmetic using secure AST parsing (no code injection)
2. **Krebs Cycle**: Boolean and logical operations
3. **Oxidative Phosphorylation**: External tool running (sandboxed)
4. **Beta Oxidation**: Data transformations (JSON/literals)
5. **ROS Management**: Error tracking and self-repair

Biological Analogy:
- Glycolysis happens fast in the cytoplasm (simple math)
- Krebs cycle runs in the mitochondrial matrix (complex logic)
- Oxidative phosphorylation produces the most ATP (tool calls)
- Beta oxidation breaks down fatty acids (data transformation)
- ROS (reactive oxygen species) accumulate from errors (damage tracking)
- Mitophagy repairs damaged mitochondria (self-healing)

SECURITY NOTE: This implementation uses AST-based parsing, NOT dangerous
arbitrary code running. All expressions are safely evaluated through
the ast module's parsing capabilities.
"""

from operon_ai import (
    Mitochondria,
    MetabolicPathway,
    SimpleTool,
    Capability,
)


def main():
    print("=" * 60)
    print("Multi-Pathway Mitochondria - Safe Computation Demo")
    print("=" * 60)

    # =================================================================
    # SECTION 1: Glycolysis (Safe Math)
    # =================================================================
    print("\n--- 1. GLYCOLYSIS (Safe Math) ---")
    print("Using AST-based parsing for secure arithmetic...\n")

    mito = Mitochondria(silent=True)

    # Basic arithmetic
    expressions = [
        "2 + 2",
        "10 * 5 - 3",
        "2 ** 10",
        "100 / 4",
        "(15 + 5) * 2",
    ]

    for expr in expressions:
        result = mito.metabolize(expr, MetabolicPathway.GLYCOLYSIS)
        if result.success and result.atp:
            print(f"  {expr} = {result.atp.value}")
        else:
            print(f"  {expr} = ERROR: {result.error}")

    # Math functions
    print("\n  Math functions available:")
    math_expressions = [
        "sqrt(16)",
        "sin(0) + cos(0)",
        "log(e)",
        "pi * 2",
        "factorial(5)",
        "ceil(3.7)",
        "floor(3.7)",
    ]

    for expr in math_expressions:
        result = mito.metabolize(expr, MetabolicPathway.GLYCOLYSIS)
        if result.success and result.atp:
            print(f"  {expr} = {result.atp.value}")

    # =================================================================
    # SECTION 2: Krebs Cycle (Boolean Logic)
    # =================================================================
    print("\n--- 2. KREBS CYCLE (Boolean Logic) ---")
    print("Evaluating logical expressions...\n")

    logic_expressions = [
        ("5 > 3", True),
        ("10 == 10", True),
        ("5 <= 3", False),
        ("True and True", True),
        ("True or False", True),
        ("5 > 3 and 10 < 20", True),
        ("not False", True),
    ]

    for expr, expected in logic_expressions:
        result = mito.metabolize(expr, MetabolicPathway.KREBS_CYCLE)
        if result.success and result.atp:
            status = "OK" if result.atp.value == expected else "WRONG"
            print(f"  [{status}] {expr} = {result.atp.value}")

    # =================================================================
    # SECTION 3: Oxidative Phosphorylation (Tool Running)
    # =================================================================
    print("\n--- 3. OXIDATIVE PHOSPHORYLATION (Tools) ---")
    print("Running registered tools safely...\n")

    # Create a fresh mitochondria and register tools
    mito_tools = Mitochondria(silent=True)

    # Register some tools using SimpleTool
    mito_tools.engulf_tool(SimpleTool(
        name="reverse",
        description="Reverse a string",
        func=lambda s: s[::-1]
    ))

    mito_tools.engulf_tool(SimpleTool(
        name="uppercase",
        description="Convert to uppercase",
        func=lambda s: s.upper()
    ))

    mito_tools.engulf_tool(SimpleTool(
        name="count_words",
        description="Count words in a string",
        func=lambda s: len(s.split())
    ))

    # Or use the convenience method
    mito_tools.register_function(
        "double",
        lambda x: x * 2,
        "Double a number"
    )

    mito_tools.register_function(
        "greet",
        lambda name, greeting="Hello": f"{greeting}, {name}!",
        "Generate a greeting"
    )

    print("  Registered tools:")
    for tool in mito_tools.list_tools():
        print(f"    - {tool['name']}: {tool['description']}")

    print("\n  Tool invocation:")
    tool_calls = [
        'reverse("hello")',
        'uppercase("world")',
        'count_words("The quick brown fox")',
        'double(21)',
        'greet("Alice")',
        'greet("Bob", greeting="Hi")',
    ]

    for call in tool_calls:
        result = mito_tools.metabolize(call, MetabolicPathway.OXIDATIVE)
        if result.success and result.atp:
            print(f"  {call} = {result.atp.value}")
        else:
            print(f"  {call} = ERROR: {result.error}")

    # Capability gating (least-privilege)
    print("\n  Capability gating (least-privilege):")
    restricted = Mitochondria(allowed_capabilities=set(), silent=True)
    restricted.register_function(
        "fetch_url",
        lambda url: f"(mock) fetched {url}",
        "Mock network fetch",
        required_capabilities={Capability.NET},
    )
    restricted_result = restricted.metabolize('fetch_url(\"https://example.com\")')
    print(f"  fetch_url(\"https://example.com\") -> success={restricted_result.success}")
    if not restricted_result.success:
        print(f"    error: {restricted_result.error}")

    # =================================================================
    # SECTION 4: Beta Oxidation (Data Transformation)
    # =================================================================
    print("\n--- 4. BETA OXIDATION (Data Transform) ---")
    print("Parsing JSON and Python literals safely...\n")

    data_inputs = [
        '{"name": "Alice", "age": 30}',
        '["apple", "banana", "cherry"]',
        "{'key': 'value'}",  # Python dict literal
        "(1, 2, 3)",  # Python tuple
    ]

    for data in data_inputs:
        result = mito.metabolize(data, MetabolicPathway.BETA_OXIDATION)
        if result.success and result.atp:
            print(f"  Input: {data[:40]}...")
            print(f"  Output: {result.atp.value} (type: {type(result.atp.value).__name__})")
            print()

    # =================================================================
    # SECTION 5: Auto-Detection
    # =================================================================
    print("\n--- 5. AUTO-DETECTION ---")
    print("Mitochondria automatically chooses the right pathway...\n")

    auto_inputs = [
        "2 + 2",                           # -> Glycolysis
        "5 > 3",                           # -> Krebs Cycle
        '{"x": 1}',                        # -> Beta Oxidation
        'double(5)',                       # -> Oxidative (if tool registered)
    ]

    for expr in auto_inputs:
        # Use mito_tools which has tools registered
        result = mito_tools.metabolize(expr)  # No pathway specified
        if result.success and result.atp:
            pathway = result.atp.pathway.value
            print(f"  \"{expr}\" -> {pathway.upper()}")
            print(f"           Result: {result.atp.value}")
        else:
            print(f"  \"{expr}\" -> ERROR: {result.error}")

    # =================================================================
    # SECTION 6: ROS Management (Error Handling)
    # =================================================================
    print("\n--- 6. ROS MANAGEMENT (Error Handling) ---")
    print("Tracking accumulated damage and self-repair...\n")

    # Create a fresh mitochondria with low ROS threshold
    fragile_mito = Mitochondria(max_ros=0.3, silent=True)

    print(f"  Initial ROS level: {fragile_mito.get_ros_level():.2f}")
    print(f"  Max ROS before dysfunction: 0.30")

    # Cause some errors to accumulate ROS
    bad_expressions = [
        "invalid syntax",
        "undefined_variable",
        "1 / 0",  # Division by zero
    ]

    print("\n  Causing metabolic errors:")
    for expr in bad_expressions:
        result = fragile_mito.metabolize(expr)
        print(f"  Error from '{expr}': ROS = {result.ros_level:.2f}")

    # Check health status
    stats = fragile_mito.get_statistics()
    print(f"\n  Health status: {stats['health']}")

    # If dysfunctional, operations fail
    result = fragile_mito.metabolize("2 + 2")
    if not result.success:
        print(f"  Even safe operations fail: {result.error}")

    # Repair the mitochondria (mitophagy)
    print("\n  Initiating repair (mitophagy)...")
    fragile_mito.repair(amount=0.5)
    print(f"  ROS after repair: {fragile_mito.get_ros_level():.2f}")

    # Now it works again
    result = fragile_mito.metabolize("2 + 2")
    if result.success:
        print(f"  Operations restored: 2 + 2 = {result.atp.value}")

    # =================================================================
    # SECTION 7: Legacy API Compatibility
    # =================================================================
    print("\n--- 7. LEGACY API ---")
    print("Backward compatible digest_glucose() method...\n")

    # Original simple API still works
    result = mito.digest_glucose("2 + 2")
    print(f"  digest_glucose('2 + 2') = {result}")

    result = mito.digest_glucose("sqrt(144)")
    print(f"  digest_glucose('sqrt(144)') = {result}")

    result = mito.digest_glucose("bad expression")
    print(f"  digest_glucose('bad') = {result}")

    # =================================================================
    # SECTION 8: Statistics
    # =================================================================
    print("\n--- 8. STATISTICS ---")
    stats = mito_tools.get_statistics()
    print(f"  Operations count: {stats['operations_count']}")
    print(f"  Total ATP produced: {stats['total_atp_produced']:.2f}")
    print(f"  ROS level: {stats['ros_level']:.2f}")
    print(f"  Tools available: {stats['tools_available']}")
    print(f"  Health: {stats['health']}")

    print("\n" + "=" * 60)
    print("Mitochondria demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
