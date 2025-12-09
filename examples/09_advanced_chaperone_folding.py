"""
Example 9: Advanced Chaperone Folding
=====================================

Demonstrates the enhanced Chaperone's multi-strategy validation:

1. **STRICT**: Exact JSON parsing, no modifications
2. **EXTRACTION**: Find JSON in markdown blocks or surrounding text
3. **LENIENT**: Type coercion (string "42" -> int 42)
4. **REPAIR**: Fix common JSON errors (trailing commas, single quotes)
5. **Confidence Scoring**: Track how much we trust the result
6. **Co-chaperones**: Domain-specific preprocessors

Biological Analogy:
- HSP70/HSP90 are general-purpose chaperones (like our strategies)
- GroEL/GroES provide an isolation chamber for difficult folds (retry logic)
- Ubiquitin tags misfolded proteins for degradation (error tracking)
- Co-chaperones assist with specific protein types (preprocessors)
"""

from pydantic import BaseModel
from operon_ai import (
    Chaperone,
    FoldingStrategy,
)


# Define some schemas to validate against
class UserProfile(BaseModel):
    name: str
    age: int
    email: str


class APIResponse(BaseModel):
    status: str
    code: int
    data: dict


class TaskItem(BaseModel):
    title: str
    completed: bool
    priority: int


def main():
    print("=" * 60)
    print("Advanced Chaperone Folding - Multi-Strategy Demo")
    print("=" * 60)

    # =================================================================
    # SECTION 1: Strict Folding
    # =================================================================
    print("\n--- 1. STRICT FOLDING ---")
    print("Exact JSON parsing, no modifications allowed...\n")

    chap = Chaperone(silent=True)

    # Perfect JSON - passes strict
    perfect_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
    result = chap.fold(perfect_json, UserProfile, strategies=[FoldingStrategy.STRICT])
    print(f"  Perfect JSON:")
    print(f"    Valid: {result.valid}")
    if result.valid:
        print(f"    Name: {result.structure.name}")
        print(f"    Age: {result.structure.age}")

    # Imperfect JSON - fails strict
    imperfect_json = "{'name': 'Bob', 'age': 25, 'email': 'bob@example.com'}"  # Single quotes
    result = chap.fold(imperfect_json, UserProfile, strategies=[FoldingStrategy.STRICT])
    print(f"\n  Single-quoted JSON (strict):")
    print(f"    Valid: {result.valid}")
    print(f"    Error: {result.error_trace[:60]}...")

    # =================================================================
    # SECTION 2: Extraction Strategy
    # =================================================================
    print("\n--- 2. EXTRACTION STRATEGY ---")
    print("Finding JSON buried in markdown or text...\n")

    # JSON in markdown code block
    markdown_response = '''
    Here's the user data you requested:

    ```json
    {"name": "Charlie", "age": 35, "email": "charlie@example.com"}
    ```

    Let me know if you need anything else!
    '''

    result = chap.fold(markdown_response, UserProfile)
    print(f"  JSON in markdown block:")
    print(f"    Valid: {result.valid}")
    if result.valid:
        print(f"    Extracted: {result.structure.name}, age {result.structure.age}")

    # JSON with XML-style tags
    xml_tagged = '''
    Processing complete.
    <json>{"name": "Diana", "age": 28, "email": "diana@example.com"}</json>
    End of response.
    '''

    result = chap.fold(xml_tagged, UserProfile)
    print(f"\n  JSON in XML tags:")
    print(f"    Valid: {result.valid}")
    if result.valid:
        print(f"    Extracted: {result.structure.name}")

    # =================================================================
    # SECTION 3: Lenient Strategy (Type Coercion)
    # =================================================================
    print("\n--- 3. LENIENT STRATEGY ---")
    print("Automatic type coercion...\n")

    # Age as string instead of int
    wrong_types = '{"name": "Eve", "age": "42", "email": "eve@example.com"}'

    # Strict fails
    result_strict = chap.fold(wrong_types, UserProfile, strategies=[FoldingStrategy.STRICT])
    print(f"  Wrong types (strict): Valid = {result_strict.valid}")

    # Lenient succeeds with coercion
    result_lenient = chap.fold(wrong_types, UserProfile, strategies=[FoldingStrategy.LENIENT])
    print(f"  Wrong types (lenient): Valid = {result_lenient.valid}")
    if result_lenient.valid:
        print(f"    Age coerced to: {result_lenient.structure.age} (type: {type(result_lenient.structure.age).__name__})")

    # =================================================================
    # SECTION 4: Repair Strategy
    # =================================================================
    print("\n--- 4. REPAIR STRATEGY ---")
    print("Fixing common JSON errors...\n")

    malformed_examples = [
        # Trailing commas
        ('{"name": "Frank", "age": 50, "email": "frank@example.com",}', "trailing comma"),
        # Single quotes
        ("{'name': 'Grace', 'age': 45, 'email': 'grace@example.com'}", "single quotes"),
        # Python literals
        ('{"name": "Henry", "age": 55, "email": None}', "None instead of null"),
        # Unquoted keys
        ('{name: "Iris", age: 33, email: "iris@example.com"}', "unquoted keys"),
    ]

    for malformed, issue in malformed_examples:
        result = chap.fold(malformed, UserProfile)
        status = "REPAIRED" if result.valid else "FAILED"
        print(f"  {issue}: [{status}]")
        if result.valid:
            print(f"    -> {result.structure.name}")

    # =================================================================
    # SECTION 5: Enhanced Folding with Confidence
    # =================================================================
    print("\n--- 5. CONFIDENCE SCORING ---")
    print("Track how confident we are in the result...\n")

    test_cases = [
        # Perfect JSON - high confidence
        ('{"title": "Task 1", "completed": true, "priority": 1}', "Perfect JSON"),
        # Extracted - medium confidence
        ('```json\n{"title": "Task 2", "completed": false, "priority": 2}\n```', "Extracted"),
        # Coerced types - lower confidence
        ('{"title": "Task 3", "completed": "true", "priority": "3"}', "Type coercion"),
        # Repaired - lowest confidence
        ("{'title': 'Task 4', 'completed': True, 'priority': 5,}", "Heavy repair"),
    ]

    for raw, description in test_cases:
        result = chap.fold_enhanced(raw, TaskItem)
        if result.valid:
            print(f"  {description}:")
            print(f"    Confidence: {result.confidence:.0%}")
            print(f"    Strategy: {result.strategy_used.value if result.strategy_used else 'N/A'}")
            if result.coercions_applied:
                print(f"    Coercions: {result.coercions_applied}")
            print()

    # =================================================================
    # SECTION 6: Co-chaperones (Preprocessors)
    # =================================================================
    print("\n--- 6. CO-CHAPERONES ---")
    print("Domain-specific preprocessing...\n")

    # Define a custom schema
    class CodeBlock(BaseModel):
        language: str
        code: str

    # Register a co-chaperone that extracts code from markdown
    def code_preprocessor(raw: str) -> str:
        """Extract code block and wrap in JSON."""
        import re
        match = re.search(r'```(\w+)\n([\s\S]*?)\n```', raw)
        if match:
            lang, code = match.groups()
            # Escape the code for JSON
            code_escaped = code.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            return f'{{"language": "{lang}", "code": "{code_escaped}"}}'
        return raw

    chap.register_co_chaperone(CodeBlock, code_preprocessor)

    # Now we can fold raw markdown directly
    markdown_code = '''
    Here's a Python example:

    ```python
    def hello():
        print("Hello, World!")
    ```
    '''

    result = chap.fold(markdown_code, CodeBlock)
    print(f"  Co-chaperone preprocessing:")
    print(f"    Valid: {result.valid}")
    if result.valid:
        print(f"    Language: {result.structure.language}")
        print(f"    Code: {result.structure.code[:30]}...")

    # =================================================================
    # SECTION 7: Misfold Callbacks
    # =================================================================
    print("\n--- 7. MISFOLD CALLBACKS ---")
    print("Handle failures gracefully...\n")

    misfold_log = []

    def on_misfold(result):
        misfold_log.append({
            "raw": result.raw_peptide_chain[:50],
            "attempts": len(result.attempts),
        })

    chap_with_callback = Chaperone(on_misfold=on_misfold, silent=True)

    # Try to fold something that can't be fixed
    hopeless_cases = [
        "This is not JSON at all",
        "name=Alice,age=30",  # Key-value but not JSON
        "<user><name>Bob</name></user>",  # XML not JSON
    ]

    for raw in hopeless_cases:
        result = chap_with_callback.fold(raw, UserProfile)
        print(f"  '{raw[:30]}...'")
        print(f"    Valid: {result.valid}")

    print(f"\n  Misfold callback captured {len(misfold_log)} failures")

    # =================================================================
    # SECTION 8: Statistics
    # =================================================================
    print("\n--- 8. STATISTICS ---")
    stats = chap.get_statistics()
    print(f"  Total folds: {stats['total_folds']}")
    print(f"  Successful: {stats['successful_folds']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"\n  Strategy success rates:")
    for strategy, rate in stats['strategy_success_rates'].items():
        attempts = stats['strategy_attempts'].get(strategy, 0)
        if attempts > 0:
            print(f"    {strategy}: {rate:.1%} ({stats['strategy_success'][strategy]}/{attempts})")

    print("\n" + "=" * 60)
    print("Chaperone demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
