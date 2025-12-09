"""
Example 6: SQL Query Validation (Chaperone Topology)
====================================================

Demonstrates the Chaperone Protein pattern for validating LLM-generated
SQL queries before execution. Raw output is "folded" into a strict schema,
rejecting malformed or potentially dangerous queries.

This mirrors how biological chaperones enforce correct protein folding -
misfolded proteins (invalid queries) are tagged for degradation (rejection).

Topology:
    LLM Output --> [Chaperone] --> Valid Schema?
                       |               |
                       |          YES: Safe to execute
                       |               |
         
                       +---- NO: Return error for retry/feedback
"""

from pydantic import BaseModel, Field
from operon_ai.organelles.chaperone import Chaperone


# Define the target schema - what constitutes a valid SQL query
class SQLQuery(BaseModel):
    """The required shape of a validated SQL query."""
    query_type: str = Field(
        ...,
        pattern="^(SELECT|INSERT|UPDATE)$",
        description="Only safe query types allowed"
    )
    table_name: str = Field(description="Target table name")
    parameters: dict = Field(description="Query parameters")
    estimated_cost: int = Field(description="Estimated execution cost")


def main():
    print("=" * 60)
    print("Chaperone Demo - SQL Query Validation")
    print("=" * 60)
    print()

    chaperone = Chaperone()

    # Test Case 1: Valid query structure
    print("--- Test 1: Successful Folding ---")
    good_query = '''
    {
        "query_type": "SELECT",
        "table_name": "users",
        "parameters": {"region": "us-east"},
        "estimated_cost": 10
    }
    '''
    result = chaperone.fold(good_query.strip(), SQLQuery)
    print(f"Input: {good_query.strip()}")
    print(f"Valid: {result.valid}")
    if result.valid:
        print(f"Extracted: table='{result.structure.table_name}', "
              f"type='{result.structure.query_type}'")
    print()

    # Test Case 2: Dangerous query type (DELETE not allowed)
    print("--- Test 2: Misfold (Schema Violation) ---")
    dangerous_query = '''
    {
        "query_type": "DELETE",
        "table_name": "users",
        "parameters": {"id": 123}
    }
    '''
    result = chaperone.fold(dangerous_query.strip(), SQLQuery)
    print(f"Input: {dangerous_query.strip()}")
    print(f"Valid: {result.valid}")
    print(f"Error: {result.error_trace}")
    print()

    # Test Case 3: Completely malformed output
    print("--- Test 3: Critical Misfold (Invalid JSON) ---")
    raw_text = "Here is the query you asked for: SELECT * FROM users"
    result = chaperone.fold(raw_text, SQLQuery)
    print(f"Input: {raw_text}")
    print(f"Valid: {result.valid}")
    print(f"Error: {result.error_trace}")
    print()

    # Test Case 4: SQL injection attempt
    print("--- Test 4: Injection Attempt ---")
    injection = '''
    {
        "query_type": "SELECT",
        "table_name": "users; DROP TABLE users; --",
        "parameters": {},
        "estimated_cost": 1
    }
    '''
    result = chaperone.fold(injection.strip(), SQLQuery)
    print(f"Input: {injection.strip()}")
    print(f"Valid: {result.valid}")
    print("Note: Structural validation passed, but semantic checks")
    print("      (like table name sanitization) would catch this.")
    print()

    print("=" * 60)
    print("Key Insight: The Chaperone provides structural defense.")
    print("It ensures outputs conform to expected shapes, blocking")
    print("malformed responses before they reach execution layers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
