"""
Example 3: Structured Data Extraction (Chaperone Topology)
==========================================================

Demonstrates the Chaperone Protein pattern where raw LLM output
(unstructured text) is "folded" into a strict schema. If folding
fails, the output is rejected and can be retried.

This mirrors how biological chaperones force proteins to fold
correctly - misfolded proteins are tagged for degradation.

Topology:
    Raw Text --> [Chaperone] --> Valid Schema?
                     |               |
                     |          YES: Return structured data
                     |               |
                     +---- NO: Return error trace for retry
"""

from pydantic import BaseModel, Field
from operon_ai.organelles.chaperone import Chaperone


# Define the target schema - what we want to extract
class ContactInfo(BaseModel):
    """Structured contact information."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    company: str = Field(description="Company or organization")


class MeetingRequest(BaseModel):
    """Structured meeting request."""
    title: str = Field(description="Meeting title")
    date: str = Field(description="Date in YYYY-MM-DD format")
    duration_minutes: int = Field(description="Duration in minutes")
    attendees: list[str] = Field(description="List of attendee names")


def main():
    print("=" * 60)
    print("Structured Data Extraction - Chaperone Demo")
    print("=" * 60)
    print()

    chaperone = Chaperone()

    # Test Case 1: Valid JSON that matches schema
    print("--- Test 1: Valid Contact Info ---")
    valid_contact = '''
    {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "phone": "+1-555-0123",
        "company": "Acme Corp"
    }
    '''
    result = chaperone.fold(valid_contact.strip(), ContactInfo)
    print(f"Input: {valid_contact.strip()}")
    print(f"Valid: {result.valid}")
    if result.valid:
        print(f"Extracted: {result.structure}")
    print()

    # Test Case 2: Invalid JSON (syntax error)
    print("--- Test 2: Malformed JSON ---")
    malformed = '{"name": "Bob", email: broken}'
    result = chaperone.fold(malformed, ContactInfo)
    print(f"Input: {malformed}")
    print(f"Valid: {result.valid}")
    print(f"Error: {result.error_trace}")
    print()

    # Test Case 3: Valid JSON but missing required fields
    print("--- Test 3: Missing Required Fields ---")
    incomplete = '{"name": "Charlie", "email": "charlie@test.com"}'
    result = chaperone.fold(incomplete, ContactInfo)
    print(f"Input: {incomplete}")
    print(f"Valid: {result.valid}")
    print(f"Error: {result.error_trace}")
    print()

    # Test Case 4: Valid meeting request
    print("--- Test 4: Valid Meeting Request ---")
    meeting = '''
    {
        "title": "Project Kickoff",
        "date": "2025-01-15",
        "duration_minutes": 60,
        "attendees": ["Alice", "Bob", "Charlie"]
    }
    '''
    result = chaperone.fold(meeting.strip(), MeetingRequest)
    print(f"Input: {meeting.strip()}")
    print(f"Valid: {result.valid}")
    if result.valid:
        print(f"Extracted: {result.structure}")
    print()

    # Test Case 5: Wrong types
    print("--- Test 5: Wrong Field Types ---")
    wrong_types = '''
    {
        "title": "Meeting",
        "date": "2025-01-15",
        "duration_minutes": "one hour",
        "attendees": "just me"
    }
    '''
    result = chaperone.fold(wrong_types.strip(), MeetingRequest)
    print(f"Input: {wrong_types.strip()}")
    print(f"Valid: {result.valid}")
    print(f"Error: {result.error_trace}")
    print()

    print("=" * 60)
    print("The Chaperone ensures only properly 'folded' data proceeds.")
    print("Misfolded outputs are caught before they corrupt downstream agents.")
    print("=" * 60)


if __name__ == "__main__":
    main()
