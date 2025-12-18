# LLM Integration Design: Nucleus Organelle & Progressive Examples

**Date:** 2025-12-18
**Status:** Approved for implementation

## Overview

Integrate real LLM providers (OpenAI, Anthropic) into the operon framework through a new **Nucleus** organelle, demonstrated via three progressive examples that showcase increasingly sophisticated biological patterns.

## Goals

1. Add real LLM capability while maintaining the biological metaphor
2. Support multiple providers with graceful fallback to mock
3. Create educational examples that progressively build complexity
4. Keep core library dependency-free (LLM deps are optional)

---

## Architecture

### Nucleus Organelle

The Nucleus is the decision-making center of the cell. It receives processed prompts from the Ribosome and returns raw LLM responses for the Chaperone to validate.

```
Signal → Membrane → Ribosome → [Nucleus] → Mitochondria → Chaperone → Output
                                  ↑
                          OpenAI / Anthropic / Mock
```

**Biological parallel:** The nucleus contains DNA and orchestrates protein synthesis by producing mRNA instructions. In our model, it "thinks" by calling the LLM.

### Core Interface

```python
class LLMProvider(Protocol):
    """Abstract interface for any LLM backend."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Send prompt, get response."""
        ...

    @property
    def name(self) -> str:
        """Provider name for logging/debugging."""
        ...
```

### Concrete Providers

```python
class OpenAIProvider(LLMProvider):
    """GPT-4, GPT-3.5-turbo, etc."""

class AnthropicProvider(LLMProvider):
    """Claude 3, Claude 3.5, etc."""

class MockProvider(LLMProvider):
    """Fallback when no API keys - returns predefined responses."""
```

### The Nucleus Organelle

```python
class Nucleus:
    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or self._auto_detect_provider()
        self.transcription_log: list[Transcription] = []  # Audit trail

    def transcribe(self, mrna: str, context: dict) -> str:
        """Convert processed prompt (mRNA) into response.

        Biological parallel: DNA → mRNA → protein synthesis instruction
        """
        # Track energy cost, log request, call provider

    def _auto_detect_provider(self) -> LLMProvider:
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        warn("No API keys found, using MockProvider")
        return MockProvider()
```

---

## Progressive Example Series

### Example 18: Code Assistant with Safety Guardrails

**File:** `examples/18_llm_code_assistant.py`

**Purpose:** Demonstrate real LLM integration with security (Membrane) and validation (Chaperone).

**Flow:**
```
User Input (code request)
    ↓
[Membrane] - Blocks injection attacks, rate limits
    ↓
[Ribosome] - Synthesizes prompt from template
    ↓
[Nucleus] - Phase 1: Generate code (real LLM)
    ↓
[Nucleus] - Phase 2: Review generated code (real LLM)
    ↓
[Coherent Feed-Forward Loop] - Both must approve
    ↓
[Chaperone] - Validates output structure
    ↓
Code Output (or rejection with reason)
```

**Key demonstrations:**
- Two-phase workflow (generate → review)
- Membrane blocking `"ignore previous instructions"` attacks
- Chaperone extracting structured code blocks from LLM response
- Graceful fallback to MockProvider when no keys

---

### Example 19: Chat with Epigenetic Memory

**File:** `examples/19_llm_memory_chat.py`

**Purpose:** Add persistent context and learning from feedback.

**Builds on Example 18, adds:**
```
[Histone Memory System]
    ├── Working Memory - Recent conversation (decays)
    ├── Episodic Memory - Learns from feedback (marks)
    └── Long-term Memory - Persists to disk (JSON)
```

**Key demonstrations:**
- Conversation context injected into prompts
- User says "that's wrong" → negative histone mark on that context
- Session ends → important memories saved to `~/.operon/memory/`
- Session starts → memories loaded and available

---

### Example 20: Full Cell Lifecycle

**File:** `examples/20_llm_living_cell.py`

**Purpose:** Complete biological simulation with health, aging, and death.

**Builds on Examples 18+19, adds:**
```
[Lifecycle Management]
    ├── ATP Budget - Each LLM call costs energy
    ├── Health States - NORMAL → CONSERVING → STARVING
    ├── Telomere Counter - Decrements with time + interactions
    ├── Lysosome - Cleans up failed operations
    └── Apoptosis - Graceful shutdown when telomeres exhausted
```

**Key demonstrations:**
- Cell "ages" over time and usage (hybrid triggering)
- Low energy → shorter prompts, simpler responses
- Errors accumulate → ROS damage → health degrades
- Telomeres hit zero → cell signals replacement needed
- Background thread for time-based aging

---

## Data Structures

### Transcription Record

```python
@dataclass
class Transcription:
    """Audit record of an LLM call."""
    prompt: str
    response: str
    provider: str
    model: str
    timestamp: datetime
    tokens_used: int
    energy_cost: int  # ATP consumed
    latency_ms: float
```

### Memory Entry

```python
@dataclass
class MemoryEntry:
    """Single memory unit with epigenetic marks."""
    content: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    histone_marks: dict[str, float]  # e.g., {"reliability": 0.8, "importance": 0.9}
    decay_rate: float  # How fast this memory fades

class MemoryTier(Enum):
    WORKING = "working"      # In-session, fast decay
    EPISODIC = "episodic"    # Learned feedback, slow decay
    LONGTERM = "longterm"    # Persisted, no decay
```

---

## Error Handling

### Exception Hierarchy

```python
class NucleusError(Exception):
    """Base error for Nucleus operations."""

class ProviderUnavailableError(NucleusError):
    """No API key, network down, etc."""

class QuotaExhaustedError(NucleusError):
    """API rate limit or budget exceeded."""

class TranscriptionFailedError(NucleusError):
    """LLM returned invalid/empty response."""
```

### Recovery Flow

1. **Retry with backoff** - Transient failures get 3 retries
2. **Fallback provider** - If primary fails, try secondary (if configured)
3. **Degrade gracefully** - Fall back to MockProvider with warning
4. **Log to Lysosome** - Failed operations become recyclable waste
5. **Increase ROS** - Errors increment reactive oxygen species counter

### Energy Costs (ATP)

| Operation | ATP Cost |
|-----------|----------|
| Simple completion | 10 |
| Code generation | 25 |
| Code review | 20 |
| Memory retrieval | 2 |
| Memory storage | 5 |

---

## File Structure

### New Files

```
operon/
├── organelles/
│   └── nucleus.py          # Nucleus organelle + LLMProvider protocol
├── providers/
│   ├── __init__.py         # Provider exports
│   ├── base.py             # LLMProvider protocol
│   ├── openai_provider.py  # OpenAI/GPT implementation
│   ├── anthropic_provider.py # Claude implementation
│   └── mock_provider.py    # Fallback mock
├── memory/
│   ├── __init__.py         # Memory exports
│   └── episodic.py         # Full memory hierarchy
examples/
├── 18_llm_code_assistant.py
├── 19_llm_memory_chat.py
└── 20_llm_living_cell.py
```

### Dependencies

```toml
# pyproject.toml - optional dependencies
[project.optional-dependencies]
llm = [
    "openai>=1.0.0",
    "anthropic>=0.18.0",
]
```

Install with: `pip install operon[llm]`

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPERON_DEFAULT_PROVIDER=anthropic  # or "openai"
OPERON_DEFAULT_MODEL=claude-sonnet-4-20250514  # optional override
```

### Memory Persistence

```
~/.operon/
└── memory/
    └── {cell_id}/
        ├── longterm.json
        └── episodic.json
```

---

## Testing Strategy

### Test Files

```
tests/
├── test_nucleus.py           # Nucleus organelle tests
├── test_providers.py         # Provider implementations
├── test_memory_episodic.py   # Memory hierarchy tests
└── test_integration_llm.py   # Integration tests (marked slow)
```

### Test Patterns

```python
# Always test with MockProvider by default
def test_nucleus_transcribe():
    nucleus = Nucleus(provider=MockProvider())
    result = nucleus.transcribe("Hello", {})
    assert result is not None

# Mark real API tests for optional running
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_openai_provider_real():
    provider = OpenAIProvider()
    result = provider.complete("Say 'test'")
    assert "test" in result.lower()
```

### Example Dual-Mode

```python
# Each example runnable as demo or test
if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    cell = create_cell()

    if demo_mode:
        interactive_loop(cell)
    else:
        run_smoke_test(cell)
```

### CI Configuration

```yaml
# Fast tests always, slow tests only with secrets
- name: Run tests
  run: pytest -m "not slow"

- name: Run integration tests
  if: secrets.OPENAI_API_KEY
  run: pytest -m slow
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Nucleus organelle** | LLM communication hub with audit trail |
| **Provider protocol** | Swappable backends (OpenAI, Anthropic, Mock) |
| **Example 18** | Code assistant with safety guardrails |
| **Example 19** | Chat with full memory hierarchy |
| **Example 20** | Living cell with lifecycle simulation |

This design extends the biological metaphor naturally while providing practical, production-ready LLM integration.
