"""BFCL live suite — evaluates Chaperone folding on real LLM function-call output."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from operon_ai.organelles.chaperone import Chaperone
from operon_ai.providers.base import ProviderConfig, LLMResponse
from eval.utils import Counter
from eval.suites.bfcl_folding import FALLBACK_SCHEMAS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BfclLiveConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    categories: list[str] = field(
        default_factory=lambda: ["simple", "multiple", "parallel"]
    )
    max_samples: int = 50
    temperature: float = 0.001
    use_chaperone: bool = True


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _make_provider(provider_name: str, model: str) -> Any:
    """Instantiate an LLM provider by name."""
    # Clear empty base URL vars that interfere with SDK defaults
    for key in ("OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT"):
        if os.environ.get(key, None) == "":
            del os.environ[key]

    if provider_name == "openai":
        from operon_ai.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if provider_name == "anthropic":
        from operon_ai.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    if provider_name == "gemini":
        from operon_ai.providers.gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    raise ValueError(f"Unknown provider: {provider_name}")


# ---------------------------------------------------------------------------
# Synthetic test cases (fallback when BFCL data is unavailable)
# ---------------------------------------------------------------------------

SYNTHETIC_SIMPLE: list[dict[str, Any]] = [
    {
        "schemas": ["calculate_area"],
        "prompt": "Calculate the area of a rectangle with width 12.5 and height 8.0.",
        "expected": [{"calculate_area": {"shape": "rectangle", "width": 12.5, "height": 8.0}}],
    },
    {
        "schemas": ["calculate_area"],
        "prompt": "What is the area of a circle with width 10 and height 10?",
        "expected": [{"calculate_area": {"shape": "circle", "width": 10, "height": 10}}],
    },
    {
        "schemas": ["search_items"],
        "prompt": "Search for 'wireless headphones' and show me the top 5 results that are in stock.",
        "expected": [{"search_items": {"query": "wireless headphones", "max_results": 5, "in_stock": True}}],
    },
    {
        "schemas": ["search_items"],
        "prompt": "Find all available laptops.",
        "expected": [{"search_items": {"query": "laptops", "in_stock": True}}],
    },
    {
        "schemas": ["send_message"],
        "prompt": "Send 'Hello, how are you?' to alice@example.com with priority 3.",
        "expected": [{"send_message": {"recipient": "alice@example.com", "body": "Hello, how are you?", "priority": 3}}],
    },
    {
        "schemas": ["send_message"],
        "prompt": "Send an urgent message to bob saying 'Meeting at 3pm'.",
        "expected": [{"send_message": {"recipient": "bob", "body": "Meeting at 3pm", "urgent": True}}],
    },
    {
        "schemas": ["create_event"],
        "prompt": "Create a meeting called 'Sprint Planning' on 2026-03-01 for 60 minutes in Room 4A.",
        "expected": [{"create_event": {"title": "Sprint Planning", "date": "2026-03-01", "duration_minutes": 60, "location": "Room 4A"}}],
    },
    {
        "schemas": ["create_event"],
        "prompt": "Schedule a virtual standup called 'Daily Sync' on 2026-02-20 for 15 minutes.",
        "expected": [{"create_event": {"title": "Daily Sync", "date": "2026-02-20", "duration_minutes": 15, "is_virtual": True}}],
    },
    {
        "schemas": ["get_weather"],
        "prompt": "What is the weather in Tokyo in celsius with forecast?",
        "expected": [{"get_weather": {"city": "Tokyo", "units": "celsius", "include_forecast": True}}],
    },
    {
        "schemas": ["get_weather"],
        "prompt": "Get the current weather in London.",
        "expected": [{"get_weather": {"city": "London"}}],
    },
]

SYNTHETIC_MULTIPLE: list[dict[str, Any]] = [
    {
        "schemas": ["calculate_area", "search_items", "get_weather"],
        "prompt": "What is the weather in Paris using fahrenheit?",
        "expected": [{"get_weather": {"city": "Paris", "units": "fahrenheit"}}],
    },
    {
        "schemas": ["send_message", "create_event", "get_weather"],
        "prompt": "Create a 90 minute event called 'Workshop' on 2026-04-15.",
        "expected": [{"create_event": {"title": "Workshop", "date": "2026-04-15", "duration_minutes": 90}}],
    },
    {
        "schemas": ["calculate_area", "search_items", "send_message"],
        "prompt": "Search for 'running shoes' and return at most 10 results.",
        "expected": [{"search_items": {"query": "running shoes", "max_results": 10}}],
    },
    {
        "schemas": ["calculate_area", "get_weather", "create_event"],
        "prompt": "Calculate the area of a triangle with width 6 and height 9.",
        "expected": [{"calculate_area": {"shape": "triangle", "width": 6, "height": 9}}],
    },
    {
        "schemas": ["send_message", "search_items", "get_weather"],
        "prompt": "Send a message to carol saying 'See you tomorrow' with priority 1.",
        "expected": [{"send_message": {"recipient": "carol", "body": "See you tomorrow", "priority": 1}}],
    },
]

SYNTHETIC_PARALLEL: list[dict[str, Any]] = [
    {
        "schemas": ["get_weather"],
        "prompt": "Get the weather in both New York and San Francisco in celsius.",
        "expected": [
            {"get_weather": {"city": "New York", "units": "celsius"}},
            {"get_weather": {"city": "San Francisco", "units": "celsius"}},
        ],
    },
    {
        "schemas": ["send_message"],
        "prompt": "Send 'Team lunch at noon' to both dave and eve.",
        "expected": [
            {"send_message": {"recipient": "dave", "body": "Team lunch at noon"}},
            {"send_message": {"recipient": "eve", "body": "Team lunch at noon"}},
        ],
    },
    {
        "schemas": ["create_event", "send_message"],
        "prompt": "Create a virtual event called 'Demo Day' on 2026-05-01 for 120 minutes, and send a message to frank saying 'Demo Day is coming!'.",
        "expected": [
            {"create_event": {"title": "Demo Day", "date": "2026-05-01", "duration_minutes": 120, "is_virtual": True}},
            {"send_message": {"recipient": "frank", "body": "Demo Day is coming!"}},
        ],
    },
    {
        "schemas": ["calculate_area"],
        "prompt": "Calculate the area of a square with width 5 and height 5, and a rectangle with width 3 and height 7.",
        "expected": [
            {"calculate_area": {"shape": "square", "width": 5, "height": 5}},
            {"calculate_area": {"shape": "rectangle", "width": 3, "height": 7}},
        ],
    },
    {
        "schemas": ["search_items", "get_weather"],
        "prompt": "Search for 'umbrellas' with max 3 results, and also get the weather in Seattle.",
        "expected": [
            {"search_items": {"query": "umbrellas", "max_results": 3}},
            {"get_weather": {"city": "Seattle"}},
        ],
    },
]


# ---------------------------------------------------------------------------
# Schema lookup and Pydantic model construction
# ---------------------------------------------------------------------------

_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {s["name"]: s for s in FALLBACK_SCHEMAS}


def _build_call_model() -> type:
    """Build a Pydantic model for a function call response.

    The model expects: {"name": str, "arguments": dict}
    """
    from pydantic import BaseModel

    class FunctionCall(BaseModel):
        name: str
        arguments: dict[str, Any]

    return FunctionCall


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_schema_text(schema: dict[str, Any]) -> str:
    """Format a single function schema for inclusion in a prompt."""
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    param_lines = []
    for pname, pspec in properties.items():
        req = " (required)" if pname in required else ""
        param_lines.append(
            f"  - {pname}: {pspec.get('type', 'string')} — {pspec.get('description', '')}{req}"
        )
    param_block = "\n".join(param_lines) if param_lines else "  (none)"

    return (
        f"Function: {schema['name']}\n"
        f"Description: {schema.get('description', '')}\n"
        f"Parameters:\n{param_block}"
    )


def _build_prompt(
    schemas: list[dict[str, Any]], user_query: str, parallel: bool = False
) -> str:
    """Build the full prompting-mode prompt."""
    schema_texts = "\n\n".join(_format_schema_text(s) for s in schemas)

    if parallel:
        output_instruction = (
            "Respond ONLY with a JSON array of function call objects. "
            "Each object must have \"name\" (string) and \"arguments\" (object) keys.\n"
            "Example: [{\"name\": \"func1\", \"arguments\": {\"a\": 1}}, "
            "{\"name\": \"func2\", \"arguments\": {\"b\": 2}}]"
        )
    else:
        output_instruction = (
            "Respond ONLY with a single JSON object with \"name\" (string) "
            "and \"arguments\" (object) keys.\n"
            "Example: {\"name\": \"func_name\", \"arguments\": {\"param\": \"value\"}}"
        )

    return (
        f"You have access to the following functions:\n\n"
        f"{schema_texts}\n\n"
        f"User query: {user_query}\n\n"
        f"{output_instruction}\n"
        f"Do not include any other text, explanation, or markdown formatting."
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison (case-insensitive strings, etc.)."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 2)
    return v


def _compare_call(
    actual: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Compare a single function call against expected.

    actual:   {"func_name": {"arg": "val"}} (BFCL format)
    expected: {"func_name": {"arg": "val"}} (BFCL format)

    Checks: function name matches and all expected arguments are present with
    correct values. Extra arguments in actual are tolerated.
    """
    if not actual or not expected:
        return False

    # Each dict has one key: the function name
    actual_name = next(iter(actual), None)
    expected_name = next(iter(expected), None)
    if actual_name is None or expected_name is None:
        return False
    if _normalize_value(actual_name) != _normalize_value(expected_name):
        return False

    actual_args = actual.get(actual_name, {})
    expected_args = expected.get(expected_name, {})
    if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
        return False

    # All expected args must be present and match
    for key, exp_val in expected_args.items():
        if key not in actual_args:
            return False
        act_val = actual_args[key]
        if _normalize_value(act_val) != _normalize_value(exp_val):
            return False

    return True


def _compare_calls(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
) -> bool:
    """Compare lists of function calls.

    For parallel calls, we check that every expected call has a match
    in the actual calls (order-independent).
    """
    if len(actual_calls) < len(expected_calls):
        return False

    matched = [False] * len(expected_calls)
    for act in actual_calls:
        for i, exp in enumerate(expected_calls):
            if not matched[i] and _compare_call(act, exp):
                matched[i] = True
                break

    return all(matched)


# ---------------------------------------------------------------------------
# Parsing raw LLM output into function calls
# ---------------------------------------------------------------------------

def _raw_parse(text: str) -> list[dict[str, Any]] | None:
    """Parse raw LLM text into BFCL-format function call list.

    Tries to extract JSON and convert {"name": "f", "arguments": {...}}
    into [{"f": {...}}].
    """
    text = text.strip()

    # Try to find JSON in common wrappers
    import re
    for pattern in [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]:
        m = re.search(pattern, text)
        if m:
            text = m.group(1).strip()
            break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    calls: list[dict[str, Any]] = []

    if isinstance(data, dict):
        # Single call: {"name": "f", "arguments": {...}}
        name = data.get("name")
        args = data.get("arguments", {})
        if name:
            calls.append({name: args})
        else:
            # Maybe already in BFCL format: {"f": {...}}
            for k, v in data.items():
                if isinstance(v, dict):
                    calls.append({k: v})

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("name")
                args = item.get("arguments", {})
                if name:
                    calls.append({name: args})
                else:
                    for k, v in item.items():
                        if isinstance(v, dict):
                            calls.append({k: v})

    return calls if calls else None


def _chaperone_parse(
    text: str, chaperone: Chaperone, call_model: type
) -> list[dict[str, Any]] | None:
    """Parse raw LLM text using Chaperone cascade folding.

    Folds into the FunctionCall Pydantic model, then converts to BFCL format.
    """
    text = text.strip()

    # Handle arrays: try to fold each element
    # First check if it looks like an array
    import re
    inner = text
    for pattern in [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]:
        m = re.search(pattern, inner)
        if m:
            inner = m.group(1).strip()
            break

    # Try as array first
    try:
        arr = json.loads(inner)
        if isinstance(arr, list):
            calls: list[dict[str, Any]] = []
            for item in arr:
                raw_item = json.dumps(item) if not isinstance(item, str) else item
                folded = chaperone.fold(raw_item, call_model)
                if folded.valid and folded.structure is not None:
                    calls.append({folded.structure.name: folded.structure.arguments})
            if calls:
                return calls
    except (json.JSONDecodeError, Exception):
        pass

    # Try as single object
    folded = chaperone.fold(text, call_model)
    if folded.valid and folded.structure is not None:
        return [{folded.structure.name: folded.structure.arguments}]

    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_bfcl_live(config: BfclLiveConfig) -> dict:
    """Run the live BFCL evaluation suite.

    Calls a real LLM provider and compares raw parsing vs Chaperone folding
    of function-call output.
    """
    provider = _make_provider(config.provider, config.model)

    if not provider.is_available():
        return {
            "error": f"Provider {config.provider} is not available (missing API key?)",
            "provider": config.provider,
            "model": config.model,
            "samples": 0,
            "raw": Counter("raw").as_dict(),
            "chaperone": Counter("chaperone").as_dict(),
        }

    chaperone = Chaperone(silent=True)
    prov_config = ProviderConfig(
        temperature=config.temperature,
        max_tokens=1024,
        system_prompt=None,
    )

    raw_counter = Counter("raw")
    chaperone_counter = Counter("chaperone")
    category_results: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    errors: list[str] = []

    # Build test cases per category
    test_suite = _build_test_suite(config.categories, config.max_samples)

    for category, test_cases in test_suite.items():
        cat_raw = Counter(f"{category}_raw")
        cat_chap = Counter(f"{category}_chaperone")

        for tc in test_cases:
            # Resolve schemas
            schema_names = tc["schemas"]
            schemas = []
            for name in schema_names:
                if name in _SCHEMA_BY_NAME:
                    schemas.append(_SCHEMA_BY_NAME[name])
            if not schemas:
                continue

            is_parallel = category == "parallel"
            prompt = _build_prompt(schemas, tc["prompt"], parallel=is_parallel)
            expected = tc["expected"]

            # Call the LLM
            try:
                response: LLMResponse = provider.complete(prompt, prov_config)
            except Exception as e:
                errors.append(f"LLM call failed: {e}")
                raw_counter.record(False)
                chaperone_counter.record(False)
                cat_raw.record(False)
                cat_chap.record(False)
                continue

            total_tokens += response.tokens_used
            raw_text = response.content

            # Raw parse attempt
            raw_calls = _raw_parse(raw_text)
            raw_ok = raw_calls is not None and _compare_calls(raw_calls, expected)
            raw_counter.record(raw_ok)
            cat_raw.record(raw_ok)

            # Chaperone parse attempt
            if config.use_chaperone:
                call_model = _build_call_model()
                chap_calls = _chaperone_parse(raw_text, chaperone, call_model)
                chap_ok = chap_calls is not None and _compare_calls(chap_calls, expected)
            else:
                chap_ok = raw_ok
            chaperone_counter.record(chap_ok)
            cat_chap.record(chap_ok)

        category_results[category] = {
            "raw_rate": cat_raw.rate,
            "chaperone_rate": cat_chap.rate,
            "samples": cat_raw.total,
        }

    raw_stats = raw_counter.as_dict()
    chap_stats = chaperone_counter.as_dict()

    return {
        "provider": config.provider,
        "model": config.model,
        "samples": raw_counter.total,
        "raw": raw_stats,
        "chaperone": chap_stats,
        "delta": chap_stats["rate"] - raw_stats["rate"],
        "by_category": category_results,
        "cost": {
            "total_tokens": total_tokens,
        },
        "errors": errors[:10] if errors else [],
        "source": "synthetic",
    }


# ---------------------------------------------------------------------------
# Test suite construction
# ---------------------------------------------------------------------------

def _build_test_suite(
    categories: list[str], max_samples: int
) -> dict[str, list[dict[str, Any]]]:
    """Build the test suite from synthetic cases, capped per category."""
    suite: dict[str, list[dict[str, Any]]] = {}

    pools: dict[str, list[dict[str, Any]]] = {
        "simple": SYNTHETIC_SIMPLE,
        "multiple": SYNTHETIC_MULTIPLE,
        "parallel": SYNTHETIC_PARALLEL,
    }

    for cat in categories:
        pool = pools.get(cat, [])
        suite[cat] = pool[:max_samples]

    return suite
