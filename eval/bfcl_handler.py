"""
BFCL Model Handler: LLM + Operon Chaperone
==========================================

Standalone handler compatible with the gorilla repo's BaseHandler interface.
Wraps any LLM provider with Operon's Chaperone cascade folding to improve
function-call accuracy in prompting mode.

To use with BFCL:
1. Copy this file to gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/
2. Register the model in bfcl_eval/constants/model_config.py
3. Run: bfcl generate --model operon-chaperone-gpt-4o-mini --test-category simple
4. Run: bfcl evaluate --model operon-chaperone-gpt-4o-mini --test-category simple
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# Chaperone-lite: self-contained JSON repair for BFCL handler
# (Avoids requiring operon-ai as a dependency in the gorilla repo)
# ---------------------------------------------------------------------------

# Common JSON repairs (mirrors operon_ai/organelles/chaperone.py)
_JSON_REPAIRS = [
    (r',\s*}', '}'),
    (r',\s*]', ']'),
    (r"'([^']*)'(?=\s*:)", r'"\1"'),
    (r":\s*'([^']*)'", r': "\1"'),
    (r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
    (r'\bNone\b', 'null'),
    (r'\bTrue\b', 'true'),
    (r'\bFalse\b', 'false'),
    (r':\s*undefined\b', ': null'),
    (r':\s*NaN\b', ': null'),
]

_EXTRACTION_PATTERNS = [
    r'```json\s*([\s\S]*?)\s*```',
    r'```\s*([\s\S]*?)\s*```',
    r'<json>([\s\S]*?)</json>',
]


def _extract_json(text: str) -> str:
    """Extract JSON from surrounding text (markdown, XML tags, etc.)."""
    for pattern in _EXTRACTION_PATTERNS:
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return text.strip()


def _repair_json(text: str) -> str:
    """Apply common JSON repairs."""
    for pattern, replacement in _JSON_REPAIRS:
        text = re.sub(pattern, replacement, text)
    return text


def _cascade_parse(raw: str) -> Any | None:
    """Cascade JSON parsing: strict → extract → repair.

    Mirrors the Chaperone's STRICT → EXTRACTION → LENIENT → REPAIR pipeline.
    Returns parsed JSON data or None.
    """
    # STRICT: try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # EXTRACTION: find JSON in wrapper text
    extracted = _extract_json(raw)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # REPAIR: fix common errors, then parse
    repaired = _repair_json(extracted)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # REPAIR on original (in case extraction lost context)
    repaired_orig = _repair_json(raw.strip())
    try:
        return json.loads(repaired_orig)
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_functions_for_prompt(functions: list[dict[str, Any]]) -> str:
    """Format function definitions for a text prompt."""
    parts = []
    for fn in functions:
        params = fn.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        param_lines = []
        for pname, pspec in properties.items():
            req_mark = " (required)" if pname in required else ""
            desc = pspec.get("description", "")
            ptype = pspec.get("type", "string")
            param_lines.append(f"  - {pname}: {ptype} — {desc}{req_mark}")

        param_block = "\n".join(param_lines) if param_lines else "  (none)"
        parts.append(
            f"Function: {fn['name']}\n"
            f"Description: {fn.get('description', '')}\n"
            f"Parameters:\n{param_block}"
        )

    return "\n\n".join(parts)


def build_prompt(functions: list[dict[str, Any]], user_query: str) -> str:
    """Build a prompting-mode prompt for function calling."""
    fn_text = _format_functions_for_prompt(functions)
    return (
        f"You have access to the following functions:\n\n"
        f"{fn_text}\n\n"
        f"User query: {user_query}\n\n"
        f'Respond ONLY with a JSON array of function call objects. '
        f'Each object must have "name" (string) and "arguments" (object) keys.\n'
        f'Example: [{{"name": "func_name", "arguments": {{"param": "value"}}}}]\n'
        f"Do not include any other text, explanation, or markdown formatting."
    )


# ---------------------------------------------------------------------------
# decode helpers
# ---------------------------------------------------------------------------

def _normalize_call(data: dict[str, Any]) -> dict[str, Any] | None:
    """Convert {"name": "f", "arguments": {...}} to {"f": {...}} (BFCL format)."""
    name = data.get("name")
    args = data.get("arguments", {})
    if name and isinstance(args, dict):
        return {name: args}
    return None


def decode_ast_chaperone(raw_result: str) -> list[dict[str, Any]]:
    """Parse raw LLM output into BFCL AST format using Chaperone cascade.

    Returns list of {"func_name": {"arg": "val"}} dicts.
    """
    parsed = _cascade_parse(raw_result)
    if parsed is None:
        return []

    calls: list[dict[str, Any]] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                call = _normalize_call(item)
                if call:
                    calls.append(call)
                else:
                    # Maybe already in BFCL format: {"func": {args}}
                    for k, v in item.items():
                        if isinstance(v, dict):
                            calls.append({k: v})

    elif isinstance(parsed, dict):
        call = _normalize_call(parsed)
        if call:
            calls.append(call)
        else:
            for k, v in parsed.items():
                if isinstance(v, dict):
                    calls.append({k: v})

    return calls


def decode_execute_chaperone(raw_result: str) -> list[str]:
    """Convert raw LLM output to executable function call strings.

    Returns list of "func_name(arg1=val1, arg2=val2)" strings.
    """
    ast = decode_ast_chaperone(raw_result)
    result = []
    for call_dict in ast:
        for func_name, args in call_dict.items():
            if isinstance(args, dict):
                arg_parts = []
                for k, v in args.items():
                    arg_parts.append(f"{k}={v!r}")
                result.append(f"{func_name}({', '.join(arg_parts)})")
    return result


# ---------------------------------------------------------------------------
# Standalone evaluation (can be run without BFCL installed)
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.001) -> str:
    """Call OpenAI API and return raw text response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required: pip install openai")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    # Quick demo: test the handler with a simple function call
    test_functions = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "units": {"type": "string", "description": "Temperature units."},
                },
                "required": ["city"],
            },
        }
    ]

    prompt = build_prompt(test_functions, "What's the weather in Tokyo in celsius?")
    print("=== Prompt ===")
    print(prompt)
    print()

    raw = _call_openai(prompt)
    print("=== Raw LLM output ===")
    print(raw)
    print()

    ast = decode_ast_chaperone(raw)
    print("=== decode_ast (Chaperone) ===")
    print(json.dumps(ast, indent=2))
    print()

    exe = decode_execute_chaperone(raw)
    print("=== decode_execute (Chaperone) ===")
    for call in exe:
        print(f"  {call}")
