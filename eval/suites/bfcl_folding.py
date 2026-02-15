"""BFCL folding suite — tests Chaperone with realistic function-call schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import importlib.resources
import json
import os
import pathlib
import random
import re

from pydantic import create_model

from operon_ai.organelles.chaperone import Chaperone, FoldingStrategy
from eval.utils import Counter, random_word


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BfclFoldingConfig:
    categories: list[str] = field(default_factory=lambda: ["simple", "multiple", "parallel"])
    max_samples: int = 200
    min_corruptions: int = 1
    max_corruptions: int = 3
    wrap_text_prob: float = 0.2


# ---------------------------------------------------------------------------
# Hardcoded fallback schemas (used when no BFCL data files are found)
# ---------------------------------------------------------------------------

FALLBACK_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "calculate_area",
        "description": "Calculate the area of a geometric shape.",
        "parameters": {
            "type": "dict",
            "properties": {
                "shape": {"type": "string", "description": "The shape type."},
                "width": {"type": "number", "description": "Width of the shape."},
                "height": {"type": "number", "description": "Height of the shape."},
            },
            "required": ["shape", "width", "height"],
        },
    },
    {
        "name": "search_items",
        "description": "Search for items in a catalogue.",
        "parameters": {
            "type": "dict",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Maximum results to return."},
                "in_stock": {"type": "boolean", "description": "Filter for in-stock items only."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a text message to a recipient.",
        "parameters": {
            "type": "dict",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient identifier."},
                "body": {"type": "string", "description": "Message body."},
                "priority": {"type": "integer", "description": "Priority level (1-5)."},
                "urgent": {"type": "boolean", "description": "Mark as urgent."},
            },
            "required": ["recipient", "body"],
        },
    },
    {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "dict",
            "properties": {
                "title": {"type": "string", "description": "Event title."},
                "date": {"type": "string", "description": "Event date in ISO format."},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes."},
                "location": {"type": "string", "description": "Event location."},
                "is_virtual": {"type": "boolean", "description": "Whether the event is virtual."},
            },
            "required": ["title", "date", "duration_minutes"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "dict",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "units": {"type": "string", "description": "Temperature units."},
                "include_forecast": {"type": "boolean", "description": "Include forecast."},
            },
            "required": ["city"],
        },
    },
]


# ---------------------------------------------------------------------------
# JSON Schema type -> Python type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _json_type_to_python(json_type: str) -> type:
    """Map a JSON Schema type string to a Python type."""
    return _TYPE_MAP.get(json_type, str)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CATEGORY_FILE_PATTERNS: dict[str, list[str]] = {
    "simple": ["BFCL_v4_simple_python.json", "BFCL_v3_simple.json"],
    "multiple": ["BFCL_v4_multiple_python.json", "BFCL_v3_multiple.json"],
    "parallel": ["BFCL_v4_parallel.json", "BFCL_v3_parallel.json"],
}


def _try_load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning a list of parsed dicts."""
    results: list[dict[str, Any]] = []
    if not path.is_file():
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _load_bfcl_data(categories: list[str]) -> tuple[list[dict[str, Any]], str]:
    """Load BFCL function schemas.

    Returns (schemas, source) where source is 'bfcl' or 'fallback'.
    """
    schemas: list[dict[str, Any]] = []

    # Strategy 1: BFCL_PROJECT_ROOT env var -> data/ subdir
    project_root = os.environ.get("BFCL_PROJECT_ROOT")
    if project_root:
        data_dir = pathlib.Path(project_root) / "data"
        if data_dir.is_dir():
            schemas = _load_from_dir(data_dir, categories)
            if schemas:
                return schemas, "bfcl"

    # Strategy 2: importlib.resources from bfcl_eval package -> data/
    try:
        bfcl_pkg = importlib.resources.files("bfcl_eval")  # type: ignore[attr-defined]
        data_path = bfcl_pkg / "data"
        if hasattr(data_path, "_path"):
            data_dir = pathlib.Path(str(data_path._path))  # type: ignore[union-attr]
        else:
            data_dir = pathlib.Path(str(data_path))
        if data_dir.is_dir():
            schemas = _load_from_dir(data_dir, categories)
            if schemas:
                return schemas, "bfcl"
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    # Strategy 3: Bundled fallback directory
    bundled_dir = pathlib.Path(__file__).parent.parent / "data" / "bfcl"
    if bundled_dir.is_dir():
        schemas = _load_from_dir(bundled_dir, categories)
        if schemas:
            return schemas, "bfcl"

    # Strategy 4: Hardcoded fallback schemas
    return FALLBACK_SCHEMAS, "fallback"


def _load_from_dir(data_dir: pathlib.Path, categories: list[str]) -> list[dict[str, Any]]:
    """Load function schemas from a directory of JSONL files."""
    schemas: list[dict[str, Any]] = []
    for cat in categories:
        patterns = _CATEGORY_FILE_PATTERNS.get(cat, [f"BFCL_v4_{cat}.json"])
        for pattern in patterns:
            entries = _try_load_jsonl(data_dir / pattern)
            if entries:
                for entry in entries:
                    funcs = entry.get("function", [])
                    if isinstance(funcs, list):
                        schemas.extend(funcs)
                    elif isinstance(funcs, dict):
                        schemas.append(funcs)
                break  # found a file for this category, skip alternatives
    return schemas


# ---------------------------------------------------------------------------
# Schema -> Pydantic model
# ---------------------------------------------------------------------------

def _schema_to_model(
    schema: dict[str, Any],
    model_idx: int,
) -> tuple[type, dict[str, type]]:
    """Convert a BFCL function schema to a Pydantic model.

    Returns (model_class, field_types_dict).
    """
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    fields: dict[str, tuple[type, Any]] = {}
    types: dict[str, type] = {}

    for prop_name, prop_spec in properties.items():
        json_type = prop_spec.get("type", "string")
        py_type = _json_type_to_python(json_type)
        types[prop_name] = py_type
        if prop_name in required:
            fields[prop_name] = (py_type, ...)
        else:
            fields[prop_name] = (py_type, None)  # type: ignore[assignment]

    name = schema.get("name", f"BfclSchema{model_idx}")
    # Sanitize the model name for Pydantic (must be valid Python identifier)
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe_name and safe_name[0].isdigit():
        safe_name = f"Bfcl_{safe_name}"
    safe_name = f"{safe_name}_{model_idx}"

    model = create_model(safe_name, **fields)  # type: ignore[call-overload]
    return model, types


# ---------------------------------------------------------------------------
# Random value generation for schema fields
# ---------------------------------------------------------------------------

def _random_value(rng: random.Random, typ: type) -> Any:
    """Generate a random value for the given Python type."""
    if typ is int:
        return rng.randint(0, 100)
    if typ is float:
        return round(rng.random() * 100, 3)
    if typ is bool:
        return rng.choice([True, False])
    # str
    return " ".join(random_word(rng) for _ in range(rng.randint(2, 6)))


# ---------------------------------------------------------------------------
# Corruption functions (mirroring eval/suites/folding.py)
# ---------------------------------------------------------------------------

def _apply_dict_corruptions(
    rng: random.Random,
    data: dict[str, Any],
    types: dict[str, type],
    count: int,
) -> tuple[dict[str, Any], list[str]]:
    """Apply dict-level corruptions: remove_field, type_swap, extra_field."""
    ops: list[str] = []
    keys = list(data.keys())
    for _ in range(count):
        if not keys:
            break
        op = rng.choice(["remove_field", "type_swap", "extra_field"])
        if op == "remove_field" and keys:
            key = rng.choice(keys)
            data.pop(key, None)
            keys.remove(key)
            ops.append("remove_field")
        elif op == "type_swap" and keys:
            key = rng.choice(keys)
            typ = types.get(key)
            if typ is int or typ is float:
                data[key] = str(data[key])
            elif typ is bool:
                data[key] = "true" if data[key] else "false"
            else:
                data[key] = rng.randint(0, 100)
            ops.append("type_swap")
        elif op == "extra_field":
            extra_key = f"extra_{random_word(rng, 4, 6)}"
            data[extra_key] = rng.randint(0, 10)
            ops.append("extra_field")
    return data, ops


def _apply_string_corruptions(
    rng: random.Random,
    raw: str,
    count: int,
    wrap_text_prob: float,
) -> tuple[str, list[str]]:
    """Apply string-level corruptions: trailing_comma, single_quotes, etc."""
    ops: list[str] = []
    for _ in range(count):
        op = rng.choice(["trailing_comma", "single_quotes", "unquoted_keys", "wrap_text"])
        if op == "trailing_comma":
            raw = re.sub(r"\}\s*$", " ,}", raw)
            raw = re.sub(r"\]\s*$", " ,]", raw)
            ops.append("trailing_comma")
        elif op == "single_quotes":
            raw = raw.replace('"', "'")
            ops.append("single_quotes")
        elif op == "unquoted_keys":
            raw = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', r"\1:", raw)
            ops.append("unquoted_keys")
        elif op == "wrap_text" and rng.random() < wrap_text_prob:
            raw = f"Here is the data:\n```json\n{raw}\n```"
            ops.append("wrap_text")
    return raw, ops


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_bfcl_folding(config: BfclFoldingConfig, rng: random.Random) -> dict:
    """Run the BFCL folding evaluation suite.

    Returns a dict with keys: samples, strict, cascade, source.
    """
    chaperone = Chaperone(silent=True)
    strict_counter = Counter("strict")
    cascade_counter = Counter("cascade")

    # Load schemas
    schemas, source = _load_bfcl_data(config.categories)

    # Build (model, types) pairs from schemas
    schema_pool: list[tuple[type, dict[str, type]]] = []
    for idx, schema in enumerate(schemas):
        try:
            model, types = _schema_to_model(schema, idx)
            if types:  # skip empty schemas
                schema_pool.append((model, types))
        except Exception:
            continue  # skip schemas that fail conversion

    # If no valid schemas, nothing to do
    if not schema_pool:
        return {
            "samples": 0,
            "strict": strict_counter.as_dict(),
            "cascade": cascade_counter.as_dict(),
            "source": source,
        }

    num_samples = min(config.max_samples, max(1, config.max_samples))

    for i in range(num_samples):
        # Pick a schema from the pool
        model, types = schema_pool[i % len(schema_pool)]

        # Generate valid data matching the schema
        data = {key: _random_value(rng, typ) for key, typ in types.items()}

        # Apply dict corruptions
        dict_corruptions = rng.randint(0, max(0, config.max_corruptions - 1))
        data, _ = _apply_dict_corruptions(rng, data, types, dict_corruptions)

        # Serialize to JSON
        raw = json.dumps(data)

        # Apply string corruptions
        string_corruptions = rng.randint(config.min_corruptions, config.max_corruptions)
        raw, _ = _apply_string_corruptions(rng, raw, string_corruptions, config.wrap_text_prob)

        # Fold with STRICT
        strict = chaperone.fold(raw, model, strategies=[FoldingStrategy.STRICT])
        strict_counter.record(strict.valid)

        # Fold with cascade (default)
        cascade = chaperone.fold(raw, model)
        cascade_counter.record(cascade.valid)

    return {
        "samples": num_samples,
        "strict": strict_counter.as_dict(),
        "cascade": cascade_counter.as_dict(),
        "source": source,
    }
