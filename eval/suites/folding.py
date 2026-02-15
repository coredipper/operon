from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import random
import re

from pydantic import BaseModel, create_model

from operon_ai.organelles.chaperone import Chaperone, FoldingStrategy
from eval.utils import Counter, random_word


@dataclass
class FoldingConfig:
    samples: int = 200
    min_fields: int = 3
    max_fields: int = 8
    min_corruptions: int = 1
    max_corruptions: int = 3
    wrap_text_prob: float = 0.2


FIELD_TYPES = [int, float, str, bool]


def _random_value(rng: random.Random, typ: type) -> Any:
    if typ is int:
        return rng.randint(0, 100)
    if typ is float:
        return round(rng.random() * 100, 3)
    if typ is bool:
        return rng.choice([True, False])
    return " ".join(random_word(rng) for _ in range(rng.randint(2, 6)))


def _make_schema(rng: random.Random, idx: int, field_count: int) -> tuple[type[BaseModel], dict[str, type]]:
    fields: dict[str, tuple[type, Any]] = {}
    types: dict[str, type] = {}
    for i in range(field_count):
        name = f"field_{idx}_{i}"
        typ = rng.choice(FIELD_TYPES)
        fields[name] = (typ, ...)
        types[name] = typ
    model = create_model(f"EvalSchema{idx}", **fields)
    return model, types


def _apply_dict_corruptions(
    rng: random.Random,
    data: dict[str, Any],
    types: dict[str, type],
    count: int,
) -> tuple[dict[str, Any], list[str]]:
    ops = []
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
    ops = []
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


def run_folding(config: FoldingConfig, rng: random.Random) -> dict:
    chaperone = Chaperone(silent=True)
    strict_counter = Counter("strict")
    cascade_counter = Counter("cascade")
    corruption_counts: dict[str, int] = {}

    for i in range(config.samples):
        field_count = rng.randint(config.min_fields, config.max_fields)
        schema, types = _make_schema(rng, i, field_count)

        data = {key: _random_value(rng, typ) for key, typ in types.items()}
        dict_corruptions = rng.randint(0, max(0, config.max_corruptions - 1))
        data, dict_ops = _apply_dict_corruptions(rng, data, types, dict_corruptions)

        raw = json.dumps(data)
        string_corruptions = rng.randint(config.min_corruptions, config.max_corruptions)
        raw, str_ops = _apply_string_corruptions(rng, raw, string_corruptions, config.wrap_text_prob)

        for op in dict_ops + str_ops:
            corruption_counts[op] = corruption_counts.get(op, 0) + 1

        strict = chaperone.fold(raw, schema, strategies=[FoldingStrategy.STRICT])
        cascade = chaperone.fold(raw, schema)

        strict_counter.record(strict.valid)
        cascade_counter.record(cascade.valid)

    return {
        "samples": config.samples,
        "strict": strict_counter.as_dict(),
        "cascade": cascade_counter.as_dict(),
        "corruptions": corruption_counts,
    }
