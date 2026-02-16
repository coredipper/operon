"""
BFCL Model Handler: LLM + Operon Chaperone (Gemini)
====================================================

Standalone handler compatible with the gorilla repo's BaseHandler interface.
Wraps a Gemini model in prompting mode with Operon's Chaperone cascade
folding to improve function-call accuracy.

Subclasses GeminiHandler and overrides decode_ast/decode_execute to use the
Chaperone cascade (STRICT -> EXTRACTION -> REPAIR) for recovering valid
function calls from malformed or wrapped LLM output.

BFCL v4 Results (Gemini-2.5-Flash + Chaperone):
  Non-Live: 88.65%  |  Live: 78.31%

To use with BFCL:
1. Copy this file to gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/
2. Register the model in bfcl_eval/constants/model_config.py
3. Run: bfcl generate --model operon-chaperone-gemini-2.5-flash --test-category non_live
4. Run: bfcl evaluate --model operon-chaperone-gemini-2.5-flash --test-category non_live

See also: eval/bfcl_handler.py for the OpenAI variant.
See: https://github.com/coredipper/operon
"""
import json
import os
import re
from typing import Any

from bfcl_eval.model_handler.api_inference.gemini import GeminiHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)


# ---------------------------------------------------------------------------
# Chaperone-lite: self-contained JSON repair cascade
# Mirrors operon_ai/organelles/chaperone.py without requiring the package.
# ---------------------------------------------------------------------------

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


def _cascade_parse(raw: str) -> Any | None:
    """Parse JSON with Chaperone cascade: strict -> extract -> repair."""
    raw = raw.strip()

    # STRICT
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # EXTRACTION
    for pattern in _EXTRACTION_PATTERNS:
        m = re.search(pattern, raw, re.MULTILINE | re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

    # REPAIR
    repaired = raw
    for pattern, replacement in _JSON_REPAIRS:
        repaired = re.sub(pattern, replacement, repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # REPAIR on extracted text
    for pattern in _EXTRACTION_PATTERNS:
        m = re.search(pattern, raw, re.MULTILINE | re.DOTALL)
        if m:
            extracted = m.group(1).strip()
            repaired = extracted
            for pat, repl in _JSON_REPAIRS:
                repaired = re.sub(pat, repl, repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class OpeornChaperoneGeminiHandler(GeminiHandler):
    """
    Wraps a Gemini model in prompting mode with Operon's Chaperone cascade
    for robust JSON repair in the decode step.

    Inherits all Gemini prompting/FC methods from GeminiHandler.
    Overrides decode_ast and decode_execute to use the Chaperone cascade.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        # Map GEMINI_API_KEY to GOOGLE_API_KEY if needed
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    # ----- decode with Chaperone cascade -----

    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode model output using Chaperone cascade parsing."""
        if not isinstance(result, str):
            return super().decode_ast(result, language, has_tool_call_tag)

        # Strip Gemini's tool_code blocks first
        result = result.replace("```tool_code\n", "").replace("\n```", "")

        parsed = _cascade_parse(result)
        if parsed is not None:
            if isinstance(parsed, list):
                decoded = [item for item in parsed if isinstance(item, dict)]
                if decoded:
                    return decoded
            elif isinstance(parsed, dict):
                return [parsed]

        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        """Decode model output to executable function call strings."""
        if not isinstance(result, str):
            return super().decode_execute(result, has_tool_call_tag)

        result = result.replace("```tool_code\n", "").replace("\n```", "")

        parsed = _cascade_parse(result)
        if parsed is not None:
            if isinstance(parsed, list):
                decoded = [item for item in parsed if isinstance(item, dict)]
                if decoded:
                    return convert_to_function_call(decoded)
            elif isinstance(parsed, dict):
                return convert_to_function_call([parsed])

        return default_decode_execute_prompting(result)
