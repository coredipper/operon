"""
Chaperone: Protein Quality Control System
=========================================

Biological Analogy:
- HSP70/HSP90: General purpose chaperones (our main fold method)
- GroEL/GroES: Isolation chamber for difficult folds (retry logic)
- Ubiquitin system: Tag misfolded proteins for degradation
- Unfolded Protein Response (UPR): Stress response when folding fails
- Co-chaperones: Helpers for specific protein types

The Chaperone ensures that raw LLM output (unstructured text) is
"folded" into properly structured data (Pydantic models). Misfolded
outputs are rejected or repaired before they can corrupt downstream systems.
"""

from dataclasses import dataclass, field
from typing import Type, TypeVar, Callable, Any
from enum import Enum
import json
import re
import time

from pydantic import BaseModel, ValidationError
from ..core.types import FoldedProtein

T = TypeVar('T', bound=BaseModel)


class FoldingStrategy(Enum):
    """
    Different strategies for protein folding.

    Each strategy trades off strictness vs. flexibility.
    """
    STRICT = "strict"           # Exact match required, no modifications
    EXTRACTION = "extraction"   # Extract JSON from surrounding text
    LENIENT = "lenient"         # Allow type coercion
    REPAIR = "repair"           # Attempt to fix malformed input


@dataclass
class FoldingAttempt:
    """Record of a single folding attempt."""
    strategy: FoldingStrategy
    success: bool
    duration_ms: float
    error: str | None = None


@dataclass
class EnhancedFoldedProtein:
    """
    Enhanced folding result with full provenance.

    Tracks all attempts, confidence level, and any coercions applied.
    """
    valid: bool
    structure: Any | None = None
    raw_peptide_chain: str = ""
    error_trace: str | None = None
    attempts: list[FoldingAttempt] = field(default_factory=list)
    confidence: float = 1.0  # How confident we are in the fold
    coercions_applied: list[str] = field(default_factory=list)
    strategy_used: FoldingStrategy | None = None


class Chaperone:
    """
    Protein Quality Control System.

    Transforms raw LLM output into strictly typed Pydantic models.

    Features:

    1. Multiple Folding Strategies
       - STRICT: Exact JSON parsing, no modifications
       - EXTRACTION: Find JSON in markdown blocks or surrounding text
       - LENIENT: Type coercion (string "42" → int 42)
       - REPAIR: Fix common JSON errors (trailing commas, single quotes)

    2. Retry Logic
       - Tries strategies in order until one succeeds
       - Configurable strategy list

    3. Confidence Scoring
       - 1.0 for strict matches
       - Lower for extractions and coercions
       - Tracks all modifications made

    4. Co-chaperone Support
       - Register preprocessors for specific schema types
       - Domain-specific cleanup before folding

    5. Statistics
       - Track success rates per strategy
       - Identify problematic output patterns

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> chap = Chaperone()
        >>> result = chap.fold('{"name": "Alice", "age": 30}', Person)
        >>> result.valid
        True
        >>> result.structure.name
        'Alice'

    Example with extraction:
        >>> raw = '''Here's the data:
        ... ```json
        ... {"name": "Bob", "age": 25}
        ... ```
        ... '''
        >>> result = chap.fold(raw, Person)
        >>> result.valid
        True
        >>> result.confidence
        0.9
    """

    # Patterns for extracting JSON from text
    JSON_EXTRACTION_PATTERNS = [
        (r'```json\s*([\s\S]*?)\s*```', "markdown_json_block"),
        (r'```\s*([\s\S]*?)\s*```', "markdown_code_block"),
        (r'<json>([\s\S]*?)</json>', "xml_json_tag"),
        (r'\{[^{}]*\}', "bare_json_object"),
        (r'\[[^\[\]]*\]', "bare_json_array"),
    ]

    # Common JSON repairs
    JSON_REPAIRS = [
        # Fix trailing commas
        (r',\s*}', '}', "removed_trailing_comma_object"),
        (r',\s*]', ']', "removed_trailing_comma_array"),
        # Fix single quotes to double quotes
        (r"'([^']*)'(?=\s*:)", r'"\1"', "fixed_single_quote_key"),
        (r":\s*'([^']*)'", r': "\1"', "fixed_single_quote_value"),
        # Fix unquoted keys
        (r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', "quoted_unquoted_key"),
        # Fix Python literals
        (r'\bNone\b', 'null', "converted_none_to_null"),
        (r'\bTrue\b', 'true', "converted_true"),
        (r'\bFalse\b', 'false', "converted_false"),
        # Fix common typos
        (r':\s*undefined\b', ': null', "converted_undefined"),
        (r':\s*NaN\b', ': null', "converted_nan"),
    ]

    def __init__(
        self,
        max_retries: int = 3,
        strategies: list[FoldingStrategy] | None = None,
        co_chaperones: dict[Type, Callable[[str], str]] | None = None,
        on_misfold: Callable[[EnhancedFoldedProtein], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Chaperone.

        Args:
            max_retries: Maximum retry attempts (unused, kept for compatibility)
            strategies: Order of strategies to try
            co_chaperones: Type-specific preprocessors
            on_misfold: Callback when all strategies fail
            silent: Suppress console output
        """
        self.max_retries = max_retries
        self.strategies = strategies or [
            FoldingStrategy.STRICT,
            FoldingStrategy.EXTRACTION,
            FoldingStrategy.LENIENT,
            FoldingStrategy.REPAIR,
        ]
        self.co_chaperones = co_chaperones or {}
        self.on_misfold = on_misfold
        self.silent = silent

        # Statistics
        self._total_folds = 0
        self._successful_folds = 0
        self._strategy_success: dict[FoldingStrategy, int] = {s: 0 for s in FoldingStrategy}
        self._strategy_attempts: dict[FoldingStrategy, int] = {s: 0 for s in FoldingStrategy}

    def fold(
        self,
        raw_peptide_chain: str,
        target_schema: Type[T],
        strategies: list[FoldingStrategy] | None = None
    ) -> FoldedProtein[T]:
        """
        Attempt to fold raw output into a structured schema.

        Tries multiple strategies in order until one succeeds.

        Args:
            raw_peptide_chain: The raw text to fold
            target_schema: Pydantic model to validate against
            strategies: Override default strategy order

        Returns:
            FoldedProtein with the result (use .valid to check success)
        """
        self._total_folds += 1
        strategies = strategies or self.strategies
        attempts: list[FoldingAttempt] = []

        # Apply co-chaperone preprocessing if available
        processed_input = raw_peptide_chain
        if target_schema in self.co_chaperones:
            processed_input = self.co_chaperones[target_schema](raw_peptide_chain)

        for strategy in strategies:
            self._strategy_attempts[strategy] += 1
            start = time.time()

            try:
                result = self._attempt_fold(processed_input, target_schema, strategy)
                duration = (time.time() - start) * 1000

                if result.valid:
                    self._successful_folds += 1
                    self._strategy_success[strategy] += 1

                    # Return with attempt history
                    return FoldedProtein(
                        valid=True,
                        structure=result.structure,
                        raw_peptide_chain=raw_peptide_chain,
                        error_trace=None
                    )
                else:
                    attempts.append(FoldingAttempt(strategy, False, duration, result.error_trace))

            except Exception as e:
                duration = (time.time() - start) * 1000
                attempts.append(FoldingAttempt(strategy, False, duration, str(e)))

        # All strategies failed
        error_summary = "; ".join(
            f"{a.strategy.value}: {a.error}" for a in attempts if a.error
        )

        result = FoldedProtein(
            valid=False,
            structure=None,
            raw_peptide_chain=raw_peptide_chain,
            error_trace=f"All {len(strategies)} folding strategies failed. {error_summary}"
        )

        if self.on_misfold:
            enhanced = EnhancedFoldedProtein(
                valid=False,
                raw_peptide_chain=raw_peptide_chain,
                error_trace=result.error_trace,
                attempts=attempts,
                confidence=0.0
            )
            self.on_misfold(enhanced)

        return result

    def fold_enhanced(
        self,
        raw_peptide_chain: str,
        target_schema: Type[T],
        strategies: list[FoldingStrategy] | None = None
    ) -> EnhancedFoldedProtein:
        """
        Enhanced fold with full provenance tracking.

        Returns detailed information about the folding process.
        """
        self._total_folds += 1
        strategies = strategies or self.strategies
        attempts: list[FoldingAttempt] = []

        # Apply co-chaperone preprocessing
        processed_input = raw_peptide_chain
        if target_schema in self.co_chaperones:
            processed_input = self.co_chaperones[target_schema](raw_peptide_chain)

        for strategy in strategies:
            self._strategy_attempts[strategy] += 1
            start = time.time()

            try:
                result = self._attempt_fold_enhanced(processed_input, target_schema, strategy)
                duration = (time.time() - start) * 1000

                if result.valid:
                    self._successful_folds += 1
                    self._strategy_success[strategy] += 1

                    result.attempts = attempts + [FoldingAttempt(strategy, True, duration)]
                    result.raw_peptide_chain = raw_peptide_chain
                    return result
                else:
                    attempts.append(FoldingAttempt(strategy, False, duration, result.error_trace))

            except Exception as e:
                duration = (time.time() - start) * 1000
                attempts.append(FoldingAttempt(strategy, False, duration, str(e)))

        # All strategies failed
        result = EnhancedFoldedProtein(
            valid=False,
            raw_peptide_chain=raw_peptide_chain,
            error_trace=f"All {len(strategies)} folding strategies failed",
            attempts=attempts,
            confidence=0.0
        )

        if self.on_misfold:
            self.on_misfold(result)

        return result

    def _attempt_fold(
        self,
        raw: str,
        schema: Type[T],
        strategy: FoldingStrategy
    ) -> FoldedProtein[T]:
        """Attempt folding with a specific strategy."""
        if strategy == FoldingStrategy.STRICT:
            return self._fold_strict(raw, schema)
        elif strategy == FoldingStrategy.EXTRACTION:
            return self._fold_extraction(raw, schema)
        elif strategy == FoldingStrategy.LENIENT:
            return self._fold_lenient(raw, schema)
        elif strategy == FoldingStrategy.REPAIR:
            return self._fold_repair(raw, schema)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _attempt_fold_enhanced(
        self,
        raw: str,
        schema: Type[T],
        strategy: FoldingStrategy
    ) -> EnhancedFoldedProtein:
        """Attempt folding with enhanced result tracking."""
        if strategy == FoldingStrategy.STRICT:
            return self._fold_strict_enhanced(raw, schema)
        elif strategy == FoldingStrategy.EXTRACTION:
            return self._fold_extraction_enhanced(raw, schema)
        elif strategy == FoldingStrategy.LENIENT:
            return self._fold_lenient_enhanced(raw, schema)
        elif strategy == FoldingStrategy.REPAIR:
            return self._fold_repair_enhanced(raw, schema)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _fold_strict(self, raw: str, schema: Type[T]) -> FoldedProtein[T]:
        """Strict JSON parsing - no modifications."""
        try:
            data = json.loads(raw.strip())
            structure = schema.model_validate(data)
            return FoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw
            )
        except json.JSONDecodeError as e:
            return FoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=f"JSON: {e}")
        except ValidationError as e:
            return FoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=f"Validation: {e}")

    def _fold_strict_enhanced(self, raw: str, schema: Type[T]) -> EnhancedFoldedProtein:
        """Strict folding with enhanced tracking."""
        try:
            data = json.loads(raw.strip())
            structure = schema.model_validate(data)
            return EnhancedFoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw,
                confidence=1.0,
                strategy_used=FoldingStrategy.STRICT
            )
        except json.JSONDecodeError as e:
            return EnhancedFoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=f"JSON: {e}")
        except ValidationError as e:
            return EnhancedFoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=f"Validation: {e}")

    def _fold_extraction(self, raw: str, schema: Type[T]) -> FoldedProtein[T]:
        """Extract JSON from surrounding text."""
        for pattern, _ in self.JSON_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, raw, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    structure = schema.model_validate(data)
                    return FoldedProtein(
                        valid=True,
                        structure=structure,
                        raw_peptide_chain=raw
                    )
                except (json.JSONDecodeError, ValidationError):
                    continue

        return FoldedProtein(
            valid=False,
            raw_peptide_chain=raw,
            error_trace="No valid JSON found in text"
        )

    def _fold_extraction_enhanced(self, raw: str, schema: Type[T]) -> EnhancedFoldedProtein:
        """Extraction with enhanced tracking."""
        for pattern, pattern_name in self.JSON_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, raw, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    structure = schema.model_validate(data)
                    return EnhancedFoldedProtein(
                        valid=True,
                        structure=structure,
                        raw_peptide_chain=raw,
                        confidence=0.9,
                        coercions_applied=[f"extracted_via_{pattern_name}"],
                        strategy_used=FoldingStrategy.EXTRACTION
                    )
                except (json.JSONDecodeError, ValidationError):
                    continue

        return EnhancedFoldedProtein(
            valid=False,
            raw_peptide_chain=raw,
            error_trace="No valid JSON found in text"
        )

    def _fold_lenient(self, raw: str, schema: Type[T]) -> FoldedProtein[T]:
        """Lenient parsing with type coercion."""
        # First extract JSON
        extracted = self._extract_json(raw)
        if extracted is None:
            return FoldedProtein(valid=False, raw_peptide_chain=raw, error_trace="No JSON found")

        data = self._coerce_types(extracted, schema)

        try:
            structure = schema.model_validate(data)
            return FoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw
            )
        except ValidationError as e:
            return FoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=str(e))

    def _fold_lenient_enhanced(self, raw: str, schema: Type[T]) -> EnhancedFoldedProtein:
        """Lenient folding with enhanced tracking."""
        extracted = self._extract_json(raw)
        if extracted is None:
            return EnhancedFoldedProtein(valid=False, raw_peptide_chain=raw, error_trace="No JSON found")

        data, coercions = self._coerce_types_tracked(extracted, schema)

        try:
            structure = schema.model_validate(data)
            confidence = max(0.5, 0.85 - (len(coercions) * 0.05))
            return EnhancedFoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw,
                confidence=confidence,
                coercions_applied=coercions,
                strategy_used=FoldingStrategy.LENIENT
            )
        except ValidationError as e:
            return EnhancedFoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=str(e))

    def _fold_repair(self, raw: str, schema: Type[T]) -> FoldedProtein[T]:
        """Attempt to repair malformed JSON."""
        repaired = raw.strip()

        for pattern, replacement, _ in self.JSON_REPAIRS:
            repaired = re.sub(pattern, replacement, repaired)

        try:
            data = json.loads(repaired)
            structure = schema.model_validate(data)
            return FoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw
            )
        except (json.JSONDecodeError, ValidationError) as e:
            return FoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=str(e))

    def _fold_repair_enhanced(self, raw: str, schema: Type[T]) -> EnhancedFoldedProtein:
        """Repair folding with enhanced tracking."""
        repaired = raw.strip()
        repairs_applied = []

        for pattern, replacement, repair_name in self.JSON_REPAIRS:
            new_repaired = re.sub(pattern, replacement, repaired)
            if new_repaired != repaired:
                repairs_applied.append(repair_name)
                repaired = new_repaired

        try:
            data = json.loads(repaired)
            structure = schema.model_validate(data)
            confidence = max(0.4, 0.75 - (len(repairs_applied) * 0.05))
            return EnhancedFoldedProtein(
                valid=True,
                structure=structure,
                raw_peptide_chain=raw,
                confidence=confidence,
                coercions_applied=repairs_applied,
                strategy_used=FoldingStrategy.REPAIR
            )
        except (json.JSONDecodeError, ValidationError) as e:
            return EnhancedFoldedProtein(valid=False, raw_peptide_chain=raw, error_trace=str(e))

    def _extract_json(self, raw: str) -> dict | list | None:
        """Extract first valid JSON from text."""
        for pattern, _ in self.JSON_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, raw, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try the whole string
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    def _coerce_types(self, data: dict | list, schema: Type[T]) -> dict | list:
        """Coerce data types to match schema (without tracking)."""
        result, _ = self._coerce_types_tracked(data, schema)
        return result

    def _coerce_types_tracked(
        self,
        data: dict | list,
        schema: Type[T]
    ) -> tuple[dict | list, list[str]]:
        """Coerce data types to match schema with tracking."""
        coercions: list[str] = []

        if isinstance(data, list):
            return data, coercions

        result = dict(data)

        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
            for field_name, field_info in fields.items():
                if field_name not in result:
                    continue

                value = result[field_name]
                annotation = field_info.annotation

                # String to int
                if annotation == int and isinstance(value, str):
                    try:
                        result[field_name] = int(value)
                        coercions.append(f"{field_name}_str_to_int")
                    except ValueError:
                        pass

                # String to float
                elif annotation == float and isinstance(value, str):
                    try:
                        result[field_name] = float(value)
                        coercions.append(f"{field_name}_str_to_float")
                    except ValueError:
                        pass

                # Number to string
                elif annotation == str and isinstance(value, (int, float)):
                    result[field_name] = str(value)
                    coercions.append(f"{field_name}_num_to_str")

                # String to bool
                elif annotation == bool and isinstance(value, str):
                    lower = value.lower()
                    if lower in ('true', '1', 'yes'):
                        result[field_name] = True
                        coercions.append(f"{field_name}_str_to_bool")
                    elif lower in ('false', '0', 'no'):
                        result[field_name] = False
                        coercions.append(f"{field_name}_str_to_bool")

                # String to list (comma-separated)
                elif hasattr(annotation, '__origin__') and annotation.__origin__ == list:
                    if isinstance(value, str):
                        result[field_name] = [v.strip() for v in value.split(',')]
                        coercions.append(f"{field_name}_str_to_list")

        return result, coercions

    def register_co_chaperone(
        self,
        schema: Type[T],
        preprocessor: Callable[[str], str]
    ):
        """
        Register a co-chaperone for a specific schema type.

        Co-chaperones preprocess the raw input before folding.
        Useful for domain-specific cleanup.

        Args:
            schema: The Pydantic model type
            preprocessor: Function to preprocess raw input
        """
        self.co_chaperones[schema] = preprocessor

    def get_statistics(self) -> dict:
        """Get folding statistics."""
        return {
            "total_folds": self._total_folds,
            "successful_folds": self._successful_folds,
            "success_rate": self._successful_folds / max(1, self._total_folds),
            "strategy_success": {s.value: c for s, c in self._strategy_success.items()},
            "strategy_attempts": {s.value: c for s, c in self._strategy_attempts.items()},
            "strategy_success_rates": {
                s.value: self._strategy_success[s] / max(1, self._strategy_attempts[s])
                for s in FoldingStrategy
            }
        }

    def reset_statistics(self):
        """Reset all statistics."""
        self._total_folds = 0
        self._successful_folds = 0
        self._strategy_success = {s: 0 for s in FoldingStrategy}
        self._strategy_attempts = {s: 0 for s in FoldingStrategy}
