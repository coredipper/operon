"""
Ribosome: Protein Synthesis Machinery
=====================================

Biological Analogy:
- mRNA: Prompt templates with variable slots
- tRNA: Variable bindings (context → template slots)
- Amino acids: Text fragments being assembled
- Translation: Template rendering with context
- Post-translational modification: Output processing

The Ribosome translates "genetic sequences" (prompt templates)
into "proteins" (formatted prompts ready for LLM consumption).

This is where prompts are assembled from templates and context,
supporting variable substitution, conditional sections, and
composition of multiple templates.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import re
import copy


class CodonType(Enum):
    """Types of codons (template elements) the ribosome can process."""
    VARIABLE = "variable"           # Simple variable substitution
    CONDITIONAL = "conditional"     # If/else blocks
    LOOP = "loop"                   # Repeated sections
    INCLUDE = "include"             # Include other templates
    FILTER = "filter"               # Transform output


@dataclass
class Codon:
    """
    A single unit of genetic code (template element).

    Represents a variable slot, conditional, or other template construct.
    """
    codon_type: CodonType
    name: str
    default: Any = None
    required: bool = True


@dataclass
class mRNA:
    """
    Messenger RNA: A prompt template ready for translation.

    Contains the template string and metadata about required variables.
    """
    sequence: str
    codons: list[Codon] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def __post_init__(self):
        """Auto-detect codons from template."""
        if not self.codons:
            self.codons = self._detect_codons()

    def _detect_codons(self) -> list[Codon]:
        """Detect variable slots in the template."""
        codons = []

        # Simple variables: {{variable_name}}
        for match in re.finditer(r'\{\{(\w+)\}\}', self.sequence):
            codons.append(Codon(
                codon_type=CodonType.VARIABLE,
                name=match.group(1)
            ))

        # Optional variables: {{?variable_name}}
        for match in re.finditer(r'\{\{\?(\w+)\}\}', self.sequence):
            codons.append(Codon(
                codon_type=CodonType.VARIABLE,
                name=match.group(1),
                required=False
            ))

        # Variables with defaults: {{variable_name|default_value}}
        for match in re.finditer(r'\{\{(\w+)\|([^}]*)\}\}', self.sequence):
            codons.append(Codon(
                codon_type=CodonType.VARIABLE,
                name=match.group(1),
                default=match.group(2),
                required=False
            ))

        return codons

    def get_required_variables(self) -> list[str]:
        """Get list of required variable names."""
        return [c.name for c in self.codons if c.required and c.codon_type == CodonType.VARIABLE]


@dataclass
class Protein:
    """
    The synthesized output: a fully rendered prompt.

    Contains the final text plus metadata about the synthesis process.
    """
    sequence: str
    source_mrna: str = ""
    variables_bound: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class tRNA:
    """
    Transfer RNA: Binds a specific variable to a value.

    Used for providing context to template synthesis.
    """
    anticodon: str  # Variable name
    amino_acid: Any  # Value to substitute


class Ribosome:
    """
    Protein Synthesis: Translates templates into formatted prompts.

    The Ribosome takes mRNA (prompt templates) and tRNA (variable bindings)
    to produce Proteins (rendered prompts).

    Features:

    1. Variable Substitution
       - Simple: {{name}} → value
       - Optional: {{?name}} → value or empty
       - Default: {{name|default}} → value or default

    2. Conditional Sections
       - {{#if condition}}...{{/if}}
       - {{#if condition}}...{{#else}}...{{/if}}

    3. Loops
       - {{#each items}}...{{/each}}
       - Access current item as {{.}} or {{item}}

    4. Includes
       - {{>template_name}} includes another registered template

    5. Filters
       - {{name|upper}} → UPPERCASE
       - {{name|lower}} → lowercase
       - {{name|trim}} → stripped whitespace

    Example:
        >>> ribosome = Ribosome()
        >>> template = mRNA(
        ...     sequence="Hello {{name}}, you have {{count}} messages.",
        ...     name="greeting"
        ... )
        >>> ribosome.register_template(template)
        >>> protein = ribosome.translate("greeting", name="Alice", count=5)
        >>> protein.sequence
        'Hello Alice, you have 5 messages.'
    """

    # Built-in filters for post-translational modification
    BUILTIN_FILTERS: dict[str, Callable[[Any], str]] = {
        'upper': lambda x: str(x).upper(),
        'lower': lambda x: str(x).lower(),
        'trim': lambda x: str(x).strip(),
        'title': lambda x: str(x).title(),
        'length': lambda x: str(len(x)),
        'json': lambda x: __import__('json').dumps(x),
        'repr': lambda x: repr(x),
    }

    def __init__(
        self,
        templates: dict[str, mRNA] | None = None,
        filters: dict[str, Callable[[Any], str]] | None = None,
        strict: bool = False,
        silent: bool = False,
    ):
        """
        Initialize the Ribosome.

        Args:
            templates: Pre-registered templates
            filters: Custom filters for post-translational modification
            strict: Raise errors for missing variables (vs. warnings)
            silent: Suppress console output
        """
        self.templates: dict[str, mRNA] = templates or {}
        self.filters = {**self.BUILTIN_FILTERS}
        if filters:
            self.filters.update(filters)
        self.strict = strict
        self.silent = silent

        # Statistics
        self._translations_count = 0
        self._errors_count = 0

    def register_template(self, template: mRNA, name: str | None = None):
        """
        Register an mRNA template for later use.

        Args:
            template: The mRNA template to register
            name: Override the template's name
        """
        template_name = name or template.name
        if not template_name:
            raise ValueError("Template must have a name")
        self.templates[template_name] = template
        if not self.silent:
            print(f"🧬 [Ribosome] Registered template: {template_name}")

    def create_template(
        self,
        sequence: str,
        name: str,
        description: str = ""
    ) -> mRNA:
        """
        Create and register a new template.

        Args:
            sequence: The template string
            name: Template name for later reference
            description: Human-readable description

        Returns:
            The created mRNA template
        """
        template = mRNA(sequence=sequence, name=name, description=description)
        self.register_template(template)
        return template

    def translate(
        self,
        template: str | mRNA,
        **context: Any
    ) -> Protein:
        """
        Translate a template into a protein (rendered prompt).

        Args:
            template: Template name (string) or mRNA object
            **context: Variable bindings (tRNA)

        Returns:
            Protein with the rendered sequence
        """
        self._translations_count += 1
        warnings: list[str] = []

        # Resolve template
        if isinstance(template, str):
            if template not in self.templates:
                self._errors_count += 1
                raise ValueError(f"Unknown template: {template}")
            mrna = self.templates[template]
        else:
            mrna = template

        sequence = mrna.sequence

        # Check required variables
        for var_name in mrna.get_required_variables():
            if var_name not in context:
                msg = f"Missing required variable: {var_name}"
                if self.strict:
                    self._errors_count += 1
                    raise ValueError(msg)
                warnings.append(msg)

        # Process conditionals first
        sequence = self._process_conditionals(sequence, context)

        # Process loops
        sequence = self._process_loops(sequence, context)

        # Process includes
        sequence = self._process_includes(sequence, context)

        # Process variable substitutions
        sequence = self._process_variables(sequence, context, warnings)

        return Protein(
            sequence=sequence,
            source_mrna=mrna.name,
            variables_bound=context,
            warnings=warnings
        )

    def synthesize(self, sequence: str, **context: Any) -> Protein:
        """
        Direct synthesis without registering a template.

        Convenience method for one-off translations.
        """
        mrna = mRNA(sequence=sequence, name="_direct_")
        return self.translate(mrna, **context)

    def _process_variables(
        self,
        sequence: str,
        context: dict[str, Any],
        warnings: list[str]
    ) -> str:
        """Process variable substitutions."""
        result = sequence

        # Variables with filters: {{name|filter}}
        def replace_filtered(match: re.Match) -> str:
            var_name = match.group(1)
            filter_name = match.group(2)

            if var_name in context:
                value = context[var_name]
                if filter_name in self.filters:
                    return self.filters[filter_name](value)
                warnings.append(f"Unknown filter: {filter_name}")
                return str(value)
            return match.group(0)

        result = re.sub(r'\{\{(\w+)\|(\w+)\}\}', replace_filtered, result)

        # Variables with defaults: {{name|default_value}} (not a filter)
        def replace_with_default(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)

            if var_name in context:
                return str(context[var_name])
            # Check if it's a filter or a default value
            if default not in self.filters:
                return default
            return match.group(0)

        # Only process if not already handled by filter
        for match in re.finditer(r'\{\{(\w+)\|([^}]+)\}\}', result):
            var_name = match.group(1)
            value_or_default = match.group(2)
            if value_or_default not in self.filters:
                if var_name in context:
                    result = result.replace(match.group(0), str(context[var_name]))
                else:
                    result = result.replace(match.group(0), value_or_default)

        # Optional variables: {{?name}}
        def replace_optional(match: re.Match) -> str:
            var_name = match.group(1)
            return str(context.get(var_name, ""))

        result = re.sub(r'\{\{\?(\w+)\}\}', replace_optional, result)

        # Simple variables: {{name}}
        def replace_simple(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in context:
                return str(context[var_name])
            warnings.append(f"Unbound variable: {var_name}")
            return match.group(0)

        result = re.sub(r'\{\{(\w+)\}\}', replace_simple, result)

        return result

    def _process_conditionals(self, sequence: str, context: dict[str, Any]) -> str:
        """Process conditional blocks."""
        result = sequence

        # If/else blocks: {{#if var}}...{{#else}}...{{/if}}
        pattern = r'\{\{#if\s+(\w+)\}\}(.*?)(?:\{\{#else\}\}(.*?))?\{\{/if\}\}'

        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            if_content = match.group(2)
            else_content = match.group(3) or ""

            value = context.get(var_name)
            if value:  # Truthy check
                return if_content
            return else_content

        result = re.sub(pattern, replace_conditional, result, flags=re.DOTALL)

        return result

    def _process_loops(self, sequence: str, context: dict[str, Any]) -> str:
        """Process loop blocks."""
        result = sequence

        # Each blocks: {{#each items}}...{{/each}}
        pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'

        def replace_loop(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)

            items = context.get(var_name, [])
            if not isinstance(items, (list, tuple)):
                return ""

            output_parts = []
            for i, item in enumerate(items):
                # Create loop context
                loop_context = {
                    '.': item,
                    'item': item,
                    'index': i,
                    'first': i == 0,
                    'last': i == len(items) - 1,
                }

                # If item is a dict, merge its keys
                if isinstance(item, dict):
                    loop_context.update(item)

                # Process the content with loop context
                part = content
                for key, value in loop_context.items():
                    part = part.replace(f"{{{{{key}}}}}", str(value))

                output_parts.append(part)

            return "".join(output_parts)

        result = re.sub(pattern, replace_loop, result, flags=re.DOTALL)

        return result

    def _process_includes(self, sequence: str, context: dict[str, Any]) -> str:
        """Process include directives."""
        result = sequence

        # Include: {{>template_name}}
        pattern = r'\{\{>(\w+)\}\}'

        def replace_include(match: re.Match) -> str:
            template_name = match.group(1)
            if template_name in self.templates:
                protein = self.translate(template_name, **context)
                return protein.sequence
            return f"[Unknown template: {template_name}]"

        result = re.sub(pattern, replace_include, result)

        return result

    def get_statistics(self) -> dict:
        """Get synthesis statistics."""
        return {
            "translations_count": self._translations_count,
            "errors_count": self._errors_count,
            "templates_registered": len(self.templates),
            "template_names": list(self.templates.keys()),
        }

    def list_templates(self) -> list[dict]:
        """List all registered templates."""
        return [
            {
                "name": name,
                "description": t.description,
                "required_variables": t.get_required_variables(),
            }
            for name, t in self.templates.items()
        ]
