"""
Comprehensive tests for the Ribosome prompt template engine.

Tests cover:
- Basic template creation and translation
- Variable substitution (required, optional, defaults)
- Filters (upper, lower, trim, chained)
- Conditionals (if/else)
- Loops (each with index access)
- Template includes
- Edge cases and error handling
"""

import pytest
from operon_ai.organelles.ribosome import (
    Ribosome,
    mRNA,
    Protein,
    Codon,
    CodonType,
    tRNA,
)


class TestRibosomeBasics:
    """Test basic functionality of the Ribosome."""

    def test_create_template(self):
        """Test creating a template with create_template()."""
        ribosome = Ribosome(silent=True)
        template = ribosome.create_template(
            sequence="Hello {{name}}",
            name="greeting",
            description="A simple greeting"
        )
        assert template.name == "greeting"
        assert template.description == "A simple greeting"
        assert template.sequence == "Hello {{name}}"
        assert "greeting" in ribosome.templates

    def test_simple_translation(self):
        """Test translating a simple template."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template(
            sequence="Hello {{name}}, welcome!",
            name="welcome"
        )
        protein = ribosome.translate("welcome", name="Alice")
        assert protein.sequence == "Hello Alice, welcome!"
        assert protein.source_mrna == "welcome"
        assert protein.variables_bound == {"name": "Alice"}

    def test_missing_required_variable_strict(self):
        """Test that missing required variables raise in strict mode."""
        ribosome = Ribosome(strict=True, silent=True)
        ribosome.create_template(
            sequence="Hello {{name}}",
            name="greeting"
        )
        with pytest.raises(ValueError, match="Missing required variable: name"):
            ribosome.translate("greeting")

    def test_missing_required_variable_warning(self):
        """Test that missing required variables produce warnings in non-strict mode."""
        ribosome = Ribosome(strict=False, silent=True)
        ribosome.create_template(
            sequence="Hello {{name}}",
            name="greeting"
        )
        protein = ribosome.translate("greeting")
        assert len(protein.warnings) > 0
        assert any("Missing required variable: name" in w for w in protein.warnings)

    def test_optional_variable_with_default(self):
        """Test optional variables with default values."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template(
            sequence="Hello {{name|Guest}}",
            name="greeting"
        )
        # With value
        protein1 = ribosome.translate("greeting", name="Alice")
        assert protein1.sequence == "Hello Alice"

        # Without value (uses default)
        protein2 = ribosome.translate("greeting")
        assert protein2.sequence == "Hello Guest"

    def test_optional_variable_question_mark(self):
        """Test optional variables with ? syntax."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template(
            sequence="Hello {{?name}}",
            name="greeting"
        )
        # With value
        protein1 = ribosome.translate("greeting", name="Alice")
        assert protein1.sequence == "Hello Alice"

        # Without value (empty string)
        protein2 = ribosome.translate("greeting")
        assert protein2.sequence == "Hello "

    def test_multiple_variables(self):
        """Test templates with multiple variables."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template(
            sequence="{{greeting}} {{name}}, you have {{count}} messages.",
            name="message"
        )
        protein = ribosome.translate("message", greeting="Hello", name="Bob", count=5)
        assert protein.sequence == "Hello Bob, you have 5 messages."

    def test_synthesize_direct(self):
        """Test direct synthesis without registering template."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("Quick {{action}} test", action="brown fox")
        assert protein.sequence == "Quick brown fox test"
        assert protein.source_mrna == "_direct_"


class TestRibosomeFilters:
    """Test filter application in templates."""

    def test_upper_filter(self):
        """Test upper filter."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{name|upper}}", name="alice")
        assert protein.sequence == "ALICE"

    def test_lower_filter(self):
        """Test lower filter."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{name|lower}}", name="ALICE")
        assert protein.sequence == "alice"

    def test_trim_filter(self):
        """Test trim filter."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{name|trim}}", name="  Alice  ")
        assert protein.sequence == "Alice"

    def test_title_filter(self):
        """Test title filter."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{name|title}}", name="alice smith")
        assert protein.sequence == "Alice Smith"

    def test_multiple_filters(self):
        """Test multiple filters in one template."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{first|upper}} and {{second|lower}}",
            first="alice",
            second="BOB"
        )
        assert protein.sequence == "ALICE and bob"

    def test_unknown_filter_warning(self):
        """Test that unknown filters produce warnings."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{name|unknown}}", name="Alice")
        assert "Unknown filter: unknown" in protein.warnings

    def test_custom_filter(self):
        """Test registering custom filters."""
        def reverse_filter(x):
            return str(x)[::-1]

        ribosome = Ribosome(filters={"reverse": reverse_filter}, silent=True)
        protein = ribosome.synthesize("{{name|reverse}}", name="Alice")
        assert protein.sequence == "ecilA"


class TestRibosomeConditionals:
    """Test if/else processing."""

    def test_if_true(self):
        """Test if block with true condition."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#if show}}Visible{{/if}}",
            show=True
        )
        assert protein.sequence == "Visible"

    def test_if_false(self):
        """Test if block with false condition."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#if show}}Visible{{/if}}",
            show=False
        )
        assert protein.sequence == ""

    def test_if_else_true_branch(self):
        """Test if/else with true condition."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#if premium}}Gold{{#else}}Silver{{/if}}",
            premium=True
        )
        assert protein.sequence == "Gold"

    def test_if_else_false_branch(self):
        """Test if/else with false condition."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#if premium}}Gold{{#else}}Silver{{/if}}",
            premium=False
        )
        assert protein.sequence == "Silver"

    def test_if_truthy_values(self):
        """Test if with truthy values."""
        ribosome = Ribosome(silent=True)

        # Non-empty string is truthy
        protein1 = ribosome.synthesize("{{#if value}}Yes{{/if}}", value="hello")
        assert protein1.sequence == "Yes"

        # Non-zero number is truthy
        protein2 = ribosome.synthesize("{{#if value}}Yes{{/if}}", value=42)
        assert protein2.sequence == "Yes"

    def test_if_falsy_values(self):
        """Test if with falsy values."""
        ribosome = Ribosome(silent=True)

        # Empty string is falsy
        protein1 = ribosome.synthesize("{{#if value}}Yes{{/if}}", value="")
        assert protein1.sequence == ""

        # Zero is falsy
        protein2 = ribosome.synthesize("{{#if value}}Yes{{/if}}", value=0)
        assert protein2.sequence == ""

        # None is falsy
        protein3 = ribosome.synthesize("{{#if value}}Yes{{/if}}", value=None)
        assert protein3.sequence == ""


class TestRibosomeLoops:
    """Test each loops."""

    def test_basic_loop(self):
        """Test basic iteration over a list."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each items}}{{.}} {{/each}}",
            items=["a", "b", "c"]
        )
        assert protein.sequence == "a b c "

    def test_loop_with_index(self):
        """Test loop with index access."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each items}}{{index}}:{{.}} {{/each}}",
            items=["a", "b", "c"]
        )
        assert protein.sequence == "0:a 1:b 2:c "

    def test_loop_with_item_name(self):
        """Test loop with 'item' variable."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each items}}Item: {{item}} {{/each}}",
            items=["x", "y"]
        )
        assert protein.sequence == "Item: x Item: y "

    def test_loop_empty_list(self):
        """Test loop with empty list."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each items}}{{.}}{{/each}}",
            items=[]
        )
        assert protein.sequence == ""

    def test_loop_dict_items(self):
        """Test loop with dictionary items."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each users}}{{name}}: {{age}} {{/each}}",
            users=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        )
        assert protein.sequence == "Alice: 30 Bob: 25 "

    def test_loop_first_last_markers(self):
        """Test loop with first/last markers."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "{{#each items}}{{first}}{{last}}{{/each}}",
            items=["a", "b", "c"]
        )
        # First item: first=True, last=False
        # Middle item: first=False, last=False
        # Last item: first=False, last=True
        assert protein.sequence == "TrueFalseFalseFalseFalseTrue"


class TestRibosomeIncludes:
    """Test template inclusion."""

    def test_include_template(self):
        """Test including another template."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("Header: {{title}}", name="header")
        ribosome.create_template("{{>header}}\nBody", name="page")

        protein = ribosome.translate("page", title="Welcome")
        assert protein.sequence == "Header: Welcome\nBody"

    def test_include_nonexistent_template(self):
        """Test including a nonexistent template."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize("{{>missing}}")
        assert "[Unknown template: missing]" in protein.sequence

    def test_nested_includes(self):
        """Test nested template includes."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("{{name}}", name="name_part")
        ribosome.create_template("Hello {{>name_part}}", name="greeting")
        ribosome.create_template("{{>greeting}}!", name="full")

        protein = ribosome.translate("full", name="Alice")
        assert protein.sequence == "Hello Alice!"


class TestRibosomeEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_template_raises(self):
        """Test that translating unknown template raises error."""
        ribosome = Ribosome(silent=True)
        with pytest.raises(ValueError, match="Unknown template: nonexistent"):
            ribosome.translate("nonexistent")

    def test_empty_template(self):
        """Test empty template."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("", name="empty")
        protein = ribosome.translate("empty")
        assert protein.sequence == ""

    def test_template_no_variables(self):
        """Test template with no variables."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("Just plain text", name="plain")
        protein = ribosome.translate("plain")
        assert protein.sequence == "Just plain text"

    def test_special_characters(self):
        """Test templates with special characters."""
        ribosome = Ribosome(silent=True)
        protein = ribosome.synthesize(
            "Special: !@#$%^&*() {{value}}",
            value="test"
        )
        assert protein.sequence == "Special: !@#$%^&*() test"

    def test_register_template_without_name(self):
        """Test that registering template without name raises error."""
        ribosome = Ribosome(silent=True)
        template = mRNA(sequence="test")
        with pytest.raises(ValueError, match="Template must have a name"):
            ribosome.register_template(template)

    def test_codon_detection(self):
        """Test automatic codon detection in mRNA."""
        template = mRNA(sequence="Hello {{name}}, {{?optional}}, {{default|value}}")
        assert len(template.codons) == 3
        assert template.codons[0].name == "name"
        assert template.codons[0].required is True
        assert template.codons[1].name == "optional"
        assert template.codons[1].required is False
        assert template.codons[2].name == "default"
        assert template.codons[2].default == "value"

    def test_get_required_variables(self):
        """Test getting required variables from mRNA."""
        template = mRNA(sequence="{{required}} {{?optional}} {{default|value}}")
        required = template.get_required_variables()
        assert "required" in required
        assert "optional" not in required
        assert "default" not in required

    def test_statistics(self):
        """Test getting synthesis statistics."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("{{value}}", name="test1")
        ribosome.create_template("{{value}}", name="test2")

        ribosome.translate("test1", value="a")
        ribosome.translate("test2", value="b")

        stats = ribosome.get_statistics()
        assert stats["translations_count"] == 2
        assert stats["templates_registered"] == 2
        assert "test1" in stats["template_names"]
        assert "test2" in stats["template_names"]

    def test_list_templates(self):
        """Test listing all templates."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template(
            "{{name}}",
            name="greeting",
            description="A greeting"
        )

        templates = ribosome.list_templates()
        assert len(templates) == 1
        assert templates[0]["name"] == "greeting"
        assert templates[0]["description"] == "A greeting"
        assert "name" in templates[0]["required_variables"]

    def test_protein_dataclass(self):
        """Test Protein dataclass structure."""
        ribosome = Ribosome(silent=True)
        ribosome.create_template("Hello {{name}}", name="test")
        protein = ribosome.translate("test", name="Alice")

        assert isinstance(protein, Protein)
        assert protein.sequence == "Hello Alice"
        assert protein.source_mrna == "test"
        assert protein.variables_bound == {"name": "Alice"}
        assert isinstance(protein.warnings, list)

    def test_trna_dataclass(self):
        """Test tRNA dataclass structure."""
        transfer = tRNA(anticodon="name", amino_acid="Alice")
        assert transfer.anticodon == "name"
        assert transfer.amino_acid == "Alice"

    def test_codon_dataclass(self):
        """Test Codon dataclass structure."""
        codon = Codon(
            codon_type=CodonType.VARIABLE,
            name="test",
            default="value",
            required=False
        )
        assert codon.codon_type == CodonType.VARIABLE
        assert codon.name == "test"
        assert codon.default == "value"
        assert codon.required is False
