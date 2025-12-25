"""
Example 10: Ribosome Prompt Factory
===================================

Demonstrates the Ribosome's template synthesis capabilities:

1. **Variable Substitution**: {{name}}, {{?optional}}, {{name|default}}
2. **Conditionals**: {{#if condition}}...{{#else}}...{{/if}}
3. **Loops**: {{#each items}}...{{/each}}
4. **Includes**: {{>template_name}} for composition
5. **Filters**: |upper, |lower, |trim, |json, |title

Biological Analogy:
- mRNA carries the genetic code (prompt template)
- tRNA brings amino acids (variable bindings)
- Ribosome assembles them into proteins (rendered prompts)
- Post-translational modification = filters
- Template includes = protein subunit assembly

This is where prompts are synthesized from reusable components,
enabling DRY (Don't Repeat Yourself) prompt engineering.
"""

from operon_ai import (
    Protein,
    Ribosome,
    mRNA,
)


def main():
    print("=" * 60)
    print("Ribosome Prompt Factory - Template Synthesis Demo")
    print("=" * 60)

    ribosome = Ribosome(silent=True)

    # =================================================================
    # SECTION 1: Simple Variable Substitution
    # =================================================================
    print("\n--- 1. SIMPLE VARIABLES ---")
    print("Basic {{variable}} substitution...\n")

    # Direct synthesis (one-off)
    template = "Hello {{name}}, welcome to {{place}}!"
    protein = ribosome.synthesize(template, name="Alice", place="Operon")
    print(f"  Template: {template}")
    print(f"  Result:   {protein.sequence}")

    # Multiple variables
    template2 = "User {{user}} requested {{action}} on {{resource}}"
    protein2 = ribosome.synthesize(template2, user="bob", action="READ", resource="/api/data")
    print(f"\n  Template: {template2}")
    print(f"  Result:   {protein2.sequence}")

    # =================================================================
    # SECTION 2: Optional Variables and Defaults
    # =================================================================
    print("\n--- 2. OPTIONAL & DEFAULTS ---")
    print("{{?optional}} and {{name|default}}...\n")

    # Optional variable (empty if not provided)
    template = "Hello {{name}}{{?title}}"
    p1 = ribosome.synthesize(template, name="Alice", title=", PhD")
    p2 = ribosome.synthesize(template, name="Bob")
    print(f"  With title: {p1.sequence}")
    print(f"  Without:    {p2.sequence}")

    # Default values
    template = "Priority: {{priority|normal}}, Status: {{status|pending}}"
    p1 = ribosome.synthesize(template, priority="high")
    p2 = ribosome.synthesize(template)
    print(f"\n  Custom priority: {p1.sequence}")
    print(f"  All defaults:    {p2.sequence}")

    # =================================================================
    # SECTION 3: Filters
    # =================================================================
    print("\n--- 3. FILTERS ---")
    print("Transform variables with |filter...\n")

    filter_examples = [
        ("Name: {{name|upper}}", {"name": "alice"}, "uppercase"),
        ("Name: {{name|lower}}", {"name": "ALICE"}, "lowercase"),
        ("Name: {{name|title}}", {"name": "alice smith"}, "title case"),
        ("Name: {{name|trim}}", {"name": "  alice  "}, "trim whitespace"),
        ("Items: {{count|length}}", {"count": [1,2,3,4,5]}, "length"),
    ]

    for template, context, description in filter_examples:
        protein = ribosome.synthesize(template, **context)
        print(f"  {description}: {protein.sequence}")

    # JSON filter for complex data
    data = {"user": "Alice", "items": ["a", "b", "c"]}
    protein = ribosome.synthesize("Data: {{data|json}}", data=data)
    print(f"\n  JSON filter: {protein.sequence}")

    # =================================================================
    # SECTION 4: Conditionals
    # =================================================================
    print("\n--- 4. CONDITIONALS ---")
    print("{{#if}}...{{#else}}...{{/if}}...\n")

    # Simple if
    template = "Status: {{#if active}}Online{{#else}}Offline{{/if}}"
    p1 = ribosome.synthesize(template, active=True)
    p2 = ribosome.synthesize(template, active=False)
    print(f"  Active=True:  {p1.sequence}")
    print(f"  Active=False: {p2.sequence}")

    # If without else
    template = "Welcome{{#if vip}}, VIP member{{/if}}!"
    p1 = ribosome.synthesize(template, vip=True)
    p2 = ribosome.synthesize(template, vip=False)
    print(f"\n  VIP=True:  {p1.sequence}")
    print(f"  VIP=False: {p2.sequence}")

    # Nested conditionals in a system prompt
    template = """You are a {{role}} assistant.
{{#if strict}}You must follow rules exactly.{{#else}}You can be flexible.{{/if}}
{{#if verbose}}Provide detailed explanations.{{/if}}"""

    protein = ribosome.synthesize(template, role="helpful", strict=True, verbose=True)
    print(f"\n  Complex conditional:")
    for line in protein.sequence.strip().split('\n'):
        print(f"    {line}")

    # =================================================================
    # SECTION 5: Loops
    # =================================================================
    print("\n--- 5. LOOPS ---")
    print("{{#each items}}...{{/each}}...\n")

    # Simple list iteration
    template = "Tasks: {{#each tasks}}\n- {{.}}{{/each}}"
    protein = ribosome.synthesize(template, tasks=["Buy milk", "Walk dog", "Code review"])
    print(f"  Simple list:")
    print(f"    {protein.sequence.replace(chr(10), chr(10) + '    ')}")

    # Object iteration with properties
    template = "Users:{{#each users}}\n  - {{name}} ({{role}}){{/each}}"
    users = [
        {"name": "Alice", "role": "Admin"},
        {"name": "Bob", "role": "User"},
        {"name": "Charlie", "role": "Guest"},
    ]
    protein = ribosome.synthesize(template, users=users)
    print(f"\n  Object list:")
    print(f"    {protein.sequence.replace(chr(10), chr(10) + '    ')}")

    # Loop with index
    template = "Steps:{{#each steps}}\n{{index}}. {{item}}{{/each}}"
    protein = ribosome.synthesize(template, steps=["Initialize", "Process", "Complete"])
    print(f"\n  With index:")
    print(f"    {protein.sequence.replace(chr(10), chr(10) + '    ')}")

    # =================================================================
    # SECTION 6: Template Registration & Includes
    # =================================================================
    print("\n--- 6. TEMPLATE INCLUDES ---")
    print("{{>template_name}} for composition...\n")

    # Register base templates
    ribosome.create_template(
        name="system_base",
        sequence="You are a {{role}} AI assistant.",
        description="Base system prompt"
    )

    ribosome.create_template(
        name="user_message",
        sequence="User: {{user}}\nQuery: {{query}}",
        description="Standard user message format"
    )

    ribosome.create_template(
        name="format_instruction",
        sequence="Please respond in {{format|text}} format.",
        description="Response format instruction"
    )

    # Compose a full prompt using includes
    ribosome.create_template(
        name="full_prompt",
        sequence="""{{>system_base}}

{{>format_instruction}}

{{>user_message}}""",
        description="Complete prompt with all components"
    )

    protein = ribosome.translate(
        "full_prompt",
        role="helpful coding",
        format="JSON",
        user="Alice",
        query="How do I sort a list in Python?"
    )

    print(f"  Composed prompt from 3 templates:")
    for line in protein.sequence.strip().split('\n'):
        print(f"    {line}")

    # =================================================================
    # SECTION 7: mRNA Objects
    # =================================================================
    print("\n--- 7. mRNA OBJECTS ---")
    print("Template metadata and introspection...\n")

    # Create an mRNA object with rich metadata
    agent_template = mRNA(
        sequence="""System: {{system_prompt}}

Context: {{context}}

User Query: {{query}}

{{#if examples}}Examples:{{#each examples}}
- {{.}}{{/each}}{{/if}}

Please respond thoughtfully.""",
        name="agent_prompt",
        description="Standard agent prompt template"
    )

    print(f"  Template: {agent_template.name}")
    print(f"  Description: {agent_template.description}")
    print(f"  Required variables: {agent_template.get_required_variables()}")
    print(f"  Total codons: {len(agent_template.codons)}")

    # Register and use
    ribosome.register_template(agent_template)
    protein = ribosome.translate(
        "agent_prompt",
        system_prompt="You are a helpful assistant",
        context="Python programming",
        query="How do I read a file?",
        examples=["open('file.txt')", "with open('file.txt') as f:"]
    )
    print(f"\n  Rendered ({len(protein.sequence)} chars)")

    # =================================================================
    # SECTION 8: Warnings and Strict Mode
    # =================================================================
    print("\n--- 8. WARNINGS & STRICT MODE ---")
    print("Handling missing variables...\n")

    # Non-strict mode - warnings but still renders
    template = "Hello {{name}}, your ID is {{id}}"
    protein = ribosome.synthesize(template, name="Alice")  # Missing 'id'
    print(f"  Non-strict mode:")
    print(f"    Result: {protein.sequence}")
    print(f"    Warnings: {protein.warnings}")

    # Strict mode - raises error
    strict_ribosome = Ribosome(strict=True, silent=True)
    try:
        strict_ribosome.synthesize(template, name="Bob")
    except ValueError as e:
        print(f"\n  Strict mode:")
        print(f"    Error: {e}")

    # =================================================================
    # SECTION 9: Custom Filters
    # =================================================================
    print("\n--- 9. CUSTOM FILTERS ---")
    print("Adding domain-specific transformations...\n")

    # Create ribosome with custom filters
    custom_ribosome = Ribosome(
        filters={
            'snake': lambda x: x.lower().replace(' ', '_'),
            'camel': lambda x: ''.join(w.title() for w in x.split()),
            'truncate': lambda x: x[:20] + '...' if len(str(x)) > 20 else str(x),
            'code': lambda x: f'`{x}`',
        },
        silent=True
    )

    template = "Variable: {{name|snake}}, Class: {{name|camel}}, Code: {{name|code}}"
    protein = custom_ribosome.synthesize(template, name="my variable name")
    print(f"  Custom filters: {protein.sequence}")

    # Truncate filter
    long_text = "This is a very long piece of text that needs truncation"
    protein = custom_ribosome.synthesize("Summary: {{text|truncate}}", text=long_text)
    print(f"  Truncate: {protein.sequence}")

    # =================================================================
    # SECTION 10: Statistics
    # =================================================================
    print("\n--- 10. STATISTICS ---")
    stats = ribosome.get_statistics()
    print(f"  Translations: {stats['translations_count']}")
    print(f"  Errors: {stats['errors_count']}")
    print(f"  Templates registered: {stats['templates_registered']}")
    print(f"  Template names: {stats['template_names']}")

    print("\n  Registered templates:")
    for t in ribosome.list_templates():
        print(f"    - {t['name']}: {t['description']}")
        if t['required_variables']:
            print(f"      Required: {t['required_variables']}")

    print("\n" + "=" * 60)
    print("Ribosome demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
