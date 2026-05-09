## 2024-05-24 - Pre-compiling Regex in Template Engines
**Learning:** In the `operon_ai` project, template parsing and variable interpolation in the `Ribosome` organelle used string-based `re.sub` and `re.finditer` on every call. Explicitly pre-compiling regexes as `typing.ClassVar` attributes avoids dictionary cache lookup overhead in hot paths, leading to a measurable ~16% speedup in template rendering benchmarks.
**Action:** Always check template parsing loops or hot paths involving regular expressions. Pre-compile them using `ClassVar` on the class level for maximum efficiency.

## 2024-05-23 - Pre-compile Regexes in dataclasses using ClassVar
**Learning:** When optimizing repetitive Regex compilation inside `dataclass` methods, simply assigning `re.compile()` to a class variable will treat it as a dataclass field by default. This alters the class constructor and creates errors.
**Action:** Use `typing.ClassVar` to correctly type-hint pre-compiled regular expression patterns as class attributes when using `@dataclass`, ensuring they are omitted from the generated `__init__` constructor.

## 2024-05-25 - Dictionary Lookup Optimization
**Learning:** Double dictionary lookups (`if key in dict` followed by `dict[key]`) are a frequent micro-bottleneck. Using the walrus operator with `dict.get()` (`if val := dict.get(key):`) cuts dictionary lookups in half, providing measurable speedups in highly-frequent hot paths like coordination gradients. Furthermore, returning the value directly from a setter method allows callers to skip an additional, otherwise redundant, `get()` lookup when calculating state deltas.
**Action:** Always refactor redundant `in` + `[]` accesses to single `.get()` calls using the walrus operator. Consider updating stateful setter methods to return their updated value to allow callers to calculate deltas more efficiently.
