## 2024-05-14 - Gradio Manual HTML Table Accessibility
**Learning:** Standard Gradio components (like `gr.Button`) often do not natively expose properties to set standard ARIA attributes. However, when building manual HTML strings for output components (like `gr.HTML()`), we can apply standard accessibility attributes such as `scope="col"` to `<th>` tags directly.
**Action:** When working on Gradio apps, prioritize accessibility enhancements in raw HTML string constructors or specifically supported UI components when raw properties are unsupported by the framework API.

## 2026-05-16 - Add scope attributes to table headers
**Learning:** Gradio UI does not allow setting scope directly on table headers in Markdown/HTML. Custom tables created with HTML need manual ARIA attributes.
**Action:** Always add scope="col" manually when constructing HTML tables in Gradio apps.

## 2026-05-18 - Add scope attributes to manually constructed table row headers
**Learning:** In manual HTML table construction (e.g. key-value tables) within Gradio or HTML strings, the leading column acting as a header must be a `<th scope="row">` tag rather than a `<td>` tag. Otherwise, screen readers will not properly associate the row header with the row data cell.
**Action:** When inspecting manual HTML tables, verify that key/label columns use `<th scope="row">` instead of just `<td>`. If they don't, replace them and apply visual normalization styles (like `font-weight:normal; text-align:left`) so they match the surrounding text style and don't introduce visual regressions while fixing the semantic structure.
