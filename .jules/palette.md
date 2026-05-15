## 2024-05-14 - Gradio Manual HTML Table Accessibility
**Learning:** Standard Gradio components (like `gr.Button`) often do not natively expose properties to set standard ARIA attributes. However, when building manual HTML strings for output components (like `gr.HTML()`), we can apply standard accessibility attributes such as `scope="col"` to `<th>` tags directly.
**Action:** When working on Gradio apps, prioritize accessibility enhancements in raw HTML string constructors or specifically supported UI components when raw properties are unsupported by the framework API.
