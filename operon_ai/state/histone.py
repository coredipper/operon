from typing import List

class HistoneStore:
    """
    Acts as the 'Chromatin State' (Memory).
    Stores 'Markers' that bias the agent against repeating mistakes.
    """
    def __init__(self):
        self.methylations: List[str] = []

    def add_marker(self, lesson: str):
        """Methylate: Add a permanent constraint."""
        print(f"📜 [Epigenetics] Methylating: '{lesson}'")
        self.methylations.append(lesson)

    def retrieve_context(self, query: str) -> str:
        """The Lens: Filters global state to find relevant constraints."""
        if not self.methylations:
            return ""
        return "⚠️ PREVIOUS FAILURES:\n" + "\n".join(f"- {m}" for m in self.methylations)
