"""
Example 20: LLM Chat with Epigenetic Memory
===========================================

Builds on Example 19, adding a three-tier memory system:
- Working Memory: Recent conversation (decays)
- Episodic Memory: Learns from feedback (histone marks)
- Long-term Memory: Persists to disk across sessions

Key demonstrations:
- Conversation context injected into prompts
- User feedback affects memory reliability marks
- Session persistence for long-term memories
- Memory decay over time

Environment Variables:
    ANTHROPIC_API_KEY: For Claude models (preferred)
    OPENAI_API_KEY: For GPT models (fallback)

Usage:
    python examples/20_llm_memory_chat.py --demo    # Interactive mode
    python examples/20_llm_memory_chat.py           # Smoke test mode
"""

import sys
from pathlib import Path
from dataclasses import dataclass

from operon_ai import ATP_Store, Signal, Membrane
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import ProviderConfig
from operon_ai.memory import EpisodicMemory, MemoryTier


@dataclass
class ChatResponse:
    """Response from the memory-enabled chat."""
    content: str
    memories_used: int
    energy_consumed: int


class MemoryChat:
    """
    Chat assistant with three-tier epigenetic memory.

    Each conversation turn:
    1. Retrieves relevant memories
    2. Injects them as context
    3. Gets LLM response
    4. Stores interaction in working memory
    5. Promotes important memories based on feedback
    """

    def __init__(
        self,
        budget: ATP_Store | None = None,
        persistence_path: str | Path | None = None,
        silent: bool = False,
    ):
        self.budget = budget or ATP_Store(budget=1000)
        self.silent = silent

        # Organelles
        self.membrane = Membrane()
        self.nucleus = Nucleus(base_energy_cost=15)

        # Memory system
        default_path = Path.home() / ".operon" / "memory" / "chat"
        self.memory = EpisodicMemory(
            persistence_path=persistence_path or default_path
        )

        # Try to load existing memories
        self.memory.load()

        # Chat config
        self.chat_config = ProviderConfig(
            system_prompt=(
                "You are a helpful assistant with memory of past conversations. "
                "Use the provided memories to give contextually relevant responses. "
                "If a memory seems unreliable, mention your uncertainty."
            ),
            temperature=0.7,
            max_tokens=512,
        )

    def _log(self, msg: str) -> None:
        if not self.silent:
            print(msg)

    def chat(self, user_message: str) -> ChatResponse:
        """Process a chat message with memory context."""
        # Input filtering
        signal = Signal(content=user_message)
        filter_result = self.membrane.filter(signal)
        if not filter_result.allowed:
            return ChatResponse(
                content="I can't process that message.",
                memories_used=0,
                energy_consumed=0,
            )

        # Retrieve relevant memories
        context = self.memory.format_context(user_message)
        memories_used = len(self.memory.retrieve(user_message))

        # Build prompt with memory context
        prompt_parts = []
        if context:
            prompt_parts.append(context)
            prompt_parts.append("")
        prompt_parts.append(f"User: {user_message}")
        prompt = "\n".join(prompt_parts)

        self._log(f"💭 Using {memories_used} memories for context")

        # Get response
        if not self.budget.consume(cost=15):
            return ChatResponse(
                content="I'm too tired to respond right now.",
                memories_used=memories_used,
                energy_consumed=0,
            )

        response = self.nucleus.transcribe(prompt, config=self.chat_config)

        # Store this interaction in working memory
        interaction = f"User asked: {user_message[:50]}... Response: {response.content[:50]}..."
        self.memory.store(interaction, tier=MemoryTier.WORKING)

        # Apply decay to simulate time passing
        self.memory.decay_all()

        return ChatResponse(
            content=response.content,
            memories_used=memories_used,
            energy_consumed=self.nucleus.get_total_energy_consumed(),
        )

    def feedback(self, feedback_type: str, context: str = "") -> None:
        """
        Process user feedback to adjust memory marks.

        feedback_type: "good", "bad", "wrong", "important"
        """
        # Find recent working memories
        recent = self.memory.get_tier(MemoryTier.WORKING)
        if not recent:
            return

        last_memory = recent[-1]

        if feedback_type == "wrong":
            # Mark as unreliable
            self.memory.add_mark(last_memory.id, "reliability", 0.2)
            self._log("📝 Marked last response as unreliable")

        elif feedback_type == "good":
            # Mark as reliable and promote to episodic
            self.memory.add_mark(last_memory.id, "reliability", 0.9)
            self.memory.promote(last_memory.id, MemoryTier.EPISODIC)
            self._log("📝 Promoted to episodic memory")

        elif feedback_type == "important":
            # Promote to long-term
            self.memory.add_mark(last_memory.id, "importance", 1.0)
            self.memory.promote(last_memory.id, MemoryTier.LONGTERM)
            self._log("📝 Saved to long-term memory")

    def save(self) -> None:
        """Save long-term memories to disk."""
        self.memory.save()
        self._log("💾 Long-term memories saved")

    def stats(self) -> dict:
        """Get memory statistics."""
        return {
            "working": len(self.memory.get_tier(MemoryTier.WORKING)),
            "episodic": len(self.memory.get_tier(MemoryTier.EPISODIC)),
            "longterm": len(self.memory.get_tier(MemoryTier.LONGTERM)),
            "total": len(self.memory.memories),
        }


def run_demo():
    """Interactive demo mode."""
    print("=" * 60)
    print("LLM Memory Chat - Interactive Demo")
    print("=" * 60)
    print()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        budget = ATP_Store(budget=500)
        chat = MemoryChat(budget=budget, persistence_path=tmpdir)

        print(f"Using provider: {chat.nucleus.provider.name}")
        print(f"Budget: {budget.atp} ATP")
        print()
        print("Commands:")
        print("  /good    - Mark last response as good (promotes memory)")
        print("  /wrong   - Mark last response as wrong (reduces reliability)")
        print("  /important - Save to long-term memory")
        print("  /stats   - Show memory statistics")
        print("  /save    - Save long-term memories")
        print("  /quit    - Exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd = user_input[1:].lower()
                    if cmd in ("quit", "exit", "q"):
                        chat.save()
                        break
                    elif cmd == "good":
                        chat.feedback("good")
                    elif cmd == "wrong":
                        chat.feedback("wrong")
                    elif cmd == "important":
                        chat.feedback("important")
                    elif cmd == "stats":
                        stats = chat.stats()
                        print(f"📊 Memories: {stats}")
                    elif cmd == "save":
                        chat.save()
                    continue

                response = chat.chat(user_input)
                print(f"\nAssistant: {response.content}")
                print(f"   [{response.memories_used} memories used, {budget.atp} ATP remaining]")
                print()

            except KeyboardInterrupt:
                chat.save()
                print("\nSaved and exiting...")
                break


def run_smoke_test():
    """Automated smoke test."""
    print("Running smoke test...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        budget = ATP_Store(budget=200)
        chat = MemoryChat(budget=budget, persistence_path=tmpdir, silent=True)

        # Test 1: Basic chat
        response = chat.chat("Hello, how are you?")
        assert response.content, "Should get response"
        print(f"✓ Basic chat works")

        # Test 2: Memory storage
        stats = chat.stats()
        assert stats["working"] > 0, "Should store in working memory"
        print(f"✓ Memory storage works: {stats}")

        # Test 3: Feedback mechanism
        chat.feedback("important")
        stats = chat.stats()
        assert stats["longterm"] > 0, "Should promote to long-term"
        print(f"✓ Feedback promotes memory: {stats}")

        # Test 4: Persistence
        chat.save()
        chat2 = MemoryChat(persistence_path=tmpdir, silent=True)
        stats2 = chat2.stats()
        assert stats2["longterm"] > 0, "Should load persisted memories"
        print(f"✓ Persistence works: {stats2}")

        print("\nSmoke test passed!")


def main():
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
