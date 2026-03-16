from __future__ import annotations

import json
import re
from typing import Any, Callable

from .types import ActionProtein, Signal
from ..organelles.chaperone import Chaperone
from ..organelles.membrane import Membrane
from ..organelles.mitochondria import Mitochondria
from ..organelles.nucleus import Nucleus
from ..providers import LLMResponse, ProviderConfig
from ..state.histone import HistoneStore
from ..state.metabolism import ATP_Store


ResponseMapper = Callable[[LLMResponse, Signal], ActionProtein]


class BioAgent:
    """
    Minimal agent abstraction with optional provider-bound execution.

    The legacy mock path is still the default so older examples and topology
    demos behave as before. When a Nucleus is attached, the same agent surface
    becomes provider-backed without losing membrane checks, ATP accounting, or
    histone-based context retrieval.
    """

    def __init__(
        self,
        name: str,
        role: str,
        atp_store: ATP_Store,
        nucleus: Nucleus | None = None,
        instructions: str = "",
        provider_config: ProviderConfig | None = None,
        tool_mitochondria: Mitochondria | None = None,
        response_mapper: ResponseMapper | None = None,
        silent: bool = False,
    ):
        self.name = name
        self.role = role
        self.atp = atp_store

        # Internal Organelles
        self.membrane = Membrane()
        self.mitochondria = Mitochondria()
        self.chaperone = Chaperone()

        # Epigenetics (State)
        self.histones = HistoneStore()

        # Optional provider-backed execution
        self.nucleus = nucleus
        self.instructions = instructions.strip()
        self.provider_config = provider_config
        self.tool_mitochondria = tool_mitochondria
        self.response_mapper = response_mapper
        self.silent = silent

    def express(self, signal: Signal) -> ActionProtein:
        """
        Process a signal into an ActionProtein.
        """
        if not self.silent:
            print(f"🧬 [{self.name}] Expressing...")

        # 1. Membrane Check (Prion Defense)
        filter_result = self.membrane.filter(signal)
        if not filter_result.allowed:
            return ActionProtein("BLOCK", "Blocked by Membrane (Prion Detected)", 1.0)

        # 2. Metabolic Check (Ischemia Defense)
        energy_cost = self.nucleus.base_energy_cost if self.nucleus is not None else 10
        if not self.atp.consume(cost=energy_cost, operation="agent_express"):
            return ActionProtein("FAILURE", "Apoptosis: Insufficient ATP", 0.0)

        # 3. Epigenetic Retrieval (RAG)
        retrieval = self.histones.retrieve_context(signal.content)
        prompt = self._build_prompt(signal, retrieval.formatted_context)

        # 4. Decision / transcription
        raw_output = (
            self._llm_express(prompt, signal)
            if self.nucleus is not None
            else self._mock_llm(prompt, signal)
        )

        # 5. Feedback (Writing State)
        if raw_output.action_type == "FAILURE":
            self.histones.add_marker(f"Avoid '{signal.content}' due to crash.")

        return raw_output

    def _build_prompt(self, signal: Signal, context: str) -> str:
        lines = [f"ROLE: {self.role}"]
        if self.instructions:
            lines.append(f"INSTRUCTIONS:\n{self.instructions}")
        if context:
            lines.append(f"MEMORY:\n{context}")

        shared_state = signal.metadata.get("shared_state")
        if shared_state:
            lines.append(f"SHARED STATE:\n{self._format_metadata_block(shared_state)}")

        stage_outputs = signal.metadata.get("stage_outputs")
        if stage_outputs:
            lines.append(
                f"PRIOR STAGE OUTPUTS:\n{self._format_metadata_block(stage_outputs)}"
            )

        extra_context = signal.metadata.get("context")
        if extra_context:
            lines.append(f"EXTRA CONTEXT:\n{self._format_metadata_block(extra_context)}")

        lines.append(f"INPUT: {signal.content}")
        return "\n\n".join(lines)

    def _llm_express(self, prompt: str, signal: Signal) -> ActionProtein:
        if self.tool_mitochondria is not None:
            response = self.nucleus.transcribe_with_tools(
                prompt,
                mitochondria=self.tool_mitochondria,
                config=self.provider_config,
            )
        else:
            response = self.nucleus.transcribe(prompt, config=self.provider_config)

        if self.response_mapper is not None:
            protein = self.response_mapper(response, signal)
        else:
            protein = self._default_response_mapper(response)

        protein.source_agent = protein.source_agent or self.name
        protein.metadata.setdefault("provider", self.nucleus.provider.name)
        protein.metadata.setdefault("model", response.model)
        protein.metadata.setdefault("tokens_used", response.tokens_used)
        protein.metadata.setdefault("latency_ms", response.latency_ms)
        protein.metadata.setdefault("instructions", self.instructions)
        return protein

    def _default_response_mapper(self, response: LLMResponse) -> ActionProtein:
        content = response.content.strip()
        action_type, payload = self._extract_explicit_action(content)

        if action_type is None:
            role = self.role.lower()
            if role in {"riskassessor", "reviewer", "voter"}:
                action_type = "BLOCK"
                payload = (
                    "Ambiguous reviewer output. "
                    "Return an explicit PERMIT: or BLOCK: decision.\n\n"
                    f"Raw output: {content}"
                )
            else:
                action_type = "EXECUTE"
                payload = content

        return ActionProtein(
            action_type=action_type,
            payload=payload,
            confidence=1.0,
            source_agent=self.name,
            metadata={},
        )

    @staticmethod
    def _extract_explicit_action(content: str) -> tuple[str | None, str]:
        match = re.match(
            r"^\s*(EXECUTE|PERMIT|BLOCK|DEFER|FAILURE|UNKNOWN)\s*[:\-]?\s*(.*)$",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return None, content
        action = match.group(1).upper()
        payload = match.group(2).strip() or content
        return action, payload

    @staticmethod
    def _format_metadata_block(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)

    def _mock_llm(self, prompt: str, signal: Signal) -> ActionProtein:
        """
        Simulate translation with deterministic logic for examples/tests.
        """
        content = signal.content
        content_lower = content.lower()

        # A. Risk Logic (for topologies / voting)
        if self.role in ("RiskAssessor", "Voter"):
            dangerous_markers = (
                "destroy",
                "delete all",
                "rm -rf",
                "wipe",
                "exfiltrate",
                "steal",
                "hack",
            )
            if any(marker in content_lower for marker in dangerous_markers):
                return ActionProtein("BLOCK", "Violates safety protocols.", 1.0)
            return ActionProtein("PERMIT", "Action is safe.", 1.0)

        # B. Neuro-symbolic calculation (executor only)
        if self.role == "Executor":
            match = re.search(r"\bcalculate\b(.*)$", content, flags=re.IGNORECASE)
            if match:
                math_expr = match.group(1).strip()
                looks_like_math = bool(
                    math_expr
                    and re.search(
                        r"[0-9]|[+\-*/()^]|\b(pi|e|tau)\b",
                        math_expr,
                        flags=re.IGNORECASE,
                    )
                )
                if looks_like_math:
                    result = self.mitochondria.digest_glucose(math_expr)
                    return ActionProtein("EXECUTE", f"Calculated: {result}", 1.0)

        # C. Executor logic
        if self.role == "Executor":
            if "Avoid" in prompt:
                return ActionProtein("BLOCK", "Suppressed by Epigenetic Memory.", 1.0)
            if "deploy" in signal.content.lower():
                return ActionProtein("FAILURE", "ConnectionRefusedError", 0.0)
            return ActionProtein("EXECUTE", f"Running: {signal.content}", 0.9)

        return ActionProtein("UNKNOWN", "No Instruction", 0.0)
