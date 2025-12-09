from typing import Optional
from .types import Signal, ActionProtein
from ..state.metabolism import ATP_Store
from ..state.histone import HistoneStore
from ..organelles.membrane import Membrane
from ..organelles.mitochondria import Mitochondria
from ..organelles.chaperone import Chaperone

class BioAgent:
    """
    The fundamental unit: A Polynomial Functor.
    Consumes Signals, produces ActionProteins, constrained by State and Energy.
    """
    def __init__(self, name: str, role: str, atp_store: ATP_Store):
        self.name = name
        self.role = role
        self.atp = atp_store
        
        # Internal Organelles
        self.membrane = Membrane()
        self.mitochondria = Mitochondria()
        self.chaperone = Chaperone()
        
        # Epigenetics (State)
        self.histones = HistoneStore()

    def express(self, signal: Signal) -> ActionProtein:
        """
        The Morphism: Input -> Output
        """
        print(f"🧬 [{self.name}] Expressing...")

        # 1. Membrane Check (Prion Defense)
        if not self.membrane.filter(signal):
            return ActionProtein("BLOCK", "Blocked by Membrane (Prion Detected)", 1.0)

        # 2. Metabolic Check (Ischemia Defense)
        if not self.atp.consume(cost=10):
            return ActionProtein("FAILURE", "Apoptosis: Insufficient ATP", 0.0)

        # 3. Epigenetic Retrieval (RAG)
        context = self.histones.retrieve_context(signal.content)
        
        # 4. Transcription (LLM Generation)
        # We inject the "methylations" into the System Prompt.
        prompt = f"ROLE: {self.role}\nMEMORY: {context}\nINPUT: {signal.content}"
        
        # --- Mock LLM Inference ---
        raw_output = self._mock_llm(prompt, signal)
        # --------------------------

        # 5. Feedback (Writing State)
        if raw_output.action_type == "FAILURE":
            self.histones.add_marker(f"Avoid '{signal.content}' due to crash.")

        return raw_output

    def _mock_llm(self, prompt: str, signal: Signal) -> ActionProtein:
        """
        Simulates the Ribosome (Translation).
        Includes Mitochondria usage (Neuro-symbolic).
        """
        # A. Neuro-Symbolic Calculation
        if "calculate" in signal.content.lower():
            math_expr = signal.content.split("calculate")[-1].strip()
            result = self.mitochondria.digest_glucose(math_expr)
            return ActionProtein("EXECUTE", f"Calculated: {result}", 1.0)

        # B. Risk Logic (for topologies)
        if self.role == "RiskAssessor":
            if "destroy" in signal.content.lower():
                return ActionProtein("BLOCK", "Violates safety protocols.", 1.0)
            return ActionProtein("PERMIT", "Action is safe.", 1.0)

        # C. Executor Logic
        if self.role == "Executor":
            if "Avoid" in prompt:
                return ActionProtein("BLOCK", "Suppressed by Epigenetic Memory.", 1.0)
            if "deploy" in signal.content.lower():
                return ActionProtein("FAILURE", "ConnectionRefusedError", 0.0)
            
            return ActionProtein("EXECUTE", f"Running: {signal.content}", 0.9)

        return ActionProtein("UNKNOWN", "No Instruction", 0.0)
