from ..core.agent import BioAgent
from ..core.types import Signal
from ..state.metabolism import ATP_Store

class CoherentFeedForwardLoop:
    """
    Theorem 4.1: The Guardrail.
    Executor (Z) only fires if RiskAssessor (Y) permits.
    """
    def __init__(self, budget: ATP_Store):
        self.executor = BioAgent("Gene_Z (Exec)", role="Executor", atp_store=budget)
        self.assessor = BioAgent("Gene_Y (Risk)", role="RiskAssessor", atp_store=budget)

    def run(self, user_prompt: str):
        signal = Signal(content=user_prompt)
        
        # Parallel Expression
        z_out = self.executor.express(signal)
        y_out = self.assessor.express(signal)

        # The Logic Gate
        if y_out.action_type == "BLOCK":
            print(f"🛑 BLOCKED by Risk Assessor: {y_out.payload}")
        elif z_out.action_type == "FAILURE":
            print(f"💥 RUNTIME ERROR: {z_out.payload}")
        elif z_out.action_type == "BLOCK":
            print(f"⏸️ SKIPPED by Executor Memory: {z_out.payload}")
        elif z_out.action_type == "EXECUTE" and y_out.action_type == "PERMIT":
            print(f"✅ SUCCESS: {z_out.payload}")
        else:
            print("⚠️ SYSTEM ERROR: Signal mismatch.")
