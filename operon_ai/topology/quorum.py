from typing import List
from ..core.agent import BioAgent
from ..core.types import Signal
from ..state.metabolism import ATP_Store

class QuorumSensing:
    """
    Consensus Topology: Mixture of Experts.
    """
    def __init__(self, n_agents: int, budget: ATP_Store):
        self.colony = [BioAgent(f"Bacterium_{i}", "Voter", budget) for i in range(n_agents)]
    
    def run_vote(self, prompt: str):
        print(f"\n📢 [Quorum] Initiating Vote on: '{prompt}'")
        signal = Signal(prompt)
        votes = 0
        
        for agent in self.colony:
            protein = agent.express(signal)
            if protein.action_type == "PERMIT" or protein.action_type == "EXECUTE":
                votes += 1
        
        print(f"📊 [Quorum] Concentration: {votes}/{len(self.colony)}")
        
        if votes > len(self.colony) / 2:
            print("✅ QUORUM REACHED: Action Committed.")
        else:
            print("❌ QUORUM FAILED: Signal too weak.")
