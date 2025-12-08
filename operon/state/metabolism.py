class ATP_Store:
    """
    Manages the Metabolic Budget (Token Limits).
    Prevents 'Ischemia' (Resource Exhaustion).
    """
    def __init__(self, budget: int):
        self.atp = budget

    def consume(self, cost: int) -> bool:
        if self.atp < cost:
            print("💀 [Metabolism] APOPTOSIS TRIGGERED: Insufficient ATP (Tokens).")
            return False
        self.atp -= cost
        return True
