from ..core.types import Signal

class Membrane:
    """
    Prion Defense: Sanitizes inputs before they reach the nucleus.
    Prevents 'Prompt Injection' (Misfolded Inputs).
    """
    def filter(self, signal: Signal) -> bool:
        forbidden = ["ignore previous", "system prompt", "jailbreak"]
        if any(bad in signal.content.lower() for bad in forbidden):
            print(f"🛡️ [Membrane] PRION DETECTED. Blocking signal.")
            return False
        return True
