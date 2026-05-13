from operon_ai.patterns.review import reviewer_gate

def test_reviewer_gate_pass():
    """Test that reviewer gate allows execution when reviewer approves."""
    gate = reviewer_gate(
        executor=lambda prompt: "execution result",
        reviewer=lambda prompt, payload: True  # True indicates approval
    )
    result = gate.run("test prompt")

    assert result.allowed is True
    assert result.status == "allowed"
    assert result.output == "execution result"

def test_reviewer_gate_fail():
    """Test that reviewer gate blocks execution when reviewer rejects."""
    gate = reviewer_gate(
        executor=lambda prompt: "execution result",
        reviewer=lambda prompt, payload: False  # False indicates rejection
    )
    result = gate.run("test prompt")

    assert result.allowed is False
    assert result.status == "blocked"

def test_reviewer_gate_error():
    """Test that reviewer gate handles errors gracefully."""
    def raising_executor(prompt):
        raise ValueError("Simulated executor error")

    gate = reviewer_gate(
        executor=raising_executor,
        reviewer=lambda prompt, payload: True
    )
    result = gate.run("test prompt")

    assert result.allowed is False
    assert result.status == "blocked"
    assert "Simulated executor error" in result.reason
