"""Tests for WAgent typed wiring and approval tokens."""

import hashlib

import pytest

from operon_ai.core.types import Capability, DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, WiringDiagram, WiringError
from operon_ai.state.metabolism import ATP_Store
from operon_ai.topology.loops import CoherentFeedForwardLoop


def test_port_type_integrity_flow():
    """Integrity labels must not flow from lower -> higher."""
    untrusted_text = PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)
    trusted_text = PortType(DataType.TEXT, IntegrityLabel.TRUSTED)

    assert trusted_text.can_flow_to(untrusted_text) is True
    assert untrusted_text.can_flow_to(trusted_text) is False

    with pytest.raises(WiringError):
        untrusted_text.require_flow_to(trusted_text)


def test_wiring_diagram_checks_types_and_labels():
    """WiringDiagram rejects ill-typed or integrity-violating wires."""
    diagram = WiringDiagram()

    diagram.add_module(
        ModuleSpec(
            name="source",
            outputs={"out": PortType(DataType.TEXT, IntegrityLabel.UNTRUSTED)},
        )
    )
    diagram.add_module(
        ModuleSpec(
            name="sink",
            inputs={"in": PortType(DataType.TEXT, IntegrityLabel.TRUSTED)},
            capabilities={Capability.WRITE_FS},
        )
    )

    with pytest.raises(WiringError):
        diagram.connect("source", "out", "sink", "in")

    assert diagram.required_capabilities() == {Capability.WRITE_FS}


def test_cffl_mints_approval_token_on_permit():
    """CFFL success path includes an ApprovalToken from the assessor."""
    budget = ATP_Store(budget=1000, silent=True)
    cffl = CoherentFeedForwardLoop(budget=budget, silent=True)

    prompt = "List all files"
    result = cffl.run(prompt)

    assert result.blocked is False
    assert result.approval_token is not None
    assert result.approval_token.issuer == cffl.assessor.name
    assert result.approval_token.request_hash == hashlib.sha256(prompt.encode()).hexdigest()[:16]

