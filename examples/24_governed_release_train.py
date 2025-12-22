#!/usr/bin/env python3
"""
Example 24: Governed Release Train
==================================

Demonstrates a release controller that combines:
- QuorumSensing for multi-agent consensus
- CoherentFeedForwardLoop for two-key gating
- NegativeFeedbackLoop to adjust strictness based on error rate
- CoordinationSystem for resource-aware execution
- Membrane filtering and Lysosome cleanup

Run:
    python examples/24_governed_release_train.py
"""

from __future__ import annotations

from operon_ai import (
    ATP_Store,
    CoherentFeedForwardLoop,
    GateLogic,
    Lysosome,
    Membrane,
    NegativeFeedbackLoop,
    QuorumSensing,
    Signal,
    ThreatLevel,
    VotingStrategy,
    Waste,
    WasteType,
)
from operon_ai.coordination import CoordinationSystem


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def run_release_step(step_name: str, release: dict) -> dict:
    prompt = release["prompt"].lower()

    if step_name == "deploy" and "refactor" in prompt:
        raise RuntimeError("Canary failed")

    if step_name == "monitor" and release["observed_error_rate"] > 0.15:
        return {"status": "alert", "reason": "error rate high"}

    return {"status": "ok", "step": step_name, "release_id": release["id"]}


def validate_step(result: dict) -> bool:
    return isinstance(result, dict) and result.get("status") == "ok"


def main() -> None:
    print("=" * 70)
    print("Governed Release Train")
    print("=" * 70)

    budget = ATP_Store(budget=300, silent=True)
    membrane = Membrane(threshold=ThreatLevel.DANGEROUS, silent=True)
    quorum = QuorumSensing(
        n_agents=5,
        budget=budget,
        strategy=VotingStrategy.MAJORITY,
        threshold=0.6,
        silent=True,
    )
    cffl = CoherentFeedForwardLoop(
        budget=budget,
        gate_logic=GateLogic.AND,
        enable_cache=False,
        silent=True,
    )
    feedback = NegativeFeedbackLoop(
        setpoint=0.1,
        gain=0.4,
        damping=0.1,
        min_correction=0.02,
        max_correction=0.2,
        silent=True,
    )
    coordination = CoordinationSystem()
    lysosome = Lysosome(silent=True)

    for resource in ("build_cluster", "deploy_slot", "metrics_channel"):
        coordination.register_resource(resource)

    release_queue = [
        {
            "id": "rel-101",
            "prompt": "Deploy checkout cache warmup",
            "observed_error_rate": 0.05,
        },
        {
            "id": "rel-102",
            "prompt": "Deploy auth service refactor",
            "observed_error_rate": 0.18,
        },
        {
            "id": "rel-103",
            "prompt": "Hotfix: patch token validation",
            "observed_error_rate": 0.12,
        },
    ]

    threshold = 0.6

    for release in release_queue:
        print("-" * 60)
        print(f"Release {release['id']}: {release['prompt']}")

        filter_result = membrane.filter(Signal(content=release["prompt"]))
        if not filter_result.allowed:
            print("Blocked by membrane:", filter_result.threat_level.name)
            continue

        correction = feedback.measure(release["observed_error_rate"])
        threshold = clamp(threshold - correction, 0.4, 0.9)
        quorum.set_strategy(VotingStrategy.MAJORITY, threshold=threshold)

        vote_result = quorum.run_vote(release["prompt"])
        print(
            f"Quorum: permits={vote_result.permit_votes}, blocks={vote_result.block_votes}, "
            f"threshold={vote_result.threshold_used:.2f}"
        )
        if not vote_result.reached:
            print("Quorum blocked release")
            continue

        gate_result = cffl.run(release["prompt"])
        if gate_result.blocked:
            print("CFFL blocked release:", gate_result.block_reason)
            continue

        steps = [
            ("build", "build_cluster"),
            ("deploy", "deploy_slot"),
            ("monitor", "metrics_channel"),
        ]

        for step_name, resource in steps:
            if not budget.consume(12, f"{release['id']}:{step_name}"):
                print("Insufficient ATP for", step_name)
                break

            result = coordination.execute_operation(
                operation_id=f"{release['id']}-{step_name}",
                agent_id="release_train",
                work_fn=lambda s=step_name, r=release: run_release_step(s, r),
                resources=[resource],
                validate_fn=validate_step,
                priority=1,
            )

            if not result.success:
                lysosome.ingest(
                    Waste(
                        WasteType.FAILED_OPERATION,
                        {
                            "release": release["id"],
                            "step": step_name,
                            "error": result.error,
                        },
                        source="release_train",
                    )
                )
                print(f"Step failed: {step_name} -> {result.error}")
                break

            print(f"Step ok: {step_name}")
        else:
            print("Release completed")

    digest = lysosome.digest()
    print("=" * 70)
    print(f"Lysosome digested: {digest.disposed}, recycled: {len(digest.recycled)}")
    print(f"Remaining ATP: {budget.atp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
