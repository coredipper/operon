#!/usr/bin/env python3
"""
Example 23: Resilient Incident Response Pipeline
===============================================

Demonstrates a multi-organelle incident response flow:
- Membrane filtering for unsafe inputs
- Ribosome prompt synthesis
- Nucleus (MockProvider) for deterministic triage/plan outputs
- Chaperone for schema enforcement
- CoordinationSystem for resource-aware execution
- Mitochondria for safe computation
- Proteasome for quality gating and repair
- Lysosome cleanup for failures

Run:
    python examples/23_resilient_incident_response.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from operon_ai import (
    ATP_Store,
    Chaperone,
    Lysosome,
    Membrane,
    Mitochondria,
    Ribosome,
    Signal,
    ThreatLevel,
    Waste,
    WasteType,
)
from operon_ai.coordination import CoordinationSystem
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
from operon_ai.quality import (
    ChaperoneRepair,
    DegronType,
    DegradationResult,
    Proteasome,
    ProvenanceContext,
    UbiquitinPool,
)


class TriageReport(BaseModel):
    severity: str
    impact: str
    suspected_causes: list[str]
    sla_breach_risk: float = Field(ge=0.0, le=1.0)
    recommended_response: str


class PlanStep(BaseModel):
    id: int
    action: str
    resource: str
    risk: str


class RemediationPlan(BaseModel):
    summary: str
    steps: list[PlanStep]
    rollback: str


def build_nucleus() -> Nucleus:
    triage_response = """{
      "severity": "SEV-2",
      "impact": "Elevated 5xx on checkout",
      "suspected_causes": ["cache eviction", "db pool saturation"],
      "sla_breach_risk": 0.35,
      "recommended_response": "Scale web tier and warm cache"
    }"""

    plan_response = """{
      "summary": "Stabilize checkout errors and restore latency",
      "steps": [
        {"id": 1, "action": "compute_error_rate", "resource": "metrics_db", "risk": "medium"},
        {"id": 2, "action": "increase_replicas", "resource": "orchestrator", "risk": "low"},
        {"id": 3, "action": "flip_traffic", "resource": "traffic_manager", "risk": "high"}
      ],
      "rollback": "Revert traffic shift and reset autoscaler"
    }"""

    responses = {
        "triage_report": triage_response,
        "remediation_plan": plan_response,
    }

    return Nucleus(provider=MockProvider(responses=responses))


def build_repair_chaperone() -> ChaperoneRepair:
    def can_repair(data: dict, _tag) -> bool:
        return isinstance(data, dict) and isinstance(data.get("error_rate_pct"), str)

    def repair(data: dict, _tag) -> tuple[dict, bool]:
        raw_value = data.get("error_rate_pct", "")
        value = raw_value.rstrip("%")
        try:
            repaired = dict(data)
            repaired["error_rate_pct"] = float(value)
            return repaired, True
        except ValueError:
            return data, False

    return ChaperoneRepair(
        name="metric_normalizer",
        can_repair=can_repair,
        repair=repair,
        confidence_boost=0.35,
    )


def run_step(step: PlanStep, mito: Mitochondria) -> dict:
    if step.action == "compute_error_rate":
        metrics = {"errors": 72, "requests": 1400}
        expression = f"{metrics['errors']} / {metrics['requests']} * 100"
        result = mito.metabolize(expression)
        if not result.success or not result.atp:
            raise RuntimeError(result.error or "Mitochondria failed")
        return {"error_rate_pct": f"{result.atp.value:.2f}%"}

    if step.action == "increase_replicas":
        return {"replicas": 12, "status": "scaled"}

    if step.action == "flip_traffic":
        raise RuntimeError("Traffic manager timed out")

    return {"status": "noop"}


def validate_for_step(step: PlanStep):
    def validate(result: dict) -> bool:
        if step.action == "compute_error_rate":
            return "error_rate_pct" in result
        if step.action == "increase_replicas":
            return "replicas" in result
        if step.action == "flip_traffic":
            return "traffic_shift" in result
        return True

    return validate


def main() -> None:
    print("=" * 70)
    print("Resilient Incident Response Pipeline")
    print("=" * 70)

    incident = "Checkout error spikes after cache deploy"

    membrane = Membrane(threshold=ThreatLevel.DANGEROUS, silent=True)
    filter_result = membrane.filter(Signal(content=incident))
    if not filter_result.allowed:
        print("Blocked by membrane:", filter_result.threat_level.name)
        return

    budget = ATP_Store(budget=120, silent=True)
    ribosome = Ribosome()
    nucleus = build_nucleus()
    chaperone = Chaperone(silent=True)
    mito = Mitochondria(silent=True)
    coordination = CoordinationSystem()
    lysosome = Lysosome(silent=True)

    for resource in ("metrics_db", "orchestrator", "traffic_manager"):
        coordination.register_resource(resource)

    pool = UbiquitinPool(capacity=10)
    proteasome = Proteasome(degradation_threshold=0.3, block_threshold=0.1)
    proteasome.chaperones.append(build_repair_chaperone())
    proteasome.fallback_strategy = lambda data, _tag: {
        "status": "degraded",
        "data": data,
    }

    ribosome.create_template(
        name="triage_prompt",
        sequence=(
            "TRIAGE_REPORT\n"
            "Incident: {{incident}}\n"
            "Return JSON with severity, impact, suspected_causes, sla_breach_risk, recommended_response."
        ),
        description="Incident triage report template"
    )
    ribosome.create_template(
        name="plan_prompt",
        sequence=(
            "REMEDIATION_PLAN\n"
            "Incident: {{incident}}\n"
            "Triage: {{triage}}\n"
            "Return JSON with summary, steps[id,action,resource,risk], rollback."
        ),
        description="Remediation plan generation template"
    )

    if not budget.consume(15, "triage"):
        print("Insufficient ATP for triage")
        return

    triage_prompt = ribosome.translate("triage_prompt", incident=incident).sequence
    triage_response = nucleus.transcribe(
        triage_prompt,
        config=ProviderConfig(temperature=0.0, max_tokens=256),
    )
    triage_fold = chaperone.fold_enhanced(triage_response.content, TriageReport)
    if not triage_fold.valid or not triage_fold.structure:
        lysosome.ingest(Waste(WasteType.MISFOLDED_PROTEIN, triage_response.content, source="triage"))
        print("Triage output failed schema validation")
        return

    triage = triage_fold.structure
    print("Triage:", triage.model_dump())

    if not budget.consume(20, "planning"):
        print("Insufficient ATP for planning")
        return

    plan_prompt = ribosome.translate(
        "plan_prompt",
        incident=incident,
        triage=triage.model_dump_json(),
    ).sequence
    plan_response = nucleus.transcribe(
        plan_prompt,
        config=ProviderConfig(temperature=0.0, max_tokens=512),
    )
    plan_fold = chaperone.fold_enhanced(plan_response.content, RemediationPlan)
    if not plan_fold.valid or not plan_fold.structure:
        lysosome.ingest(Waste(WasteType.MISFOLDED_PROTEIN, plan_response.content, source="plan"))
        print("Plan output failed schema validation")
        return

    plan = plan_fold.structure
    print("Plan summary:", plan.summary)

    risk_confidence = {
        "low": 0.9,
        "medium": 0.25,
        "high": 0.15,
    }

    for step in plan.steps:
        cost = 10 if step.risk == "low" else 15
        if not budget.consume(cost, f"step_{step.id}"):
            print(f"Insufficient ATP for step {step.id}")
            break

        print("-" * 60)
        print(f"Executing step {step.id}: {step.action} ({step.resource})")

        result = coordination.execute_operation(
            operation_id=f"incident-step-{step.id}",
            agent_id="ops_pipeline",
            work_fn=lambda s=step: run_step(s, mito),
            resources=[step.resource],
            validate_fn=validate_for_step(step),
            priority=1,
        )

        if not result.success:
            lysosome.ingest(
                Waste(
                    WasteType.FAILED_OPERATION,
                    {"step": step.id, "error": result.error},
                    source="coordination",
                )
            )
            print(f"Step failed: {result.error}")
            continue

        tag = pool.allocate(
            origin="ops_pipeline",
            confidence=risk_confidence.get(step.risk, 0.5),
            degron=DegronType.NORMAL,
        )
        if not tag:
            print("No ubiquitin tags available, skipping quality gate")
            continue

        context = ProvenanceContext(
            tag=tag,
            source_module=step.resource,
            target_module="incident_pipeline",
            metadata={"action": step.action},
        )
        inspected, updated_tag, status = proteasome.inspect(
            result.result,
            tag,
            context,
            pool,
        )

        print(f"Quality gate: {status.value} (confidence {updated_tag.confidence:.2f})")

        if status in (DegradationResult.BLOCKED, DegradationResult.QUEUED_REVIEW):
            lysosome.ingest(
                Waste(
                    WasteType.MISFOLDED_PROTEIN,
                    {"step": step.id, "output": result.result, "status": status.value},
                    source="proteasome",
                )
            )
            print("Output held for review")
            continue

        print("Step output:", inspected)

    digest = lysosome.digest()
    print("=" * 70)
    print(f"Lysosome digested: {digest.disposed}, recycled: {len(digest.recycled)}")
    print(f"Remaining ATP: {budget.atp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
