"""Speedup theorem (Theorem 3) empirical validation.

Validates that the predicted parallel speedup from epistemic topology
analysis matches measured wall-clock speedup with real latencies.

Uses time.sleep() handlers (no LLM calls needed) to create measurable
latencies, then compares:
  predicted = parallel_speedup(diagram).speedup
  measured  = sequential_time / parallel_time

Usage: python eval/speedup_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.core.epistemic import parallel_speedup
from operon_ai.core.types import DataType, IntegrityLabel
from operon_ai.core.wagent import ModuleSpec, PortType, ResourceCost, WiringDiagram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sleep_handler(ms: int):
    """Create a handler that sleeps for a fixed duration."""
    def handler(task, state, outputs, stage):
        time.sleep(ms / 1000.0)
        return f"done_{stage.name}_{ms}ms"
    return handler


def _build_diagram(stage_groups, latencies: dict[str, int]) -> WiringDiagram:
    """Build a WiringDiagram matching the organism's topology.

    Each stage becomes a module with cost = ResourceCost(atp=1, latency_ms=ms).
    Edges follow the stage_groups structure: sequential between groups,
    independent within groups.
    """
    port_in = PortType(DataType.JSON, IntegrityLabel.VALIDATED)
    port_out = PortType(DataType.JSON, IntegrityLabel.VALIDATED)

    diagram = WiringDiagram()
    for group in stage_groups:
        for stage in group:
            ms = latencies.get(stage.name, 100)
            diagram.add_module(ModuleSpec(
                name=stage.name,
                inputs={"in": port_in},
                outputs={"out": port_out},
                cost=ResourceCost(atp=1, latency_ms=ms),
            ))

    # Wire: sequential between groups, no wires within parallel groups
    flat_groups = list(stage_groups)
    for i in range(len(flat_groups) - 1):
        # Last stage of group i → first stage of group i+1
        src = flat_groups[i][-1].name
        dst = flat_groups[i + 1][0].name
        diagram.connect(src, "out", dst, "in")

    return diagram


@dataclass
class SpeedupResult:
    name: str
    predicted: float
    measured: float
    sequential_ms: float
    parallel_ms: float
    within_bound: bool
    overhead_pct: float


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def _make_nucleus():
    return Nucleus(provider=MockProvider())


CONFIGS = [
    {
        "name": "2-parallel-equal",
        "parallel_stages": [[
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
        ]],
        "sequential_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "latencies": {"a": 100, "b": 100},
    },
    {
        "name": "3-parallel-equal",
        "parallel_stages": [[
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ]],
        "sequential_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "latencies": {"a": 100, "b": 100, "c": 100},
    },
    {
        "name": "mixed-2par+1seq",
        "parallel_stages": [
            [
                SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
                SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            ],
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "sequential_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "latencies": {"a": 100, "b": 100, "c": 100},
    },
    {
        "name": "uneven-2par",
        "parallel_stages": [[
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(200), mode="fixed"),
        ]],
        "sequential_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(200), mode="fixed"),
        ],
        "latencies": {"a": 100, "b": 200},
    },
    {
        "name": "sequential-baseline",
        "parallel_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "sequential_stages": [
            SkillStage(name="a", role="A", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="b", role="B", handler=_make_sleep_handler(100), mode="fixed"),
            SkillStage(name="c", role="C", handler=_make_sleep_handler(100), mode="fixed"),
        ],
        "latencies": {"a": 100, "b": 100, "c": 100},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Speedup Theorem (Theorem 3) Empirical Validation")
    print("=" * 60)

    nucleus = _make_nucleus()
    results: list[SpeedupResult] = []

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n--- {name} ---")

        # Build diagram and compute predicted speedup
        par_org = skill_organism(
            stages=cfg["parallel_stages"],
            fast_nucleus=nucleus, deep_nucleus=nucleus,
        )
        diagram = _build_diagram(par_org.stage_groups, cfg["latencies"])
        predicted = parallel_speedup(diagram)

        print(f"  predicted speedup: {predicted.speedup:.2f}x")

        # Measure sequential
        seq_org = skill_organism(
            stages=cfg["sequential_stages"],
            fast_nucleus=nucleus, deep_nucleus=nucleus,
        )
        t0 = time.monotonic()
        seq_org.run("test")
        sequential_ms = (time.monotonic() - t0) * 1000

        # Measure parallel
        t0 = time.monotonic()
        par_org.run("test")
        parallel_ms = (time.monotonic() - t0) * 1000

        measured = sequential_ms / max(parallel_ms, 1)
        within_bound = measured <= predicted.speedup * 1.1  # 10% tolerance
        overhead = ((1.0 / measured - 1.0 / predicted.speedup) /
                    (1.0 / predicted.speedup)) * 100 if predicted.speedup > 1 else 0

        r = SpeedupResult(
            name=name,
            predicted=predicted.speedup,
            measured=measured,
            sequential_ms=sequential_ms,
            parallel_ms=parallel_ms,
            within_bound=within_bound,
            overhead_pct=overhead,
        )
        results.append(r)

        status = "OK" if within_bound else "EXCEEDS BOUND"
        print(f"  sequential:  {sequential_ms:.0f}ms")
        print(f"  parallel:    {parallel_ms:.0f}ms")
        print(f"  measured:    {measured:.2f}x")
        print(f"  status:      {status}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Config':<25s} {'Predicted':>10s} {'Measured':>10s} {'Status':>10s}")
    print(f"  {'-'*55}")
    for r in results:
        status = "OK" if r.within_bound else "EXCEEDS"
        print(f"  {r.name:<25s} {r.predicted:>9.2f}x {r.measured:>9.2f}x {status:>10s}")

    # Correlation
    if len(results) >= 3:
        from scipy.stats import spearmanr
        predicted = [r.predicted for r in results]
        measured = [r.measured for r in results]
        rho, p = spearmanr(predicted, measured)
        print(f"\n  Spearman rho: {rho:.3f} (p={p:.4f})")
        if rho > 0.8:
            print("  Strong positive correlation — theorem predicts measured speedup")
        elif rho > 0.5:
            print("  Moderate correlation")
        else:
            print("  Weak or no correlation")

    all_within = all(r.within_bound for r in results)
    print(f"\n  All within bound: {all_within}")

    # Save
    out_path = Path("eval/results/speedup_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import datetime
    out_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": [
            {
                "name": r.name,
                "predicted": r.predicted,
                "measured": round(r.measured, 3),
                "sequential_ms": round(r.sequential_ms, 1),
                "parallel_ms": round(r.parallel_ms, 1),
                "within_bound": r.within_bound,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
