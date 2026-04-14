"""
Example 115: Quality-Based Escalation
=======================================

Demonstrates the adaptive immune layer: VerifierComponent evaluates
stage output quality via a rubric, and WatcherComponent escalates
from fast → deep model when quality falls below threshold.

1. Build a 2-stage organism with VerifierComponent + WatcherComponent
2. Configure a rubric that scores fast-model output low
3. Run — watch the verifier emit a low-quality signal
4. WatcherComponent detects low quality on fast model → ESCALATE
5. Organism re-runs the stage with the deep nucleus

Biological analogy:
  Innate immunity (WatcherComponent) detects generic anomalies.
  Adaptive immunity (VerifierComponent) evaluates against a specific
  rubric — like B-cells producing antibodies tailored to an antigen.

References:
  Rosset et al. (arXiv:2604.06240) — dual reward signals
  Ma et al. (arXiv:2604.05013) — atomic coding skills

Usage: python examples/115_quality_escalation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.verifier import VerifierComponent, VerifierConfig
from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig


# -----------------------------------------------------------------------
# Rubric: scores fast-model output low, deep-model output high
# -----------------------------------------------------------------------

FAST_RESPONSE = "quick fix: add try/except"
DEEP_RESPONSE = "Root-cause analysis: the session token was not refreshed on 401 retry. Fix: add token refresh in the retry interceptor, with exponential backoff."


def quality_rubric(output: str, stage_name: str) -> float:
    """Score output quality 0.0-1.0.

    The fast model produces a shallow fix (low quality).
    The deep model produces a thorough analysis (high quality).
    """
    if stage_name != "fix":
        return 0.8  # non-fix stages pass
    if "root-cause" in output.lower() or len(output) > 100:
        return 0.95  # thorough → high quality
    return 0.3  # shallow → low quality


def main():
    print("=" * 60)
    print("Quality-Based Escalation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build organism with verifier + watcher
    # ------------------------------------------------------------------
    # Use unique markers in instructions to avoid MockProvider
    # substring collisions across stages.
    fast = Nucleus(provider=MockProvider(responses={
        "xdiag7": "Likely a session bug",
        "xpatch9": FAST_RESPONSE,
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "xpatch9": DEEP_RESPONSE,
    }))

    watcher = WatcherComponent(config=WatcherConfig())
    verifier = VerifierComponent(
        rubric=quality_rubric,
        config=VerifierConfig(quality_low_threshold=0.5),
    )

    org = skill_organism(
        stages=[
            SkillStage(
                name="diagnose",
                role="Diagnostician",
                instructions="[xdiag7] Diagnose the reported issue.",
                mode="fixed",
            ),
            SkillStage(
                name="fix",
                role="Engineer",
                instructions="[xpatch9] Fix the diagnosed issue with a thorough solution.",
                mode="fixed",   # starts on fast model
            ),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        budget=ATP_Store(budget=1000, silent=True),
        components=[watcher, verifier],
    )

    # ------------------------------------------------------------------
    # 2. Run — expect escalation on the "fix" stage
    # ------------------------------------------------------------------
    print("\n--- Run ---")
    result = org.run("Login fails intermittently after session timeout")

    print(f"  final output:   {result.final_output[:80]}...")
    print(f"  stages:         {[sr.stage_name for sr in result.stage_results]}")

    # ------------------------------------------------------------------
    # 3. Inspect verifier quality scores
    # ------------------------------------------------------------------
    print("\n--- Verifier Quality Scores ---")
    for stage_name, quality in verifier.quality_scores:
        verdict = "PASS" if quality >= 0.5 else "LOW → triggers escalation"
        print(f"  {stage_name}: {quality:.2f} ({verdict})")

    print(f"  mean quality:   {verifier.mean_quality():.2f}")

    # ------------------------------------------------------------------
    # 4. Inspect watcher interventions
    # ------------------------------------------------------------------
    print("\n--- Watcher Interventions ---")
    if watcher.interventions:
        for intv in watcher.interventions:
            print(f"  {intv.stage_name}: {intv.kind.value} — {intv.reason}")
    else:
        print("  (none)")

    # ------------------------------------------------------------------
    # 5. Inspect watcher signals
    # ------------------------------------------------------------------
    print("\n--- Watcher Signals (verifier-sourced) ---")
    verifier_signals = [s for s in watcher.signals if s.source == "verifier"]
    for sig in verifier_signals:
        print(f"  stage={sig.stage_name} value={sig.value:.2f} "
              f"quality={sig.detail.get('quality', '?'):.2f} "
              f"below_threshold={sig.detail.get('below_threshold')}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n--- Assertions ---")

    # Verifier scored fix stage low initially
    fix_scores = [(s, q) for s, q in verifier.quality_scores if s == "fix"]
    assert len(fix_scores) >= 1, "verifier should score fix stage"
    initial_fix_quality = fix_scores[0][1]
    assert initial_fix_quality < 0.5, (
        f"fast model fix quality should be < 0.5, got {initial_fix_quality}"
    )

    # Watcher escalated
    escalations = [i for i in watcher.interventions
                   if i.kind.value == "escalate"]
    assert len(escalations) >= 1, "watcher should escalate on low quality"
    assert escalations[0].stage_name == "fix"

    # Final output is from deep model (the escalated response)
    assert "root-cause" in result.final_output.lower() or len(result.final_output) > 100, (
        "final output should be from deep model after escalation"
    )

    # Verifier signal for fix stage was deposited with correct metadata
    fix_signals = [s for s in verifier_signals if s.stage_name == "fix"]
    assert len(fix_signals) >= 1, "watcher should have verifier signal for fix"
    assert fix_signals[0].detail.get("below_threshold") is True

    print("  all assertions passed ✓")


if __name__ == "__main__":
    main()
