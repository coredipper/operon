"""
Operon Compliance Review Pipeline -- Interactive Gradio Demo
=============================================================

Simulates a multi-stage compliance review pipeline where:
- MorphogenGradient carries document metadata (complexity, risk) across stages
- Cascade stages extract clauses, assess risk, and make decisions
- QuorumSensing activates when confidence is LOW for multi-agent voting
- NegativeFeedbackLoop adjusts confidence based on voting outcomes

Run locally:
    pip install gradio
    python space-compliance-review/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    MorphogenType,
    MorphogenGradient,
    ATP_Store,
)
from operon_ai.topology.quorum import QuorumSensing, VotingStrategy


# ── Data Structures ──────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"


VERDICT_STYLES: dict[ReviewVerdict, tuple[str, str]] = {
    ReviewVerdict.APPROVED: ("#22c55e", "APPROVED"),
    ReviewVerdict.CONDITIONAL: ("#eab308", "CONDITIONAL"),
    ReviewVerdict.REJECTED: ("#ef4444", "REJECTED"),
}


@dataclass
class DocumentClause:
    id: int
    text: str
    risk_indicators: list[str] = field(default_factory=list)
    risk_score: float = 0.0


RISK_KEYWORDS = {
    "indemnify": 0.3,
    "liability": 0.2,
    "penalty": 0.3,
    "termination": 0.2,
    "unlimited": 0.4,
    "irrevocable": 0.3,
    "waive": 0.3,
    "exclusive": 0.2,
    "non-compete": 0.3,
    "damages": 0.2,
    "confidential": 0.1,
    "warranty": 0.1,
}


# ── Presets ───────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "(custom)": {
        "description": "Configure your own document and parameters.",
        "document": "",
        "initial_complexity": 0.5,
        "initial_risk": 0.3,
        "quorum_budget": 500,
        "confidence_threshold": 0.5,
    },
    "Simple contract": {
        "description": "Low-risk supply agreement -- no risky clauses, quick approval.",
        "document": (
            "Agreement for Widget Supply\n\n"
            "This agreement is between Acme Corp and Widget Co for the supply "
            "of standard widgets.\n\n"
            "The supplier agrees to deliver 1000 widgets per month at "
            "$5.00 per unit.\n\n"
            "Payment terms are net 30 days from invoice date.\n\n"
            "Both parties agree to maintain confidential any proprietary "
            "information shared during this engagement."
        ),
        "initial_complexity": 0.2,
        "initial_risk": 0.1,
        "quorum_budget": 500,
        "confidence_threshold": 0.5,
    },
    "Ambiguous clauses": {
        "description": "Partnership agreement with termination penalties and indemnification -- triggers quorum.",
        "document": (
            "Strategic Partnership Agreement\n\n"
            "This partnership agreement covers joint development of "
            "the new product line.\n\n"
            "Either party may terminate this agreement with 30 days notice. "
            "The termination clause includes a penalty of 2x monthly revenue.\n\n"
            "All intellectual property developed jointly shall be shared "
            "equally, with exclusive licensing rights to each party in "
            "their respective territories.\n\n"
            "Both parties agree to indemnify each other against claims "
            "arising from the use of shared technology.\n\n"
            "A non-compete clause prevents either party from developing "
            "competing products for 24 months after termination."
        ),
        "initial_complexity": 0.5,
        "initial_risk": 0.3,
        "quorum_budget": 500,
        "confidence_threshold": 0.5,
    },
    "High-risk document": {
        "description": "Dangerous licensing agreement with unlimited liability and waived warranties.",
        "document": (
            "Enterprise Licensing Agreement\n\n"
            "The licensee accepts unlimited liability for any damages arising "
            "from use of the software, including consequential damages.\n\n"
            "The licensee irrevocably waives all warranty rights and agrees "
            "to indemnify the licensor against any claims.\n\n"
            "Termination may occur at licensor's sole discretion with "
            "a penalty equal to the remaining contract value.\n\n"
            "The licensee agrees to exclusive use of licensor's platform "
            "and a non-compete preventing use of alternative solutions.\n\n"
            "All disputes shall be resolved in licensor's jurisdiction. "
            "The licensee waives right to jury trial."
        ),
        "initial_complexity": 0.8,
        "initial_risk": 0.7,
        "quorum_budget": 500,
        "confidence_threshold": 0.5,
    },
    "Low confidence with quorum": {
        "description": "Moderate document but very low initial confidence -- forces quorum activation.",
        "document": (
            "Service Level Agreement\n\n"
            "Provider guarantees 99.9% uptime with penalty credits "
            "for any downtime exceeding the threshold.\n\n"
            "Liability for service failures is limited to the monthly "
            "service fee paid by the customer.\n\n"
            "Either party may terminate with 60 days written notice. "
            "Early termination incurs a penalty of remaining contract value.\n\n"
            "The customer agrees to indemnify the provider against "
            "third-party claims arising from misuse of the service."
        ),
        "initial_complexity": 0.4,
        "initial_risk": 0.3,
        "quorum_budget": 800,
        "confidence_threshold": 0.7,
    },
}


def _load_preset(name: str) -> tuple[str, float, float, int, float]:
    p = PRESETS.get(name, PRESETS["(custom)"])
    return (
        p["document"],
        p["initial_complexity"],
        p["initial_risk"],
        p["quorum_budget"],
        p["confidence_threshold"],
    )


# ── Core Simulation ──────────────────────────────────────────────────────


def _extract_clauses(
    document: str,
    gradient: MorphogenGradient,
) -> tuple[list[DocumentClause], list[str]]:
    """Stage 1: Extract clauses and update gradient."""
    paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
    clauses: list[DocumentClause] = []
    log_lines: list[str] = []
    total_risk = 0.0

    for i, para in enumerate(paragraphs):
        indicators = []
        risk_score = 0.0
        for keyword, weight in RISK_KEYWORDS.items():
            if keyword.lower() in para.lower():
                indicators.append(keyword)
                risk_score += weight

        clause = DocumentClause(
            id=i + 1,
            text=para[:200],
            risk_indicators=indicators,
            risk_score=min(1.0, risk_score),
        )
        clauses.append(clause)
        total_risk += risk_score

    num_clauses = len(clauses)
    complexity = min(1.0, num_clauses / 10.0)
    avg_risk = total_risk / max(1, num_clauses)

    gradient.set(MorphogenType.COMPLEXITY, complexity,
                 description=f"Document complexity ({num_clauses} clauses)")
    gradient.set(MorphogenType.RISK, min(1.0, avg_risk),
                 description=f"Average clause risk ({avg_risk:.2f})")

    log_lines.append(f"Extracted **{num_clauses}** clauses from document")
    log_lines.append(f"Complexity: **{complexity:.2f}** | Avg risk: **{avg_risk:.2f}**")

    risky = [c for c in clauses if c.risk_indicators]
    if risky:
        log_lines.append(f"Risky clauses found: **{len(risky)}**")
        for c in risky[:5]:
            log_lines.append(f"- Clause {c.id}: {', '.join(c.risk_indicators)} "
                             f"(score {c.risk_score:.2f})")

    return clauses, log_lines


def _assess_risk(
    clauses: list[DocumentClause],
    gradient: MorphogenGradient,
    budget: ATP_Store,
    confidence_threshold: float,
) -> tuple[RiskLevel, float, bool, str | None, list[str]]:
    """Stage 2: Risk assessment with optional quorum."""
    confidence = gradient.get(MorphogenType.CONFIDENCE)
    complexity = gradient.get(MorphogenType.COMPLEXITY)
    max_risk = max((c.risk_score for c in clauses), default=0.0)
    avg_risk = sum(c.risk_score for c in clauses) / max(1, len(clauses))
    log_lines: list[str] = []

    log_lines.append(f"Confidence: **{confidence:.2f}** | Complexity: **{complexity:.2f}**")

    need_quorum = confidence < confidence_threshold or (complexity > 0.6 and max_risk > 0.3)
    quorum_details = None

    if need_quorum and budget.consume(30, "quorum_review"):
        log_lines.append(f"Confidence ({confidence:.2f}) < threshold ({confidence_threshold:.2f}) "
                         "-- **activating Quorum (3 reviewers)**")

        quorum_budget = ATP_Store(budget=300, silent=True)
        quorum = QuorumSensing(
            n_agents=3,
            budget=quorum_budget,
            strategy=VotingStrategy.MAJORITY,
            silent=True,
        )

        risky_clauses = [c for c in clauses if c.risk_score > 0.2]
        risk_summary = ", ".join(
            f"clause {c.id} ({c.risk_score:.1f})"
            for c in risky_clauses[:5]
        )
        question = f"Should this contract be approved? Risky clauses: {risk_summary}"
        result = quorum.run_vote(question)

        quorum_details = (
            f"Decision: **{result.decision.value}** | "
            f"Reached: **{result.reached}** | "
            f"Score: **{result.weighted_score:.2f}**"
        )

        for vote in result.votes:
            log_lines.append(f"- {vote.agent_id}: **{vote.vote_type.value}** "
                             f"(confidence {vote.confidence:.2f})")

        if result.decision.value == "permit":
            gradient.set(MorphogenType.CONFIDENCE,
                         min(1.0, confidence + 0.2))
            log_lines.append("Quorum permitted -- confidence boosted +0.2")
        else:
            gradient.set(MorphogenType.CONFIDENCE,
                         max(0.0, confidence - 0.1))
            log_lines.append("Quorum blocked -- confidence reduced -0.1")
    elif not need_quorum:
        log_lines.append(f"Confidence ({confidence:.2f}) >= threshold ({confidence_threshold:.2f}) "
                         "-- **single reviewer (fast path)**")
        budget.consume(10, "single_review")
    else:
        log_lines.append("Insufficient budget for quorum -- falling back to single reviewer")
        budget.consume(10, "single_review_fallback")

    if max_risk >= 0.6 or avg_risk >= 0.4:
        risk_level = RiskLevel.CRITICAL
    elif max_risk >= 0.4 or avg_risk >= 0.25:
        risk_level = RiskLevel.HIGH
    elif max_risk >= 0.2 or avg_risk >= 0.1:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    gradient.set(MorphogenType.RISK, max_risk,
                 description=f"Max clause risk: {max_risk:.2f}")

    log_lines.append(f"Risk level: **{risk_level.value}** "
                     f"(max={max_risk:.2f}, avg={avg_risk:.2f})")

    return risk_level, confidence, need_quorum, quorum_details, log_lines


def _make_decision(
    risk_level: RiskLevel,
    clauses: list[DocumentClause],
    gradient: MorphogenGradient,
) -> tuple[ReviewVerdict, str, list[str]]:
    """Stage 3: Final decision with feedback-adjusted threshold."""
    confidence = gradient.get(MorphogenType.CONFIDENCE)
    risk = gradient.get(MorphogenType.RISK)
    log_lines: list[str] = []

    base_threshold = 0.4
    adjusted_threshold = base_threshold * (0.5 + confidence * 0.5)

    log_lines.append(f"Risk: **{risk:.2f}** | Confidence: **{confidence:.2f}** | "
                     f"Approval threshold: **{adjusted_threshold:.2f}**")
    log_lines.append(f"_(Lower confidence = stricter threshold via negative feedback)_")

    risk_factors = list(set(
        ind for c in clauses for ind in c.risk_indicators
    ))

    if risk <= adjusted_threshold and risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        verdict = ReviewVerdict.APPROVED
        recommendation = "Document meets review standards."
    elif risk <= adjusted_threshold * 1.5:
        verdict = ReviewVerdict.CONDITIONAL
        recommendation = f"Conditional approval. Address: {', '.join(risk_factors[:3])}"
    else:
        verdict = ReviewVerdict.REJECTED
        recommendation = f"Rejected due to: {', '.join(risk_factors[:5])}"

    log_lines.append(f"Verdict: **{verdict.value}** | {recommendation}")

    return verdict, recommendation, log_lines


def run_compliance_review(
    preset_name: str,
    document: str,
    initial_complexity: float,
    initial_risk: float,
    quorum_budget: int,
    confidence_threshold: float,
) -> tuple[str, str, str, str]:
    """Run the full compliance review pipeline.

    Returns (decision_banner_html, pipeline_stages_md, gradient_evolution_md, quorum_results_md).
    """
    if not document.strip():
        return "Enter a document to review.", "", "", ""

    # Initialize
    gradient = MorphogenGradient()
    gradient.set(MorphogenType.COMPLEXITY, initial_complexity,
                 description="Initial document complexity estimate")
    gradient.set(MorphogenType.RISK, initial_risk,
                 description="Initial risk estimate")
    gradient.set(MorphogenType.CONFIDENCE, 1.0 - initial_complexity * 0.5,
                 description="Initial confidence (inverse of complexity)")

    budget = ATP_Store(budget=int(quorum_budget), silent=True)

    # Track gradient evolution
    gradient_snapshots: list[dict] = []

    def snapshot(label: str) -> None:
        gradient_snapshots.append({
            "stage": label,
            "complexity": gradient.get(MorphogenType.COMPLEXITY),
            "confidence": gradient.get(MorphogenType.CONFIDENCE),
            "risk": gradient.get(MorphogenType.RISK),
        })

    snapshot("Initial")

    # ── Stage 1: Extract clauses ────────────────────────────────────────
    clauses, stage1_log = _extract_clauses(document, gradient)
    snapshot("After Stage 1: Extraction")

    # ── Stage 2: Risk assessment ────────────────────────────────────────
    risk_level, confidence, quorum_used, quorum_details, stage2_log = _assess_risk(
        clauses, gradient, budget, confidence_threshold,
    )
    snapshot("After Stage 2: Assessment")

    # ── Stage 3: Final decision ─────────────────────────────────────────
    verdict, recommendation, stage3_log = _make_decision(
        risk_level, clauses, gradient,
    )

    # Apply feedback: adjust confidence for future reviews
    if verdict == ReviewVerdict.APPROVED:
        current = gradient.get(MorphogenType.CONFIDENCE)
        gradient.set(MorphogenType.CONFIDENCE, min(1.0, current + 0.05))
    elif verdict == ReviewVerdict.REJECTED:
        current = gradient.get(MorphogenType.CONFIDENCE)
        gradient.set(MorphogenType.CONFIDENCE, max(0.0, current - 0.1))

    snapshot("After Stage 3: Decision + Feedback")

    # ── Decision banner ─────────────────────────────────────────────────
    color, label = VERDICT_STYLES.get(verdict, ("#888", str(verdict)))
    risk_color = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}.get(
        risk_level.value, "#888"
    )

    banner = (
        f'<div style="padding:12px 16px;border-radius:8px;'
        f'background:{color}20;border:2px solid {color};margin-bottom:8px">'
        f'<span style="font-size:1.3em;font-weight:700;color:{color}">'
        f'{label}</span>'
        f'<span style="color:#888;margin-left:12px">'
        f'Risk: <span style="color:{risk_color};font-weight:600">{risk_level.value}</span>'
        f' | Confidence: {gradient.get(MorphogenType.CONFIDENCE):.2f}'
        f' | Clauses: {len(clauses)}'
        f' | Quorum: {"yes" if quorum_used else "no"}</span>'
        f'<br><span style="color:#666;font-size:0.9em;margin-top:4px;display:inline-block">'
        f'{recommendation}</span></div>'
    )

    # ── Pipeline stages markdown ────────────────────────────────────────
    pipeline_parts = []
    pipeline_parts.append("### Stage 1: Clause Extraction\n")
    for line in stage1_log:
        pipeline_parts.append(f"- {line}")
    pipeline_parts.append("\n### Stage 2: Risk Assessment\n")
    for line in stage2_log:
        pipeline_parts.append(f"- {line}")
    pipeline_parts.append("\n### Stage 3: Final Decision\n")
    for line in stage3_log:
        pipeline_parts.append(f"- {line}")
    pipeline_md = "\n".join(pipeline_parts)

    # ── Gradient evolution markdown ─────────────────────────────────────
    gradient_lines = [
        "### Morphogen Gradient Evolution\n",
        "| Stage | Complexity | Confidence | Risk |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for snap in gradient_snapshots:
        c_color = "#ef4444" if snap["complexity"] > 0.6 else "#22c55e"
        conf_color = "#ef4444" if snap["confidence"] < 0.4 else "#22c55e"
        r_color = "#ef4444" if snap["risk"] > 0.4 else "#22c55e"
        gradient_lines.append(
            f'| {snap["stage"]} '
            f'| <span style="color:{c_color}">{snap["complexity"]:.2f}</span> '
            f'| <span style="color:{conf_color}">{snap["confidence"]:.2f}</span> '
            f'| <span style="color:{r_color}">{snap["risk"]:.2f}</span> |'
        )
    gradient_lines.append("\n### How Gradients Drive Decisions\n")
    gradient_lines.append("- **COMPLEXITY** rises with more clauses -- affects review depth")
    gradient_lines.append("- **CONFIDENCE** determines single-reviewer vs quorum path")
    gradient_lines.append("- **RISK** accumulates from keyword analysis -- drives final verdict")
    gradient_lines.append("- Negative feedback loop: low confidence tightens approval threshold")
    gradient_md = "\n".join(gradient_lines)

    # ── Quorum results markdown ─────────────────────────────────────────
    if quorum_used and quorum_details:
        quorum_parts = [
            "### Quorum Voting Results\n",
            quorum_details,
            "",
            "Quorum was activated because confidence fell below the threshold.",
            "Three independent reviewers voted on the document.",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Reviewers | 3 |",
            f"| Strategy | Majority |",
            f"| Budget consumed | 30 ATP |",
            f"| Confidence before | {confidence:.2f} |",
            f"| Confidence after | {gradient.get(MorphogenType.CONFIDENCE):.2f} |",
        ]
        quorum_md = "\n".join(quorum_parts)
    else:
        quorum_md = (
            "### Quorum Voting\n\n"
            "*Quorum was not activated.* Confidence was high enough for "
            "single-reviewer assessment."
        )

    return banner, pipeline_md, gradient_md, quorum_md


# ── Gradio UI ─────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Compliance Review Pipeline") as app:
        gr.Markdown(
            "# Compliance Review Pipeline\n"
            "Multi-stage document review using **MorphogenGradient** coordination, "
            "**QuorumSensing** for low-confidence escalation, and "
            "**NegativeFeedbackLoop** threshold adjustment.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Simple contract",
                label="Preset",
                scale=2,
            )
            run_btn = gr.Button("Run Review", variant="primary", scale=1)

        document_tb = gr.Textbox(
            lines=8,
            label="Document content",
            placeholder="Enter document text here (paragraphs separated by blank lines)...",
        )

        with gr.Row():
            complexity_sl = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05,
                label="Initial complexity",
            )
            risk_sl = gr.Slider(
                0.0, 1.0, value=0.1, step=0.05,
                label="Initial risk",
            )

        with gr.Row():
            budget_sl = gr.Slider(
                100, 2000, value=500, step=50,
                label="Quorum budget (ATP)",
            )
            threshold_sl = gr.Slider(
                0.3, 0.9, value=0.5, step=0.05,
                label="Confidence threshold (below = quorum)",
            )

        banner_html = gr.HTML(label="Decision")
        pipeline_md = gr.Markdown(label="Pipeline Stages")

        with gr.Row():
            with gr.Column():
                gradient_md = gr.Markdown(label="Gradient Evolution")
            with gr.Column():
                quorum_md = gr.Markdown(label="Quorum Results")

        # ── Event wiring ─────────────────────────────────────────────────
        preset_dd.change(
            fn=_load_preset,
            inputs=[preset_dd],
            outputs=[document_tb, complexity_sl, risk_sl, budget_sl, threshold_sl],
        )

        run_btn.click(
            fn=run_compliance_review,
            inputs=[preset_dd, document_tb, complexity_sl, risk_sl, budget_sl, threshold_sl],
            outputs=[banner_html, pipeline_md, gradient_md, quorum_md],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
