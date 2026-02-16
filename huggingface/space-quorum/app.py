"""
Operon Quorum Sensing -- Multi-Agent Voting Simulator
=====================================================

Configure a panel of agents (name, weight, vote, confidence), pick a
voting strategy, and run the vote.  The app shows the aggregated result
and a comparison across all 7 strategies.

No API keys required -- votes are supplied directly, and the real
aggregation code from QuorumSensing._aggregate_votes() does the math.

Run locally:
    pip install gradio
    python space-quorum/app.py

Deploy to HuggingFace Spaces:
    Copy this directory to a new HF Space with sdk=gradio.
"""

import sys
from pathlib import Path

import gradio as gr

# Allow importing operon_ai from the repo root when running locally
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from operon_ai import (
    QuorumSensing,
    VotingStrategy,
    VoteType,
    Vote,
    QuorumResult,
    ATP_Store,
)


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

VOTE_MAP = {"Permit": VoteType.PERMIT, "Block": VoteType.BLOCK, "Abstain": VoteType.ABSTAIN}

# Each preset: list of (name, weight, vote_str, confidence)
PRESETS: dict[str, dict] = {
    "(custom)": {"agents": [], "strategy": "Majority", "threshold": ""},
    "Unanimous agreement": {
        "agents": [
            ("Sentinel", 1.0, "Permit", 0.95),
            ("Guardian", 1.0, "Permit", 0.90),
            ("Watcher", 1.0, "Permit", 0.85),
            ("Auditor", 1.0, "Permit", 0.92),
            ("Verifier", 1.0, "Permit", 0.88),
        ],
        "strategy": "Unanimous",
        "threshold": "",
    },
    "Split vote (3-2)": {
        "agents": [
            ("Alpha", 1.0, "Permit", 0.80),
            ("Beta", 1.0, "Permit", 0.75),
            ("Gamma", 1.0, "Permit", 0.70),
            ("Delta", 1.0, "Block", 0.85),
            ("Epsilon", 1.0, "Block", 0.90),
        ],
        "strategy": "Majority",
        "threshold": "",
    },
    "Expert override": {
        "agents": [
            ("Expert", 5.0, "Block", 0.95),
            ("Junior-1", 1.0, "Permit", 0.70),
            ("Junior-2", 1.0, "Permit", 0.65),
            ("Junior-3", 1.0, "Permit", 0.60),
            ("Junior-4", 1.0, "Permit", 0.55),
        ],
        "strategy": "Weighted",
        "threshold": "",
    },
    "Low confidence": {
        "agents": [
            ("Uncertain-1", 1.0, "Permit", 0.20),
            ("Uncertain-2", 1.0, "Permit", 0.25),
            ("Uncertain-3", 1.0, "Permit", 0.15),
            ("Hesitant", 1.0, "Permit", 0.30),
            ("Guessing", 1.0, "Permit", 0.10),
        ],
        "strategy": "Confidence",
        "threshold": "",
    },
    "Emergency quorum": {
        "agents": [
            ("Responder-1", 1.0, "Permit", 0.90),
            ("Responder-2", 1.0, "Permit", 0.85),
            ("Offline-1", 1.0, "Abstain", 0.00),
            ("Offline-2", 1.0, "Abstain", 0.00),
            ("Offline-3", 1.0, "Abstain", 0.00),
        ],
        "strategy": "Threshold",
        "threshold": "2",
    },
    "Byzantine voting": {
        "agents": [
            ("Malicious-1", 2.0, "Block", 0.95),
            ("Malicious-2", 2.0, "Block", 0.90),
            ("Honest-1", 1.0, "Permit", 0.85),
            ("Honest-2", 1.0, "Permit", 0.80),
            ("Honest-3", 1.0, "Permit", 0.75),
        ],
        "strategy": "Weighted",
        "threshold": "",
    },
    "Abstention majority": {
        "agents": [
            ("Abstainer-1", 1.0, "Abstain", 0.50),
            ("Abstainer-2", 1.0, "Abstain", 0.50),
            ("Abstainer-3", 1.0, "Abstain", 0.50),
            ("Voter-1", 1.0, "Permit", 0.80),
            ("Voter-2", 1.0, "Block", 0.85),
        ],
        "strategy": "Threshold",
        "threshold": "1",
    },
    "Dictatorial weight": {
        "agents": [
            ("Dictator", 100.0, "Block", 0.99),
            ("Citizen-1", 1.0, "Permit", 0.90),
            ("Citizen-2", 1.0, "Permit", 0.85),
            ("Citizen-3", 1.0, "Permit", 0.80),
            ("Citizen-4", 1.0, "Permit", 0.75),
        ],
        "strategy": "Weighted",
        "threshold": "",
    },
}

STRATEGY_MAP = {
    "Majority": VotingStrategy.MAJORITY,
    "Supermajority": VotingStrategy.SUPERMAJORITY,
    "Unanimous": VotingStrategy.UNANIMOUS,
    "Weighted": VotingStrategy.WEIGHTED,
    "Confidence": VotingStrategy.CONFIDENCE,
    "Bayesian": VotingStrategy.BAYESIAN,
    "Threshold": VotingStrategy.THRESHOLD,
}

STRATEGY_DESCRIPTIONS = {
    "Majority": ">50% permits required",
    "Supermajority": ">66% permits required",
    "Unanimous": "All must permit (zero blocks)",
    "Weighted": "Weight-adjusted majority (weight * confidence)",
    "Confidence": "Only votes above 0.3 confidence count",
    "Bayesian": "Bayesian belief update from uniform prior",
    "Threshold": "Fixed count of permits required",
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_votes(
    names: list[str],
    weights: list[float],
    vote_strs: list[str],
    confidences: list[float],
) -> list[Vote]:
    """Build Vote objects from parallel lists of agent data."""
    votes = []
    for name, weight, vote_str, conf in zip(names, weights, vote_strs, confidences):
        if not name.strip():
            continue
        vote_type = VOTE_MAP.get(vote_str, VoteType.ABSTAIN)
        votes.append(Vote(
            agent_id=name.strip(),
            vote_type=vote_type,
            confidence=conf,
            weight=weight,
        ))
    return votes


def _run_with_strategy(
    quorum: QuorumSensing,
    votes: list[Vote],
    strategy: VotingStrategy,
    threshold: float | None,
) -> QuorumResult:
    """Run aggregation with a specific strategy."""
    quorum.strategy = strategy
    quorum.custom_threshold = threshold
    return quorum._aggregate_votes(votes)


def run_simulation(
    name1, weight1, vote1, conf1,
    name2, weight2, vote2, conf2,
    name3, weight3, vote3, conf3,
    name4, weight4, vote4, conf4,
    name5, weight5, vote5, conf5,
    strategy_str, threshold_str,
) -> tuple[str, str, str]:
    """Run the quorum vote simulation.

    Returns (decision_html, vote_breakdown_md, strategy_comparison_md).
    """
    names = [name1, name2, name3, name4, name5]
    weights = [weight1, weight2, weight3, weight4, weight5]
    vote_strs = [vote1, vote2, vote3, vote4, vote5]
    confidences = [conf1, conf2, conf3, conf4, conf5]

    votes = _build_votes(names, weights, vote_strs, confidences)
    if not votes:
        return "Configure at least one agent.", "", ""

    strategy = STRATEGY_MAP.get(strategy_str, VotingStrategy.MAJORITY)
    threshold = float(threshold_str) if threshold_str.strip() else None

    # Create QuorumSensing with matching agent count
    budget = ATP_Store(budget=500)
    n = len(votes)
    quorum = QuorumSensing(n_agents=n, budget=budget, strategy=strategy, threshold=threshold, silent=True)

    # Run primary vote
    result = _run_with_strategy(quorum, votes, strategy, threshold)

    # --- Decision banner ---
    if result.reached:
        decision_color = "#16a34a" if result.decision == VoteType.PERMIT else "#dc2626"
        decision_bg = "#f0fdf4" if result.decision == VoteType.PERMIT else "#fef2f2"
        border = "#22c55e" if result.decision == VoteType.PERMIT else "#ef4444"
        reached_text = "QUORUM REACHED"
    else:
        decision_color = "#9ca3af"
        decision_bg = "#f9fafb"
        border = "#d1d5db"
        reached_text = "QUORUM NOT REACHED"

    decision_label = result.decision.value.upper()

    decision_html = (
        f'<div style="padding:16px;border-radius:8px;border:2px solid {border};background:{decision_bg};">'
        f'<div style="font-size:1.4em;font-weight:700;color:{decision_color};margin-bottom:4px;">'
        f'{decision_label}</div>'
        f'<div style="font-size:0.95em;color:{decision_color};">{reached_text}</div>'
        f'<div style="margin-top:8px;display:flex;gap:20px;flex-wrap:wrap;font-size:0.9em;">'
        f'<span>Strategy: <b>{strategy_str}</b></span>'
        f'<span>Score: <b>{result.weighted_score:.2%}</b></span>'
        f'<span>Threshold: <b>{result.threshold_used:.2%}</b></span>'
        f'<span>Confidence: <b>{result.confidence_score:.2%}</b></span>'
        f'</div>'
        f'</div>'
    )

    # --- Vote breakdown table ---
    breakdown_md = "| Agent | Vote | Weight | Confidence | Effective Weight | Aligned |\n"
    breakdown_md += "|-------|------|--------|------------|------------------|--------|\n"
    for v in votes:
        vote_emoji = {"permit": "+", "block": "-", "abstain": "~"}.get(v.vote_type.value, "?")
        aligned = "Yes" if v.vote_type == result.decision else ("--" if v.vote_type == VoteType.ABSTAIN else "No")
        breakdown_md += (
            f"| {v.agent_id} | {vote_emoji} {v.vote_type.value.capitalize()} "
            f"| {v.weight:.1f} | {v.confidence:.2f} "
            f"| {v.effective_weight:.2f} | {aligned} |\n"
        )
    breakdown_md += (
        f"\n**Totals:** {result.permit_votes} permit, "
        f"{result.block_votes} block, {result.abstain_votes} abstain "
        f"(of {result.total_votes} votes)"
    )

    # --- All-strategies comparison ---
    comparison_md = "| Strategy | Decision | Reached | Score | Threshold | Description |\n"
    comparison_md += "|----------|----------|---------|-------|-----------|-------------|\n"
    for s_name, s_enum in STRATEGY_MAP.items():
        # Use the custom threshold only for the selected strategy; others use defaults
        s_threshold = threshold if s_name == strategy_str else None
        s_result = _run_with_strategy(quorum, votes, s_enum, s_threshold)
        reached_icon = "Yes" if s_result.reached else "No"
        s_desc = STRATEGY_DESCRIPTIONS[s_name]
        marker = " **<<**" if s_name == strategy_str else ""
        comparison_md += (
            f"| {s_name}{marker} | {s_result.decision.value.capitalize()} "
            f"| {reached_icon} | {s_result.weighted_score:.2%} "
            f"| {s_result.threshold_used:.2%} | {s_desc} |\n"
        )

    return decision_html, breakdown_md, comparison_md


def load_preset(preset_name: str):
    """Load a preset scenario into the agent fields.

    Returns a flat tuple of (name1, w1, v1, c1, ..., name5, w5, v5, c5, strategy, threshold).
    """
    preset = PRESETS.get(preset_name)
    if not preset or not preset["agents"]:
        # Return defaults
        defaults = []
        default_names = ["Agent-1", "Agent-2", "Agent-3", "Agent-4", "Agent-5"]
        for name in default_names:
            defaults.extend([name, 1.0, "Permit", 0.80])
        defaults.extend(["Majority", ""])
        return defaults

    agents = preset["agents"]
    result = []
    for i in range(5):
        if i < len(agents):
            name, weight, vote, conf = agents[i]
            result.extend([name, weight, vote, conf])
        else:
            result.extend(["", 1.0, "Abstain", 0.50])
    result.extend([preset["strategy"], preset["threshold"]])
    return result


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    vote_choices = ["Permit", "Block", "Abstain"]

    with gr.Blocks(title="Operon Quorum Sensing") as app:
        gr.Markdown(
            "# Operon Quorum Sensing -- Voting Simulator\n"
            "Multi-agent consensus with **7 voting strategies**: Majority, Supermajority, "
            "Unanimous, Weighted, Confidence, Bayesian, and Threshold.\n\n"
            "Configure agents, pick a strategy, and run the vote to see results "
            "and a cross-strategy comparison.\n\n"
            "[GitHub](https://github.com/coredipper/operon) | "
            "[Paper](https://github.com/coredipper/operon/tree/main/article)"
        )

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="(custom)",
                label="Load Preset",
                scale=2,
            )
            strategy_dropdown = gr.Dropdown(
                choices=list(STRATEGY_MAP.keys()),
                value="Majority",
                label="Voting Strategy",
                scale=2,
            )
            threshold_input = gr.Textbox(
                label="Custom Threshold",
                placeholder="e.g. 2 for Threshold, 0.6 for Majority",
                scale=1,
            )

        # --- Agent rows ---
        agent_components = []  # flat list: name, weight, vote, conf for each agent
        default_names = ["Sentinel", "Guardian", "Watcher", "Auditor", "Verifier"]

        for i in range(5):
            with gr.Row():
                name = gr.Textbox(
                    label=f"Agent {i+1} Name",
                    value=default_names[i],
                    scale=2,
                )
                weight = gr.Slider(
                    minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                    label="Weight",
                    scale=1,
                )
                vote = gr.Radio(
                    choices=vote_choices,
                    value="Permit",
                    label="Vote",
                    scale=1,
                )
                conf = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.80, step=0.05,
                    label="Confidence",
                    scale=1,
                )
                agent_components.extend([name, weight, vote, conf])

        run_btn = gr.Button("Run Vote", variant="primary", size="lg")

        decision_html = gr.HTML(label="Decision")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Vote Breakdown")
                breakdown_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("### All Strategies Comparison")
                comparison_md = gr.Markdown()

        # All inputs for the simulation function
        sim_inputs = agent_components + [strategy_dropdown, threshold_input]
        sim_outputs = [decision_html, breakdown_md, comparison_md]

        # Wire events
        run_btn.click(fn=run_simulation, inputs=sim_inputs, outputs=sim_outputs)

        # Preset loading outputs: all agent fields + strategy + threshold
        preset_outputs = agent_components + [strategy_dropdown, threshold_input]
        preset_dropdown.change(fn=load_preset, inputs=[preset_dropdown], outputs=preset_outputs)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
