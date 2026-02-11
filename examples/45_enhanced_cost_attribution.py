#!/usr/bin/env python3
"""
Example 45: Enhanced Cost Attribution with Team Rollups
=======================================================

Extends Example 38 with:
- **Team-level rollups**: Aggregate costs by Linear team
- **Morphogen Gradients**: Coordinate budget limits across teams
- **Trend Analysis**: Compare week-over-week spending and predict exhaustion
- **Rich Reporting**: Markdown reports with summary tables

This is a production-ready pattern for LLM cost management in multi-team orgs.

Architecture:

```
Data Sources
+-- Anthropic CSV (costs by user/day)
+-- Git History (commits by branch/ticket)
+-- Linear API (tickets, estimates, teams)
    |
[Cascade Pipeline]
+-- Stage 1: Parse & Validate
+-- Stage 2: Correlate (user->ticket, ticket->team)
+-- Stage 3: Attribute (split costs by commit proportion)
+-- Stage 4: Aggregate (ticket->team->org rollups)
    |
[Morphogen Coordination]
+-- Team A budget signal (0.0-1.0 ratio remaining)
+-- Team B budget signal
+-- Shared org budget pool
    |
[Alert Dispatcher]
+-- Per-ticket alerts (CONSERVING, STARVING)
+-- Per-team alerts (approaching budget)
+-- Trend alerts (spending acceleration)
```

Usage:
    python examples/45_enhanced_cost_attribution.py              # Demo with mock
    python examples/45_enhanced_cost_attribution.py --test       # Smoke test
    python examples/45_enhanced_cost_attribution.py --csv <file> # Real data

Prerequisites:
- Example 38 for basic cost attribution patterns
- Example 42 for Morphogen gradient coordination
"""

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable
from enum import Enum

from operon_ai import ATP_Store
from operon_ai.state.metabolism import MetabolicState
from operon_ai.coordination.morphogen import MorphogenGradient, MorphogenType


# =============================================================================
# Data Structures
# =============================================================================


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TeamConfig:
    """Configuration for a team's budget."""
    team_id: str
    team_name: str
    monthly_budget_usd: float
    dollars_per_point: float = 5.00
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9


@dataclass
class TicketCost:
    """Cost attributed to a single ticket."""
    ticket_id: str
    team_id: str
    estimate_points: int
    budget_usd: float
    spent_usd: float
    utilization: float
    state: MetabolicState


@dataclass
class TeamSummary:
    """Summary of a team's spending."""
    team_id: str
    team_name: str
    budget_usd: float
    spent_usd: float
    utilization: float
    state: MetabolicState
    ticket_count: int
    over_budget_tickets: int
    avg_efficiency: float  # dollars per point


@dataclass
class TrendData:
    """Spending trend analysis."""
    current_week_usd: float
    previous_week_usd: float
    week_over_week_change: float  # percentage
    daily_burn_rate: float
    projected_monthly_usd: float
    days_until_exhaustion: int | None
    acceleration: str  # "increasing", "stable", "decreasing"


@dataclass
class CostReport:
    """Complete cost attribution report."""
    report_date: date
    org_total_budget: float
    org_total_spent: float
    org_utilization: float
    teams: list[TeamSummary]
    tickets: list[TicketCost]
    trend: TrendData
    alerts: list[tuple[AlertLevel, str]]


# =============================================================================
# Mock Data for Demo
# =============================================================================


def create_mock_teams() -> list[TeamConfig]:
    """Create mock team configurations."""
    return [
        TeamConfig("ENG", "Engineering", monthly_budget_usd=5000.0),
        TeamConfig("DESIGN", "Design", monthly_budget_usd=2000.0),
        TeamConfig("DATA", "Data Science", monthly_budget_usd=8000.0),
    ]


def create_mock_ticket_costs() -> list[dict]:
    """Create mock ticket cost data."""
    return [
        # Engineering tickets
        {"ticket_id": "ENG-101", "team_id": "ENG", "estimate": 5, "spent": 35.00},
        {"ticket_id": "ENG-102", "team_id": "ENG", "estimate": 3, "spent": 28.00},
        {"ticket_id": "ENG-103", "team_id": "ENG", "estimate": 8, "spent": 65.00},
        {"ticket_id": "ENG-104", "team_id": "ENG", "estimate": 2, "spent": 45.00},  # Over budget
        {"ticket_id": "ENG-105", "team_id": "ENG", "estimate": 5, "spent": 12.00},

        # Design tickets
        {"ticket_id": "DESIGN-50", "team_id": "DESIGN", "estimate": 3, "spent": 18.00},
        {"ticket_id": "DESIGN-51", "team_id": "DESIGN", "estimate": 5, "spent": 22.00},

        # Data Science tickets (high spenders)
        {"ticket_id": "DATA-200", "team_id": "DATA", "estimate": 13, "spent": 180.00},
        {"ticket_id": "DATA-201", "team_id": "DATA", "estimate": 8, "spent": 120.00},
        {"ticket_id": "DATA-202", "team_id": "DATA", "estimate": 5, "spent": 85.00},
    ]


def create_mock_trend_data() -> dict:
    """Create mock trend data."""
    return {
        "current_week": 610.00,
        "previous_week": 520.00,
        "daily_history": [85.0, 92.0, 78.0, 88.0, 95.0, 82.0, 90.0],
    }


# =============================================================================
# Team Budget Tracker
# =============================================================================


class TeamBudgetTracker:
    """
    Tracks budget for a team using ATP_Store and Morphogen gradients.

    Each team has:
    - An ATP_Store representing their monthly budget (in cents)
    - A contribution to the org-wide Morphogen gradient
    - Alerts when thresholds are crossed
    """

    def __init__(
        self,
        config: TeamConfig,
        gradient: MorphogenGradient,
        on_alert: Callable[[AlertLevel, str], None] | None = None,
        silent: bool = False,
    ):
        self.config = config
        self.gradient = gradient
        self.on_alert = on_alert
        self.silent = silent

        # Create ATP store (budget in cents for precision)
        budget_cents = int(config.monthly_budget_usd * 100)
        self.budget = ATP_Store(
            budget=budget_cents,
            on_state_change=self._handle_state_change,
            silent=silent,
        )

        self._tickets: list[TicketCost] = []
        self._total_spent = 0

    def _handle_state_change(self, new_state: MetabolicState) -> None:
        """Handle budget state transitions."""
        if not self.on_alert:
            return

        if new_state == MetabolicState.CONSERVING:
            self.on_alert(
                AlertLevel.WARNING,
                f"Team {self.config.team_name} has consumed {self.get_utilization():.0%} of budget",
            )
        elif new_state == MetabolicState.STARVING:
            self.on_alert(
                AlertLevel.CRITICAL,
                f"Team {self.config.team_name} is nearly out of budget ({self.get_utilization():.0%})!",
            )

    def record_ticket_cost(
        self,
        ticket_id: str,
        estimate_points: int,
        cost_usd: float,
    ) -> TicketCost:
        """Record cost for a ticket."""
        cost_cents = int(cost_usd * 100)
        self._total_spent += cost_cents

        # Try to consume from budget
        self.budget.consume(cost_cents, operation=f"ticket:{ticket_id}")

        # Calculate ticket budget (points * $/point)
        ticket_budget = estimate_points * self.config.dollars_per_point
        ticket_utilization = cost_usd / ticket_budget if ticket_budget > 0 else 1.0

        # Determine ticket state
        if ticket_utilization > 1.0:
            ticket_state = MetabolicState.STARVING  # Over budget
        elif ticket_utilization > self.config.warning_threshold:
            ticket_state = MetabolicState.CONSERVING
        else:
            ticket_state = MetabolicState.NORMAL

        ticket_cost = TicketCost(
            ticket_id=ticket_id,
            team_id=self.config.team_id,
            estimate_points=estimate_points,
            budget_usd=ticket_budget,
            spent_usd=cost_usd,
            utilization=ticket_utilization,
            state=ticket_state,
        )
        self._tickets.append(ticket_cost)

        # Update morphogen gradient with team budget ratio
        self._update_gradient()

        return ticket_cost

    def _update_gradient(self) -> None:
        """Update morphogen gradient with team's budget status."""
        utilization = self.get_utilization()
        remaining_ratio = 1.0 - min(utilization, 1.0)

        # The gradient uses MorphogenType.BUDGET for budget ratios
        # We'll use a custom description to identify this team
        self.gradient.set(
            MorphogenType.BUDGET,
            remaining_ratio,
            description=f"Team {self.config.team_id} budget remaining",
        )

    def get_utilization(self) -> float:
        """Get team budget utilization."""
        budget_cents = int(self.config.monthly_budget_usd * 100)
        if budget_cents == 0:
            return 1.0
        return self._total_spent / budget_cents

    def get_summary(self) -> TeamSummary:
        """Get team summary."""
        utilization = self.get_utilization()
        spent_usd = self._total_spent / 100

        # Calculate average efficiency
        total_points = sum(t.estimate_points for t in self._tickets)
        avg_efficiency = spent_usd / total_points if total_points > 0 else 0.0

        # Count over-budget tickets
        over_budget = sum(1 for t in self._tickets if t.utilization > 1.0)

        return TeamSummary(
            team_id=self.config.team_id,
            team_name=self.config.team_name,
            budget_usd=self.config.monthly_budget_usd,
            spent_usd=spent_usd,
            utilization=utilization,
            state=self.budget.get_state(),
            ticket_count=len(self._tickets),
            over_budget_tickets=over_budget,
            avg_efficiency=avg_efficiency,
        )


# =============================================================================
# Trend Analyzer
# =============================================================================


class TrendAnalyzer:
    """Analyzes spending trends and predicts exhaustion."""

    def __init__(self, org_budget: float):
        self.org_budget = org_budget

    def analyze(
        self,
        current_week: float,
        previous_week: float,
        daily_history: list[float],
    ) -> TrendData:
        """Analyze spending trends."""
        # Week over week change
        if previous_week > 0:
            wow_change = (current_week - previous_week) / previous_week
        else:
            wow_change = 0.0

        # Daily burn rate (average of last 7 days)
        if daily_history:
            daily_burn = sum(daily_history) / len(daily_history)
        else:
            daily_burn = current_week / 7

        # Projected monthly spend
        projected_monthly = daily_burn * 30

        # Days until exhaustion
        remaining = self.org_budget - sum(daily_history)
        if daily_burn > 0:
            days_until = int(remaining / daily_burn)
            days_until = max(0, days_until)
        else:
            days_until = None

        # Determine acceleration
        if wow_change > 0.15:
            acceleration = "increasing"
        elif wow_change < -0.15:
            acceleration = "decreasing"
        else:
            acceleration = "stable"

        return TrendData(
            current_week_usd=current_week,
            previous_week_usd=previous_week,
            week_over_week_change=wow_change,
            daily_burn_rate=daily_burn,
            projected_monthly_usd=projected_monthly,
            days_until_exhaustion=days_until,
            acceleration=acceleration,
        )


# =============================================================================
# Cost Attribution Pipeline
# =============================================================================


class CostAttributionPipeline:
    """
    Production-ready cost attribution pipeline.

    Orchestrates:
    1. Team budget tracking with ATP_Store
    2. Cross-team coordination via Morphogen gradients
    3. Trend analysis and alerting
    4. Report generation
    """

    def __init__(
        self,
        teams: list[TeamConfig],
        silent: bool = False,
    ):
        self.silent = silent
        self.teams = {t.team_id: t for t in teams}

        # Shared morphogen gradient for cross-team coordination
        self.gradient = MorphogenGradient()

        # Alerts collected during processing
        self._alerts: list[tuple[AlertLevel, str]] = []

        # Team trackers
        self._trackers: dict[str, TeamBudgetTracker] = {}
        for team in teams:
            self._trackers[team.team_id] = TeamBudgetTracker(
                config=team,
                gradient=self.gradient,
                on_alert=self._record_alert,
                silent=silent,
            )

        # Trend analyzer
        org_budget = sum(t.monthly_budget_usd for t in teams)
        self.trend_analyzer = TrendAnalyzer(org_budget)

    def _record_alert(self, level: AlertLevel, message: str) -> None:
        """Record an alert."""
        self._alerts.append((level, message))
        if not self.silent:
            icon = {"info": "INFO", "warning": "WARN", "critical": "CRIT"}
            print(f"  [{icon.get(level, 'INFO')}] {message}")

    def process_ticket_costs(self, ticket_data: list[dict]) -> None:
        """Process ticket cost data."""
        if not self.silent:
            print("\n--- Processing ticket costs ---")

        for data in ticket_data:
            team_id = data.get("team_id")
            if team_id not in self._trackers:
                if not self.silent:
                    print(f"  Warning: Unknown team {team_id}")
                continue

            tracker = self._trackers[team_id]
            ticket_cost = tracker.record_ticket_cost(
                ticket_id=data["ticket_id"],
                estimate_points=data.get("estimate", 3),
                cost_usd=data.get("spent", 0.0),
            )

            if not self.silent:
                state_str = ticket_cost.state.value
                if ticket_cost.utilization > 1.0:
                    state_str = "OVER"
                print(
                    f"  {ticket_cost.ticket_id}: "
                    f"${ticket_cost.spent_usd:.2f} / ${ticket_cost.budget_usd:.2f} "
                    f"({ticket_cost.utilization:.0%}) [{state_str}]"
                )

    def generate_report(
        self,
        trend_data: dict | None = None,
    ) -> CostReport:
        """Generate comprehensive cost report."""
        # Collect team summaries
        team_summaries = [
            tracker.get_summary()
            for tracker in self._trackers.values()
        ]

        # Collect all ticket costs
        all_tickets = []
        for tracker in self._trackers.values():
            all_tickets.extend(tracker._tickets)

        # Calculate org totals
        org_budget = sum(t.budget_usd for t in team_summaries)
        org_spent = sum(t.spent_usd for t in team_summaries)
        org_utilization = org_spent / org_budget if org_budget > 0 else 0.0

        # Analyze trends
        if trend_data:
            trend = self.trend_analyzer.analyze(
                current_week=trend_data.get("current_week", 0),
                previous_week=trend_data.get("previous_week", 0),
                daily_history=trend_data.get("daily_history", []),
            )
        else:
            trend = TrendData(
                current_week_usd=org_spent,
                previous_week_usd=0,
                week_over_week_change=0,
                daily_burn_rate=org_spent / 7,
                projected_monthly_usd=org_spent * 4,
                days_until_exhaustion=None,
                acceleration="unknown",
            )

        # Add trend-based alerts
        if trend.acceleration == "increasing" and trend.week_over_week_change > 0.25:
            self._record_alert(
                AlertLevel.WARNING,
                f"Spending accelerating: {trend.week_over_week_change:+.0%} week-over-week",
            )

        if trend.days_until_exhaustion is not None and trend.days_until_exhaustion < 14:
            self._record_alert(
                AlertLevel.CRITICAL,
                f"Budget exhaustion projected in {trend.days_until_exhaustion} days!",
            )

        return CostReport(
            report_date=date.today(),
            org_total_budget=org_budget,
            org_total_spent=org_spent,
            org_utilization=org_utilization,
            teams=team_summaries,
            tickets=all_tickets,
            trend=trend,
            alerts=self._alerts.copy(),
        )

    def format_markdown_report(self, report: CostReport) -> str:
        """Format report as Markdown."""
        lines = []

        # Header
        lines.append(f"# Cost Attribution Report - {report.report_date}")
        lines.append("")

        # Organization Summary
        lines.append("## Organization Summary")
        lines.append("")
        lines.append(f"- **Total Budget**: ${report.org_total_budget:,.2f}")
        lines.append(f"- **Total Spent**: ${report.org_total_spent:,.2f}")
        lines.append(f"- **Utilization**: {report.org_utilization:.0%}")
        lines.append("")

        # Alerts
        if report.alerts:
            lines.append("## Alerts")
            lines.append("")
            for level, message in report.alerts:
                icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "")
                lines.append(f"- {icon} **{level.upper()}**: {message}")
            lines.append("")

        # Team Summary
        lines.append("## Team Summary")
        lines.append("")
        lines.append("| Team | Budget | Spent | Util | Tickets | Over Budget | $/Point |")
        lines.append("|------|--------|-------|------|---------|-------------|---------|")

        for team in sorted(report.teams, key=lambda t: t.utilization, reverse=True):
            lines.append(
                f"| {team.team_name} | "
                f"${team.budget_usd:,.0f} | "
                f"${team.spent_usd:,.0f} | "
                f"{team.utilization:.0%} | "
                f"{team.ticket_count} | "
                f"{team.over_budget_tickets} | "
                f"${team.avg_efficiency:.2f} |"
            )
        lines.append("")

        # Trend Analysis
        lines.append("## Trend Analysis")
        lines.append("")
        lines.append(f"- **Current Week**: ${report.trend.current_week_usd:,.2f}")
        lines.append(f"- **Previous Week**: ${report.trend.previous_week_usd:,.2f}")
        lines.append(f"- **Week-over-Week**: {report.trend.week_over_week_change:+.0%}")
        lines.append(f"- **Daily Burn Rate**: ${report.trend.daily_burn_rate:,.2f}")
        lines.append(f"- **Projected Monthly**: ${report.trend.projected_monthly_usd:,.2f}")
        if report.trend.days_until_exhaustion is not None:
            lines.append(f"- **Days Until Exhaustion**: {report.trend.days_until_exhaustion}")
        lines.append(f"- **Trend**: {report.trend.acceleration}")
        lines.append("")

        # Top Over-Budget Tickets
        over_budget_tickets = [t for t in report.tickets if t.utilization > 1.0]
        if over_budget_tickets:
            lines.append("## Over-Budget Tickets")
            lines.append("")
            lines.append("| Ticket | Team | Estimate | Budget | Spent | Overage |")
            lines.append("|--------|------|----------|--------|-------|---------|")

            for ticket in sorted(over_budget_tickets, key=lambda t: t.utilization, reverse=True)[:10]:
                overage = ticket.spent_usd - ticket.budget_usd
                lines.append(
                    f"| {ticket.ticket_id} | "
                    f"{ticket.team_id} | "
                    f"{ticket.estimate_points}pts | "
                    f"${ticket.budget_usd:.2f} | "
                    f"${ticket.spent_usd:.2f} | "
                    f"+${overage:.2f} |"
                )
            lines.append("")

        # Morphogen Gradient Status
        lines.append("## Budget Signals (Morphogen Gradient)")
        lines.append("")
        lines.append("Current coordination signals:")
        lines.append(f"- Budget Remaining Ratio: {self.gradient.get(MorphogenType.BUDGET):.2f}")
        lines.append(f"- Budget Level: {self.gradient.get_level(MorphogenType.BUDGET)}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Operon Cost Attribution Pipeline*")

        return "\n".join(lines)


# =============================================================================
# Demo and Tests
# =============================================================================


def run_demo():
    """Run demonstration with mock data."""
    print("=" * 70)
    print("Example 45: Enhanced Cost Attribution - Demo")
    print("=" * 70)

    # Create pipeline
    teams = create_mock_teams()
    pipeline = CostAttributionPipeline(teams=teams, silent=False)

    # Process ticket costs
    ticket_data = create_mock_ticket_costs()
    pipeline.process_ticket_costs(ticket_data)

    # Generate report
    print("\n--- Generating report ---")
    trend_data = create_mock_trend_data()
    report = pipeline.generate_report(trend_data=trend_data)

    # Print summary
    print("\n" + "=" * 70)
    print("Organization Summary")
    print("=" * 70)
    print(f"Total Budget: ${report.org_total_budget:,.2f}")
    print(f"Total Spent:  ${report.org_total_spent:,.2f}")
    print(f"Utilization:  {report.org_utilization:.1%}")

    print("\n" + "-" * 40)
    print("Team Summary")
    print("-" * 40)
    for team in report.teams:
        state = team.state.value
        if team.utilization > 0.9:
            state = "STARVING"
        elif team.utilization > 0.7:
            state = "CONSERVING"
        print(
            f"  {team.team_name:15} "
            f"${team.spent_usd:>8,.2f} / ${team.budget_usd:>8,.2f} "
            f"({team.utilization:>5.1%}) [{state}]"
        )

    print("\n" + "-" * 40)
    print("Trend Analysis")
    print("-" * 40)
    print(f"  Week-over-Week: {report.trend.week_over_week_change:+.0%} ({report.trend.acceleration})")
    print(f"  Daily Burn Rate: ${report.trend.daily_burn_rate:.2f}")
    print(f"  Projected Monthly: ${report.trend.projected_monthly_usd:,.2f}")

    if report.alerts:
        print("\n" + "-" * 40)
        print("Alerts")
        print("-" * 40)
        for level, message in report.alerts:
            print(f"  [{level.value.upper()}] {message}")

    # Show Markdown report
    print("\n" + "=" * 70)
    print("Markdown Report Preview")
    print("=" * 70)
    markdown = pipeline.format_markdown_report(report)
    # Show first 50 lines
    for line in markdown.split("\n")[:50]:
        print(line)
    print("... (truncated)")


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Pipeline creation
    teams = create_mock_teams()
    pipeline = CostAttributionPipeline(teams=teams, silent=True)
    assert len(pipeline._trackers) == 3, "Should have 3 team trackers"
    print("  Test 1: Pipeline creation - PASSED")

    # Test 2: Ticket processing
    ticket_data = create_mock_ticket_costs()
    pipeline.process_ticket_costs(ticket_data)

    eng_tracker = pipeline._trackers["ENG"]
    assert len(eng_tracker._tickets) == 5, "ENG should have 5 tickets"
    print("  Test 2: Ticket processing - PASSED")

    # Test 3: Report generation
    report = pipeline.generate_report()
    assert report.org_total_spent > 0, "Should have some spending"
    assert len(report.teams) == 3, "Should have 3 teams"
    print("  Test 3: Report generation - PASSED")

    # Test 4: Team summary
    eng_summary = next(t for t in report.teams if t.team_id == "ENG")
    assert eng_summary.ticket_count == 5, "ENG should have 5 tickets"
    assert eng_summary.over_budget_tickets >= 1, "Should have over-budget tickets"
    print("  Test 4: Team summary - PASSED")

    # Test 5: Trend analysis
    trend_data = create_mock_trend_data()
    report = pipeline.generate_report(trend_data=trend_data)
    assert report.trend.week_over_week_change > 0, "Should show increase"
    assert report.trend.acceleration == "increasing", "Should be increasing"
    print("  Test 5: Trend analysis - PASSED")

    # Test 6: Morphogen gradient
    budget_ratio = pipeline.gradient.get(MorphogenType.BUDGET)
    assert 0.0 <= budget_ratio <= 1.0, f"Budget ratio should be 0-1: {budget_ratio}"
    print("  Test 6: Morphogen gradient - PASSED")

    # Test 7: Markdown report
    markdown = pipeline.format_markdown_report(report)
    assert "# Cost Attribution Report" in markdown, "Should have title"
    assert "## Team Summary" in markdown, "Should have team section"
    print("  Test 7: Markdown report - PASSED")

    # Test 8: Over-budget detection
    over_budget = [t for t in report.tickets if t.utilization > 1.0]
    assert len(over_budget) >= 1, "Should detect over-budget tickets"
    print("  Test 8: Over-budget detection - PASSED")

    print("\nSmoke tests passed!")


def main():
    """Main entry point."""
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        raise
