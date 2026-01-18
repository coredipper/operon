"""
Example 38: Linear Budget Tracking (Cost Attribution via Git Commits)
======================================================================

Tracks Claude Code token costs against Linear ticket estimates using:
- Anthropic Console CSV exports for cost data
- Git commit timestamps for ticket attribution
- Linear API for ticket estimates (optional)
- ATP_Store for metabolic budget management

This demonstrates how Operon's biological abstractions apply to real-world
resource management: token budgets as ATP, budget states as metabolic states,
and predictive alerts as ischemia detection.

Architecture (Mermaid):

```mermaid
flowchart TB
    subgraph inputs["Data Sources"]
        CSV[("Anthropic CSV<br/>cost/user/day")]
        GIT[("Git History<br/>commits/branches")]
        LINEAR[("Linear API<br/>tickets/estimates")]
    end

    subgraph parsing["Parsing Layer"]
        CSV --> CSVParser["CostCSVParser<br/>━━━━━━━━━━━━━<br/>• Extract username from api_key<br/>• Aggregate cost by user/day"]
        GIT --> GitAnalyzer["GitTicketAnalyzer<br/>━━━━━━━━━━━━━<br/>• Parse branch → ticket ID<br/>• Map commits to dates"]
    end

    subgraph attribution["Attribution"]
        CSVParser --> Attributor
        GitAnalyzer --> Attributor
        Attributor["CostAttributor<br/>━━━━━━━━━━━━━<br/>• Correlate user+date<br/>• Split cost by commit count"]
    end

    subgraph budgeting["Budget Tracking"]
        LINEAR --> BudgetConfig["BudgetConfig<br/>━━━━━━━━━━━━━<br/>• $/point multipliers<br/>• Type adjustments"]
        BudgetConfig --> Tracker
        Attributor --> Tracker
        Tracker["TicketBudgetTracker<br/>━━━━━━━━━━━━━<br/>• ATP_Store per ticket<br/>• Metabolic states"]
    end

    subgraph states["Metabolic States"]
        Tracker --> Normal["NORMAL<br/>< 70%"]
        Tracker --> Conserving["CONSERVING<br/>70-90%"]
        Tracker --> Starving["STARVING<br/>> 90%"]
        Tracker --> Over["OVER BUDGET<br/>> 100%"]
    end

    subgraph outputs["Alert Outputs"]
        Normal --> Dashboard["Dashboard JSON"]
        Conserving --> Slack["Slack Webhook"]
        Starving --> Slack
        Over --> LinearComment["Linear Comment"]
        Starving --> LinearComment
    end

    style CSV fill:#e1f5fe
    style GIT fill:#e1f5fe
    style LINEAR fill:#e1f5fe
    style Normal fill:#c8e6c9
    style Conserving fill:#fff9c4
    style Starving fill:#ffccbc
    style Over fill:#ffcdd2
```

Sequence (Mermaid):

```mermaid
sequenceDiagram
    participant User
    participant Tracker as LinearBudgetTracker
    participant CSV as CostCSVParser
    participant Git as GitTicketAnalyzer
    participant Attr as CostAttributor
    participant ATP as ATP_Store
    participant Alert as AlertDispatcher

    User->>Tracker: analyze(csv_path, repo_path)

    rect rgb(225, 245, 254)
        Note over CSV,Git: Data Collection
        Tracker->>CSV: parse_csv()
        CSV-->>Tracker: daily_costs[user][date]
        Tracker->>Git: get_commits_by_date_range()
        Git-->>Tracker: commit_activities[]
    end

    rect rgb(243, 229, 245)
        Note over Attr: Attribution
        Tracker->>Attr: attribute_costs()
        Attr-->>Tracker: ticket_costs[ticket_id]
    end

    rect rgb(232, 245, 233)
        Note over ATP,Alert: Budget Tracking
        loop For each ticket
            Tracker->>ATP: create ATP_Store(budget)
            Tracker->>ATP: consume(cost)
            alt State changed
                ATP->>Alert: dispatch(level, message)
            end
        end
    end

    Tracker-->>User: reports[ticket_id]
```

Prerequisites:
- Example 04 for basic ATP budgeting concepts
- Example 37 for metabolic coalgebra patterns

Usage:
    python examples/38_linear_budget_tracking.py                    # Demo with mock data
    python examples/38_linear_budget_tracking.py --csv path.csv     # Demo with real CSV
    python examples/38_linear_budget_tracking.py --test             # Smoke test
"""

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Callable
from urllib.request import Request, urlopen
from urllib.error import URLError

from operon_ai import ATP_Store
from operon_ai.state.metabolism import MetabolicState


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DailyCost:
    """Single row from the Anthropic cost CSV."""
    date: date
    user: str
    model: str
    cost_usd: Decimal
    token_type: str
    workspace: str


@dataclass
class CommitActivity:
    """Git commit activity for a user on a specific ticket/day."""
    date: date
    author: str
    ticket_id: str
    commit_count: int
    branch: str


@dataclass
class LinearTicket:
    """Linear ticket with budget-relevant fields."""
    id: str
    title: str
    type: str  # bug, feature, task, story
    estimate: int | None  # story points
    state: str
    team: str | None = None


@dataclass
class BudgetConfig:
    """Configuration for estimate-to-cost mapping."""
    dollars_per_point: float = 5.00
    type_multipliers: dict = field(default_factory=lambda: {
        "bug": 0.5,
        "feature": 1.5,
        "task": 1.0,
        "story": 1.2,
    })
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9
    default_estimate: int = 3  # for tickets without estimates

    def estimate_to_dollars(self, estimate: int | None, ticket_type: str = "task") -> int:
        """Convert story point estimate to dollar budget (as integer cents for ATP)."""
        points = estimate or self.default_estimate
        multiplier = self.type_multipliers.get(ticket_type, 1.0)
        dollars = points * self.dollars_per_point * multiplier
        return int(dollars * 100)  # cents for ATP granularity


@dataclass
class TicketBudgetReport:
    """Budget status report for a ticket."""
    ticket_id: str
    estimate_points: int
    budget_dollars: float
    spent_dollars: float
    remaining_dollars: float
    utilization: float
    state: MetabolicState
    efficiency: float  # dollars per point consumed
    predicted_exhaustion: date | None


# =============================================================================
# CSV Parser
# =============================================================================

class CostCSVParser:
    """
    Parses Anthropic Console CSV exports.

    Expected format:
        usage_date_utc,model,workspace,api_key,usage_type,context_window,token_type,cost_usd,cost_type
        2025-12-01,Claude Haiku 4.5,Claude Code,claude_code_key_dmytro.baltak_jivb,message,...,0.13,token
    """

    @staticmethod
    def extract_user_from_api_key(api_key: str) -> str:
        """
        Extract username from api_key field.

        Examples:
            claude_code_key_dmytro.baltak_jivb -> dmytro.baltak
            claude_code_key_andy_hcgi -> andy
        """
        if not api_key.startswith("claude_code_key_"):
            return api_key

        # Remove prefix
        remainder = api_key[len("claude_code_key_"):]

        # Remove trailing random suffix (4 chars after last underscore)
        parts = remainder.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 4:
            return parts[0]
        return remainder

    @classmethod
    def parse_csv(cls, path: str) -> list[DailyCost]:
        """Parse CSV file into DailyCost records."""
        costs = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Skip non-Claude Code entries
                if row.get("workspace") != "Claude Code":
                    continue

                # Parse date
                date_str = row.get("usage_date_utc", "")
                try:
                    usage_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                # Parse cost
                try:
                    cost = Decimal(row.get("cost_usd", "0"))
                except:
                    continue

                # Extract user
                api_key = row.get("api_key", "")
                user = cls.extract_user_from_api_key(api_key)

                costs.append(DailyCost(
                    date=usage_date,
                    user=user,
                    model=row.get("model", "unknown"),
                    cost_usd=cost,
                    token_type=row.get("token_type", "unknown"),
                    workspace=row.get("workspace", ""),
                ))

        return costs

    @staticmethod
    def aggregate_by_user_day(costs: list[DailyCost]) -> dict[str, dict[date, Decimal]]:
        """Aggregate costs by user and day."""
        result: dict[str, dict[date, Decimal]] = {}

        for cost in costs:
            if cost.user not in result:
                result[cost.user] = {}
            if cost.date not in result[cost.user]:
                result[cost.user][cost.date] = Decimal("0")
            result[cost.user][cost.date] += cost.cost_usd

        return result


# =============================================================================
# Git Analyzer
# =============================================================================

class GitTicketAnalyzer:
    """
    Analyzes git commits to map activity to Linear tickets.

    Uses branch names to identify ticket IDs (e.g., ENG-123-feature-name).
    """

    # Patterns to extract ticket ID from branch names (case-insensitive)
    TICKET_PATTERNS = [
        r"^([A-Za-z]+-\d+)",           # ENG-123-feature-name
        r"/([A-Za-z]+-\d+)",           # feature/ENG-123-name or user/ENG-123-name
        r"_([A-Za-z]+-\d+)",           # feature_ENG-123_name
        r"-([A-Za-z]+-\d+)",           # prefix-ENG-123-name
    ]

    @classmethod
    def extract_ticket_from_branch(cls, branch: str) -> str | None:
        """Extract ticket ID like ENG-123 from branch name (case-insensitive, uppercased)."""
        for pattern in cls.TICKET_PATTERNS:
            match = re.search(pattern, branch, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # Normalize to uppercase
        return None

    @staticmethod
    def get_current_branch(repo_path: str = ".") -> str | None:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    @classmethod
    def get_commits_by_date_range(
        cls,
        repo_path: str,
        start_date: date,
        end_date: date,
    ) -> list[CommitActivity]:
        """
        Get commit activity grouped by author, date, and ticket.

        Strategy: Enumerate branches with ticket IDs, then find commits on each.
        """
        activities: list[CommitActivity] = []
        activity_map: dict[tuple[str, date, str], int] = {}
        branch_map: dict[tuple[str, date, str], str] = {}

        try:
            # Step 1: Get all branches (local and remote)
            result = subprocess.run(
                ["git", "branch", "-a", "--format=%(refname:short)"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return activities

            branches = result.stdout.strip().split("\n")

            # Step 2: For each branch with a ticket ID, get commits in date range
            for branch in branches:
                branch = branch.strip()
                if not branch:
                    continue

                # Clean up remote prefix for display
                display_branch = branch
                if branch.startswith("origin/"):
                    display_branch = branch[7:]

                ticket_id = cls.extract_ticket_from_branch(display_branch)
                if not ticket_id:
                    continue

                # Get commits on this branch in date range
                result = subprocess.run(
                    [
                        "git", "log",
                        branch,
                        f"--since={start_date.isoformat()}",
                        f"--until={end_date.isoformat()}",
                        "--format=%ad|%an",
                        "--date=short",
                    ],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    continue

                for line in result.stdout.strip().split("\n"):
                    if not line or "|" not in line:
                        continue

                    parts = line.split("|", 1)
                    if len(parts) < 2:
                        continue

                    date_str, author = parts

                    try:
                        commit_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue

                    key = (author, commit_date, ticket_id)
                    activity_map[key] = activity_map.get(key, 0) + 1
                    branch_map[key] = display_branch

            # Convert to CommitActivity objects
            for (author, commit_date, ticket_id), count in activity_map.items():
                activities.append(CommitActivity(
                    date=commit_date,
                    author=author,
                    ticket_id=ticket_id,
                    commit_count=count,
                    branch=branch_map[(author, commit_date, ticket_id)],
                ))

        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return activities


# =============================================================================
# Cost Attribution
# =============================================================================

class CostAttributor:
    """
    Correlates daily costs with commit activity to attribute costs to tickets.

    Attribution logic:
    - If user made commits on ticket X on day D, attribute that day's cost to X
    - If multiple tickets on same day, split proportionally by commit count
    """

    @staticmethod
    def normalize_author_name(name: str) -> str:
        """
        Normalize author name for matching.

        Handles various formats:
        - "Mark Stewart" -> "markstewart"
        - "mark.stewart" -> "markstewart"
        - "markstewart" -> "markstewart"
        - "David Klajbar" -> "davidklajbar"
        - "david.klajbar" -> "davidklajbar"
        """
        return name.lower().replace(".", "").replace(" ", "").replace("-", "")

    @classmethod
    def attribute_costs(
        cls,
        daily_costs: dict[str, dict[date, Decimal]],
        commit_activity: list[CommitActivity],
    ) -> dict[str, Decimal]:
        """
        Attribute costs to tickets based on commit activity.

        Returns:
            dict mapping ticket_id -> total attributed cost
        """
        ticket_costs: dict[str, Decimal] = {}

        # Build activity index: (normalized_author, date) -> [(ticket, commits)]
        activity_index: dict[tuple[str, date], list[tuple[str, int]]] = {}
        for activity in commit_activity:
            key = (cls.normalize_author_name(activity.author), activity.date)
            if key not in activity_index:
                activity_index[key] = []
            activity_index[key].append((activity.ticket_id, activity.commit_count))

        # Attribute each user's daily cost to their tickets
        for user, date_costs in daily_costs.items():
            normalized_user = cls.normalize_author_name(user)

            for cost_date, cost in date_costs.items():
                key = (normalized_user, cost_date)

                if key not in activity_index:
                    # No commits on this day - cost is unattributed
                    # Could track as "overhead" but we'll skip for now
                    continue

                # Calculate total commits across all tickets this day
                tickets = activity_index[key]
                total_commits = sum(commits for _, commits in tickets)

                if total_commits == 0:
                    continue

                # Split cost proportionally
                for ticket_id, commits in tickets:
                    portion = cost * Decimal(commits) / Decimal(total_commits)
                    ticket_costs[ticket_id] = ticket_costs.get(ticket_id, Decimal("0")) + portion

        return ticket_costs


# =============================================================================
# Linear Client
# =============================================================================

class LinearClient:
    """
    HTTP client for Linear GraphQL API.

    Uses urllib to avoid external dependencies.
    """

    API_URL = "https://api.linear.app/graphql"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("LINEAR_API_KEY")

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self._api_key)

    def _query(self, query: str, variables: dict | None = None) -> dict | None:
        """Execute GraphQL query."""
        if not self._api_key:
            return None

        payload = json.dumps({
            "query": query,
            "variables": variables or {},
        }).encode("utf-8")

        request = Request(
            self.API_URL,
            data=payload,
            headers={
                "Authorization": self._api_key,
                "Content-Type": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, json.JSONDecodeError):
            return None

    def get_issue(self, issue_id: str) -> LinearTicket | None:
        """Fetch issue details from Linear API."""
        query = """
        query GetIssue($id: String!) {
            issue(id: $id) {
                id
                identifier
                title
                state { name }
                team { key }
                estimate
                labels { nodes { name } }
            }
        }
        """

        result = self._query(query, {"id": issue_id})
        if not result or "data" not in result or not result.get("data") or not result["data"].get("issue"):
            return None

        issue = result["data"]["issue"]

        # Determine ticket type from labels
        ticket_type = "task"
        labels = [l["name"].lower() for l in issue.get("labels", {}).get("nodes", [])]
        if "bug" in labels:
            ticket_type = "bug"
        elif "feature" in labels:
            ticket_type = "feature"
        elif "story" in labels:
            ticket_type = "story"

        return LinearTicket(
            id=issue.get("identifier", issue_id),
            title=issue.get("title", ""),
            type=ticket_type,
            estimate=issue.get("estimate"),
            state=issue.get("state", {}).get("name", "unknown"),
            team=issue.get("team", {}).get("key"),
        )

    def add_comment(self, issue_id: str, comment: str) -> bool:
        """Add a comment to a Linear issue."""
        query = """
        mutation AddComment($issueId: String!, $body: String!) {
            commentCreate(input: { issueId: $issueId, body: $body }) {
                success
            }
        }
        """

        result = self._query(query, {"issueId": issue_id, "body": comment})
        return bool(
            result
            and result.get("data")
            and result["data"].get("commentCreate", {}).get("success")
        )


# =============================================================================
# Budget Tracker
# =============================================================================

class TicketBudgetTracker:
    """
    Tracks budget consumption for a Linear ticket using ATP_Store.

    Maps the biological metaphor:
    - ATP = dollar budget (in cents)
    - MetabolicState = budget warning level
    - Apoptosis = budget exhaustion
    """

    def __init__(
        self,
        ticket: LinearTicket,
        config: BudgetConfig | None = None,
        on_alert: Callable[[str, str, dict], None] | None = None,
    ):
        self.ticket = ticket
        self.config = config or BudgetConfig()
        self.on_alert = on_alert

        # Calculate budget from estimate
        budget_cents = self.config.estimate_to_dollars(ticket.estimate, ticket.type)

        # Create ATP store with state change callback
        self.budget = ATP_Store(
            budget=budget_cents,
            on_state_change=self._handle_state_change,
            silent=True,
        )

        self._initial_budget = budget_cents
        self._total_spent = 0  # Track actual spending regardless of budget
        self._history: list[tuple[datetime, int]] = []
        self._start_time = datetime.now()

    def _handle_state_change(self, new_state: MetabolicState) -> None:
        """React to metabolic state transitions."""
        if not self.on_alert:
            return

        level = "info"
        message = f"Ticket {self.ticket.id} budget state: {new_state.value}"

        if new_state == MetabolicState.CONSERVING:
            level = "warning"
            message = f"Ticket {self.ticket.id} has consumed {self.get_utilization():.0%} of budget"
        elif new_state == MetabolicState.STARVING:
            level = "critical"
            message = f"Ticket {self.ticket.id} is nearly out of budget!"

        self.on_alert(level, message, self.get_report().__dict__)

    def record_cost(self, cost_usd: Decimal) -> bool:
        """
        Record cost consumption from Claude Code.

        Returns True if within budget, False if over budget.
        Tracks actual spending regardless of budget for reporting.
        """
        cost_cents = int(cost_usd * 100)
        self._total_spent += cost_cents  # Always track actual spend
        self._history.append((datetime.now(), cost_cents))

        # Try to consume from ATP budget
        success = self.budget.consume(cost_cents, operation=f"ticket:{self.ticket.id}")

        # If over budget, trigger critical alert
        if not success and self.on_alert:
            overage = (self._total_spent - self._initial_budget) / 100
            self.on_alert(
                "critical",
                f"Ticket {self.ticket.id} is ${overage:.2f} OVER BUDGET!",
                self.get_report().__dict__,
            )

        return success

    def get_utilization(self) -> float:
        """Get budget utilization (can exceed 1.0 if over budget)."""
        if self._initial_budget == 0:
            return 1.0
        return self._total_spent / self._initial_budget

    def get_efficiency(self) -> float:
        """Get dollars spent per story point."""
        spent_dollars = self._total_spent / 100
        points = self.ticket.estimate or self.config.default_estimate
        if points == 0:
            return 0.0
        return spent_dollars / points

    def predict_exhaustion(self) -> date | None:
        """Predict when budget will be exhausted based on burn rate."""
        if len(self._history) < 2:
            return None

        # Calculate average daily burn rate
        total_spent = sum(cost for _, cost in self._history)
        days = (datetime.now() - self._start_time).days or 1
        daily_rate = total_spent / days

        if daily_rate <= 0:
            return None

        remaining = self.budget.atp
        days_remaining = int(remaining / daily_rate)

        return (datetime.now() + timedelta(days=days_remaining)).date()

    def get_report(self) -> TicketBudgetReport:
        """Generate comprehensive budget report."""
        spent = self._total_spent / 100
        budget = self._initial_budget / 100
        remaining = budget - spent  # Can be negative if over budget

        return TicketBudgetReport(
            ticket_id=self.ticket.id,
            estimate_points=self.ticket.estimate or self.config.default_estimate,
            budget_dollars=budget,
            spent_dollars=spent,
            remaining_dollars=remaining,
            utilization=self.get_utilization(),
            state=self.budget.get_state(),
            efficiency=self.get_efficiency(),
            predicted_exhaustion=self.predict_exhaustion(),
        )


# =============================================================================
# Alert Dispatcher
# =============================================================================

class AlertDispatcher:
    """
    Multi-channel alert dispatcher.

    Outputs to:
    - Console (always)
    - JSON file (optional)
    - Slack webhook (optional, via SLACK_WEBHOOK_URL)
    - Linear comment (optional, via LinearClient)
    """

    def __init__(
        self,
        slack_webhook_url: str | None = None,
        linear_client: LinearClient | None = None,
        output_file: str | None = None,
    ):
        self.slack_webhook = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.linear_client = linear_client
        self.output_file = output_file
        self._alerts: list[dict] = []

    def dispatch(
        self,
        level: str,
        message: str,
        data: dict | None = None,
        ticket_id: str | None = None,
    ) -> None:
        """Send alert to all configured channels."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        }
        self._alerts.append(alert)

        # Console output
        icons = {"info": "INFO", "warning": "WARN", "critical": "CRIT"}
        print(f"  [{icons.get(level, 'INFO')}] {message}")

        # Slack webhook
        if self.slack_webhook and level in ("warning", "critical"):
            self._send_slack(level, message, data)

        # Linear comment
        if self.linear_client and ticket_id and level == "critical":
            self._add_linear_comment(ticket_id, level, message, data)

        # JSON output
        if self.output_file:
            self._write_json()

    def _send_slack(self, level: str, message: str, data: dict | None) -> None:
        """Send Slack webhook notification."""
        if not self.slack_webhook:
            return

        color = "#ff0000" if level == "critical" else "#ffcc00"
        payload = json.dumps({
            "attachments": [{
                "color": color,
                "title": f"Budget Alert: {level.upper()}",
                "text": message,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in (data or {}).items()
                    if k in ("ticket_id", "utilization", "remaining_dollars")
                ],
            }]
        }).encode("utf-8")

        try:
            request = Request(
                self.slack_webhook,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urlopen(request, timeout=5)
        except URLError:
            pass

    def _add_linear_comment(
        self,
        ticket_id: str,
        level: str,
        message: str,
        data: dict | None,
    ) -> None:
        """Add comment to Linear ticket."""
        if not self.linear_client:
            return

        comment = f"**Budget Alert ({level.upper()})**\n\n{message}"
        if data:
            comment += "\n\n| Metric | Value |\n|--------|-------|\n"
            for k, v in data.items():
                comment += f"| {k} | {v} |\n"

        self.linear_client.add_comment(ticket_id, comment)

    def _write_json(self) -> None:
        """Write alerts to JSON file."""
        if not self.output_file:
            return

        with open(self.output_file, "w") as f:
            json.dump(self._alerts, f, indent=2, default=str)

    def get_alerts(self) -> list[dict]:
        """Get all dispatched alerts."""
        return self._alerts


# =============================================================================
# Main Application
# =============================================================================

class LinearBudgetTracker:
    """
    Main application coordinating all components.

    Workflow:
    1. Parse CSV cost data
    2. Analyze git commits for ticket attribution
    3. Attribute costs to tickets
    4. Track budgets using ATP_Store
    5. Alert on threshold crossings
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        linear_client: LinearClient | None = None,
        alert_dispatcher: AlertDispatcher | None = None,
    ):
        self.config = config or BudgetConfig()
        self.linear_client = linear_client or LinearClient()
        self.dispatcher = alert_dispatcher or AlertDispatcher(linear_client=self.linear_client)

        self._trackers: dict[str, TicketBudgetTracker] = {}

    def analyze(
        self,
        csv_path: str,
        repo_path: str = ".",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, TicketBudgetReport]:
        """
        Run full analysis pipeline.

        Returns:
            dict mapping ticket_id -> budget report
        """
        print("\n--- Parsing CSV cost data ---")
        costs = CostCSVParser.parse_csv(csv_path)
        print(f"  Loaded {len(costs)} cost records")

        daily_costs = CostCSVParser.aggregate_by_user_day(costs)
        users = list(daily_costs.keys())
        print(f"  Users: {', '.join(users[:5])}{'...' if len(users) > 5 else ''}")

        # Determine date range from CSV if not provided
        if costs:
            all_dates = [c.date for c in costs]
            start_date = start_date or min(all_dates)
            end_date = end_date or max(all_dates)
        else:
            start_date = start_date or date.today() - timedelta(days=30)
            end_date = end_date or date.today()

        print(f"\n--- Analyzing git commits ({start_date} to {end_date}) ---")
        commits = GitTicketAnalyzer.get_commits_by_date_range(repo_path, start_date, end_date)
        print(f"  Found {len(commits)} commit activities")

        tickets = set(c.ticket_id for c in commits)
        print(f"  Tickets: {', '.join(list(tickets)[:5])}{'...' if len(tickets) > 5 else ''}")

        print("\n--- Attributing costs to tickets ---")
        ticket_costs = CostAttributor.attribute_costs(daily_costs, commits)
        print(f"  Attributed costs to {len(ticket_costs)} tickets")

        print("\n--- Tracking budgets ---")
        reports = {}

        for ticket_id, cost in ticket_costs.items():
            # Try to fetch ticket details from Linear
            ticket = None
            if self.linear_client.is_available():
                ticket = self.linear_client.get_issue(ticket_id)

            # Fall back to mock ticket if Linear unavailable
            if not ticket:
                ticket = LinearTicket(
                    id=ticket_id,
                    title=f"Ticket {ticket_id}",
                    type="task",
                    estimate=None,
                    state="in_progress",
                )

            # Create tracker
            tracker = TicketBudgetTracker(
                ticket=ticket,
                config=self.config,
                on_alert=lambda l, m, d: self.dispatcher.dispatch(l, m, d, ticket_id),
            )

            # Record cost
            tracker.record_cost(cost)

            # Store tracker and report
            self._trackers[ticket_id] = tracker
            reports[ticket_id] = tracker.get_report()

        return reports

    def print_summary(self, reports: dict[str, TicketBudgetReport]) -> None:
        """Print summary table of all ticket budgets."""
        print("\n" + "=" * 70)
        print("Budget Summary")
        print("=" * 70)
        print(f"{'Ticket':<12} {'Est':>4} {'Budget':>8} {'Spent':>8} {'Util':>6} {'State':<12}")
        print("-" * 70)

        for ticket_id, report in sorted(reports.items()):
            state_str = report.state.value
            if report.utilization > 1.0:
                state_str = "OVER BUDGET!"
            elif report.state == MetabolicState.STARVING:
                state_str = "STARVING!"
            elif report.state == MetabolicState.CONSERVING:
                state_str = "conserving"

            # Check if using default estimate
            tracker = self._trackers.get(ticket_id)
            est_str = f"{report.estimate_points:>4}"
            if tracker and tracker.ticket.estimate is None:
                est_str = f"{report.estimate_points:>3}*"  # asterisk = default

            print(
                f"{ticket_id:<12} "
                f"{est_str} "
                f"${report.budget_dollars:>6.2f} "
                f"${report.spent_dollars:>6.2f} "
                f"{report.utilization:>5.0%} "
                f"{state_str:<12}"
            )

        print("-" * 70)

        total_budget = sum(r.budget_dollars for r in reports.values())
        total_spent = sum(r.spent_dollars for r in reports.values())
        print(f"{'TOTAL':<12} {'':>4} ${total_budget:>6.2f} ${total_spent:>6.2f}")
        print()
        print("* = no estimate in Linear, using default")


# =============================================================================
# Demo and Tests
# =============================================================================

def create_mock_data() -> tuple[list[DailyCost], list[CommitActivity], list[LinearTicket]]:
    """Create mock data for demonstration."""
    # Mock costs (usernames extracted from api_key field)
    costs = [
        DailyCost(date(2025, 12, 1), "alice.smith", "Claude Haiku 4.5", Decimal("2.50"), "input", "Claude Code"),
        DailyCost(date(2025, 12, 1), "alice.smith", "Claude Haiku 4.5", Decimal("1.20"), "output", "Claude Code"),
        DailyCost(date(2025, 12, 2), "alice.smith", "Claude Opus 4", Decimal("8.00"), "input", "Claude Code"),
        DailyCost(date(2025, 12, 1), "bob.jones", "Claude Haiku 4.5", Decimal("3.00"), "input", "Claude Code"),
        DailyCost(date(2025, 12, 2), "bob.jones", "Claude Haiku 4.5", Decimal("4.50"), "input", "Claude Code"),
        DailyCost(date(2025, 12, 3), "bob.jones", "Claude Opus 4", Decimal("12.00"), "input", "Claude Code"),
    ]

    # Mock commits (author names normalized: "Alice Smith" -> "alicesmith")
    commits = [
        CommitActivity(date(2025, 12, 1), "Alice Smith", "ENG-101", 3, "ENG-101-auth-flow"),
        CommitActivity(date(2025, 12, 2), "Alice Smith", "ENG-101", 2, "ENG-101-auth-flow"),
        CommitActivity(date(2025, 12, 1), "Bob Jones", "ENG-102", 2, "ENG-102-api-endpoint"),
        CommitActivity(date(2025, 12, 2), "Bob Jones", "ENG-102", 4, "ENG-102-api-endpoint"),
        CommitActivity(date(2025, 12, 3), "Bob Jones", "ENG-103", 1, "ENG-103-bugfix"),
    ]

    # Mock tickets
    tickets = [
        LinearTicket("ENG-101", "Implement auth flow", "feature", 5, "in_progress"),
        LinearTicket("ENG-102", "Build API endpoint", "task", 3, "in_progress"),
        LinearTicket("ENG-103", "Fix login bug", "bug", 1, "in_progress"),
    ]

    return costs, commits, tickets


def run_demo(csv_path: str | None = None):
    """Run demonstration with mock or real data."""
    print("=" * 70)
    print("Linear Budget Tracking - Demo")
    print("=" * 70)

    config = BudgetConfig(
        dollars_per_point=5.00,
        warning_threshold=0.7,
        critical_threshold=0.9,
    )

    if csv_path:
        # Real data mode
        print(f"\nUsing real CSV: {csv_path}")
        tracker = LinearBudgetTracker(config=config)
        reports = tracker.analyze(csv_path)
        tracker.print_summary(reports)
    else:
        # Mock data mode
        print("\nUsing mock data (pass --csv <path> for real data)")

        costs, commits, tickets = create_mock_data()

        # Aggregate costs
        daily_costs = CostCSVParser.aggregate_by_user_day(costs)

        # Attribute to tickets
        ticket_costs = CostAttributor.attribute_costs(daily_costs, commits)

        print("\n--- Cost Attribution ---")
        for ticket_id, cost in ticket_costs.items():
            print(f"  {ticket_id}: ${cost:.2f}")

        # Track budgets
        dispatcher = AlertDispatcher()
        reports = {}

        print("\n--- Budget Tracking ---")
        for ticket in tickets:
            tracker = TicketBudgetTracker(
                ticket=ticket,
                config=config,
                on_alert=lambda l, m, d: dispatcher.dispatch(l, m, d),
            )

            if ticket.id in ticket_costs:
                tracker.record_cost(ticket_costs[ticket.id])

            reports[ticket.id] = tracker.get_report()

        # Summary
        print("\n" + "=" * 70)
        print("Budget Summary")
        print("=" * 70)
        print(f"{'Ticket':<12} {'Type':<8} {'Est':>4} {'Budget':>8} {'Spent':>8} {'Util':>6} {'State':<12}")
        print("-" * 70)

        for ticket in tickets:
            report = reports[ticket.id]
            state_str = report.state.value
            if report.utilization > 1.0:
                state_str = "OVER BUDGET!"
            elif report.state == MetabolicState.STARVING:
                state_str = "STARVING!"
            elif report.state == MetabolicState.CONSERVING:
                state_str = "conserving"

            print(
                f"{ticket.id:<12} "
                f"{ticket.type:<8} "
                f"{report.estimate_points:>4} "
                f"${report.budget_dollars:>6.2f} "
                f"${report.spent_dollars:>6.2f} "
                f"{report.utilization:>5.0%} "
                f"{state_str:<12}"
            )

        print("-" * 70)

    print("\n" + "=" * 70)
    print("Key Insight: Metabolic states map to budget warnings.")
    print("NORMAL (plenty) -> CONSERVING (70%+) -> STARVING (90%+)")
    print("=" * 70)


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Username extraction
    assert CostCSVParser.extract_user_from_api_key("claude_code_key_dmytro.baltak_jivb") == "dmytro.baltak"
    assert CostCSVParser.extract_user_from_api_key("claude_code_key_andy_hcgi") == "andy"
    assert CostCSVParser.extract_user_from_api_key("other_key") == "other_key"
    print("  Test 1: Username extraction - PASSED")

    # Test 2: Branch ticket extraction
    assert GitTicketAnalyzer.extract_ticket_from_branch("ENG-123-feature-name") == "ENG-123"
    assert GitTicketAnalyzer.extract_ticket_from_branch("feature/ENG-456-fix") == "ENG-456"
    assert GitTicketAnalyzer.extract_ticket_from_branch("main") is None
    assert GitTicketAnalyzer.extract_ticket_from_branch("prefix-ABC-789-task") == "ABC-789"
    assert GitTicketAnalyzer.extract_ticket_from_branch("user/dat-1003-some-feature") == "DAT-1003"  # lowercase
    assert GitTicketAnalyzer.extract_ticket_from_branch("b/dat-1575-youtube") == "DAT-1575"  # short prefix
    print("  Test 2: Branch ticket extraction - PASSED")

    # Test 3: Budget config
    config = BudgetConfig(dollars_per_point=10.00)
    assert config.estimate_to_dollars(3, "task") == 3000  # 3 * $10 * 1.0 = $30 = 3000 cents
    assert config.estimate_to_dollars(2, "feature") == 3000  # 2 * $10 * 1.5 = $30
    assert config.estimate_to_dollars(4, "bug") == 2000  # 4 * $10 * 0.5 = $20
    print("  Test 3: Budget config - PASSED")

    # Test 4: Cost attribution
    daily_costs = {
        "alice": {date(2025, 12, 1): Decimal("10.00")},
        "bob": {date(2025, 12, 1): Decimal("6.00")},
    }
    commits = [
        CommitActivity(date(2025, 12, 1), "Alice", "ENG-1", 2, "ENG-1-feat"),
        CommitActivity(date(2025, 12, 1), "Alice", "ENG-2", 1, "ENG-2-fix"),
        CommitActivity(date(2025, 12, 1), "Bob", "ENG-1", 3, "ENG-1-feat"),
    ]
    attributed = CostAttributor.attribute_costs(daily_costs, commits)
    # Alice: $10 split 2:1 between ENG-1 and ENG-2 = $6.67 and $3.33
    # Bob: $6 all to ENG-1
    # ENG-1 total: ~$6.67 + $6 = ~$12.67
    assert "ENG-1" in attributed
    assert "ENG-2" in attributed
    assert attributed["ENG-1"] > attributed["ENG-2"]
    print("  Test 4: Cost attribution - PASSED")

    # Test 5: Budget tracker state transitions
    ticket = LinearTicket("TEST-1", "Test", "task", 2, "in_progress")
    config = BudgetConfig(dollars_per_point=5.00)  # $10 budget
    tracker = TicketBudgetTracker(ticket=ticket, config=config)

    assert tracker.budget.get_state() == MetabolicState.NORMAL
    tracker.record_cost(Decimal("7.50"))  # 75% consumed
    assert tracker.budget.get_state() == MetabolicState.CONSERVING
    tracker.record_cost(Decimal("2.00"))  # 95% consumed
    assert tracker.budget.get_state() == MetabolicState.STARVING
    print("  Test 5: Budget tracker states - PASSED")

    # Test 6: Utilization calculation
    assert tracker.get_utilization() > 0.9
    print("  Test 6: Utilization calculation - PASSED")

    print("\nSmoke test passed!")


def main():
    """Main entry point."""
    if "--test" in sys.argv:
        run_smoke_test()
    elif "--csv" in sys.argv:
        idx = sys.argv.index("--csv")
        if idx + 1 < len(sys.argv):
            run_demo(csv_path=sys.argv[idx + 1])
        else:
            print("Error: --csv requires a path argument")
            sys.exit(1)
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
