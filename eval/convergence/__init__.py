"""C6 Convergence Evaluation Harness.

Evaluates Operon's structural guidance across multiple compilation targets
(Swarms, DeerFlow, Ralph, Scion) by building real organisms, compiling them
through the real compilers and adapters, and measuring risk scores, token
costs, and convergence rates.

Key components:
    - TaskDefinition / get_benchmark_tasks -- 20 benchmark tasks
    - ConfigurationSpec / get_configurations -- 7 framework configurations
    - MockEvaluator -- exercises real Operon compilers and adapters
    - ConvergenceHarness -- orchestrates full evaluation runs
    - generate_convergence_report -- renders results as JSON/Markdown
"""

from .configurations import ConfigurationSpec, get_configurations
from .credit_assignment import StageCredit, aggregate_credit, assign_credit
from .harness import ConvergenceHarness, HarnessConfig
from .metrics import AggregateMetrics, RunMetrics, collect_metrics, compare_configs
from .mock_evaluator import MockEvaluator
from .report import generate_convergence_report, ranking_table
from .structural_variation import topology_distance, variation_summary
from .tasks import TaskDefinition, get_benchmark_tasks, task_to_fingerprint

__all__ = [
    # Tasks
    "TaskDefinition",
    "get_benchmark_tasks",
    "task_to_fingerprint",
    # Configurations
    "ConfigurationSpec",
    "get_configurations",
    # Metrics
    "RunMetrics",
    "AggregateMetrics",
    "collect_metrics",
    "compare_configs",
    # Structural variation
    "topology_distance",
    "variation_summary",
    # Credit assignment
    "StageCredit",
    "assign_credit",
    "aggregate_credit",
    # Mock evaluator
    "MockEvaluator",
    # Harness
    "HarnessConfig",
    "ConvergenceHarness",
    # Report
    "generate_convergence_report",
    "ranking_table",
]
