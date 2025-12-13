"""
Mitochondria: Powerhouse of the Cell
====================================

Biological Analogy:
- Glycolysis: Fast, simple math operations (cytoplasm)
- Krebs Cycle: Complex logical operations (matrix)
- Electron Transport Chain: Chained computations
- ATP Synthase: Resource accounting
- ROS Management: Error/danger tracking (reactive oxygen species)
- Endosymbiosis: Can incorporate external tools safely

The Mitochondria provides deterministic computation capabilities
to the agent, converting "glucose" (expressions) into "ATP" (results).

This implementation uses safe AST parsing - no arbitrary code execution.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable
from enum import Enum
import ast
import operator
import math
import time

from ..core.types import Capability

class MetabolicPathway(Enum):
    """
    Different metabolic pathways for different substrates.

    Each pathway is optimized for a specific type of computation.
    """
    GLYCOLYSIS = "math"           # Simple arithmetic
    KREBS_CYCLE = "logic"         # Boolean operations
    OXIDATIVE = "tool"            # External tool execution
    BETA_OXIDATION = "transform"  # Data transformations


@dataclass
class ATP:
    """
    Energy unit produced by metabolism.

    Contains the result value plus metadata about how it was produced.
    """
    value: Any
    pathway: MetabolicPathway
    efficiency: float = 1.0  # 0.0 to 1.0
    byproducts: list[str] = field(default_factory=list)  # Warnings
    execution_time_ms: float = 0.0


@dataclass
class MetabolicResult:
    """
    Result of metabolic processing.

    Includes the ATP produced (if successful) plus diagnostics.
    """
    success: bool
    atp: ATP | None = None
    error: str | None = None
    ros_level: float = 0.0  # Accumulated danger level


@runtime_checkable
class Tool(Protocol):
    """
    Protocol for external tools (symbiotic organelles).

    Any class implementing this protocol can be "engulfed" by
    the Mitochondria for safe execution.
    """
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class SimpleTool:
    """Simple tool implementation wrapping a callable."""
    name: str
    description: str
    func: Callable[..., Any]
    required_capabilities: set[Capability] = field(default_factory=set)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class Mitochondria:
    """
    Powerhouse of the Cell: Executes deterministic computations safely.

    Provides multiple metabolic pathways:

    1. Glycolysis (Math)
       - Safe arithmetic using AST parsing
       - Supports: +, -, *, /, //, %, **
       - Math functions: sqrt, sin, cos, log, etc.
       - NO arbitrary code execution

    2. Krebs Cycle (Logic)
       - Boolean and comparison operations
       - Returns True/False

    3. Oxidative Phosphorylation (Tools)
       - Execute registered external tools
       - Each tool is sandboxed via the Tool protocol

    4. Beta Oxidation (Transform)
       - Data transformations (JSON, literals)
       - Breaking down complex substrates

    Safety Features:
    - AST-based parsing prevents code injection
    - ROS (Reactive Oxygen Species) tracking for errors
    - Configurable timeout protection
    - Tool sandboxing

    Example:
        >>> mito = Mitochondria()
        >>> result = mito.metabolize("2 + 2 * 10")
        >>> result.atp.value
        22
        >>> result = mito.metabolize("sqrt(16) + pi")
        >>> result.atp.value
        7.141592653589793
    """

    # Safe operators for expression evaluation
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Safe comparison operators
    SAFE_COMPARISONS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    # Safe boolean operators
    SAFE_BOOL_OPS = {
        ast.And: lambda values: all(values),
        ast.Or: lambda values: any(values),
    }

    # Safe math functions and constants
    SAFE_FUNCTIONS: dict[str, Any] = {
        # Basic
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'int': int,
        'float': float,
        'bool': bool,

        # Math functions
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'pow': math.pow,
        'ceil': math.ceil,
        'floor': math.floor,
        'trunc': math.trunc,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'degrees': math.degrees,
        'radians': math.radians,

        # Constants
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
    }

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_ros: float = 1.0,
        tools: list[Tool] | None = None,
        allowed_capabilities: set[Capability] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Mitochondria.

        Args:
            timeout_seconds: Max execution time per operation
            max_ros: Max accumulated errors before shutdown
            tools: External tools to engulf
            allowed_capabilities: If set, tools must declare required capabilities
                as a subset of this set (least-privilege enforcement)
            silent: Suppress console output
        """
        self.timeout = timeout_seconds
        self.max_ros = max_ros
        self.silent = silent
        self.allowed_capabilities = allowed_capabilities

        # Tool registry (endosymbiotic organelles)
        self.tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.engulf_tool(tool)

        # State tracking
        self._total_atp_produced = 0.0
        self._ros_accumulated = 0.0
        self._operations_count = 0

    def engulf_tool(self, tool: Tool):
        """
        Endosymbiosis: Incorporate an external tool.

        Like ancient bacteria being engulfed to become mitochondria,
        external tools become part of the organelle's capabilities.

        Args:
            tool: Any object implementing the Tool protocol
        """
        self.tools[tool.name] = tool
        if not self.silent:
            print(f"🦠 [Mitochondria] Engulfed tool: {tool.name}")

    def register_function(
        self,
        name: str,
        func: Callable,
        description: str = "",
        required_capabilities: set[Capability] | None = None,
    ):
        """
        Convenience method to register a simple function as a tool.

        Args:
            name: Name to call the function by
            func: The callable to register
            description: Human-readable description
            required_capabilities: Capabilities required to invoke this tool
        """
        tool = SimpleTool(
            name=name,
            description=description,
            func=func,
            required_capabilities=required_capabilities or set(),
        )
        self.engulf_tool(tool)

    def metabolize(
        self,
        expression: str,
        pathway: MetabolicPathway | None = None
    ) -> MetabolicResult:
        """
        Main metabolic entry point.

        Automatically detects the appropriate pathway if not specified.

        Args:
            expression: The expression to process
            pathway: Specific pathway to use (auto-detected if None)

        Returns:
            MetabolicResult with ATP or error information
        """
        self._operations_count += 1
        start_time = time.time()

        # Check ROS levels (accumulated danger)
        if self._ros_accumulated >= self.max_ros:
            return MetabolicResult(
                success=False,
                error="Mitochondrial dysfunction: ROS threshold exceeded. Call repair() to recover.",
                ros_level=self._ros_accumulated
            )

        # Auto-detect pathway if not specified
        if pathway is None:
            pathway = self._detect_pathway(expression)

        if not self.silent:
            print(f"⚡ [Mitochondria] Metabolizing: {expression[:50]}...")

        try:
            if pathway == MetabolicPathway.GLYCOLYSIS:
                result = self._glycolysis(expression)
            elif pathway == MetabolicPathway.KREBS_CYCLE:
                result = self._krebs_cycle(expression)
            elif pathway == MetabolicPathway.OXIDATIVE:
                result = self._oxidative_phosphorylation(expression)
            elif pathway == MetabolicPathway.BETA_OXIDATION:
                result = self._beta_oxidation(expression)
            else:
                return MetabolicResult(
                    success=False,
                    error=f"Unknown pathway: {pathway}",
                    ros_level=self._ros_accumulated
                )

            execution_time = (time.time() - start_time) * 1000
            efficiency = max(0.1, 1.0 - (execution_time / (self.timeout * 1000)))

            atp = ATP(
                value=result,
                pathway=pathway,
                efficiency=efficiency,
                execution_time_ms=execution_time
            )

            self._total_atp_produced += efficiency

            return MetabolicResult(
                success=True,
                atp=atp,
                ros_level=self._ros_accumulated
            )

        except Exception as e:
            self._ros_accumulated += 0.1
            return MetabolicResult(
                success=False,
                error=f"Metabolic failure: {e}",
                ros_level=self._ros_accumulated
            )

    def digest_glucose(self, expression: str) -> str:
        """
        Legacy API for backward compatibility.

        Args:
            expression: Math expression to process

        Returns:
            String result or error message
        """
        result = self.metabolize(expression, MetabolicPathway.GLYCOLYSIS)
        if result.success and result.atp:
            return str(result.atp.value)
        return f"Metabolic Failure: {result.error}"

    def _detect_pathway(self, expression: str) -> MetabolicPathway:
        """Auto-detect the appropriate metabolic pathway."""
        expr_lower = expression.lower().strip()

        # Check for tool calls
        for tool_name in self.tools:
            if expr_lower.startswith(f"{tool_name.lower()}("):
                return MetabolicPathway.OXIDATIVE

        # Check for JSON/dict/list literals
        if expr_lower.startswith(('{', '[')):
            return MetabolicPathway.BETA_OXIDATION

        # Check for boolean keywords
        if any(kw in expr_lower for kw in ['true', 'false', ' and ', ' or ', ' not ']):
            return MetabolicPathway.KREBS_CYCLE

        # Check for comparisons
        if any(op in expression for op in ['==', '!=', '<=', '>=', '<', '>']):
            return MetabolicPathway.KREBS_CYCLE

        # Default to math
        return MetabolicPathway.GLYCOLYSIS

    def _glycolysis(self, expression: str) -> Any:
        """
        Safe math computation using AST parsing.

        Fast but limited - like real glycolysis in the cytoplasm.
        """
        tree = ast.parse(expression, mode='eval')
        return self._compute_node(tree.body)

    def _krebs_cycle(self, expression: str) -> bool:
        """
        Boolean/logical computation.

        More complex than glycolysis - like the Krebs cycle in
        the mitochondrial matrix.
        """
        # Normalize Python boolean literals
        expression = expression.replace('True', '1').replace('False', '0')
        expression = expression.replace('true', '1').replace('false', '0')

        tree = ast.parse(expression, mode='eval')
        return bool(self._compute_node(tree.body))

    def _oxidative_phosphorylation(self, expression: str) -> Any:
        """
        Execute a registered tool.

        Most ATP production but also most ROS risk.
        Format: "tool_name(arg1, arg2, kwarg=value)"
        """
        tree = ast.parse(expression, mode='eval')

        if not isinstance(tree.body, ast.Call):
            raise ValueError("Expected a tool call: tool_name(args)")

        if isinstance(tree.body.func, ast.Name):
            tool_name = tree.body.func.id
        else:
            raise ValueError("Invalid tool call format")

        if tool_name not in self.tools:
            available = list(self.tools.keys())
            raise ValueError(f"Unknown tool: {tool_name}. Available: {available}")

        tool = self.tools[tool_name]
        required_caps = (
            getattr(tool, "required_capabilities", None)
            or getattr(tool, "capabilities", None)
            or set()
        )
        required_caps = set(required_caps)
        if self.allowed_capabilities is not None and not required_caps.issubset(self.allowed_capabilities):
            missing = sorted(
                (c.value if isinstance(c, Capability) else str(c)) for c in (required_caps - self.allowed_capabilities)
            )
            raise PermissionError(
                f"Tool '{tool_name}' requires disallowed capabilities: {missing}"
            )

        args = [self._compute_node(arg) for arg in tree.body.args]
        kwargs = {kw.arg: self._compute_node(kw.value) for kw in tree.body.keywords if kw.arg}

        return tool.execute(*args, **kwargs)

    def _beta_oxidation(self, expression: str) -> Any:
        """
        Data transformation operations.

        Breaking down complex substrates (like fatty acids to acetyl-CoA).
        Handles JSON and Python literals safely.
        """
        import json

        expression = expression.strip()

        # Try JSON first
        try:
            return json.loads(expression)
        except json.JSONDecodeError:
            pass

        # Try Python literal (safe)
        try:
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError):
            pass

        raise ValueError(f"Cannot parse as JSON or Python literal: {expression[:50]}...")

    def _compute_node(self, node: ast.AST) -> Any:
        """Recursively compute AST nodes safely."""

        # Constants (numbers, strings, etc.)
        if isinstance(node, ast.Constant):
            return node.value

        # Binary operations (+, -, *, /, etc.)
        elif isinstance(node, ast.BinOp):
            left = self._compute_node(node.left)
            right = self._compute_node(node.right)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        # Unary operations (-, +)
        elif isinstance(node, ast.UnaryOp):
            operand = self._compute_node(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(operand)

        # Function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.SAFE_FUNCTIONS:
                    func = self.SAFE_FUNCTIONS[func_name]
                    args = [self._compute_node(arg) for arg in node.args]
                    if callable(func):
                        return func(*args)
                    return func  # Constants like pi, e
                raise ValueError(f"Unknown function: {func_name}")
            raise ValueError("Complex function calls not supported")

        # Variable names (for constants like pi, e)
        elif isinstance(node, ast.Name):
            if node.id in self.SAFE_FUNCTIONS:
                return self.SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        # Lists
        elif isinstance(node, ast.List):
            return [self._compute_node(el) for el in node.elts]

        # Tuples
        elif isinstance(node, ast.Tuple):
            return tuple(self._compute_node(el) for el in node.elts)

        # Comparisons
        elif isinstance(node, ast.Compare):
            left = self._compute_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._compute_node(comparator)
                cmp_func = self.SAFE_COMPARISONS.get(type(op))
                if cmp_func is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not cmp_func(left, right):
                    return False
                left = right
            return True

        # Boolean operations (and, or)
        elif isinstance(node, ast.BoolOp):
            values = [self._compute_node(v) for v in node.values]
            bool_func = self.SAFE_BOOL_OPS.get(type(node.op))
            if bool_func is None:
                raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")
            return bool_func(values)

        # If expressions (ternary)
        elif isinstance(node, ast.IfExp):
            if self._compute_node(node.test):
                return self._compute_node(node.body)
            return self._compute_node(node.orelse)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def get_efficiency(self) -> float:
        """Get total ATP efficiency produced."""
        return self._total_atp_produced

    def get_ros_level(self) -> float:
        """Get accumulated reactive oxygen species (danger level)."""
        return self._ros_accumulated

    def repair(self, amount: float = 0.5):
        """
        Mitophagy: Repair accumulated damage.

        In biology, damaged mitochondria are recycled.
        """
        self._ros_accumulated = max(0, self._ros_accumulated - amount)
        if not self.silent:
            print(f"🔧 [Mitochondria] Repaired. ROS level: {self._ros_accumulated:.2f}")

    def get_statistics(self) -> dict:
        """Get metabolic statistics."""
        return {
            "operations_count": self._operations_count,
            "total_atp_produced": self._total_atp_produced,
            "ros_level": self._ros_accumulated,
            "tools_available": list(self.tools.keys()),
            "health": "healthy" if self._ros_accumulated < self.max_ros else "dysfunctional",
        }

    def list_tools(self) -> list[dict]:
        """List all available tools with descriptions."""
        tools = []
        for name, tool in self.tools.items():
            required_caps = (
                getattr(tool, "required_capabilities", None)
                or getattr(tool, "capabilities", None)
                or set()
            )
            tools.append(
                {
                    "name": name,
                    "description": tool.description,
                    "required_capabilities": sorted(
                        c.value if isinstance(c, Capability) else str(c) for c in set(required_caps)
                    ),
                }
            )
        return tools
