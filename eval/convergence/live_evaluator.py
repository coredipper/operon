"""Live evaluator for the C6 convergence evaluation harness.

Runs actual LLM calls through Operon's SkillOrganism pipeline to measure
real success rates, token costs, and latencies across guided vs unguided
configurations.

Unlike the MockEvaluator (which uses structural analysis to derive synthetic
metrics), this evaluator executes tasks end-to-end with real providers.

Supported providers:
  - gemini: Google Gemini API (requires GEMINI_API_KEY)
  - anthropic: Anthropic Claude API (requires ANTHROPIC_API_KEY)
  - openai: OpenAI API (requires OPENAI_API_KEY)
  - ollama: Local Ollama server (localhost:11434, no API key)
  - lmstudio: Local LM Studio server (localhost:1234, no API key)
  - claude: Claude Code CLI (requires `claude` binary)
  - codex: Codex CLI (requires `codex` binary)
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from operon_ai import Nucleus, SkillStage, skill_organism
from operon_ai.patterns.advisor import advise_topology
from operon_ai.patterns.cli import cli_handler
from operon_ai.providers.anthropic_provider import AnthropicProvider
from operon_ai.providers.base import ProviderConfig
from operon_ai.providers.gemini_provider import GeminiProvider
from operon_ai.providers.mock import MockProvider
from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider
from operon_ai.providers.openai_provider import OpenAIProvider

from operon_ai.convergence.swarms_adapter import analyze_external_topology
from operon_ai.convergence.types import ExternalTopology

from .tasks import TaskDefinition


# ---------------------------------------------------------------------------
# Local provider detection
# ---------------------------------------------------------------------------

def _probe_chat(base_url: str, model: str) -> bool:
    """Verify a model can handle chat completions with a tiny request."""
    try:
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read())
            return bool(data.get("choices"))
    except Exception:
        return False


def _list_models(base_url: str) -> list[str]:
    """Return model IDs from an OpenAI-compatible /v1/models endpoint."""
    try:
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def _first_chat_model(base_url: str, candidates: list[str]) -> str | None:
    """Return the first model in *candidates* that passes a chat probe."""
    for model in candidates:
        if _probe_chat(base_url, model):
            return model
    return None


def _detect_ollama() -> OpenAICompatibleProvider | None:
    """Return a provider for the local Ollama server if reachable and has a chat-capable model."""
    base = "http://localhost:11434/v1"
    models = _list_models(base)
    if not models:
        return None
    # Try preferred reasoning models first, then all others
    preferred = ["gemma4:latest", "deepseek-r1:8b", "deepseek-r1:latest", "qwen3:latest"]
    ordered = [m for m in preferred if m in models] + [m for m in models if m not in preferred]
    model = _first_chat_model(base, ordered)
    if not model:
        return None
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url=base,
        model=model,
    )


def _detect_lmstudio() -> OpenAICompatibleProvider | None:
    """Return a provider for LM Studio if reachable and has a chat-capable model."""
    base = "http://localhost:1234/v1"
    models = _list_models(base)
    if not models:
        return None
    model = _first_chat_model(base, models)
    if not model:
        return None
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url=base,
        model=model,
    )


# ---------------------------------------------------------------------------
# Concrete task prompts for all 20 benchmark tasks
# ---------------------------------------------------------------------------

_TASK_PROMPTS: dict[str, str] = {
    "easy_seq_01": (
        "Summarize the following into 3 bullet points:\n\n"
        "The James Webb Space Telescope (JWST) has detected carbon dioxide "
        "in the atmosphere of exoplanet WASP-39b, a gas giant orbiting a "
        "star 700 light-years away. This marks the first clear evidence of "
        "CO2 on a planet outside our solar system. The discovery was made "
        "using JWST's Near-Infrared Spectrograph (NIRSpec), which measures "
        "how starlight filters through a planet's atmosphere. The finding "
        "demonstrates JWST's ability to characterize exoplanet atmospheres "
        "with unprecedented precision, opening new possibilities for "
        "studying potentially habitable worlds."
    ),
    "easy_seq_02": (
        "Extract the following fields as JSON: {name, date, amount, currency}\n\n"
        "Invoice #2847 issued to Acme Corp on March 15, 2026 for consulting "
        "services rendered. Total amount: EUR 12,500.00. Payment terms: "
        "Net 30. Contact: Jane Smith, CFO."
    ),
    "easy_seq_03": (
        "Translate the following English text to French, then verify the "
        "translation preserves the technical meaning:\n\n"
        "The compiler optimizes intermediate representations by eliminating "
        "dead code and propagating constants through the control flow graph."
    ),
    "med_mix_01": (
        "Research synthesis task: Compare and contrast these two approaches "
        "to multi-agent coordination, then recommend which is better suited "
        "for a 5-agent customer support system:\n\n"
        "Approach A: Hierarchical supervisor pattern where a manager agent "
        "delegates to specialist workers (billing, technical, shipping, "
        "returns, general inquiry).\n\n"
        "Approach B: Peer-to-peer mesh where each specialist can hand off "
        "to any other specialist directly, with no central coordinator."
    ),
    "med_mix_02": (
        "Code review: Analyze this Python function for bugs, style issues, "
        "and security vulnerabilities, then merge your findings into a "
        "single report:\n\n"
        "def process_user_input(data):\n"
        "    query = f\"SELECT * FROM users WHERE name = '{data['name']}'\"\n"
        "    result = db.execute(query)\n"
        "    users = []\n"
        "    for row in result:\n"
        "        user = {'name': row[0], 'email': row[1]}\n"
        "        user['token'] = hashlib.md5(row[1].encode()).hexdigest()\n"
        "        users.append(user)\n"
        "    return json.dumps(users)\n"
    ),
    "med_mix_03": (
        "Data pipeline task: Given this raw CSV data, ingest it, clean any "
        "invalid rows, transform the date column to ISO 8601 format, and "
        "produce a final cleaned dataset ready for warehouse loading:\n\n"
        "id,name,signup_date,email,revenue\n"
        "1,Alice,2026/03/15,alice@example.com,1250.00\n"
        "2,Bob,March 20 2026,bob@example,invalid\n"
        "3,,2026-03-22,charlie@example.com,890.50\n"
        "4,Diana,2026/03/25,diana@example.com,2100.00\n"
        "5,Eve,N/A,eve@example.com,0\n"
    ),
    "med_mix_04": (
        "Content creation task: Draft a short product announcement for a "
        "new AI-powered code review tool called 'CodeLens'. Include a "
        "headline, a 2-paragraph body describing the key features "
        "(real-time suggestions, security scanning, multi-language support), "
        "suggest a hero image concept, propose a page layout, and review "
        "the final draft for tone and accuracy."
    ),
    # --- Easy sequential (remaining) ---
    "easy_seq_04": (
        "Reformat this Python code to follow PEP 8 style, then lint it:\n\n"
        "def   calcArea( w,h ):\n"
        "  result=w*h\n"
        "  if(result>100):print('large')\n"
        "  return result\n"
        "class   myClass:\n"
        " def __init__(self,x):\n"
        "  self.x =x\n"
        " def run( self):\n"
        "  return self.x**2\n"
    ),
    "easy_seq_05": (
        "Classify each email below into one of: billing, technical, shipping, "
        "returns, general. Then route it to the correct department and log "
        "the classification.\n\n"
        "Email 1: 'My credit card was charged twice for order #4821.'\n"
        "Email 2: 'The API returns 502 when I call /v2/users endpoint.'\n"
        "Email 3: 'Where is my package? Tracking says delivered but I have nothing.'\n"
        "Email 4: 'I want to return the keyboard, it has a sticky spacebar.'\n"
        "Email 5: 'Do you have any job openings in engineering?'\n"
    ),
    # --- Medium mixed (remaining) ---
    "med_mix_05": (
        "Incident triage: Given these alerts, correlate the events, diagnose "
        "the root cause, and propose a remediation plan.\n\n"
        "14:01 ALERT CPU > 95% on web-server-03\n"
        "14:02 ALERT Response time p99 > 5s on /api/search\n"
        "14:02 ALERT Connection pool exhausted on db-replica-02\n"
        "14:03 ALERT OOM kill on web-server-03 (process: java, RSS: 7.8GB)\n"
        "14:03 ALERT Deployment pipeline completed: search-service v3.2.1\n"
        "14:04 ALERT Error rate > 10% on /api/search\n\n"
        "Which event is the root cause? What's the remediation?"
    ),
    "med_mix_06": (
        "Document QA: Given this text, chunk it into sections, then answer "
        "the question by retrieving the relevant chunk.\n\n"
        "TEXT:\n"
        "Section 1 - Overview: Operon is a biomimetic agent framework with "
        "1,530 tests and 107 examples. It provides structural analysis for "
        "multi-agent topologies.\n"
        "Section 2 - Convergence: The convergence package has 23 modules "
        "and 6 adapters bridging external frameworks (Swarms, DeerFlow, "
        "AnimaWorks, Ralph, A-Evolve, Scion).\n"
        "Section 3 - Evaluation: The evaluation harness runs 20 benchmark "
        "tasks across 7 configurations with structural variation analysis.\n\n"
        "QUESTION: How many adapters does the convergence package have, "
        "and which frameworks do they support?"
    ),
    "med_mix_07": (
        "Test generation: Analyze this function, generate unit tests, run "
        "them mentally to predict pass/fail, and report coverage.\n\n"
        "def fibonacci(n: int) -> int:\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be non-negative')\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n\n"
        "Generate at least 5 test cases covering edge cases, normal cases, "
        "and error cases. Predict the expected output for each."
    ),
    "med_mix_08": (
        "Multi-lingual support: Detect the language of this query, translate "
        "it to English, answer in English, then translate the answer back "
        "to the original language.\n\n"
        "Query: 'Wie viele Adapter hat das Konvergenzpaket und welche "
        "Frameworks werden unterstuetzt?'"
    ),
    # --- Hard parallel ---
    "hard_par_01": (
        "Distributed debugging task: Three microservices are involved in a "
        "payment flow. Service A (order-api) receives the request, calls "
        "Service B (payment-gateway) which calls Service C (fraud-detector). "
        "Users report that 5% of payments fail with 'timeout' but the "
        "fraud-detector logs show all checks complete in <200ms. "
        "Service B has a 500ms timeout for downstream calls. "
        "Service A retries failed requests up to 3 times with no backoff. "
        "What is the most likely root cause, and what changes would you "
        "make to each service?"
    ),
    "hard_par_02": (
        "Competitive analysis: Compare these 5 AI agent frameworks based "
        "on the provided summaries. Aggregate into a comparison matrix and "
        "write a recommendation.\n\n"
        "Swarms: Graph-based workflow engine. Agents as nodes, edges as "
        "data flow. Sequential/parallel/hierarchical patterns. Python-only. "
        "No built-in error recovery. Deploy via pip + custom orchestrator.\n\n"
        "DeerFlow: LangGraph-based with skill system. Supervisor delegates "
        "to sub-agents with Markdown skill definitions. Thinking mode for "
        "complex tasks. Deploy via Docker + LangServe.\n\n"
        "CrewAI: Role-based with 'crew' abstraction. Agents have roles, "
        "goals, backstories. Sequential/hierarchical process. Built-in "
        "memory. Deploy via CrewAI+ cloud or self-hosted.\n\n"
        "AutoGen: Conversation-based multi-agent. Group chat with speaker "
        "selection. Code execution in Docker sandbox. Flexible but complex "
        "setup. Deploy via Azure or self-hosted.\n\n"
        "LangGraph: State machine with conditional edges. Checkpointing "
        "and human-in-the-loop. Streaming support. Production-grade via "
        "LangSmith. Deploy via LangGraph Cloud.\n\n"
        "Compare on: architecture, coordination, error handling, production "
        "readiness. Recommend for a 5-agent customer support system."
    ),
    "hard_par_03": (
        "ML pipeline: Given this dataset description, design and evaluate "
        "a classification pipeline.\n\n"
        "Dataset: 10,000 customer support tickets with columns: "
        "ticket_text (string), category (billing/technical/shipping/returns), "
        "priority (low/medium/high), resolution_time_hours (float).\n\n"
        "Tasks in parallel:\n"
        "1. Feature engineering: TF-IDF + ticket length + keyword presence\n"
        "2. Model selection: compare logistic regression, random forest, SVM\n"
        "3. Hyperparameter tuning: grid search over top model\n"
        "4. Evaluation: accuracy, precision, recall, F1, confusion matrix\n"
        "5. Deployment: describe serving strategy and monitoring\n"
        "6. Monitoring: define drift detection and alerting thresholds"
    ),
    "hard_par_04": (
        "Security audit: Analyze the following application description and "
        "bundled vulnerability data to produce a unified severity-ranked "
        "vulnerability report.\n\n"
        "Application: Python Flask REST API with PostgreSQL, deployed on "
        "AWS ECS with nginx reverse proxy.\n\n"
        "SAST findings (provided):\n"
        "- Line 42 app.py: f-string SQL query (SQL injection, CRITICAL)\n"
        "- Line 89 app.py: subprocess with user input (command injection, CRITICAL)\n"
        "- Line 156 templates/profile.html: unescaped user bio (XSS, HIGH)\n\n"
        "Dependency scan results (provided):\n"
        "- Pillow 10.0: CVE-2023-44271 (DoS via large TIFF, MEDIUM)\n"
        "- Flask 2.3: no known CVEs\n"
        "- SQLAlchemy 2.0: no known CVEs\n"
        "- PyJWT 2.8: CVE-2024-33663 (algorithm confusion, HIGH)\n"
        "- requests 2.31: no known CVEs\n\n"
        "Secrets scan (provided):\n"
        "- config.py:3 hardcoded AWS credential (CRITICAL)\n"
        "- .env.example:7 default password placeholder (LOW)\n\n"
        "DAST findings (provided):\n"
        "- /api/admin accessible without authentication (auth bypass, HIGH)\n"
        "- /api/users returns full records including password hashes (data exposure, HIGH)\n\n"
        "Compliance: Map each finding to the relevant OWASP Top 10 category. "
        "Produce a severity-ranked report with remediation recommendations."
    ),
    "hard_par_05": (
        "Multi-modal processing: Process these parallel data streams and "
        "produce a fused summary.\n\n"
        "Text stream: 'The quarterly earnings call revealed a 15% revenue "
        "increase driven by cloud services. The CEO expressed cautious "
        "optimism about AI integration.'\n\n"
        "Image description: 'A bar chart showing revenue by segment: "
        "Cloud $4.2B (up 28%), Enterprise $2.1B (flat), Consumer $1.3B "
        "(down 5%).'\n\n"
        "Audio transcript: 'Analyst question: What percentage of cloud "
        "revenue comes from AI workloads? CEO answer: Approximately 35%, "
        "up from 20% last year.'\n\n"
        "Video summary: 'CEO body language: confident, leaning forward "
        "during AI discussion. CFO appeared tense during consumer segment.'\n\n"
        "Fuse these into a single investment brief with sentiment analysis."
    ),
    "hard_par_06": (
        "Distributed testing: Run these parallel test suites and aggregate "
        "the results into a release-readiness report.\n\n"
        "Unit tests: 847 tests, 12 failing (3 in auth module, 9 in parser)\n"
        "Integration tests: 156 tests, 2 failing (database timeout in CI)\n"
        "E2E tests: 43 tests, 1 failing (flaky Selenium wait on dashboard)\n"
        "Performance tests: p50=45ms, p95=220ms, p99=890ms (SLA: p99<500ms)\n"
        "Fuzz tests: 100K iterations, 3 crashes (null pointer in XML parser)\n\n"
        "Aggregate these results. Which failures are blockers? What's the "
        "overall release risk? Recommend go/no-go with reasoning."
    ),
    "hard_par_07": (
        "Knowledge graph construction: Extract entities and relations from "
        "this text, resolve coreferences, build the graph, and validate.\n\n"
        "Text: 'Dr. Sarah Chen at MIT published a paper on protein folding "
        "with her colleague James Liu. Their work built on DeepMind's "
        "AlphaFold2 model. Chen previously worked at Stanford where she "
        "collaborated with Prof. David Baker on Rosetta. Liu joined MIT "
        "from Google Brain where he developed transformer architectures. "
        "The paper was accepted at NeurIPS 2026.'\n\n"
        "Extract: persons, organizations, publications, relations "
        "(works_at, collaborated_with, built_on, published_at). "
        "Resolve 'she' -> Sarah Chen, 'Their' -> Chen + Liu, etc. "
        "Validate: no contradictions, all relations have valid endpoints."
    ),
    "hard_par_08": (
        "Deep code review: The following Python module handles paginated API "
        "responses and caches results with financial calculations. It is "
        "already in production. Find ALL bugs — focus on logic errors, "
        "concurrency issues, precision problems, and exception handling. "
        "For each bug, explain the failure scenario and propose a fix.\n\n"
        "```python\n"
        "import threading\n"
        "from decimal import Decimal\n"
        "\n"
        "class PaginatedFetcher:\n"
        '    """Fetch all pages from a paginated API endpoint."""\n'
        "\n"
        "    def __init__(self, client, page_size=100):\n"
        "        self.client = client\n"
        "        self.page_size = page_size\n"
        "\n"
        "    def fetch_all(self, endpoint: str) -> list[dict]:\n"
        "        results = []\n"
        "        total = self.client.get_count(endpoint)\n"
        "        num_pages = total // self.page_size  # pages to fetch\n"
        "        for page in range(num_pages):\n"
        "            offset = page * self.page_size\n"
        "            batch = self.client.get(endpoint, offset=offset, limit=self.page_size)\n"
        "            results.extend(batch)\n"
        "        return results\n"
        "\n"
        "\n"
        "class ResultCache:\n"
        '    """Thread-safe cache with TTL-based expiration."""\n'
        "\n"
        "    def __init__(self):\n"
        "        self._store: dict[str, tuple[float, any]] = {}\n"
        "        self._lock = threading.Lock()\n"
        "\n"
        "    def get(self, key: str) -> any | None:\n"
        "        import time\n"
        "        with self._lock:\n"
        "            if key in self._store:\n"
        "                expires_at, value = self._store[key]\n"
        "                if time.time() < expires_at:\n"
        "                    return value\n"
        "                del self._store[key]\n"
        "        return None\n"
        "\n"
        "    def set(self, key: str, value: any, ttl: float = 60.0) -> None:\n"
        "        import time\n"
        "        expires_at = time.time() + ttl\n"
        "        self._store[key] = (expires_at, value)\n"
        "\n"
        "\n"
        "def calculate_portfolio_value(positions: list[dict]) -> float:\n"
        '    """Calculate total portfolio value from position data."""\n'
        "    total = 0.0\n"
        "    for pos in positions:\n"
        "        price = pos['price']  # float from API\n"
        "        qty = pos['quantity']\n"
        "        subtotal = price * qty\n"
        "        fee = subtotal * 0.001  # 0.1% transaction fee\n"
        "        total += subtotal - fee\n"
        "    # Round to cents for display\n"
        "    if total == int(total):\n"
        "        return int(total)\n"
        "    return round(total, 2)\n"
        "\n"
        "\n"
        "def safe_process(records: list[dict]) -> list[dict]:\n"
        '    """Process records with error isolation per record."""\n'
        "    results = []\n"
        "    for record in records:\n"
        "        try:\n"
        "            processed = transform(record)\n"
        "            results.append(processed)\n"
        "        except Exception as e:\n"
        "            results.append({'error': str(e), 'record_id': record.get('id')})\n"
        "    return results\n"
        "```\n\n"
        "Expected bugs to find (the review should identify at least these):\n"
        "1. Pagination logic error\n"
        "2. Thread safety issue in the cache\n"
        "3. Financial calculation precision problem\n"
        "4. Exception handling anti-pattern\n"
        "For each, explain the concrete failure scenario — not just 'this could be wrong'."
    ),
}


# ---------------------------------------------------------------------------
# Quality judge — uses LLM to grade output
# ---------------------------------------------------------------------------


def _judge_quality(
    task_prompt: str,
    output: str,
    provider: Any,
    *,
    fallback_factory: Any | None = None,
) -> tuple[float, str, Any]:
    """Use LLM as judge: score output quality 0.0-1.0.

    Returns ``(score, reason, actual_provider)`` — the third element is the
    provider instance that actually produced the score (may differ from
    *provider* when a fallback fires).

    Strategy:
    1. Primary provider with JSON mode (``response_format``)
    2. Fallback provider (resolved lazily via *fallback_factory*) if primary fails
    3. Regex extraction from plain text as last resort

    Parameters
    ----------
    fallback_factory:
        A zero-arg callable returning a provider or None.  Only called
        after the primary provider has exhausted all retries.
    """
    judge_prompt = (
        "You are an evaluation judge. Score the OUTPUT for the given TASK "
        "on a 0.0-1.0 scale using these criteria:\n"
        "- Correctness (most important, ~50% weight): factual accuracy, "
        "follows instructions, no hallucination\n"
        "- Completeness (~30% weight): covers all parts of the task, "
        "nothing important is missing\n"
        "- Clarity (~20% weight): well-organized, easy to understand\n\n"
        "Score anchors:\n"
        "  0.0-0.2: Wrong, off-topic, or refuses the task\n"
        "  0.3-0.5: Partially correct but major gaps or errors\n"
        "  0.6-0.7: Mostly correct, minor issues\n"
        "  0.8-0.9: Strong, nearly complete and accurate\n"
        "  1.0: Perfect — correct, complete, and clear\n\n"
        "Return ONLY JSON: {\"score\": <float>}\n\n"
        f"TASK: {task_prompt[:800]}\n"
        f"OUTPUT: {output[:2000]}\n"
    )

    providers_to_try: list[Any] = [provider]

    for idx, prov in enumerate(providers_to_try):
        # Reasoning models and Gemini need more tokens for CoT / verbose output
        _model = getattr(prov, "model", "").lower()
        is_reasoning = (
            _model.startswith(("deepseek-r1", "qwen", "gemma4"))
            or "nemotron" in _model
        )
        is_gemini = isinstance(prov, GeminiProvider)
        max_tok = 2048 if is_reasoning else 500 if is_gemini else 150

        # Only use structured output for providers known to support it.
        # Local OpenAI-compatible servers (Ollama, LM Studio) and Gemini
        # often return preamble text instead of raw JSON in JSON mode.
        is_local = isinstance(prov, OpenAICompatibleProvider)
        resp_fmt = None if (is_local or is_gemini) else {"type": "json_object"}

        for _attempt in range(3):
            try:
                resp = prov.complete(
                    judge_prompt,
                    config=ProviderConfig(
                        temperature=0.0,
                        max_tokens=max_tok,
                        response_format=resp_fmt,
                    ),
                )
                text = resp.content.strip()

                # Strip markdown code fences (e.g. ```json\n...\n```)
                fence = re.match(r"^```\w*\n(.*?)```\s*$", text, re.DOTALL)
                if fence:
                    text = fence.group(1).strip()

                # 1) Try full JSON parse — handles {"score": N}, bare
                #    numbers (0.78, 1.0), and exact 0/1
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, (int, float)):
                        return min(max(float(parsed), 0.0), 1.0), "", prov
                    score = float(parsed.get("score", 0.5))
                    return min(max(score, 0.0), 1.0), "", prov
                except (json.JSONDecodeError, TypeError, ValueError,
                        AttributeError):
                    pass

                # 2) Extract {"score": N} embedded in prose — handles
                #    models that wrap JSON in reasoning text
                jm = re.search(r'\{[^{}]*"score"\s*:\s*([0-9.]+)[^{}]*\}', text)
                if jm:
                    try:
                        return min(max(float(jm.group(1)), 0.0), 1.0), "", prov
                    except ValueError:
                        pass

            except Exception:
                time.sleep(2 ** _attempt)
                continue
        # This provider exhausted — resolve fallback lazily if not yet tried
        if fallback_factory is not None and len(providers_to_try) == 1:
            fb = fallback_factory()
            if fb is not None:
                providers_to_try.append(fb)

    return 0.5, "Judge unavailable", provider


def _provider_name(prov: Any) -> str:
    """Derive a human-readable name from a provider instance."""
    if isinstance(prov, OpenAICompatibleProvider):
        url = getattr(prov, "base_url", "")
        if "11434" in url:
            return "ollama"
        if "1234" in url:
            return "lmstudio"
        return "openai-compatible"
    name = getattr(prov, "name", "")
    if name:
        return name
    return type(prov).__name__


# ---------------------------------------------------------------------------
# Live evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveRunResult:
    """Result of a single live evaluation run."""
    task_id: str
    config_name: str
    provider_name: str
    success: bool
    quality_score: float
    quality_reason: str
    total_tokens: int
    total_latency_ms: float
    stage_count: int
    risk_score: float
    stage_details: tuple[dict[str, Any], ...]
    judge_provider: str = ""


# ---------------------------------------------------------------------------
# Live evaluator
# ---------------------------------------------------------------------------


class LiveEvaluator:
    """Runs real LLM evaluations across guided/unguided configurations."""

    def __init__(
        self,
        openai_provider: OpenAIProvider | None = None,
        gemini_provider: GeminiProvider | None = None,
        anthropic_provider: AnthropicProvider | None = None,
    ):
        self.openai = openai_provider or OpenAIProvider()
        self.gemini = gemini_provider or GeminiProvider()
        self.anthropic = anthropic_provider or AnthropicProvider()
        # Local providers detected lazily on first access (avoids startup probes)
        self._ollama: OpenAICompatibleProvider | None | bool = False
        self._lmstudio: OpenAICompatibleProvider | None | bool = False

    @property
    def ollama(self) -> OpenAICompatibleProvider | None:
        """Lazily detect Ollama on first access."""
        if self._ollama is False:
            self._ollama = _detect_ollama()
        return self._ollama  # type: ignore[return-value]

    @property
    def lmstudio(self) -> OpenAICompatibleProvider | None:
        """Lazily detect LM Studio on first access."""
        if self._lmstudio is False:
            self._lmstudio = _detect_lmstudio()
        return self._lmstudio  # type: ignore[return-value]

    def evaluate_task(
        self,
        task: TaskDefinition,
        *,
        guided: bool,
        provider_name: str = "openai",
        judge_provider: str | None = None,
    ) -> LiveRunResult:
        """Run a single task through a real SkillOrganism pipeline."""
        prompt = _TASK_PROMPTS.get(task.task_id)
        if not prompt:
            raise ValueError(f"No concrete prompt for task {task.task_id}")

        # Validate judge early — fail before expensive execution
        judge = self._pick_judge(provider_name, judge_provider=judge_provider)
        judge_name = judge_provider if judge_provider else _provider_name(judge)

        # CLI-based providers use cli_handler stages (claude, codex)
        if provider_name in ("claude", "codex"):
            return self._evaluate_cli(task, guided=guided, cli=provider_name,
                                      judge=judge, judge_name=judge_name,
                                      cross_judge=judge_provider is not None)

        # Select API providers.
        # Guided configs use distinct fast/deep models so the mode assignment
        # (fixed→fast, fuzzy→deep) produces a measurably different pipeline.
        # Unguided configs use the same model for both to isolate the effect.
        # NOTE: Cross-judged eval (Nemotron) showed guided ≤ unguided for all
        # provider orderings. Per Ao et al. 2603.26993, without new exogenous
        # signals each stage cannot outperform a single-model baseline.
        if provider_name == "openai":
            if guided:
                fast_provider = OpenAIProvider(model="gpt-5.4-mini")
                deep_provider = OpenAIProvider(model="gpt-5.4")
            else:
                fast_provider = OpenAIProvider(model="gpt-5.4-mini")
                deep_provider = OpenAIProvider(model="gpt-5.4-mini")
        elif provider_name == "gemini":
            if guided:
                fast_provider = GeminiProvider(model="gemini-2.5-flash")
                deep_provider = GeminiProvider(model="gemini-2.5-pro")
            else:
                fast_provider = GeminiProvider(model="gemini-2.5-flash")
                deep_provider = GeminiProvider(model="gemini-2.5-flash")
        elif provider_name == "anthropic":
            if guided:
                fast_provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                deep_provider = AnthropicProvider(model="claude-sonnet-4-6-20260301")
            else:
                fast_provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                deep_provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        elif provider_name == "ollama":
            if not self.ollama:
                raise ValueError("Ollama not available (localhost:11434 unreachable)")
            fast_provider = OpenAICompatibleProvider(
                api_key="not-needed", base_url="http://localhost:11434/v1",
                model=self.ollama.model,
            )
            deep_provider = fast_provider
        elif provider_name == "lmstudio":
            if not self.lmstudio:
                raise ValueError("LM Studio not available (localhost:1234 unreachable)")
            fast_provider = OpenAICompatibleProvider(
                api_key="not-needed", base_url="http://localhost:1234/v1",
                model=self.lmstudio.model,
            )
            deep_provider = fast_provider
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        fast_nucleus = Nucleus(provider=fast_provider)
        deep_nucleus = Nucleus(provider=deep_provider)

        # Build organism with or without guidance
        advice = None
        if guided:
            advice = advise_topology(
                task_shape=task.task_shape,
                tool_count=task.tool_count,
                subtask_count=task.subtask_count,
                error_tolerance=0.01,
            )

        stages = []
        n_roles = len(task.required_roles)
        for i, role in enumerate(task.required_roles):
            if guided and advice:
                rec = advice.recommended_pattern
                if "reviewer" in rec or "gate" in rec:
                    mode = "fixed" if i == n_roles - 1 else "fuzzy"
                else:
                    mode = "fuzzy"
            else:
                mode = "fuzzy"

            # Build stage with role-specific instructions (no handler = LLM call)
            instructions = (
                f"You are a {role}. {task.description}\n"
                f"Process the input and produce output relevant to your role."
            )
            stages.append(SkillStage(
                name=f"{role}_{i}",
                role=role,
                instructions=instructions,
                mode=mode,
                include_shared_state=True,
            ))

        organism = skill_organism(
            stages=stages,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
        )

        # Compute structural risk from the actual organism topology.
        # Edges reflect the real sequential pipeline that will execute.
        agent_specs = [{"name": s.name, "role": s.role, "mode": s.mode} for s in organism.stages]
        # The organism always executes stages in order (sequential pipeline).
        realized_edges = [
            (organism.stages[i].name, organism.stages[i + 1].name)
            for i in range(len(organism.stages) - 1)
        ]
        topology = ExternalTopology(
            source="operon",
            pattern_name=advice.recommended_pattern if advice else "generic_pipeline",
            agents=tuple(agent_specs),
            edges=tuple(realized_edges),
            metadata={"guided": guided, "task_shape": task.task_shape},
        )
        adapter_result = analyze_external_topology(topology)
        risk_score = adapter_result.risk_score

        # Run the organism with real LLM calls
        start = time.time()
        try:
            run_result = organism.run(prompt)
            elapsed_ms = (time.time() - start) * 1000

            # Collect stage-level metrics
            stage_details = []
            total_tokens = 0
            for sr in run_result.stage_results:
                total_tokens += sr.tokens_used
                stage_details.append({
                    "stage": sr.stage_name,
                    "role": sr.role,
                    "model": sr.model,
                    "tokens": sr.tokens_used,
                    "latency_ms": sr.latency_ms,
                    "action_type": sr.action_type,
                    "output_preview": str(sr.output)[:200] if sr.output else "",
                })

            # Judge quality — judge already validated before execution
            final_output = str(run_result.final_output or "")
            # No fallback when cross-judging (explicit provider must be authoritative)
            local_fb = None if judge_provider else (
                (lambda: self.ollama) if not isinstance(judge, OpenAICompatibleProvider) else None
            )
            quality_score, quality_reason, actual_judge = _judge_quality(
                prompt, final_output, judge, fallback_factory=local_fb,
            )
            # Derive judge name from actual provider (may differ after fallback)
            actual_judge_name = judge_provider if judge_provider else _provider_name(actual_judge)

            return LiveRunResult(
                task_id=task.task_id,
                config_name="guided" if guided else "unguided",
                provider_name=provider_name,
                success=quality_score >= 0.6,
                quality_score=quality_score,
                quality_reason=quality_reason,
                total_tokens=total_tokens,
                total_latency_ms=elapsed_ms,
                stage_count=len(stages),
                risk_score=risk_score,
                stage_details=tuple(stage_details),
                judge_provider=actual_judge_name,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return LiveRunResult(
                task_id=task.task_id,
                config_name="guided" if guided else "unguided",
                provider_name=provider_name,
                success=False,
                quality_score=0.0,
                quality_reason=f"Execution failed: {e}",
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                stage_count=0,
                risk_score=risk_score,
                stage_details=(),
                judge_provider=judge_name,
            )

    def _pick_judge(
        self,
        provider_name: str,
        judge_provider: str | None = None,
    ) -> Any:
        """Pick the best available judge provider.

        Parameters
        ----------
        provider_name:
            The execution provider name (used for self-judging fallback).
        judge_provider:
            If set, force-use this specific provider for judging
            (cross-judging). Raises ValueError if unavailable.
        """
        if judge_provider is not None:
            resolvers: dict[str, Any] = {
                "gemini": lambda: self.gemini if self.gemini.is_available() else None,
                "anthropic": lambda: self.anthropic if self.anthropic.is_available() else None,
                "openai": lambda: self.openai if self.openai.is_available() else None,
                "ollama": lambda: self.ollama,
                "lmstudio": lambda: self.lmstudio,
            }
            resolver = resolvers.get(judge_provider)
            if resolver is None:
                raise ValueError(f"Unknown judge provider: {judge_provider!r}")
            prov = resolver()
            if prov is None:
                raise ValueError(f"Judge provider {judge_provider!r} is not available")
            return prov

        # Default: self-judging with fallback
        candidates = []
        if provider_name == "gemini" and self.gemini.is_available():
            candidates.append(self.gemini)
        elif provider_name == "anthropic" and self.anthropic.is_available():
            candidates.append(self.anthropic)
        elif provider_name == "openai" and self.openai.is_available():
            candidates.append(self.openai)
        for prov in [self.gemini, self.anthropic, self.openai]:
            if prov.is_available() and prov not in candidates:
                candidates.append(prov)
        if candidates:
            return candidates[0]
        if self.ollama:
            return self.ollama
        if self.lmstudio:
            return self.lmstudio
        return self.gemini  # will fail, but gives a clear error

    def _evaluate_cli(
        self,
        task: TaskDefinition,
        *,
        guided: bool,
        cli: str,
        judge: Any = None,
        judge_name: str = "",
        cross_judge: bool = False,
    ) -> LiveRunResult:
        """Run task through Claude Code or Codex CLI as a single-stage organism.

        CLI agents are used as single-shot prompt executors:
        - claude -p --output-format text "prompt"
        - codex exec "prompt"
        """
        prompt = _TASK_PROMPTS.get(task.task_id)
        if not prompt:
            raise ValueError(f"No concrete prompt for task {task.task_id}")

        # Require a judge provider for quality scoring.
        # Skip probe only when cross-judging (judge explicitly validated).
        if not cross_judge:
            has_remote_judge = (
                self.gemini.is_available()
                or self.anthropic.is_available()
                or self.openai.is_available()
            )
            if has_remote_judge:
                has_judge = True
            else:
                has_judge = self.ollama is not None or self.lmstudio is not None
            if not has_judge:
                raise ValueError(
                    "CLI evaluation requires a judge provider. Set GEMINI_API_KEY, "
                    "ANTHROPIC_API_KEY, or OPENAI_API_KEY, or run Ollama locally."
                )

        if cli == "claude":
            cmd = ["claude", "-p", "--output-format", "text"]
        elif cli == "codex":
            cmd = ["codex", "exec"]
        else:
            raise ValueError(f"Unknown CLI: {cli}")

        # Compute structural risk (single agent = minimal topology)
        topology = ExternalTopology(
            source="cli",
            pattern_name=f"{cli}_single",
            agents=({"name": cli, "role": "generalist"},),
            edges=(),
            metadata={},
        )
        adapter_result = analyze_external_topology(topology)
        risk_score = adapter_result.risk_score

        # Build stage with CLI handler
        handler = cli_handler(
            cmd,
            timeout=120.0,
            input_mode="arg",
            sanitize_task=False,
        )

        start = time.time()
        try:
            result = handler(prompt)
            elapsed_ms = (time.time() - start) * 1000

            output = result.get("output", "")
            cli_result = result.get("cli_result")
            action_type = result.get("_action_type", "EXECUTE")

            # Estimate tokens from output length
            est_tokens = len(output.split()) + len(prompt.split())

            stage_details = ({
                "stage": f"{cli}_0",
                "role": "generalist",
                "model": cli,
                "tokens": est_tokens,
                "latency_ms": cli_result.latency_ms if cli_result else elapsed_ms,
                "action_type": action_type,
                "output_preview": str(output)[:200],
            },)

            # Judge quality — judge pre-validated by evaluate_task caller
            if judge is None:
                judge = self._pick_judge(cli)
                judge_name = _provider_name(judge)
            # No fallback when cross-judging (explicit provider must be authoritative)
            local_fb = None if cross_judge else (
                (lambda: self.ollama) if not isinstance(judge, OpenAICompatibleProvider) else None
            )
            quality_score, quality_reason, actual_judge = _judge_quality(
                prompt, str(output), judge, fallback_factory=local_fb,
            )
            actual_judge_name = judge_name if cross_judge else _provider_name(actual_judge)

            return LiveRunResult(
                task_id=task.task_id,
                config_name=f"{cli}_{'guided' if guided else 'unguided'}",
                provider_name=cli,
                success=quality_score >= 0.6,
                quality_score=quality_score,
                quality_reason=quality_reason,
                total_tokens=est_tokens,
                total_latency_ms=elapsed_ms,
                stage_count=1,
                risk_score=risk_score,
                stage_details=stage_details,
                judge_provider=actual_judge_name,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return LiveRunResult(
                task_id=task.task_id,
                config_name=f"{cli}_{'guided' if guided else 'unguided'}",
                provider_name=cli,
                success=False,
                quality_score=0.0,
                quality_reason=f"CLI failed: {e}",
                total_tokens=0,
                total_latency_ms=elapsed_ms,
                stage_count=1,
                risk_score=risk_score,
                stage_details=(),
                judge_provider=judge_name,
            )
