"""C8 Meta-Harness proposers: tournament mutation + LLM exploration.

Hybrid proposer strategy (C): programmatic mutations for exploitation,
LLM proposer for exploration when progress stalls.

Scientific hypothesis: exogenous signals (LLM reading filesystem history)
break the dominance of programmatic search (Ao et al. 2603.26993).
"""

from __future__ import annotations

import json
import random as _random
from typing import Any, Protocol, runtime_checkable

from ..providers.base import ProviderConfig
from ..providers.openai_compatible_provider import OpenAICompatibleProvider
from .meta_store import EvolutionStore
from .meta_types import (
    CandidateConfig,
    StageConfig,
    candidate_to_genome,
    genome_to_candidate,
)


@runtime_checkable
class Proposer(Protocol):
    """Protocol for candidate proposal strategies."""

    def propose(
        self,
        population: list[CandidateConfig],
        scores: dict[str, list[float]],
        iteration: int,
    ) -> CandidateConfig: ...


# ---------------------------------------------------------------------------
# Mutation vocabulary for discrete StageConfig fields
# ---------------------------------------------------------------------------

_COGNITIVE_OPTIONS = (None, "observational", "action_oriented")
_BOOL_FIELDS = ("include_stage_outputs", "include_shared_state")


def _is_localhost(url: str) -> bool:
    """Check if a URL points to a local server."""
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname or ""
    return hostname in ("localhost", "127.0.0.1")


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output.

    Handles: raw JSON, markdown fenced blocks (```json ... ```),
    and prose-wrapped JSON ("Here is the JSON: {...}").
    """
    import re
    text = text.strip()
    # 1. Try raw JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Try fenced code block
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # 3. Extract first JSON object via raw_decode
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in LLM response: {text[:200]}")
_DISCRETE_FIELDS = ("mode", "model", "include_stage_outputs",
                     "include_shared_state", "cognitive_mode")

# Default model options (can be overridden via constructor)
_DEFAULT_MODELS: tuple[str | None, ...] = (
    None, "gemini-2.5-flash", "gemini-2.5-pro",
    "gpt-5.4-mini", "gpt-5.4",
    "claude-haiku-4-5-20251001", "claude-sonnet-4-6-20260301",
)


# ---------------------------------------------------------------------------
# Programmatic proposer
# ---------------------------------------------------------------------------


class TournamentMutator:
    """Tournament-select top-K, mutate config or topology.

    Phase A: config mutations only (mode, model, booleans).
    Phase B: 70% config mutations, 30% topology mutations
    (add_stage, remove_stage, rewire).
    """

    def __init__(
        self,
        k: int = 3,
        seed: int | None = None,
        model_options: tuple[str | None, ...] = _DEFAULT_MODELS,
        topology_mutation_rate: float = 0.3,
    ) -> None:
        self._k = k
        self._rng = _random.Random(seed)
        self._model_options = model_options
        self._topology_rate = topology_mutation_rate

    def propose(
        self,
        population: list[CandidateConfig],
        scores: dict[str, list[float]],
        iteration: int,
    ) -> CandidateConfig:
        # Tournament selection
        k = min(self._k, len(population))
        contenders = self._rng.sample(population, k)
        winner = max(
            contenders,
            key=lambda c: (
                sum(scores.get(c.candidate_id, [0.0]))
                / max(len(scores.get(c.candidate_id, [0.0])), 1)
            ),
        )

        # Choose mutation type
        if self._rng.random() < self._topology_rate and len(winner.stage_configs) > 0:
            return self._topology_mutate(winner, iteration)
        else:
            return self._config_mutate(winner, iteration)

    def _config_mutate(self, winner: CandidateConfig, iteration: int) -> CandidateConfig:
        """Mutate one config field (Phase A behavior)."""
        genome = candidate_to_genome(winner)
        n_stages = len(winner.stage_configs)
        stage_idx = self._rng.randrange(n_stages)
        field_name = self._rng.choice(list(_DISCRETE_FIELDS))
        gene_name = f"stage_{stage_idx}_{field_name}"

        old_value = genome.get_value(gene_name)
        new_value = self._mutate_value(field_name, old_value)
        genome.mutate(gene_name, new_value, f"config_mutate_iter_{iteration}")

        candidate_id = f"c_{iteration:03d}_{self._rng.randint(0, 999):03d}"
        return genome_to_candidate(
            genome,
            candidate_id=candidate_id,
            parent_id=winner.candidate_id,
            iteration=iteration,
            proposer="tournament_mutate",
            reason=f"mutated {gene_name}: {old_value!r} -> {new_value!r}",
        )

    def _topology_mutate(self, winner: CandidateConfig, iteration: int) -> CandidateConfig:
        """Mutate topology: add stage, remove stage, or rewire.

        All edge names are recomputed from final stage list at the end
        to prevent stale references from add/remove index shifts.
        """
        op = self._rng.choice(["add_stage", "remove_stage", "rewire"])
        stages = list(winner.stage_configs)
        edges = list(winner.edges)
        reason = ""
        max_stages = 8

        if op == "add_stage" and len(stages) >= max_stages:
            op = "rewire"

        if op == "add_stage" or (op == "remove_stage" and len(stages) <= 1):
            roles = [sc.role for sc in stages]
            new_role = self._rng.choice(roles)
            # Wire from current last stage to new stage (names computed after append)
            prev_last_idx = len(stages) - 1
            stages.append(StageConfig(role=new_role, mode="fuzzy"))
            new_idx = len(stages) - 1
            prev_name = f"{stages[prev_last_idx].role}_{prev_last_idx}"
            new_name = f"{new_role}_{new_idx}"
            edges.append((prev_name, new_name))
            reason = f"added stage {new_name} (role={new_role})"

        elif op == "remove_stage" and len(stages) > 1:
            if len(stages) > 2:
                idx = self._rng.randint(1, len(stages) - 2)
            else:
                idx = self._rng.randint(0, len(stages) - 1)
            # Build old→new name mapping before removing
            old_names = [f"{sc.role}_{i}" for i, sc in enumerate(stages)]
            removed_name = old_names[idx]
            stages.pop(idx)
            new_names = [f"{sc.role}_{i}" for i, sc in enumerate(stages)]
            rename = {}
            j = 0
            for k, old in enumerate(old_names):
                if k == idx:
                    continue
                rename[old] = new_names[j]
                j += 1
            # Rewrite edges through mapping, prune removed stage
            edges = [
                (rename[s], rename[d])
                for s, d in edges
                if s in rename and d in rename and s != removed_name and d != removed_name
            ]
            reason = f"removed stage {removed_name}"

        elif op == "rewire" and len(stages) >= 2:
            # Add or remove a random edge
            if edges and self._rng.random() < 0.5:
                # Remove random edge
                edge_idx = self._rng.randrange(len(edges))
                removed = edges.pop(edge_idx)
                reason = f"removed edge {removed[0]}->{removed[1]}"
            else:
                # Add random edge (no self-loops)
                new_names = [f"{sc.role}_{i}" for i, sc in enumerate(stages)]
                src = self._rng.choice(new_names)
                dst = self._rng.choice([n for n in new_names if n != src])
                edges.append((src, dst))
                reason = f"added edge {src}->{dst}"
        else:
            reason = "no-op topology mutation"

        # Recompute ALL edge names from final stage list — prevents stale refs
        final_names = {f"{sc.role}_{i}" for i, sc in enumerate(stages)}
        edges = [
            (s, d) for s, d in edges
            if s in final_names and d in final_names and s != d
        ]

        candidate_id = f"c_{iteration:03d}_{self._rng.randint(0, 999):03d}"
        return CandidateConfig(
            candidate_id=candidate_id,
            parent_id=winner.candidate_id,
            iteration=iteration,
            stage_configs=tuple(stages),
            intervention_policy=winner.intervention_policy,
            topology=winner.topology,
            edges=tuple(edges),
            proposer="tournament_mutate",
            reason=reason,
        )

    def _mutate_value(self, field_name: str, old_value: Any) -> Any:
        if field_name == "mode":
            return "fixed" if old_value == "fuzzy" else "fuzzy"
        if field_name == "model":
            options = [m for m in self._model_options if m != old_value]
            return self._rng.choice(options) if options else old_value
        if field_name in _BOOL_FIELDS:
            return not old_value
        if field_name == "cognitive_mode":
            options = [c for c in _COGNITIVE_OPTIONS if c != old_value]
            return self._rng.choice(options) if options else old_value
        return old_value


# ---------------------------------------------------------------------------
# LLM proposer
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = """\
You are an evolutionary optimizer for AI multi-agent systems.

Below are the TOP PERFORMING and WORST PERFORMING organism configurations
from the evolution history, with execution metadata (token counts and
whether each stage produced output). Use the patterns to understand
which configs work.

{candidate_details}

SCORE SUMMARY (most recent last):
{score_summary}

Based on the evidence above, propose ONE new configuration that you
believe will score higher. Consider:
- Configs with mode="fixed" on non-matching roles often produce gate behavior — usually bad
- Configs with mode="fuzzy" produce natural language output — usually good
- Stages with high token counts and produced_output=True are working well
- Stages with low tokens or produced_output=False indicate problems

Return ONLY a JSON object with this exact schema:
{{
  "stage_configs": [
    {{"role": "<str>", "mode": "<fixed|fuzzy>", "model": "<str|null>",
      "include_stage_outputs": <bool>, "include_shared_state": <bool>,
      "cognitive_mode": "<str|null>"}}
  ],
  "edges": [["<src_stage_name>", "<dst_stage_name>"]],
  "intervention_policy": {{}},
  "reason": "<why you chose this config, citing trace evidence>"
}}
"""


class LLMProposer:
    """LLM-backed proposer: reads evolution history, proposes a new config.

    This is the "exogenous signal" — the LLM sees patterns in the full
    evolution trajectory that programmatic mutation cannot discover.
    Falls back to TournamentMutator on parse failure.
    """

    def __init__(
        self,
        provider: Any,
        store: EvolutionStore,
        fallback_seed: int | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._fallback = TournamentMutator(seed=fallback_seed)

    def propose(
        self,
        population: list[CandidateConfig],
        scores: dict[str, list[float]],
        iteration: int,
    ) -> CandidateConfig:
        # Build rich context from filesystem history
        index = self._store.load_index()

        # Score summary (compressed, for trajectory overview)
        summary_lines = []
        for r in index[-20:]:
            summary_lines.append(
                f"  {r.candidate_id} task={r.task_id} "
                f"score={r.score:.2f} proposer={r.proposer}"
            )
        score_summary = "\n".join(summary_lines) if summary_lines else "(no history)"

        # Rich candidate details: top 5 + bottom 3 by mean score
        ranked = sorted(
            population,
            key=lambda c: (
                sum(scores.get(c.candidate_id, [0.0]))
                / max(len(scores.get(c.candidate_id, [0.0])), 1)
            ),
            reverse=True,
        )
        showcase = ranked[:5] + ranked[-3:] if len(ranked) > 5 else ranked
        seen = set()
        candidate_blocks = []
        for cc in showcase:
            if cc.candidate_id in seen:
                continue
            seen.add(cc.candidate_id)
            mean_score = (
                sum(scores.get(cc.candidate_id, [0.0]))
                / max(len(scores.get(cc.candidate_id, [0.0])), 1)
            )
            # Config
            config_str = json.dumps({
                "stage_configs": [
                    {"role": s.role, "mode": s.mode, "model": s.model,
                     "include_stage_outputs": s.include_stage_outputs,
                     "include_shared_state": s.include_shared_state,
                     "cognitive_mode": s.cognitive_mode}
                    for s in cc.stage_configs
                ],
                "edges": [list(e) for e in cc.edges] if cc.edges else "sequential",
            }, indent=2)
            # Trace: isolate the latest run by run_step marker
            all_trace = self._store.load_trace(cc.candidate_id)
            if all_trace:
                latest_step = max(t.get("run_step", 0) for t in all_trace)
                trace = [t for t in all_trace if t.get("run_step") == latest_step]
                if not trace:
                    trace = all_trace[-6:]
            else:
                trace = []
            trace_lines = []
            for t in trace:
                if t.get("stage") == "_error":
                    trace_lines.append(f"  ERROR: {t.get('error', '?')}")
                else:
                    produced = t.get("produced_output", bool(t.get("output_preview", "").strip()))
                    trace_lines.append(
                        f"  {t.get('stage','?')}: tokens={t.get('tokens',0)}"
                        f" produced_output={produced}"
                    )
            trace_str = "\n".join(trace_lines) if trace_lines else "  (no trace)"

            candidate_blocks.append(
                f"=== {cc.candidate_id} (mean_score={mean_score:.2f}, "
                f"proposer={cc.proposer}) ===\n"
                f"Config:\n{config_str}\n"
                f"Trace:\n{trace_str}"
            )

        candidate_details = "\n\n".join(candidate_blocks)

        prompt = _LLM_PROMPT_TEMPLATE.format(
            candidate_details=candidate_details,
            score_summary=score_summary,
        )

        # Find best for parent lineage
        best = ranked[0] if ranked else population[0]

        try:
            # Local reasoning models (deepseek-r1, qwen) need /no_think
            is_local = isinstance(self._provider, OpenAICompatibleProvider) and _is_localhost(
                getattr(self._provider, "base_url", ""),
            )
            final_prompt = prompt + "\n/no_think" if is_local else prompt

            # Gemini needs more tokens (internal reasoning consumes budget)
            from ..providers.gemini_provider import GeminiProvider
            model_name = getattr(self._provider, "model", "") or ""
            # Match gemini-*, google/gemini-*, models/gemini-*
            model_tail = model_name.rsplit("/", 1)[-1].lower() if isinstance(model_name, str) else ""
            is_gemini = isinstance(self._provider, GeminiProvider) or (
                model_tail.startswith("gemini-")
            )
            max_tok = 4096 if is_gemini else 2000

            resp = self._provider.complete(
                final_prompt,
                config=ProviderConfig(temperature=0.7, max_tokens=max_tok),
            )
            data = _extract_json(resp.content)
            stage_configs = tuple(
                StageConfig(**sc) for sc in data["stage_configs"]
            )
            # Parse edges if present
            raw_edges = data.get("edges", [])
            if isinstance(raw_edges, list):
                edges = tuple(tuple(e) for e in raw_edges if isinstance(e, (list, tuple)) and len(e) == 2)
            else:
                edges = ()

            candidate_id = f"c_{iteration:03d}_llm"
            return CandidateConfig(
                candidate_id=candidate_id,
                parent_id=best.candidate_id,
                iteration=iteration,
                stage_configs=stage_configs,
                intervention_policy=data.get("intervention_policy", {}),
                edges=edges,
                proposer="llm_explore",
                reason=data.get("reason", "LLM-proposed"),
            )
        except Exception as exc:
            # Fallback to programmatic mutation — mark as llm_fallback
            # so we can distinguish "LLM didn't trigger" from "LLM failed"
            result = self._fallback.propose(population, scores, iteration)
            return CandidateConfig(
                candidate_id=result.candidate_id,
                parent_id=result.parent_id,
                iteration=result.iteration,
                stage_configs=result.stage_configs,
                intervention_policy=result.intervention_policy,
                topology=result.topology,
                proposer="llm_fallback",
                reason=f"{type(exc).__name__}: {exc}; fell back to: {result.reason}",
            )
