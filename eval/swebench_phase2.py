"""SWE-bench-lite Phase 2: Docker-based pass/fail evaluation.

Replaces Phase 1's LLM judge with the ground truth: apply the patch
in a Docker container, run FAIL_TO_PASS + PASS_TO_PASS tests, report
pass/fail.

Pipeline:
  1. Load SWE-bench-lite tasks (N instances)
  2. For each condition (baseline, organism, langgraph):
       a. Run the model, capture output
       b. Extract unified diff
       c. Write to <condition>_predictions.jsonl
  3. Invoke swebench.harness.run_evaluation for each condition
  4. Parse per-instance reports, aggregate resolved/total
  5. Write eval/results/swebench_phase2.json

Usage:
  pip install swebench datasets
  docker info  # confirm daemon is running
  python eval/swebench_phase2.py [--model gemma4:latest] [--n 10]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import run_organism_langgraph
from operon_ai.providers.base import ProviderConfig
from operon_ai.providers.openai_compatible_provider import OpenAICompatibleProvider

from eval._patch_extraction import extract_patch
from eval._patch_sanitizer import sanitize as sanitize_patch
from eval._repo_cache import RepoCacheError, ensure_repo_at
from eval._repo_grounding import (
    extract_hints,
    format_context_block,
    rank_candidate_files,
    walk_tree_paths,
)


# ---------------------------------------------------------------------------
# Provider + organism (reuse Phase 1 shape)
# ---------------------------------------------------------------------------

def _make_provider(model: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="not-needed",
        base_url="http://localhost:11434/v1",
        model=model,
    )


class ModelIdentityError(RuntimeError):
    """Raised when an immutable identity cannot be resolved for a tag."""


# Required keys for an immutable identity. The schema is locked so the
# committed artifact always matches a fresh rerun's output.
_IDENTITY_REQUIRED = ("tag", "digest", "blob_sha256", "architecture",
                      "parameters", "quantization", "source")


def _parse_ollama_show(stdout: str) -> dict[str, str]:
    """Extract architecture / parameters / quantization from ``ollama show``.

    The non-modelfile form prints indented section headers (``Model``,
    ``Capabilities``, ``Parameters``) at 2 spaces of indent with their
    row values at 4+ spaces. We parse only the keys we publish so a
    future ollama version adding rows does not change our schema.
    """
    wanted = {"architecture", "parameters", "quantization"}
    found: dict[str, str] = {}
    in_model_section = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent <= 2:
            in_model_section = stripped == "Model"
            continue
        if not in_model_section:
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2 and parts[0] in wanted:
            found[parts[0]] = parts[1].strip()
    return found


def _resolve_model_identity(model_tag: str) -> dict[str, str]:
    """Return an immutable identifier for *model_tag* from Ollama.

    The human-readable tag (e.g. ``gemma4:latest``) is a floating pointer;
    the digest and blob sha256 pin a specific set of weights so the result
    can be correlated with a concrete model version later.

    Raises :class:`ModelIdentityError` if any required field cannot be
    resolved. Callers must surface this to the user rather than silently
    proceeding — the published artifact and the paper both cite these
    fields as reproducibility anchors.
    """
    identity: dict[str, str] = {
        "tag": model_tag,
        "source": "ollama list / ollama show / ollama show --modelfile",
    }
    try:
        listed = subprocess.run(
            ["ollama", "list"], check=True, capture_output=True, text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        raise ModelIdentityError(f"`ollama list` failed: {e}") from e
    for line in listed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == model_tag:
            identity["digest"] = parts[1]
            break

    try:
        modelfile = subprocess.run(
            ["ollama", "show", "--modelfile", model_tag],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        raise ModelIdentityError(
            f"`ollama show --modelfile {model_tag}` failed: {e}"
        ) from e
    for line in modelfile.stdout.splitlines():
        if line.startswith("FROM /") and "sha256-" in line:
            identity["blob_sha256"] = line.split("sha256-", 1)[1].strip()
            break

    try:
        shown = subprocess.run(
            ["ollama", "show", model_tag],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        raise ModelIdentityError(
            f"`ollama show {model_tag}` failed: {e}"
        ) from e
    identity.update(_parse_ollama_show(shown.stdout))

    missing = [k for k in _IDENTITY_REQUIRED if k not in identity]
    if missing:
        raise ModelIdentityError(
            f"could not resolve required identity fields {missing} for "
            f"tag {model_tag!r}; refusing to publish an incomplete "
            "reproducibility record"
        )
    return identity


def _format_task(instance: dict, grounding_block: str = "") -> str:
    repo = instance["repo"]
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "") or ""
    prompt = f"Repository: {repo}\n\nIssue:\n{problem}\n"
    if hints.strip():
        prompt += f"\nHints:\n{hints[:500]}\n"
    if grounding_block:
        prompt += f"\nRepository context:\n{grounding_block}\n"
    return prompt


@dataclass
class Grounding:
    """Per-instance grounding bundle built in main() when --grounding is on."""
    repo_path: "Path | None"  # forward ref; Path imported at module top
    tree_paths: "frozenset[str] | None"
    context_block: str

    @classmethod
    def none(cls) -> "Grounding":
        return cls(repo_path=None, tree_paths=None, context_block="")


def _build_grounding(instance: dict, cache_dir: Path) -> "Grounding":
    """Clone the instance's repo at its base_commit and produce context.

    Returns a ``Grounding.none()`` if anything fails — the caller falls
    back to the Phase-A behavior for that instance with a diagnostic
    printed.
    """
    repo_slug = instance["repo"]
    base_commit = instance.get("base_commit")
    if not base_commit:
        print(f"  grounding skipped: no base_commit on {instance['instance_id']}")
        return Grounding.none()
    try:
        repo_path = ensure_repo_at(repo_slug, base_commit, cache_dir)
    except RepoCacheError as e:
        print(f"  grounding failed for {repo_slug}@{base_commit[:12]}: {e}")
        return Grounding.none()
    tree_paths = walk_tree_paths(repo_path)
    hints = extract_hints(
        instance["problem_statement"] + "\n" + (instance.get("hints_text") or "")
    )
    candidates = rank_candidate_files(repo_path, hints, k=5)
    block = format_context_block(repo_path, candidates)
    return Grounding(repo_path=repo_path, tree_paths=tree_paths, context_block=block)


#: Per-call timeout (seconds) for benchmark LLM invocations — both
#: direct (baseline via :func:`_llm_call`) and via the three-stage
#: organism path (each :class:`SkillStage` ships with a matching
#: ``provider_config``). The provider base class defaults to 120s,
#: which is a safe choice for small interactive callers but not for
#: SWE-bench prompts: the grounded v0.34.5 run had two baseline
#: instances (astropy-12907, astropy-14995) time out at the 120s
#: ceiling, and reasoning models like deepseek-r1 routinely produce
#: 5-10 minute responses (long ``<think>`` blocks before the final
#: answer). 900s is a safety net, not a tight fit — it accommodates
#: current gemma4 runs (mean ~131s, some instances higher), grounded
#: prompts with large context blocks, and reasoning-model reruns,
#: while still catching silent Ollama hangs on a ~15 min budget.
_LLM_TIMEOUT_SECONDS: float = 900.0

#: Per-call timeout (seconds) for the startup reachability probe.
#: Deliberately short: a "Say ok." round-trip shouldn't take more
#: than a cold-load + one-token generation, so anything past ~60s is
#: a configuration problem we want to surface immediately rather
#: than hide behind the 15-minute benchmark budget. Review #753.
_LLM_PROBE_TIMEOUT_SECONDS: float = 60.0


def _llm_call(
    provider,
    prompt: str,
    timeout_seconds: float | None = None,
) -> str:
    """Call the provider with the SWE-bench runner's config.

    ``timeout_seconds`` overrides the module default for cases where
    a shorter budget is appropriate (e.g. the startup reachability
    probe — see :func:`_probe_model`). Passing ``None`` keeps the
    benchmark default (:data:`_LLM_TIMEOUT_SECONDS`).
    """
    effective_timeout = (
        _LLM_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    config = ProviderConfig(
        max_tokens=4096, timeout_seconds=effective_timeout,
    )
    return provider.complete(prompt, config).content


def _probe_model(provider) -> str:
    """Minimal reachability check for the configured model.

    Uses :data:`_LLM_PROBE_TIMEOUT_SECONDS` (short) rather than the
    benchmark timeout so a misconfigured model / unreachable Ollama
    surfaces in ~1 minute, not ~15. Returns the probe response so
    callers can print it for visual confirmation.
    """
    return _llm_call(
        provider, "Say ok.",
        timeout_seconds=_LLM_PROBE_TIMEOUT_SECONDS,
    )


def _build_organism(provider):
    nucleus = Nucleus(provider=provider)
    # Every stage ships with the same per-call timeout as the baseline
    # path. Without this, BioAgent -> Nucleus.transcribe falls back to
    # ProviderConfig()'s 120s default and organism/langgraph runs time
    # out long before the model has a chance to produce a full response.
    # ProviderConfig is constructed per-stage (rather than shared) so a
    # stage that later needs different max_tokens / temperature can
    # override without disturbing the others.
    stage_config = ProviderConfig(timeout_seconds=_LLM_TIMEOUT_SECONDS)
    return skill_organism(
        stages=[
            SkillStage(
                name="localize",
                role="Bug Locator",
                instructions=(
                    "You are analyzing a bug report for a Python project. "
                    "Identify the exact file(s) and function(s) where the bug "
                    "occurs. Explain the root cause concisely."
                ),
                mode="fixed",
                provider_config=stage_config,
            ),
            SkillStage(
                name="edit",
                role="Patch Author",
                instructions=(
                    "Based on the bug localization, write a minimal fix "
                    "as a unified diff.\n\n"
                    "STRICT FORMAT — your ENTIRE response for this stage "
                    "must be a single fenced diff block and nothing "
                    "else:\n\n"
                    "```diff\n"
                    "--- a/<path-relative-to-repo-root>\n"
                    "+++ b/<path-relative-to-repo-root>\n"
                    "@@ -<real_old_start>,<real_old_count> "
                    "+<real_new_start>,<real_new_count> @@\n"
                    " <context line>\n"
                    "-<removed line>\n"
                    "+<added line>\n"
                    "```\n\n"
                    "Rules:\n"
                    "- Do not prefix the path with the repo name.\n"
                    "- Use REAL integer line numbers. Never use `XXX`, "
                    "`N`, `?`, or single letters.\n"
                    "- The body of each hunk must be complete: "
                    "additions + context must equal the new_count; "
                    "deletions + context must equal the old_count.\n"
                    "- No prose, no explanation, no 'here is the fix' "
                    "text — the fenced diff only."
                ),
                mode="fixed",
                provider_config=stage_config,
            ),
            SkillStage(
                name="verify",
                role="Patch Reviewer",
                instructions=(
                    "Review the proposed patch from the edit stage. "
                    "Do NOT re-emit the diff. Output at most three "
                    "short bullet points: (1) does it fix the issue; "
                    "(2) any regressions you see; (3) is the diff "
                    "format valid. Never include a fenced diff block "
                    "in this stage's output."
                ),
                mode="fixed",
                provider_config=stage_config,
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
        budget=ATP_Store(budget=2000, silent=True),
    )


# ---------------------------------------------------------------------------
# Run conditions
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    instance_id: str
    repo: str
    condition: str
    raw_output: str
    model_patch: str
    latency_ms: float
    extract_ok: bool
    # Set to a short string (e.g. "API timeout", "OOM") when the model
    # call raised an exception and no real model output was produced.
    # ``None`` means the model returned text (whether or not the
    # sanitizer accepted it). Used downstream to distinguish
    # ``empty_patch`` (sanitizer-rejected legitimate model output)
    # from ``runtime_error`` (model call never completed).
    error_reason: "str | None" = None


_BASELINE_RULES = (
    "\nRules for the diff:\n"
    "- Use the path exactly as it appears in the repository, relative\n"
    "  to the repo root. Never prefix the path with the repo name. For\n"
    "  repository `{repo}`, the correct form is `a/<path>`, not\n"
    "  `a/{repo}/<path>`.\n"
    "- Every hunk header must contain real line numbers,\n"
    "  e.g. `@@ -42,7 +42,9 @@`. Never use placeholders like `XXX`, `N`,\n"
    "  `?`, or single letters.\n"
    "- Every hunk body must be complete — context, additions, and\n"
    "  deletions must match the counts declared in the header. Do not\n"
    "  truncate mid-hunk.\n"
)


def _sanitize_for_submission(
    raw: str, repo_slug: str,
    tree_paths: "frozenset[str] | None" = None,
) -> str:
    """Extract and sanitize a patch from raw model output.

    When ``tree_paths`` is provided (``--grounding`` mode), the
    sanitizer also fuzzy-corrects near-miss file paths against the
    cloned repo. A rejected submission is returned as ``""`` so the
    harness counts it as ``empty_patch`` rather than a format-error
    ``error``.
    """
    extracted = extract_patch(raw)
    if not extracted:
        return ""
    cleaned = sanitize_patch(extracted, repo_slug, tree_paths=tree_paths)
    if not cleaned:
        print(f"  sanitizer: dropped patch for {repo_slug} "
              "(placeholder hunk / malformed counts / bad paths)")
    return cleaned


def run_baseline(
    provider, instance: dict, grounding: "Grounding | None" = None,
) -> Prediction:
    g = grounding or Grounding.none()
    task = _format_task(instance, grounding_block=g.context_block)
    prompt = (
        f"{task}\nFix this bug. Output a unified diff "
        "(--- a/file, +++ b/file) with your fix. Be minimal."
        + _BASELINE_RULES.format(repo=instance["repo"])
    )
    t0 = time.monotonic()
    raw = _llm_call(provider, prompt)
    elapsed = (time.monotonic() - t0) * 1000
    patch = _sanitize_for_submission(raw, instance["repo"], g.tree_paths)
    return Prediction(
        instance["instance_id"], instance["repo"], "baseline",
        raw, patch, elapsed, bool(patch),
    )


def run_organism(
    provider, instance: dict, grounding: "Grounding | None" = None,
) -> Prediction:
    g = grounding or Grounding.none()
    task = _format_task(instance, grounding_block=g.context_block)
    org = _build_organism(provider)
    t0 = time.monotonic()
    result = org.run(task)
    elapsed = (time.monotonic() - t0) * 1000
    full = "\n\n".join(
        f"[{sr.stage_name}]\n{sr.output}" for sr in result.stage_results
    )
    patch = _sanitize_for_submission(full, instance["repo"], g.tree_paths)
    return Prediction(
        instance["instance_id"], instance["repo"], "organism",
        full, patch, elapsed, bool(patch),
    )


def run_langgraph(
    provider, instance: dict, grounding: "Grounding | None" = None,
) -> Prediction:
    g = grounding or Grounding.none()
    task = _format_task(instance, grounding_block=g.context_block)
    org = _build_organism(provider)
    t0 = time.monotonic()
    result = run_organism_langgraph(org, task=task, verify_certificates=True)
    elapsed = (time.monotonic() - t0) * 1000
    parts = [f"[{name}]\n{out}" for name, out in result.stage_outputs.items()]
    full = "\n\n".join(parts) if parts else result.output
    patch = _sanitize_for_submission(full, instance["repo"], g.tree_paths)
    return Prediction(
        instance["instance_id"], instance["repo"], "langgraph",
        full, patch, elapsed, bool(patch),
    )


# ---------------------------------------------------------------------------
# Docker evaluation
# ---------------------------------------------------------------------------

def _write_predictions(preds: list[Prediction], path: Path, model_name: str) -> None:
    """Write predictions in swebench harness format (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in preds:
            f.write(json.dumps({
                "instance_id": p.instance_id,
                "model_patch": p.model_patch,
                "model_name_or_path": model_name,
            }) + "\n")


class HarnessFailed(Exception):
    """Raised when the swebench harness itself failed to run."""


def _run_harness(
    predictions_path: Path, run_id: str, report_dir: Path, timeout: int = 600
) -> None:
    """Invoke swebench.harness.run_evaluation. Report lands in *report_dir*.

    Raises :class:`HarnessFailed` if the harness exits non-zero.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "SWE-bench/SWE-bench_Lite",
        "--predictions_path", str(predictions_path.resolve()),
        "--run_id", run_id,
        "--max_workers", "2",
        "--timeout", str(timeout),
        "--cache_level", "env",
    ]
    print(f"  Harness: python -m swebench.harness.run_evaluation "
          f"--run_id {run_id} ...")
    try:
        # Harness writes {model_name}.{run_id}.json to CWD — run in report_dir
        subprocess.run(cmd, check=True, cwd=str(report_dir))
    except subprocess.CalledProcessError as e:
        raise HarnessFailed(
            f"swebench.harness.run_evaluation exited {e.returncode}"
        ) from e


# Per-instance evaluation state
EVAL_RESOLVED = "resolved"          # patch applied, tests passed
EVAL_UNRESOLVED = "unresolved"      # patch applied but tests failed
EVAL_ERROR = "error"                # patch failed to apply, timeout, etc.
EVAL_EMPTY = "empty_patch"          # model returned but no usable diff (incl. sanitizer-rejected)
EVAL_RUNTIME_ERROR = "runtime_error"  # model call itself raised (timeout, API failure, OOM)
EVAL_NOT_EVALUATED = "not_evaluated"  # harness did not report on this instance


def classify_prediction(
    error_reason: "str | None", harness_status: str,
) -> str:
    """Return the canonical eval_status for a prediction.

    ``error_reason`` (set in main() when the model call raised an
    exception) is the authoritative signal for "model never returned".
    It overrides any ``harness_status`` — including ``empty_patch``
    (the harness can't tell the difference from sanitizer-rejection),
    ``not_evaluated`` (when --skip-harness is used or the harness
    report is missing), and even ``error`` (a harness-reported
    git-apply failure on a patch that wouldn't exist if the model
    call had raised). This guarantees that runtime failures stay
    visible regardless of the harness's reporting state. Review #748.
    """
    if error_reason is not None:
        return EVAL_RUNTIME_ERROR
    return harness_status or EVAL_NOT_EVALUATED

# Harness-run states (tri-state: not a bool, because "skipped" ≠ "failed")
HARNESS_OK = "ok"
HARNESS_FAILED = "failed"
HARNESS_SKIPPED = "skipped"

# Valid values for model_identity_post_run_check.status. These are the only
# values the live writer emits. The schema test imports this set so the
# test can never drift to accept a value the writer cannot produce.
POST_RUN_CHECK_STATUSES = frozenset({"match", "mismatch", "error"})


def build_artifact(
    *,
    model: str,
    model_identity: dict,
    post_run_check: dict,
    run_id: str,
    n_instances: int,
    offset: int,
    conditions: list,
    timestamp: str,
    skip_harness: bool,
    results: list,
    summary: dict,
    dataset: str = "SWE-bench/SWE-bench_Lite",
) -> dict:
    """Assemble the SWE-bench Phase 2 artifact envelope.

    This is the SINGLE source of truth for the artifact's top-level
    shape. Both the live writer and ``--rewrite-envelope`` call it, and
    the schema test derives the expected key set from a dummy invocation
    of this function (never from a hand-maintained constant), so writer
    and test cannot drift independently of each other.
    """
    return {
        "model": model,
        "model_identity": model_identity,
        "model_identity_post_run_check": post_run_check,
        "dataset": dataset,
        "run_id": run_id,
        "n_instances": n_instances,
        "offset": offset,
        "conditions": conditions,
        "timestamp": timestamp,
        "skip_harness": skip_harness,
        "results": results,
        "summary": summary,
    }


def _compute_post_run_check(
    prior_identity: dict, model_tag: str
) -> dict:
    """Re-resolve identity and return a post-run check record."""
    try:
        current = _resolve_model_identity(model_tag)
    except ModelIdentityError as e:
        return {"status": "error", "error": str(e)}
    if current.get("digest") == prior_identity.get("digest") and \
            current.get("blob_sha256") == prior_identity.get("blob_sha256"):
        return {"status": "match"}
    return {
        "status": "mismatch",
        "digest_now": current.get("digest", ""),
        "blob_sha256_now": current.get("blob_sha256", ""),
    }


def _rewrite_envelope(
    path: Path, cli_model_tag: str, cli_model_was_default: bool
) -> None:
    """Rewrite the envelope of an existing results artifact.

    Preserves all content fields (results, summary, run_id, n_instances,
    offset, conditions, skip_harness) and re-generates the identity
    envelope (model_identity_post_run_check, timestamp) by re-resolving
    against the currently-installed ollama model. This is how historical
    artifacts get aligned with a newer writer contract without rerunning
    predictions or the Docker harness.

    The verification tag is taken from the artifact itself (``existing["model"]``),
    not from ``--model``, so a stale or default CLI flag cannot silently
    verify the wrong tag. If ``--model`` was set explicitly to a value
    that disagrees with the artifact's recorded model, the rewrite
    refuses rather than emit a status for the wrong model.
    """
    existing = json.loads(path.read_text())
    prior_identity = existing.get("model_identity")
    if not prior_identity or "digest" not in prior_identity:
        print(f"ERROR: {path} has no model_identity with a digest to verify "
              "against. Run the script without --rewrite-envelope to "
              "produce a fresh artifact.")
        sys.exit(1)

    artifact_model = existing.get("model")
    if not artifact_model:
        print(f"ERROR: {path} has no `model` field. Cannot determine which "
              "tag to verify against.")
        sys.exit(1)

    # If the user supplied --model explicitly and it disagrees with the
    # artifact, refuse. An implicit default does not disagree — we trust
    # the artifact's own record in that case.
    if not cli_model_was_default and cli_model_tag != artifact_model:
        print(f"ERROR: --model {cli_model_tag!r} disagrees with the "
              f"artifact's recorded model {artifact_model!r}. Refusing "
              "to verify against a different tag. Remove --model to use "
              "the artifact's own value, or update the artifact.")
        sys.exit(1)

    verification_tag = artifact_model
    post_run_check = _compute_post_run_check(prior_identity, verification_tag)
    out_data = build_artifact(
        model=artifact_model,
        model_identity=prior_identity,
        post_run_check=post_run_check,
        run_id=existing["run_id"],
        n_instances=existing["n_instances"],
        offset=existing["offset"],
        conditions=existing["conditions"],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        skip_harness=existing.get("skip_harness", False),
        results=existing["results"],
        summary=existing["summary"],
        dataset=existing.get("dataset", "SWE-bench/SWE-bench_Lite"),
    )
    path.write_text(json.dumps(out_data, indent=2))
    print(f"Rewrote envelope at {path} (verified against "
          f"{verification_tag!r}): "
          f"post_run_check.status={post_run_check['status']}")


def _parse_reports(
    report_dir: Path, run_id: str, model_name: str
) -> tuple[dict[str, str], bool]:
    """Parse harness output.

    Returns (status_by_instance, report_found). *status_by_instance* maps
    instance_id → one of the EVAL_* constants. Instances not mentioned in
    the report get EVAL_NOT_EVALUATED by the caller.
    """
    status: dict[str, str] = {}
    summary = report_dir / f"{model_name}.{run_id}.json"
    if not summary.exists():
        return status, False
    try:
        data = json.loads(summary.read_text())
        for inst_id in data.get("resolved_ids", []):
            status[inst_id] = EVAL_RESOLVED
        for inst_id in data.get("unresolved_ids", []):
            status[inst_id] = EVAL_UNRESOLVED
        for inst_id in data.get("error_ids", []):
            status[inst_id] = EVAL_ERROR
        for inst_id in data.get("empty_patch_ids", []):
            status[inst_id] = EVAL_EMPTY
        return status, True
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARN: failed to parse {summary}: {e}")
        return status, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemma4:latest"


def main():
    parser = argparse.ArgumentParser(description="SWE-bench-lite Phase 2 (Docker)")
    # Sentinel default lets us distinguish an explicit --model from a
    # defaulted one. The rewrite path uses this to refuse verifying
    # against a tag the user supplied that disagrees with the artifact.
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--conditions", default="baseline,organism,langgraph")
    parser.add_argument("--skip-harness", action="store_true",
                        help="Only generate predictions, skip Docker evaluation")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--rewrite-envelope", metavar="PATH", default=None,
        help=("Rewrite the envelope of an existing results artifact at "
              "PATH: preserves results/summary/run_id/etc., re-resolves "
              "model_identity against ollama, and emits an authentic "
              "post-run check. Does not rerun predictions or the harness."),
    )
    parser.add_argument(
        "--grounding", action="store_true",
        help=("Shallow-clone each instance's {repo}@{base_commit} into "
              "--cache-dir, inject candidate-file context into the prompt, "
              "and give the sanitizer a tree-oracle for fuzzy path "
              "correction. Adds disk + network cost; opt in explicitly."),
    )
    parser.add_argument(
        "--cache-dir", default=".cache/swebench",
        help=("Directory where --grounding clones are cached "
              "(default: .cache/swebench)."),
    )
    args = parser.parse_args()

    cli_model_was_default = args.model is None
    if cli_model_was_default:
        args.model = _DEFAULT_MODEL

    if args.rewrite_envelope:
        _rewrite_envelope(
            Path(args.rewrite_envelope), args.model, cli_model_was_default,
        )
        return

    conditions = [c.strip() for c in args.conditions.split(",")]

    if "langgraph" in conditions:
        from operon_ai.convergence.langgraph_compiler import HAS_LANGGRAPH
        if not HAS_LANGGRAPH:
            print("WARNING: langgraph not installed, skipping")
            conditions = [c for c in conditions if c != "langgraph"]

    # Freeze the model identity BEFORE any predictions are generated so
    # the recorded digest/blob describes the weights actually used. A
    # concurrent `ollama pull` during a 45-minute run could otherwise
    # re-point the floating tag; resolving at save time would then lie
    # about what generated the predictions.
    try:
        model_identity = _resolve_model_identity(args.model)
    except ModelIdentityError as e:
        print(f"ERROR: cannot resolve immutable model identity: {e}")
        print("Refusing to run — the results artifact cites this identity "
              "for reproducibility. Pin the model first (e.g. `ollama pull "
              f"{args.model}`) and retry.")
        sys.exit(1)
    print(f"Model identity: tag={model_identity['tag']} "
          f"digest={model_identity['digest']} "
          f"blob={model_identity['blob_sha256'][:12]}... "
          f"arch={model_identity['architecture']} "
          f"params={model_identity['parameters']} "
          f"quant={model_identity['quantization']}")

    provider = _make_provider(args.model)

    # Probe model
    try:
        probe = _probe_model(provider)
        print(f"Model probe: {probe.strip()[:30]}")
    except Exception as e:
        print(f"ERROR: cannot reach model: {e}")
        sys.exit(1)

    # Probe Docker (unless skipping)
    if not args.skip_harness:
        try:
            subprocess.run(["docker", "info"], check=True,
                           capture_output=True, timeout=10)
            print("Docker:      OK")
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired) as e:
            print(f"ERROR: Docker unavailable: {e}")
            print("Hint: start Docker Desktop, or pass --skip-harness")
            sys.exit(1)

    # Load dataset
    from datasets import load_dataset
    print(f"\nLoading SWE-bench-lite...")
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    instances = list(ds.select(range(args.offset, min(args.offset + args.n, len(ds)))))
    print(f"Selected {len(instances)} instances")

    condition_fns = {
        "baseline": run_baseline,
        "organism": run_organism,
        "langgraph": run_langgraph,
    }

    # -- Phase A: generate predictions -----------------------------------
    run_id = f"phase2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preds_by_cond: dict[str, list[Prediction]] = {c: [] for c in conditions}

    cache_dir_path = Path(args.cache_dir)
    for i, instance in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(instances)}] {instance['instance_id']}")
        print(f"  repo:  {instance['repo']}")
        print(f"  issue: {instance['problem_statement'][:70]}...")

        if args.grounding:
            grounding = _build_grounding(instance, cache_dir_path)
            if grounding.repo_path is not None:
                print(f"  grounding: repo at {grounding.repo_path}, "
                      f"{len(grounding.tree_paths or ())} files indexed, "
                      f"context_block={len(grounding.context_block)}B")
        else:
            grounding = Grounding.none()

        for cond in conditions:
            try:
                p = condition_fns[cond](provider, instance, grounding)
            except Exception as e:
                print(f"  {cond:10s} ERROR: {e}")
                # Tag the failure mode so summary aggregation can
                # distinguish "model crashed" from "model returned but
                # sanitizer rejected the output". Both serialize as an
                # empty model_patch and the harness will mark them
                # ``empty_patch``, but only this one is a true runtime
                # failure of our pipeline.
                p = Prediction(
                    instance["instance_id"], instance["repo"], cond,
                    f"ERROR: {e}", "", 0.0, False,
                    error_reason=type(e).__name__ + ": " + str(e)[:200],
                )
            preds_by_cond[cond].append(p)
            status = "ok" if p.extract_ok else "no-patch"
            print(f"  {cond:10s} extract={status:8s} latency={p.latency_ms:.0f}ms "
                  f"(patch {len(p.model_patch)}B)")

    # -- Phase B: write predictions & run harness ------------------------
    results_dir = Path("eval/results/swebench_phase2_predictions") / run_id
    status_by_cond: dict[str, dict[str, str]] = {c: {} for c in conditions}
    # Tri-state per condition: "ok" | "failed" | "skipped". Default is
    # "failed" so that unexpected paths (no report, crash before parse)
    # are not silently treated as "ok"; the skip path explicitly sets
    # HARNESS_SKIPPED so a skipped run is distinguishable from a crash.
    harness_state_by_cond: dict[str, str] = {c: HARNESS_FAILED for c in conditions}

    for cond in conditions:
        model_name = f"operon-{cond}"
        pred_path = results_dir / f"{cond}.jsonl"
        _write_predictions(preds_by_cond[cond], pred_path, model_name)
        print(f"\n  Wrote {len(preds_by_cond[cond])} predictions to {pred_path}")

        if args.skip_harness:
            harness_state_by_cond[cond] = HARNESS_SKIPPED
            continue

        print(f"\n{'='*60}")
        print(f"Running harness for {cond} ({len(preds_by_cond[cond])} instances)")
        print(f"{'='*60}")
        report_dir = Path("eval/results/swebench_phase2_reports") / f"{run_id}_{cond}"
        try:
            _run_harness(pred_path, f"{run_id}_{cond}", report_dir, args.timeout)
        except HarnessFailed as e:
            print(f"  HARNESS FAILED for {cond}: {e}")
            print(f"  Instances for {cond} will be recorded as not_evaluated")
            continue
        statuses, report_found = _parse_reports(
            report_dir, f"{run_id}_{cond}", model_name
        )
        status_by_cond[cond] = statuses
        harness_state_by_cond[cond] = HARNESS_OK if report_found else HARNESS_FAILED

    # -- Phase C: aggregate & save ---------------------------------------
    summary: dict[str, dict] = {}
    for cond in conditions:
        preds = preds_by_cond[cond]
        statuses = status_by_cond[cond]
        harness_state = harness_state_by_cond[cond]
        n = len(preds)
        extracted = sum(1 for p in preds if p.extract_ok)

        # Per-status counts. The harness can only see the model_patch
        # field of each prediction; if the model call raised an exception
        # we wrote an empty patch which the harness then categorizes as
        # ``empty_patch``. We override that to ``runtime_error`` here so
        # the summary distinguishes "sanitizer-rejected" (model produced
        # invalid output) from "model never produced output" (API
        # timeout, OOM, etc.).
        # The "evaluated" denominator stays restricted to instances that
        # reached a pass/fail harness verdict (resolved or unresolved).
        status_counts = {
            EVAL_RESOLVED: 0,
            EVAL_UNRESOLVED: 0,
            EVAL_ERROR: 0,
            EVAL_EMPTY: 0,
            EVAL_RUNTIME_ERROR: 0,
            EVAL_NOT_EVALUATED: 0,
        }
        for p in preds:
            harness_status = statuses.get(p.instance_id, EVAL_NOT_EVALUATED)
            s = classify_prediction(p.error_reason, harness_status)
            status_counts[s] = status_counts.get(s, 0) + 1

        evaluated_n = status_counts[EVAL_RESOLVED] + status_counts[EVAL_UNRESOLVED]
        passed = status_counts[EVAL_RESOLVED]

        # Mean latency over predictions that actually completed. A
        # latency of 0 means the model call raised before any timing
        # could be recorded, so including those zeros deflates the mean
        # and overstates per-call throughput.
        completed = [p for p in preds if p.error_reason is None]
        mean_latency_ms = (
            round(sum(p.latency_ms for p in completed) / len(completed), 0)
            if completed else 0
        )

        summary[cond] = {
            "n": n,
            "patch_extracted": extracted,
            "status_counts": status_counts,
            "evaluated": evaluated_n,
            "resolved": passed,
            "resolved_rate": (
                round(passed / evaluated_n, 3) if evaluated_n else None
            ),
            "harness_state": harness_state,
            "mean_latency_ms": mean_latency_ms,
            "n_runtime_errors": status_counts[EVAL_RUNTIME_ERROR],
            "n_completed": len(completed),
        }

    # Best-effort verification: re-resolve the identity and flag if the
    # floating tag moved during the run. Non-fatal — the run has already
    # executed against `model_identity`, and that is the authoritative
    # record.
    post_run_check = _compute_post_run_check(model_identity, args.model)
    if post_run_check["status"] == "mismatch":
        print("WARN: model tag moved during the run. Recorded identity "
              "describes pre-run weights; post-run digest saved as "
              "`model_identity_post_run_check.mismatch`.")

    def _result_status(p: Prediction) -> str:
        return classify_prediction(
            p.error_reason,
            status_by_cond[p.condition].get(
                p.instance_id, EVAL_NOT_EVALUATED
            ),
        )

    results = [
        {
            "instance_id": p.instance_id,
            "repo": p.repo,
            "condition": p.condition,
            "patch_extracted": p.extract_ok,
            "patch_size_bytes": len(p.model_patch),
            "latency_ms": p.latency_ms,
            "eval_status": _result_status(p),
            "error_reason": p.error_reason,
        }
        for cond in conditions for p in preds_by_cond[cond]
    ]
    out_data = build_artifact(
        model=args.model,
        model_identity=model_identity,
        post_run_check=post_run_check,
        run_id=run_id,
        n_instances=len(instances),
        offset=args.offset,
        conditions=conditions,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        skip_harness=args.skip_harness,
        results=results,
        summary=summary,
    )
    out_path = Path("eval/results/swebench_phase2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, s in summary.items():
        rate_str = (
            f"{s['resolved_rate']:.1%}"
            if s["resolved_rate"] is not None else "N/A"
        )
        harness_str = s["harness_state"]
        print(
            f"  {cond:10s} resolved={s['resolved']}/{s['evaluated']} "
            f"({rate_str})  n={s['n']}  "
            f"extracted={s['patch_extracted']}/{s['n']}  "
            f"harness={harness_str}  "
            f"latency={s['mean_latency_ms']:.0f}ms"
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
