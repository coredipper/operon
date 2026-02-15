# Operon Synthetic Evaluation Harness

This harness exercises three motifs described in the paper using deterministic,
synthetic data. It is designed to be reproducible, lightweight, and free of
external service dependencies.

## Suites

- **Folding** (`folding`): Generates Pydantic schemas and applies controlled JSON
  corruptions to estimate strict vs. cascaded folding success.
- **Immune** (`immune`): Simulates baseline agent behavior and compromised shifts
  to measure sensitivity and false positives under two-signal activation.
- **Healing** (`healing`): Runs the ChaperoneLoop with and without error-context
  feedback to estimate recovery rates under retries.

## Run

```bash
python -m eval.run --suite all --config eval/configs/default.json --out eval/results/latest.json
```

You can run a single suite:

```bash
python -m eval.run --suite folding --config eval/configs/default.json
```

## Reproducibility

- The harness is deterministic given `--seed`.
- All configuration lives in JSON under `eval/configs/`.
- Results are emitted as JSON for direct use in tables or plots.

## Output Schema

`eval/run.py` writes a JSON object with a top-level `seed` and a `suites` map.
Each suite contains the config used and computed metrics.

## Aggregation

Aggregate multiple seed runs into a summary JSON + LaTeX table:

```bash
python -m eval.report --glob "eval/results/seed-*.json" \
  --out-json eval/results/summary.json \
  --out-tex eval/results/summary.tex
```

## External Benchmarks (BFCL + AgentDojo)

The harness includes suites derived from two external benchmarks.
Install optional dependencies:

```bash
pip install -e ".[eval]"
```

Run external suites only:

```bash
python -m eval.run --suite all_external --config eval/configs/default.json
```

### BFCL Folding

Uses function-call schemas from the [Berkeley Function Calling
Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) to test
Chaperone folding with realistic schemas. Falls back to built-in schemas
if `bfcl-eval` data is not available.

### AgentDojo Immune

Uses prompt injection attack templates from
[AgentDojo](https://agentdojo.spylab.ai/) to generate realistic compromised
agent behavioral signatures. Falls back to built-in templates if `agentdojo`
is not installed.

## Notes

This harness is intentionally synthetic. It does not represent real LLM outputs
or adversarial inputs; it is meant to provide a consistent baseline for motif
behavior under controlled perturbations.
