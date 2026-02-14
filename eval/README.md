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

## Notes

This harness is intentionally synthetic. It does not represent real LLM outputs
or adversarial inputs; it is meant to provide a consistent baseline for motif
behavior under controlled perturbations.
