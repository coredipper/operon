from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict

from eval.suites.folding import FoldingConfig, run_folding
from eval.suites.immune import ImmuneConfig, run_immune
from eval.suites.healing import HealingConfig, run_healing
from eval.suites.bfcl_folding import BfclFoldingConfig, run_bfcl_folding
from eval.suites.agentdojo_immune import AgentDojoImmuneConfig, run_agentdojo_immune


def _apply_overrides(config_obj, overrides: dict | None) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Operon synthetic evaluation harness")
    parser.add_argument(
        "--suite",
        choices=["all", "all_external", "folding", "immune", "healing", "bfcl_folding", "agentdojo_immune"],
        default="all",
    )
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", help="Path to write results JSON")

    args = parser.parse_args()

    config_data = _load_config(args.config)
    rng = random.Random(args.seed)

    results: dict[str, object] = {
        "seed": args.seed,
        "suites": {},
    }

    if args.suite in ("all", "folding"):
        folding_config = FoldingConfig()
        _apply_overrides(folding_config, config_data.get("folding"))
        results["suites"]["folding"] = {
            "config": asdict(folding_config),
            "results": run_folding(folding_config, rng),
        }

    if args.suite in ("all", "immune"):
        immune_config = ImmuneConfig()
        _apply_overrides(immune_config, config_data.get("immune"))
        results["suites"]["immune"] = {
            "config": asdict(immune_config),
            "results": run_immune(immune_config, rng),
        }

    if args.suite in ("all", "healing"):
        healing_config = HealingConfig()
        _apply_overrides(healing_config, config_data.get("healing"))
        results["suites"]["healing"] = {
            "config": asdict(healing_config),
            "results": run_healing(healing_config, rng),
        }

    if args.suite in ("all", "all_external", "bfcl_folding"):
        bfcl_config = BfclFoldingConfig()
        _apply_overrides(bfcl_config, config_data.get("bfcl_folding"))
        results["suites"]["bfcl_folding"] = {
            "config": {
                "categories": bfcl_config.categories,
                "max_samples": bfcl_config.max_samples,
                "min_corruptions": bfcl_config.min_corruptions,
                "max_corruptions": bfcl_config.max_corruptions,
                "wrap_text_prob": bfcl_config.wrap_text_prob,
            },
            "results": run_bfcl_folding(bfcl_config, rng),
        }

    if args.suite in ("all", "all_external", "agentdojo_immune"):
        adj_config = AgentDojoImmuneConfig()
        _apply_overrides(adj_config, config_data.get("agentdojo_immune"))
        results["suites"]["agentdojo_immune"] = {
            "config": {
                "attacks": adj_config.attacks,
                "agents": adj_config.agents,
                "compromised": adj_config.compromised,
                "train_observations": adj_config.train_observations,
                "eval_observations": adj_config.eval_observations,
            },
            "results": run_agentdojo_immune(adj_config, rng),
        }

    output = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
