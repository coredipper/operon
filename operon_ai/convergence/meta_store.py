"""C8 Meta-Harness filesystem store: append-only evolution history.

Candidate-first layout with an append-only index for efficient querying.
Raw artifacts live on the filesystem so an LLM proposer can read them
directly — the "exogenous signal" from Ao et al. (2603.26993).

Layout:
    .operon/evolution/{run_id}/
      meta.json
      candidates/
        {candidate_id}.json          # genome.export() + config metadata
        {candidate_id}_trace.jsonl   # per-stage execution trace
      index.jsonl                    # append-only assessment records
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..state.genome import Genome
from .meta_types import AssessmentRecord, CandidateConfig, genome_to_candidate


@dataclass
class EvolutionStore:
    """Append-only filesystem store for evolution run artifacts."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._candidates_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def _index_path(self) -> Path:
        return self.root / "index.jsonl"

    # -- Candidates ----------------------------------------------------------

    def save_candidate(
        self,
        config: CandidateConfig,
        genome: Genome,
    ) -> Path:
        """Write candidate config + genome export to a JSON file."""
        path = self._candidates_dir / f"{config.candidate_id}.json"
        payload = {
            "candidate_id": config.candidate_id,
            "parent_id": config.parent_id,
            "iteration": config.iteration,
            "topology": config.topology,
            "proposer": config.proposer,
            "reason": config.reason,
            "genome": genome.export(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def load_candidate(self, candidate_id: str) -> CandidateConfig:
        """Read a candidate back from its JSON file."""
        path = self._candidates_dir / f"{candidate_id}.json"
        data = json.loads(path.read_text())
        genome = _genome_from_export(data["genome"])
        return genome_to_candidate(
            genome,
            candidate_id=data["candidate_id"],
            parent_id=data["parent_id"],
            iteration=data["iteration"],
            proposer=data["proposer"],
            reason=data["reason"],
        )

    # -- Traces --------------------------------------------------------------

    def append_trace(
        self,
        candidate_id: str,
        stage_name: str,
        data: dict[str, Any],
    ) -> None:
        """Append one stage trace line to a candidate's trace file."""
        path = self._candidates_dir / f"{candidate_id}_trace.jsonl"
        line = json.dumps({"stage": stage_name, **data}, default=str)
        with path.open("a") as f:
            f.write(line + "\n")

    # -- Assessment index ----------------------------------------------------

    def append_assessment(self, record: AssessmentRecord) -> None:
        """Append one assessment record to the index."""
        line = json.dumps(asdict(record), default=str)
        with self._index_path.open("a") as f:
            f.write(line + "\n")

    def load_index(self) -> list[AssessmentRecord]:
        """Load all assessment records from the index."""
        if not self._index_path.exists():
            return []
        records = []
        for line in self._index_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                records.append(AssessmentRecord(**d))
        return records

    # -- Run metadata --------------------------------------------------------

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Write run-level metadata (config, seed, timestamp, etc.)."""
        path = self.root / "meta.json"
        path.write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _genome_from_export(data: dict[str, Any]) -> Genome:
    """Reconstruct a Genome from its export() dict.

    Genome.from_dict() only creates bare Gene objects. We use the full
    export format to preserve GeneType and expression levels.
    """
    from ..state.genome import ExpressionLevel, ExpressionState, Gene, GeneType

    genes = []
    for g in data.get("genes", []):
        genes.append(Gene(
            name=g["name"],
            value=g["value"],
            gene_type=GeneType(g.get("gene_type", "structural")),
            description=g.get("description", ""),
            required=g.get("required", False),
            default_expression=ExpressionLevel(
                g.get("default_expression", 2)
            ),
        ))

    genome = Genome(genes=genes, allow_mutations=True, silent=True)

    for name, state in data.get("expression", {}).items():
        genome._expression[name] = ExpressionState(
            level=ExpressionLevel(state["level"]),
            modifier=state.get("modifier", ""),
        )

    return genome
