# Example 58: Tissue Architecture — Hierarchical Multi-Agent Organization

## Wiring Diagram

```mermaid
flowchart TB
    subgraph genome["Shared Genome (6 genes)"]
        g[model, temperature, classification,<br/>code_execution, verification, research_tools]
    end

    subgraph ct_tissue["ClassificationTissue"]
        ct_boundary["Boundary<br/>caps: {NET}"]
        ct_in["in: task (V/JSON)"] --> ct_cell["primary_classifier<br/>(Classifier CellType)"]
        ct_cell --> ct_out["out: label (V/JSON)"]
    end

    subgraph rt_tissue["ResearchTissue"]
        rt_boundary["Boundary<br/>caps: {NET, READ_FS}"]
        rt_gradient["Gradient<br/>COMPLEXITY=0.8"]
        rt_in["in: query (V/JSON)"] --> rt_cell1["researcher_1<br/>(Researcher CellType)"]
        rt_in --> rt_cell2["researcher_2<br/>(Researcher CellType)"]
        rt_cell1 --> rt_out["out: findings (V/JSON)"]
        rt_cell2 --> rt_out
    end

    subgraph vt_tissue["ValidationTissue"]
        vt_boundary["Boundary<br/>caps: {} (isolated)"]
        vt_in["in: draft (V/JSON)"] --> vt_cell["validator_1<br/>(Validator CellType)"]
        vt_cell --> vt_out["out: verdict (V/JSON)"]
    end

    subgraph organism["Organism (WiringDiagram)"]
        ct_out -->|"label → query"| rt_in
        rt_out -->|"findings → draft"| vt_in
    end

    genome -.->|differentiate| ct_tissue
    genome -.->|differentiate| rt_tissue
    genome -.->|differentiate| vt_tissue

    style ct_boundary fill:#fff9c4
    style rt_boundary fill:#c8e6c9
    style vt_boundary fill:#f3e5f5
    style vt_boundary stroke:#ce93d8
```

```
                              [Shared Genome]
                      /            |              \
                     v             v               v
  [ClassificationTissue]   [ResearchTissue]    [ValidationTissue]
  Boundary: {NET}          Boundary: {NET,FS}  Boundary: {} (isolated)
  ┌─────────────────┐      ┌──────────────────┐ ┌─────────────────┐
  │ task (V/JSON)   │      │ query (V/JSON)   │ │ draft (V/JSON)  │
  │   ↓             │      │   ↓         ↓    │ │   ↓             │
  │ [classifier]    │      │ [researcher_1]   │ │ [validator_1]   │
  │   ↓             │      │ [researcher_2]   │ │   ↓             │
  │ label (V/JSON)  │      │   ↓              │ │ verdict (V/JSON)│
  └────────┬────────┘      │ findings (V/JSON)│ └─────────────────┘
           │               └────────┬─────────┘          ▲
           └───── label → query ────┘                    │
                                    └── findings → draft ─┘

  Organism-level WiringDiagram: Classification → Research → Validation
  Capability enforcement: Researcher BLOCKED from ValidationTissue (needs NET+FS, allows {})
```

## Key Patterns

### Tissue Architecture (Section 6.5.3)
Cells are grouped into Tissues with shared gradients and security boundaries.
Tissues expose typed boundary ports for inter-tissue composition via WiringDiagram.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | Genome | Shared DNA for all cell types |
| 2 | CellType | Classifier, Researcher, Validator templates |
| 3 | Tissue | Groups cells with shared gradient + boundary |
| 4 | TissueBoundary | Typed input/output ports + capability set |
| 5 | MorphogenGradient | Per-tissue chemical environment |
| 6 | WiringDiagram | Organism-level composition of tissues |
| 7 | Capability enforcement | Researcher blocked from Validation (needs NET+READ_FS) |

### Biological Parallel
- Tissue = group of cells sharing extracellular environment
- Tissue boundary = basement membrane (physical security barrier)
- Inter-tissue signaling = boundary ports (typed channels)
- Organism = coordinated tissue system
- Gradient isolation = different tissues have independent chemical environments

### 4-Level Hierarchy
```
Cell → Tissue → Organ → Organism
(this example covers Cell → Tissue → Organism)
```

## Data Flow

```
Tissue
  ├─ name: str
  ├─ boundary: TissueBoundary
  │   ├─ inputs: dict[str, PortType]
  │   ├─ outputs: dict[str, PortType]
  │   └─ allowed_capabilities: set[Capability]
  ├─ gradient: MorphogenGradient (per-tissue)
  ├─ cells: dict[str, DifferentiatedCell]
  └─ as_module() → ModuleSpec (for WiringDiagram)
       ↓
WiringDiagram (Organism)
  ├─ modules: {ClassificationTissue, ResearchTissue, ValidationTissue}
  ├─ wires: [label→query, findings→draft]
  └─ capabilities isolated per tissue
```

## Pipeline Stages

| Stage | Mechanism | Input | Output | Fallback |
|-------|-----------|-------|--------|----------|
| Define genome | Genome(genes=[...]) | 6 genes | Shared genome | — |
| Define cell types | CellType with ExpressionProfile | Genome + overrides | 3 cell types | — |
| Create tissues | Tissue(boundary, gradient) | CellTypes + genome | Bounded cell groups | — |
| Register cells | tissue.add_cell | CellType + genome | Differentiated cells | — |
| Capability check | tissue.register_cell_type | CellType caps | Allow/TissueError | Block incompatible |
| Compose organism | WiringDiagram.connect | Tissue modules | Inter-tissue wiring | — |
| Gradient isolation | Per-tissue MorphogenGradient | Morphogen values | Independent environments | Default 0.0 |

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
