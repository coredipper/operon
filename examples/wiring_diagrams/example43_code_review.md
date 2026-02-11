# Example 43: Code Review Pipeline

## Wiring Diagram

```mermaid
flowchart TB
    subgraph input["Input"]
        PR[("PR Diff")]
    end

    subgraph stage1["Stage 1: Parsing"]
        PR --> Parser["DiffParser<br/>━━━━━━━━━━━━━<br/>• Parse unified diff<br/>• Extract files/hunks<br/>• Count additions"]
    end

    subgraph stage2["Stage 2: Budget"]
        Parser --> Budget["ATP_Store<br/>━━━━━━━━━━━━━<br/>• 100 ATP per file<br/>• Reject if insufficient<br/>• Track consumption"]
    end

    subgraph stage3["Stage 3: Pre-filter"]
        Budget --> Membrane["Membrane<br/>━━━━━━━━━━━━━<br/>• Critical threat scan<br/>• Pattern matching<br/>• Early rejection"]
    end

    subgraph stage4["Stage 4: CFFL Gate"]
        Membrane --> Security["SecurityReviewer<br/>━━━━━━━━━━━━━<br/>• Regex patterns<br/>• Threat scoring"]
        Membrane --> Quality["QualityReviewer<br/>━━━━━━━━━━━━━<br/>• Style patterns<br/>• Quality scoring"]
        Security --> AND["AND Gate"]
        Quality --> AND
    end

    subgraph stage5["Stage 5: Validation"]
        AND --> Chaperone["Chaperone<br/>━━━━━━━━━━━━━<br/>• Schema validation<br/>• JSON parsing"]
    end

    subgraph output["Output"]
        Chaperone --> Result[("ReviewResult<br/>approved/blocked<br/>findings[]")]
    end

    style PR fill:#e1f5fe
    style Result fill:#c8e6c9
    style AND fill:#fff9c4
```

## Key Patterns

### CFFL (Coherent Feed-Forward Loop)
Both Security and Quality reviewers must approve for the review to pass.
This is the biological "two-key" pattern - a safety interlock.

### ATP Budgeting
Each file costs 100 ATP. Large PRs (>20 files) require explicit override.
Prevents runaway costs on massive PRs.

### Membrane Pre-filter
Fast pattern matching catches critical threats before expensive analysis.
Rejects obvious dangerous patterns immediately.

## Data Flow

```
ParsedDiff
  +-- files: list[str]
  +-- hunks: list[DiffHunk]
  +-- additions: int
  +-- deletions: int
       |
       v
ReviewerResult
  +-- approved: bool
  +-- score: float
  +-- findings: list[ReviewFinding]
       |
       v
ReviewResult
  +-- approved: bool
  +-- summary: str
  +-- security_score: float
  +-- quality_score: float
  +-- findings: list[ReviewFinding]
```
