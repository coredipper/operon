# Example 59: Plasmid Registry — Horizontal Gene Transfer

## Wiring Diagram

```mermaid
flowchart TB
    subgraph registry["PlasmidRegistry (Environmental Pool)"]
        p1[Plasmid: reverse<br/>tags: text, utility<br/>caps: none]
        p2[Plasmid: word_count<br/>tags: text, analysis<br/>caps: none]
        p3[Plasmid: fetch_url<br/>tags: network, io<br/>caps: {NET}]
    end

    subgraph search["Discovery"]
        tags_search["search(tags={text})"] --> p1
        tags_search --> p2
        name_search["search('fetch')"] --> p3
    end

    subgraph mito["Mitochondria (caps: {READ_FS})"]
        acquire["acquire(name, registry)"]
        tools["tools: {}"]
        acquire -->|reverse| tools_ok1["tools: {reverse}"]
        acquire -->|word_count| tools_ok2["tools: {reverse, word_count}"]
        acquire -->|fetch_url| blocked["BLOCKED<br/>needs NET, has READ_FS"]
    end

    subgraph net_mito["Mitochondria (caps: {NET})"]
        net_acquire["acquire(fetch_url)"] --> net_ok["tools: {fetch_url}"]
    end

    subgraph execute["Execution"]
        tools_ok2 --> exec1["metabolize('reverse(hello)')"]
        exec1 --> result1["olleh"]
        tools_ok2 --> exec2["metabolize('word_count(quick fox)')"]
        exec2 --> result2["4"]
    end

    subgraph release["Plasmid Curing"]
        tools_ok2 --> rel["release('reverse')"]
        rel --> after_rel["tools: {word_count}"]
        after_rel --> reacquire["acquire('reverse')"]
        reacquire --> restored["tools: {word_count, reverse}"]
    end

    registry --> search
    registry --> acquire
    registry --> net_acquire

    style blocked fill:#ffcdd2
    style tools_ok2 fill:#c8e6c9
    style net_ok fill:#c8e6c9
```

```
[PlasmidRegistry]
  ├─ reverse     (text, utility)   caps: none
  ├─ word_count  (text, analysis)  caps: none
  └─ fetch_url   (network, io)     caps: {NET}
        │
        ├── search(tags={text}) ──> [reverse, word_count]
        ├── search("fetch")    ──> [fetch_url]
        │
        v
[Mitochondria] caps={READ_FS}
  │
  ├── acquire("reverse")    ──> SUCCESS (no caps required)
  ├── acquire("word_count") ──> SUCCESS (no caps required)
  ├── acquire("fetch_url")  ──> BLOCKED (needs NET, has READ_FS)
  │
  ├── metabolize('reverse("hello")') ──> "olleh"
  ├── metabolize('word_count("quick brown fox")') ──> 4
  │
  ├── release("reverse") ──> tools: {word_count}
  └── acquire("reverse") ──> tools: {word_count, reverse}  (re-acquisition)

[Mitochondria] caps={NET}
  └── acquire("fetch_url") ──> SUCCESS
```

## Key Patterns

### Horizontal Gene Transfer (Section 6.2)
Tools are no longer static at construction time. Agents can dynamically discover,
acquire, and release capabilities from a searchable registry.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | PlasmidRegistry | Searchable pool of available tools |
| 2 | Plasmid | Small capability unit (name, func, tags, required_caps) |
| 3 | Mitochondria | Agent execution engine with capability envelope |
| 4 | Capability gating | Prevents privilege escalation on acquire |
| 5 | Dynamic acquisition | acquire(name, registry) adds tool at runtime |
| 6 | Plasmid curing | release(name) cleanly removes tool |
| 7 | Schema export | export_tool_schemas() for LLM function calling |

### Biological Parallel
- Plasmid = small circular DNA encoding a useful gene (tool)
- Registry = environmental pool of plasmids
- Acquisition = bacterial conjugation / transformation
- Capability gating = restriction enzymes
- Release = plasmid curing (natural loss of plasmid)

## Data Flow

```
PlasmidRegistry
  └─ plasmids: dict[str, Plasmid]
       ├─ name: str
       ├─ description: str
       ├─ func: Callable
       ├─ tags: frozenset[str]
       └─ required_capabilities: frozenset[Capability]
            ↓
Mitochondria.acquire(name, registry)
  ├─ check: required_caps ⊆ allowed_caps
  ├─ success → tool added to tools dict
  └─ failure → AcquireResult(success=False, error=reason)
            ↓
Mitochondria.metabolize(expression)
  ├─ tool lookup → execute func
  └─ MetabolizeResult(atp=value, success=bool)
            ↓
Mitochondria.release(name)
  └─ tool removed from tools dict
```

## Pipeline Stages

| Stage | Mechanism | Input | Output | Fallback |
|-------|-----------|-------|--------|----------|
| Register | PlasmidRegistry.register | Plasmid definition | Indexed plasmid | — |
| Search | registry.search(tags/name) | Query | list[Plasmid] | Empty list |
| Acquire | mito.acquire(name, registry) | Plasmid name | AcquireResult | Blocked if caps insufficient |
| Execute | mito.metabolize(expr) | Tool expression | MetabolizeResult | Error if tool not found |
| Release | mito.release(name) | Tool name | Tool removed | — |
| Re-acquire | mito.acquire(name, registry) | Plasmid name | Tool restored | — |
| Export | mito.export_tool_schemas | — | list[ToolSchema] | — |

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
