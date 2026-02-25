---
title: Operon Plasmid Registry
emoji: "\U0001F9EC"
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Dynamic tool acquisition with capability gating (HGT)
---

# Operon Plasmid Registry -- Horizontal Gene Transfer

Dynamic tool acquisition from a searchable registry with capability gating (Paper §6.2, Eq. 12).

Like bacteria exchanging plasmids, agents can **discover**, **acquire**, and **release** tools at runtime. Capability gating prevents privilege escalation -- an agent can only absorb a plasmid if it has the required capabilities.

**Features:**
- Browse and search a plasmid registry
- Acquire tools into an agent (Mitochondria) with capability checks
- Execute acquired tools
- Release tools (plasmid curing)
- See how capability restrictions block unsafe acquisitions

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
