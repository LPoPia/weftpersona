# WeftPersona System Architecture & Turn Lifecycle

> 🌐 **Language**: [**简体中文**](../architecture.md) | **English**

This document describes the component contracts, data source-of-truth division, and single-turn execution flow of **WeftPersona (`sanae_mem`)** on the Hermes host runtime. For the underlying cognitive psychology and pragmatic design rationale, see [`how-to-simulate-a-living-person.md`](how-to-simulate-a-living-person.md).

---

## 1. Core Components & Host Hook Contracts

| Component / Module | Entry File | Exposed Tools | Host Hooks & Responsibility Contract |
|---|---|---|---|
| **Memory & Mood Main Entry** | [`plugins/sanae_mem/plugin_entry.py`](../../plugins/sanae_mem/plugin_entry.py) | `sanae_mem_add`, `sanae_mem_search`, `sanae_mem_forget`, `sanae_mem_compact`, `sanae_mem_marked_promoted`, `sanae_mem_distill`, `sanae_mem_report_false_trigger` | Registers `MemoryProvider` (`sync_after_delivery = True`); `system_prompt_block` injects the rendered `BEHAVIOR.md` block; `pre_llm_call` runs concurrent gated recall and meme recognition; `pre_tool_call` enforces resident memory and controlled-file write guards |
| **Forgetting Curve & 3-Stage Collapse** | [`plugins/sanae_mem/curve.py`](../../plugins/sanae_mem/curve.py), [`event_maintenance.py`](../../plugins/sanae_mem/event_maintenance.py) | Triggered via `sanae_mem_compact` | Cross-day (`≥ 12h`) spaced reactivation extends half-life via a power law (capped at 120 days); automatically transitions untouched events through **Vivid Verbatim → `gist_faded` (archives raw quotes before retaining only gist) → `dormant` (deep dormancy) or Cold Archive** |
| **Resident Memory Metabolism & Guard** | [`plugins/sanae_mem/resident.py`](../../plugins/sanae_mem/resident.py), [`llm_guard.py`](../../plugins/sanae_mem/llm_guard.py) | Intercepts host `memory` tool writes | Tracks last conversational touch per resident entry; when capacity `≥ 85%`, identifies cold entries untouched for 45 days for demotion back into episodic storage; uses a fast model at write-time to intercept meta-rules, one-off trivia, and self-criticism, diverting them to short-term inner thoughts or neutral short-term events |
| **Affective Dynamics (Fast Variables)** | [`plugins/sanae_mem/inner_state.py`](../../plugins/sanae_mem/inner_state.py), [`inner_wiring.py`](../../plugins/sanae_mem/inner_wiring.py) | `sanae_inner_state` | Asynchronously updates 1–3 sentences of first-person `[In My Heart Right Now]` (`[我此刻心里]`), continuous slow variables (`tension` / `closeness` / `energy`), pending verbatim promises (`recent_words`), unspoken associative bridges (`unspoken_link`), and Artificial DMN distant collisions (`spark`) after private reply delivery or trusted personal `cron` completion; decays naturally across real elapsed time and preserves directional warmth (`24h` window) across long sleep boundaries |
| **Holistic Impression (Slow Variable)** | [`plugins/sanae_mem/impression.py`](../../plugins/sanae_mem/impression.py) | `sanae_impression` | Piggybacks on the post-delivery Flash call; revises `[How I See Him Lately]` (`[我最近眼里的他]`) slowly only when sustained situational or relational changes are backed by direct user quotes; reads the latest revision in real time inside `Tier 2` `[The World in My Eyes]` without breaking the `Tier 0` system prompt KV prefix cache |
| **Daily Context & Curiosity Queue** | [`plugins/sanae_mem/daily_context.py`](../../plugins/sanae_mem/daily_context.py), [`daily_semantics.py`](../../plugins/sanae_mem/daily_semantics.py) | Maintained automatically via internal hooks | Extracts current-day updates (`DAILY_FACT`), active constraints (`ACTIVE_CONSTRAINT`), 2-day hard-expiring curiosity threads (`OPEN_THREAD`), and the persona's own actionable promises (`self_statements`) |
| **Memes & Sticker Memory** | [`plugins/sanae_mem/memes.py`](../../plugins/sanae_mem/memes.py) | `sanae_meme_save/search/update/delete` | Stores text memes and original sticker bytes (SHA-256 deduplication and instant inbound recognition without vision LLM calls); prepares 300px outbound thumbnails while preserving GIF frame rates and transparency |

---

## 2. Source-of-Truth Data Division (Relative to `$HERMES_HOME`)

| Data Layer | Default Path | Role & Lifecycle |
|---|---|---|
| **Episodes, Mood, Impressions & Memes** | `sanae-mem/sanae.db` | Single authoritative SQLite source of truth: stores episodic events (`events` / `event_occurrences`), first-person inner-state revision chains (`inner_state_revisions`), slow impression revisions & session views (`impression_revisions` / `impression_session_views`), daily updates (`daily_entries`), and memes (`meme_records`) |
| **Vector & Full-Text Search Indices** | `sanae-mem/chroma/` + SQLite FTS5 | Derived retrieval indices (local `bge-small-zh-v1.5` embeddings), fully rebuildable from `sanae.db` at any time |
| **Resident Long-Term Memory** | `memories/MEMORY.md`, `memories/USER.md` | Session-level snapshot injection: holds only cross-month foundational facts and enduring relational knowledge |
| **Procedural Policy & Subjectivity Kernel** | `BEHAVIOR.md`, `SOUL.md` | [`BEHAVIOR.md`](../../prompts/BEHAVIOR.md) is front-loaded in the system prompt; [`SOUL.md`](../../prompts/SOUL.md) and the character persona are anchored at the very end |
| **Pre-Deletion Cold Archive** | `memory-archive/` | Mandatory cold archive written before any event forgetting, verbatim collapse, or resident demotion (deletion aborts if archiving fails) |

---

## 3. Single-Turn Execution Flow

1. **Inbound Aggregation & Context Binding**:
   - The gateway merges rapid consecutive text/image messages from the same sender and computes the real wall-clock delta since the last interaction; known stickers match directly via raw-byte SHA-256 hashes, bypassing multimodal vision latency.
2. **Pre-LLM Dynamic Read (`pre_llm_call` & `current_turn_context`)**:
   - Loads `[Daily Context]` on session start or after a long pause, plus unconsumed cross-channel self-utterances (`[What I Just Said Elsewhere]`);
   - Triggers context-gated episodic recall (casual banter follows the Effort Prior and skips recall; when recall is warranted, executes query planning and hybrid search within a strict 3-second budget—timing out cleanly rather than falling back to noisy keyword matching);
   - Waits up to 3 seconds for any in-flight post-delivery `[In My Heart Right Now]` update from the previous turn (including autonomous activities completed since the last reply) and injects it into the current turn's transient context without polluting persistent history `api_content`.
3. **"Engineering Front-Loaded, Soul Tail-Anchored" Prompt Assembly (`wire_system_prompt`)**:
   - Front fixed prefix: Skill index → `MEMORY.md` / `USER.md` → rendered `BEHAVIOR.md` block → environment timestamp;
   - Tail immediately preceding dialogue: `SOUL.md` → character persona → session-frozen `[How I See Him Lately]`.
4. **Tool-Call Guardrails (`pre_tool_call`)**:
   - When the main model calls `memory` to write resident memory, `llm_guard.py` inspects the write: heat-of-the-moment frustration, mind-reading guesses, or self-imposed communication rules are silently diverted into short-term inner thoughts or neutral short-term episodes;
   - `write_file` / `patch` calls are checked against canonical real paths for controlled memory files (`MEMORY.md`, `USER.md`, `BEHAVIOR.md`), preventing direct file-tool overwrites that bypass the `memory` tool.
5. **Delivery Confirmation & Asynchronous Post-Turn Metabolism (`sync_turn` & `update_inner_state`)**:
   - Only after the reply bubble is **confirmed delivered** by the chat client do background worker threads launch daily semantic extraction (`daily_semantics`) and inner-state / impression / DMN spark updates (`inner_state` + `impression` + `spark`). Interrupted or failed deliveries never write undelivered replies into memory.
