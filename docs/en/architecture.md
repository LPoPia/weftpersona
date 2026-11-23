# WeftPersona System Architecture & Turn Lifecycle

> 🌐 **Language**: [**简体中文**](../architecture.md) | **English**

This document describes the component contracts, data source-of-truth division, and single-turn execution flow of **WeftPersona (`sanae_mem`)** on the Hermes host runtime. For the underlying cognitive psychology and pragmatic design rationale, see [`how-to-simulate-a-living-person.md`](how-to-simulate-a-living-person.md).

---

## 1. Core Components & Host Hook Contracts

| Component / Module | Entry File | Exposed Tools | Host Hooks & Responsibility Contract |
|---|---|---|---|
| **Memory & Mood Main Entry** | [`plugins/sanae_mem/plugin_entry.py`](../../plugins/sanae_mem/plugin_entry.py) | `sanae_mem_add`, `sanae_mem_search`, `sanae_mem_forget`, `sanae_mem_compact`, `sanae_mem_marked_promoted`, `sanae_mem_distill`, `sanae_mem_report_false_trigger` | Registers `MemoryProvider` (`sync_after_delivery = True`); `system_prompt_block` injects the rendered `BEHAVIOR.md` block; `pre_llm_call` runs concurrent gated recall and meme recognition; `pre_tool_call` enforces resident memory and controlled-file write guards |
| **Forgetting Curve & 3-Stage Collapse** | [`plugins/sanae_mem/storage/curve.py`](../../plugins/sanae_mem/storage/curve.py), [`event_maintenance.py`](../../plugins/sanae_mem/storage/event_maintenance.py) | Triggered via `sanae_mem_compact` | Cross-day (`≥ 12h`) spaced reactivation extends half-life via a power law (capped at 120 days); automatically transitions untouched events through **Vivid Verbatim → `gist_faded` (archives raw quotes before retaining only gist) → `dormant` (deep dormancy) or Cold Archive** |
| **Resident Memory Metabolism & Guard** | [`plugins/sanae_mem/governance/resident.py`](../../plugins/sanae_mem/governance/resident.py), [`llm_guard.py`](../../plugins/sanae_mem/governance/llm_guard.py) | Intercepts host `memory` tool writes | Tracks last conversational touch per resident entry; when capacity `≥ 85%`, identifies cold entries untouched for 45 days for demotion back into episodic storage; uses a fast model at write-time to intercept meta-rules, one-off trivia, and self-criticism, diverting them to short-term inner thoughts or neutral short-term events |
| **Affective Dynamics (Fast Variables)** | [`plugins/sanae_mem/cognitive/inner_state.py`](../../plugins/sanae_mem/cognitive/inner_state.py), [`inner_wiring.py`](../../plugins/sanae_mem/cognitive/inner_wiring.py) | `sanae_inner_state` | Asynchronously updates 1–3 sentences of first-person `[In My Heart Right Now]` (`[我此刻心里]`), continuous slow variables (`tension` / `closeness` / `energy`), pending verbatim promises (`recent_words`), unspoken associative bridges (`unspoken_link`), and Artificial DMN distant collisions (`spark`) after private reply delivery or trusted personal `cron` completion; decays naturally across real elapsed time and preserves directional warmth (`24h` window) across long sleep boundaries |
| **Holistic Impression (Slow Variable)** | [`plugins/sanae_mem/cognitive/impression.py`](../../plugins/sanae_mem/cognitive/impression.py) | `sanae_impression` | Piggybacks on the post-delivery Flash call; revises `[How I See Him Lately]` (`[我最近眼里的他]`) slowly only when sustained situational or relational changes are backed by direct user quotes; reads the latest revision in real time inside `Tier 2` `[The World in My Eyes]` without breaking the `Tier 0` system prompt KV prefix cache |
| **Daily Context & Curiosity Queue** | [`plugins/sanae_mem/cognitive/daily_context.py`](../../plugins/sanae_mem/cognitive/daily_context.py), [`daily_semantics.py`](../../plugins/sanae_mem/cognitive/daily_semantics.py) | Maintained automatically via internal hooks | Extracts current-day updates (`DAILY_FACT`), active constraints (`ACTIVE_CONSTRAINT`), and 2-day hard-expiring curiosity threads (`OPEN_THREAD`) |
| **Pending Commitments** | [`plugins/sanae_mem/domains/commitments.py`](../../plugins/sanae_mem/domains/commitments.py), [`ingest.py`](../../plugins/sanae_mem/runtime/ingest.py) | Post-delivery internal hook | Creates, fulfills, or cancels commitments from the current delivered turn under the trusted owner; once completion is confirmed, archives before resolving and records the completed commitment as that owner's mid-term event |
| **Memes & Sticker Memory** | [`plugins/sanae_mem/domains/memes.py`](../../plugins/sanae_mem/domains/memes.py) | `sanae_meme_save/search/update/delete` | Stores text memes and original sticker bytes (SHA-256 deduplication and instant inbound recognition without vision LLM calls); prepares 300px outbound thumbnails while preserving GIF frame rates and transparency |

---

## 2. Source-of-Truth Data Division (Relative to `$HERMES_HOME`)

| Data Layer | Default Path | Role & Lifecycle |
|---|---|---|
| **Episodes, Mood, Impressions & Memes** | `sanae-mem/sanae.db` | Single authoritative SQLite source of truth: stores episodic events (`events` / `event_occurrences`), first-person inner-state revision chains (`inner_state_revisions`), impression revisions (`impression_revisions`), daily updates (`daily_context`), commitments (`commitments`), videos (`video_watches`), and memes (`meme_records`) |
| **Vector & Full-Text Search Indices** | `sanae-mem/chroma/` + SQLite FTS5 | Derived retrieval indices (local `bge-small-zh-v1.5` embeddings), fully rebuildable from `sanae.db` at any time |
| **Resident Long-Term Memory** | `memories/MEMORY.md`, `memories/USER.md` | Session-level snapshot injection: holds only cross-month foundational facts and enduring relational knowledge |
| **Procedural Policy & Subjectivity Kernel** | `BEHAVIOR.md`, `SOUL.md` | [`BEHAVIOR.md`](../../prompts/BEHAVIOR.md) is front-loaded in the system prompt; [`SOUL.md`](../../prompts/SOUL.md) and the character persona are anchored at the very end |
| **Pre-Deletion Cold Archive** | `memory-archive/` | Mandatory cold archive written before any event forgetting, verbatim collapse, or resident demotion (deletion aborts if archiving fails) |

---

## Schema Lifecycle and Shared Foundations

Schema v4 records confirmed maintainer accounts in `maintainer_account_bindings`. Verified DMs write to the same private owner and can read the linked accounts' earlier records. Recognizing a bound speaker in a group preserves the group's own read scope. Schema v5 adds `source_context` to `impression_revisions` and `commitments`. Older rows retain their owner, with unknown provenance left empty.

The host supplies the trusted conversation, speaker and delivery result. ViveMind reads the active profile's `group_context` for shared history, long-term memory and unaddressed replies. The current speaker's impression accompanies only the current model request. Once delivery succeeds, the speaker's own words and the delivered reply supply the evidence for group events, impressions and commitments.

`database.py` and `schema.py` own installation and versioned upgrades. Existing databases receive a recoverable snapshot before upgrade; completed upgrades leave the normal runtime path. Obsolete mood, impression-session-view and legacy commitment-queue tables leave the live schema. The spark cooldown clock lives in SQLite with revision CAS and survives consumption of its displayed text.

`infra/archive.py`, `infra/clock.py` and `infra/model_output.py` own durable append, the 04:00 sleep day and model-object parsing. `infra/status_store.py` updates counters and observations under one lock; compound operations submit only their changes. The plugin's `infra/atomic_io.py` owns file locking and atomic publication.

## 3. Single-Turn Execution Flow

1. **Inbound Aggregation & Context Binding**:
   - The gateway merges rapid consecutive text/image messages from the same sender and computes the real wall-clock delta since the last interaction; known stickers match directly via raw-byte SHA-256 hashes, bypassing multimodal vision latency.
2. **Pre-LLM Dynamic Read (`pre_llm_call` & `current_turn_context`)**:
   - Loads `[Daily Context]` on session start or after a long pause, plus unconsumed cross-channel self-utterances (`[What I Just Said Elsewhere]`);
   - Triggers context-gated episodic recall (casual banter follows the Effort Prior and skips recall; when recall is warranted, executes query planning and hybrid search within a strict 3-second budget—timing out cleanly rather than falling back to noisy keyword matching);
   - Waits up to 3 seconds for any in-flight post-delivery `[In My Heart Right Now]` update from the previous turn (including autonomous activities completed since the last reply) and injects it into the current turn's transient context without polluting persistent history `api_content`.
3. **"Engineering Front-Loaded, Soul Tail-Anchored" Prompt Assembly (`wire_system_prompt`)**:
   - Front fixed prefix: Skill index → `MEMORY.md` / `USER.md` → rendered `BEHAVIOR.md` block → environment timestamp;
   - Tail immediately preceding dialogue: `SOUL.md` → character persona; the latest impression is read in each turn’s dynamic context.
4. **Tool-Call Guardrails (`pre_tool_call`)**:
   - When the main model calls `memory` to write resident memory, `llm_guard.py` inspects the write: heat-of-the-moment frustration, mind-reading guesses, or self-imposed communication rules are silently diverted into short-term inner thoughts or neutral short-term episodes;
   - `write_file` / `patch` calls are checked against canonical real paths for controlled memory files (`MEMORY.md`, `USER.md`, `BEHAVIOR.md`), preventing direct file-tool overwrites that bypass the `memory` tool.
5. **Delivery Confirmation & Asynchronous Post-Turn Metabolism (`sync_turn` & `update_inner_state`)**:
   - Only after the reply bubble is **confirmed delivered** by the chat client do background worker threads launch daily semantic extraction (`daily_semantics`) and inner-state / impression / DMN spark updates (`inner_state` + `impression` + `spark`). Interrupted or failed deliveries never write undelivered replies into memory.
