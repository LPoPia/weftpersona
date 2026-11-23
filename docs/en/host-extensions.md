# Host Runtime Capabilities & Modification Guide

> **Reference Host Baseline**: The Sanae production `hermes-agent` branch is based on upstream [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) `main` at **`e864515d2`** (2026-10-30, merged forward from `0.22.2` / `ab44d8ac`; all host file paths below match this baseline).

WeftPersona (`sanae_mem`) keeps **memory metabolism, affective dynamics, and review logic** inside the plugin, while delegating **trusted turn identity, confirmed delivery receipts, outbound wire isolation, and group participation control** to the host runtime.

Stock [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) exposes standard `MemoryProvider` and `PluginContext` hooks, which do **not** include those execution boundaries. This document states objectively:
1. What works and what is absent or degraded on unmodified upstream Hermes;
2. Exactly where and how to extend the host if you want all mechanisms active.

---

## 1. Actual Behavior on Unmodified Upstream Hermes

All host capability checks are centralized in [`plugins/sanae_mem/infra/host_bridge.py`](../../plugins/sanae_mem/infra/host_bridge.py). On stock Hermes, the plugin loads and runs, with the following boundaries:

| Mechanism | Behavior on Stock Hermes | Reason |
|---|---|---|
| **SQLite / Chroma episodic store, forgetting curve, memes, basic recall** | Works normally | Relies only on standard `MemoryProvider` hooks and local storage. |
| **Post-delivery settlement (preventing pre-send writes)** | **Inactive** (logs `delivery receipt hook skipped`) | Stock Hermes has no `register_delivery_receipt_handler` to emit transport-confirmed `DeliveryReceipt` events or defer `sync_turn` until after delivery. |
| **4-layer prompt ordering & ephemeral wire isolation** | **Unavailable** | Stock Hermes has no `CognitivePromptPolicy` hook and does not isolate `pre_llm_call` `ephemeral: True` blocks onto the outbound wire copy without persisting them into conversation history. |
| **Hard pre-write gate on resident memory & skills** | **Not enforced by host** | Stock `memory_tool` and `skill_manager_tool` do not consult `MemoryMutationGate` before mutating `MEMORY.md`, `USER.md`, or skills. |
| **Group passive listening, reply pacing & stale-reply discard** | **Unavailable** | Stock gateway lacks `GroupContextPolicy`, `GroupReplyPolicy`, and `InboundMergePolicy`. |
| **Plugin cron hot-sync & slow-channel delivery ledger** | **Unavailable** | Stock `cron` lacks `sync_plugin_cron_job` and transport-confirmed delivery bookkeeping for hot toggling with `shadow` mode. |
| **Multi-HOME isolation & `T:` test isolation** | **Partially limited** | Several stock paths resolve against a single default root, and `hermes_test_isolation` is absent. |

---

## 2. How to Extend Hermes (Exact Integration Points)

No persona names, user identities, or cognitive business rules belong in the host. Every host extension is generic infrastructure grouped into **6 seams** (full reference implementation touches ~50+ host files; for 1-on-1 DM use only, seams 1–4 are sufficient):

### 1. Add a Generic Host Contract Registry (`hermes_host_contracts.py`)
Create a domain-neutral contract module at the host root backed by `ContextVar` and process-wide registries:
- **Trusted identity**: `TurnEnvelope` (platform, chat ID, chat type, sender ID, cron/child/test-isolation flags—populated by transport/scheduler, never inferred from text) and `CronExecutionEnvelope`;
- **Policy & receipt registration**: `register_cognitive_prompt_policy`, `register_memory_mutation_gate`, `register_inbound_merge_policy`, `register_group_context_policy`, `register_group_reply_policy`, `register_delivery_receipt_handler`, `register_cron_completion_handler`.

### 2. Wire Trusted Identity & Confirmed Delivery Receipts into the Gateway
- **Host files**: `gateway/session_envelope.py`, `gateway/run_turn.py`, `gateway/platforms/base.py`, `gateway/delivery_ledger.py`, `agent/memory_manager.py`
- **Changes**:
  1. Bind a `TurnEnvelope` built from verified platform metadata before running each turn;
  2. Support `sync_after_delivery` in `MemoryManager` so turns handled by delivery receipts skip pre-send `sync_turn`;
  3. After the platform adapter confirms outbound delivery, construct a `DeliveryReceipt` (`turn_id`, `incoming_message_id`, `delivered_text`, actual `transcript`, `transport_confirmed=True`) and dispatch it to registered handlers.

### 3. Wire 4-Layer Prompt Ordering & Ephemeral Wire Isolation
- **Host files**: `agent/system_prompt.py`, `agent/turn_context.py`, `agent/conversation_loop.py`
- **Changes**:
  1. Consult `CognitivePromptPolicy` when ordering `MEMORY` / `USER` / `BEHAVIOR.md` / `SOUL.md` and deciding turn compaction;
  2. Run `turn_start` before `pre_llm_call`;
  3. When `pre_llm_call` returns `{"context": ..., "ephemeral": True}`, inject it **only into a shallow outbound wire copy** of the request messages—never persist it into session history or state DBs.

### 4. Wire Pre-Write Mutation Gating into Memory & Skill Tools
- **Host files**: `tools/memory_tool.py`, `tools/skill_manager_tool.py`, `tools/write_approval.py`
- **Changes**:
  - Before writing `MEMORY.md` / `USER.md` or mutating skills, invoke `MemoryMutationGate` and honor its allow/reject/reroute decision.

### 5. Wire Group Participation Control & Burst Merging (Group Chats Only)
- **Host files**: `gateway/session_ingress_burst.py`, `gateway/group_reply_control.py`, `gateway/group_reply_history.py`, `gateway/passive_group_context.py`, `gateway/platforms/base_group_delivery.py`
- **Changes**:
  1. Delegate burst-message coalescing to `InboundMergePolicy` (guarding against self-deadlock when a session is busy);
  2. Query `GroupContextPolicy` on inbound group messages to capture passive unmentioned context into authorized group buffers;
  3. Query `GroupReplyPolicy` before generation and before send to enforce cooldowns, discard stale replies when the channel moved on, and preserve platform reply references.

### 6. Cron Sync, Provider Failover, Dynamic `HERMES_HOME` & Atomic I/O (Operations)
- **Cron** (`cron/jobs.py`, `cron/scheduler.py`, `cron/scheduler_delivery.py`): expose `sync_plugin_cron_job` / `list_plugin_cron_jobs` / `pause_plugin_cron_job`, wait for real delivery confirmation on slow channels, and emit `CronCompletion`;
- **Provider failover** (`agent/auxiliary_client.py`, `agent/provider_failover_state.py`): support `allow_fallback` / `no_main_fallback` for auxiliary slots and endpoint-level health isolation;
- **Multi-HOME & test isolation** (`hermes_test_isolation.py`, gateway path lookups): resolve paths dynamically via `get_hermes_home()` and propagate `T:` test-isolation flags;
- **Atomic file I/O & child JSON format**: standalone `git apply` patches are included in [`patches/hermes-agent/README_en.md`](../../patches/hermes-agent/README_en.md).

---

## 3. Minimal Subset for 1-on-1 DM Deployments

If you only run **1-on-1 private chats** without multi-platform groups or dynamic cron jobs, you only need **Steps 1–4** (minimal `TurnEnvelope`, delivery receipts, `ephemeral` wire isolation, and `MemoryMutationGate`). Refer to [`plugins/sanae_mem/infra/host_bridge.py`](../../plugins/sanae_mem/infra/host_bridge.py) for the exact signatures and fields consumed by the plugin.
