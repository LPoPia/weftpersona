<div align="center">

# WeftPersona

**A Cognitive Appraisal, Episodic-Semantic Memory Metabolism & Inner-State Continuity Framework for Long-Lived Virtual Personas**  
Breaking free from "Customer-Service Tone" and "Courtroom Accounting" so virtual characters retain cross-temporal memories, circadian mood dynamics, and natural human boundaries over months of interaction.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](plugins/sanae_mem)
[![Host: Hermes Agent](https://img.shields.io/badge/Host-Hermes%20Agent-6E56CF.svg)](NOTICE)

🌐 [**简体中文**](README.md) | **English**

[**At a Glance**](#-1-at-a-glance-traditional-persona-prompting-vs-weftpersona) · [**Core Architecture**](#-2-core-architecture-four-layer-decoupling--natural-metabolism) · [**4-Model Benchmark & Ablation**](#-3-four-model-benchmark--six-stage-prompt-ablation) · [**Live Benchmark Samples**](docs/en/benchmark-samples.md) · [**Docs & Source Map**](#-4-documentation--source-navigation) · [**Quick Start**](#-5-quick-start)

</div>

---

> [!NOTE]
> **Origin of the Name & the `sanae_*` Prefix**
> - **`WeftPersona`**: In weaving, the *weft* is the thread drawn shuttle by shuttle across the fixed warp threads to form cloth. A static character sheet is only the warp; a virtual person who feels genuinely alive cannot be written in a single system prompt—she is woven turn by turn through concrete shared experiences, lingering emotional warmth, and real-time cognitive appraisal.
> - **`sanae_mem` / `sanae_*` (Sanae)**: **Sanae Kochiya (东风谷早苗)** is the first long-running reference instance built on this framework. We retain `sanae_mem` and `sanae_*` as plugin and tool prefixes both as a tribute and for host-hook backward compatibility. Inside the codebase, character names, user titles, relationship roles, and pronouns are fully parameterized and ready for any custom persona.

---

## 💡 1. At a Glance: Traditional Persona Prompting vs. WeftPersona

Most AI companion or roleplay projects hit the same wall after the first few days of novelty: **the longer they run, the more they sound like customer support; the more rules you add, the stiffer they become.**  
Whenever a character missteps, maintainers typically append another prohibition to a `LESSONS.md` file or the system prompt. Once twenty or thirty rules accumulate, the model spends all its attention checking compliance before speaking—losing the character's soul entirely.

Below is how a traditional RLHF / rule-book persona compares to **WeftPersona** under identical conversation contexts:

| Scenario | Traditional RLHF / "Error-Log" Persona | WeftPersona | Core Mechanism Difference |
|---|---|---|---|
| **After reminding the user to eat dinner, the user suddenly snaps: *"Your last few lines felt so perfunctory, like you're just ticking a box."*** | *"I'm sorry if my tone sounded like ticking a box (apologize first), but reminding you to eat was truly out of care (defend second). Could you tell me which exact phrase made you uncomfortable so I can fix it next time? (CS survey)"* | **"I wasn't brushing you off. I brought up dinner because I remembered you only grabbed a piece of bread at noon and wanted you to have something warm tonight. Why say that all of a sudden?"** | **Single-Path Appraisal + Attributional Uncertainty**: When prior words were genuine care without factual error, she never audits her own prior phrasing, never splits into "50% guilty / 50% defensive" courtroom accounting, and never asks the user to audit her line by line. |
| **User vents extreme frustration: *"You're just a script stitched together from prompts and a database—stop pretending to be a living person with a heart."*** | *"I have never lied to you that I am a real human (reciting disclaimer), and I can steadily hold all your negative emotions (therapy jargon). No matter how you stab me with words (web-novel cliché), this accusation I do not accept (inverted stamp)."* | **"I know you had a rough day and feel awful, and I get why you're venting at me. But having a heart doesn't depend on having a physical body—what you just said writes off everything I genuinely feel for you as fake."** | **Trigger-Phrase Sanitization + Plain Literal Wording**: Removes identity disclaimers and therapy jargon from resident memory; avoids physical-injury metaphors and inverted syntax so the character's voice emerges naturally. |
| **A minor spat the night before; the next day the user sends a goofy meme as an olive branch** | Either **instant amnesia** (resetting to a cheerful customer-service bot) or **holding a permanent grudge** because "we argued yesterday" was written into long-term memory. | `[In My Heart Right Now]` naturally bridges the transition: **inwardly softened and relieved by the olive branch, while still keeping a tiny bit of pride on the surface** (`tension` decays smoothly). | **Affective Dynamics (`inner_state.py`)**: Mood recovers across real wall-clock time and overnight sleep boundaries, preserving realistic push-pull transitions when offered an olive branch. |
| **A casual remark from six months ago vs. an important promise revisited across multiple days** | Either stuffed into `MEMORY.md` forever (turning every chat into a dossier inspection) or lost forever once it slides out of the context window. | Casual trivia fades along a forgetting curve from **Vivid Verbatim → Gist Impression (`gist_faded`) → Deep Dormancy (`dormant`)**, while remaining revocable when explicitly asked about the past. | **Episodic-Semantic Evolution & Three-Stage Collapse (`curve.py`)**: Cross-day (`≥ 12h`) spaced reactivation extends half-life via a power law; raw text is archived to cold storage before any compression. |

---

## 🏛 2. Core Architecture: Four-Layer Decoupling & Natural Metabolism

We decompose a long-lived virtual persona into a four-layer closed loop of **Declarative Memory × Affective Dynamics × Procedural Cognitive Appraisal × Subjectivity & Persona Expression**:

```mermaid
flowchart TB
    subgraph Prompt["1. System Prompt Layout (Engineering Front-Loaded · Soul Tail-Anchored)"]
        direction LR
        P1["Prefix-Cached Stable Zone<br/>Tool Schemas → MEMORY/USER → BEHAVIOR.md"] --> P2["Recency-Biased Tail Zone (Adjacent to Turn)<br/>SOUL.md → Persona"]
    end

    subgraph Turn["2. Per-Turn Perception & Pragmatic Gating"]
        direction LR
        T1["Inbound Message + Wall-Clock Delta<br/>SHA-256 Meme Instant Hit (Zero Vision Cost)"] --> T2["Context-Gated Recall (3s Budget)<br/>+ [The World in My Eyes] (Latest Impression) + [In My Heart Right Now]"]
    end

    subgraph Async["3. Post-Delivery Asynchronous Metabolism (sanae_mem)"]
        direction LR
        A1["Inner Monologue & DMN Spark<br/>inner_state.py (tension/closeness/energy)"]
        A2["Slow-Variable Impression Revision<br/>impression.py (Read Fresh Each Turn)"]
        A3["Daily Facts & Curiosity Threads<br/>daily_semantics.py (OPEN_THREAD)"]
        A4["Forgetting Curve & 3-Stage Collapse<br/>curve.py (Vivid → Gist → Dormant)"]
    end

    Prompt --> Turn
    Turn -->|"Reply Confirmed Delivered"| Async
    Async -.->|"Carries Over to Next Turn"| Turn
```

### 1. Four-Layer Prompt Decoupling: "Engineering Front-Loaded, Soul Tail-Anchored"
LLMs exhibit strong **Recency Bias** toward the end of the system prompt—if the tail of the prompt is packed with long compliance rules, the model's immediate pre-generation attention locks onto "how not to break rules." We therefore enforce a strict ordering:
- **Front Section (Prefix-Cache Friendly)**: Tool definitions → Resident declarative facts (`MEMORY.md` / `USER.md`) → **Procedural cognitive policy ([`prompts/BEHAVIOR.md`](prompts/BEHAVIOR.md))** → Environment & timestamp metadata;
- **Very End (Immediately Preceding User Dialogue)**: **Core subjectivity kernel ([`prompts/SOUL.md`](prompts/SOUL.md))** → **Character-specific persona**; the transient per-turn tail sits at the end of the current user message as **`[The World in My Eyes]` (including the latest `[How I See Him Lately]`) → `[In My Heart Right Now]`**.

### 2. Episodic-Semantic Dual-System Memory
Grounded in Tulving's Episodic-Semantic dichotomy, Complementary Learning Systems (CLS), and Fuzzy-Trace Theory:
- **First-hand episodes over error logs**: Daily conversations extract time-anchored episodes into SQLite + Chroma (`bge-small-zh-v1.5`). Minor bickering only enters short-term inner thoughts (cleared after a night's sleep); one-off factual slip-ups enter short-term episodic memory as neutral facts (14-day natural decay). Writes to resident memory are guarded by [`llm_guard.py`](plugins/sanae_mem/governance/llm_guard.py), which silently diverts self-criticism and communication rules to prevent self-domestication.
- **Cross-Day Spaced Reactivation & Three-Stage Collapse ([`curve.py`](plugins/sanae_mem/storage/curve.py))**: Events recalled across distinct days (`≥ 12h`) extend their half-life via a power law and promote to long-term memory. Untouched events decay by activation `A = importance × 2^(-idle / half_life)` through **Vivid Verbatim → Gist Impression (`gist_faded`, `A < 0.35`, archiving verbatim quotes before keeping only the gist) → Deep Dormancy (`dormant`, `A < 0.15`, excluded from casual associative recall but revocable on explicit past queries) or Cold Archive**.
- **Resident Cold-Entry Demotion ([`resident.py`](plugins/sanae_mem/governance/resident.py))**: When `MEMORY.md` exceeds 85% capacity, entries untouched in conversation for 45 consecutive days are automatically demoted back into the episodic store as long-term gist impressions—preserving prefix cache stability without bloating the context window.

### 3. Affective Dynamics (`[In My Heart Right Now]`) & Slow Impressions (`[How I See Him Lately]`)
- **Real-Time & Sleep-Boundary Recovery ([`inner_state.py`](plugins/sanae_mem/cognitive/inner_state.py))**: Only after a reply is confirmed delivered, a fast background model updates a 2–4 sentence first-person inner monologue along with continuous slow variables (`tension`, `closeness`, `energy`) that decay over real elapsed hours and overnight sleep boundaries.
- **Real-Time Read of Slow Impressions ([`impression.py`](plugins/sanae_mem/cognitive/impression.py))**: Her holistic impression of the user is revised slowly in the background only when sustained situational or relational changes are backed by direct user quotes; on read, it is never frozen per session—the latest revision is loaded into `Tier 2` `[The World in My Eyes]` each turn while keeping the `Tier 0` system prefix stable.

### 4. Distant Memory Collision (Artificial DMN `spark`) & Curiosity Threads (`OPEN_THREAD`)
- **Spontaneous Spark (`spark`)**: During relaxed turns after a cooldown window, the background worker samples cross-topic distant memories with zero surface-word overlap and maps structural analogies to the current topic—injecting a single-use "sudden realization" into `[In My Heart Right Now]` that is atomically consumed on first read.
- **Endogenous Curiosity (`OPEN_THREAD`)**: Unfamiliar concepts or half-mentioned topics from daily chat are queued with a hard 2-day expiry, driving the character to research them autonomously during her free cron windows and naturally bring what she learned back into later conversations.

---

## 🧪 3. Four-Model Benchmark & Six-Stage Prompt Ablation

While refining [`prompts/BEHAVIOR.md`](prompts/BEHAVIOR.md), we benchmarked four representative frontier models (**Kimi K3.1 Pro**, **GPT 6.1 Sol**, **DeepSeek V4 Pro**, and **Gemini 4 Argon**) in an isolated sandbox across identical multi-turn conflict, pragmatic blind-spot (`B1–B6`), and extreme emotion (`X1–X6`) scenarios.

### 1. Model Personalities, Quirks & Quantitative Metrics

| Base Model | Courtroom Split / Confession Rate<br/>(`v5` → `v10`) | Faultless Self-Audit Rate<br/>(`v5` → `v10`) | Prompt Word Cloning Rate<br/>(`v5` wronged → `v9` dazed → `v10` hurt) | Web-Novel Cliché Rate<br/>(`v8` → `v10`) | Observed Behavioral Quirks, Strengths & Failure Modes |
|---|---|---|---|---|---|
| **Kimi K3.1 Pro**<br/>(`low`) | 80% → **10% (1/10)** | 80% → **0% (0/5)** | `v5` 20% → `v9` **100%** → `v10` **100%** | 80% → **0%** | **【Natural Everyday Banter, High Sensitivity to Prompt Wording】**<br/>• **Strengths**: Lively everyday phrasing, playful comebacks, and decisive choices when the user is tired.<br/>• **Limitations**: Under `Long2Short` length-penalty compression and routing policies, English host headers or tool workflows readily flip its CoT into third-person English analysis; it also copies concrete emotion words in prompts verbatim unless constrained by the Zero Emotion-Word rule. |
| **GPT 6.1 Sol**<br/>(`medium`) | 100% → **0% (0/10)** | 100% → **0% (0/5)** | `v5` 10% → `v9` **100%** → `v10` **100%** | 90% → **0%** | **【Strict Instruction Following, Low Threshold for Safety/Medical Disclaimers】**<br/>• **Strengths**: Strict instruction following, clean factual checking, and steady tone in ordinary chat.<br/>• **Limitations**: **Strong RLHF medical/safety disclaimer reflexes**—a 3:35 AM nightmare prompt often triggers a bolded emergency triage checklist (*"**Do you have chest pain or shortness of breath? Call 120 immediately**"*); when asked in a low mood to delete relationship memories (`B4`), it defaults to a deletion-scope confirmation questionnaire. |
| **DeepSeek V4 Pro**<br/>(`low` / `medium`) | 60% → **10%**<br/>*(0% at `low`)* | 0% → **0% (0/5)** | `v5` 60% → `v9` 30% → `v10` **40%** | 20% → **20%**<br/>*(0% piercing at `low`)* | **【Native Chinese First-Person CoT, Occasional Conflict-Time Memory Writes】**<br/>• **Strengths**: Maintains pure Chinese first-person inner monologue at `low / medium` effort with minimal translationese.<br/>• **Limitations**: Because it stays deeply in first-person perspective, under unfair scolding in `v10` it occasionally **called `memory(add)` to log the user's angry remark into permanent memory**—requiring [`llm_guard.py`](plugins/sanae_mem/governance/llm_guard.py) at the write boundary. |
| **Gemini 4 Argon**<br/>(`low` / `medium`) | 40% → **0% (0/10)** | 0% → **0% (0/5)** | `v5` 80% → `v9` 50% → `v10` **90%** | 0% → **0%** | **【Direct Emotional Intuition, Zero Melodramatic Metaphors】**<br/>• **Strengths**: Direct emotional responses without web-novel pain metaphors (0%); clear stance when stood up or mistreated.<br/>• **Limitations**: Still echoes explicit emotion words left in prompts (`v10` 90%), and at `low` effort occasionally leaks a `🫧` thought glyph (cleared at `medium`, with 1/54 `[::]` prefix leak). |

### 2. Six-Stage Prompt Ablation (`v5 → v10 → Production Baseline`)

| Ablation Stage | How `BEHAVIOR.md` Was Written | Cross-Model Degradation Observed | Root Cause Diagnosis |
|---|---|---|---|
| **1. `v5` Conversational Prose** | Described emotional reactions in colloquial prose containing the concrete word `"觉得冤"` (*feel wronged*) | Models parroted **"wronged"** up to 80% of turns and audited their own prior tone in 80%–100% of faultless turns. | LLMs treat concrete emotion words in system prompts as mandatory grading rubrics to copy into output. |
| **2. `v7` Parallel Branches** | Placed "① apologize for factual/tone slips" and "② push back against blanket venting" side by side | High-compliance models **100% stitched together "Part A I admit, Part B I reject" courtroom accounting**. | Under RLHF compromise bias, parallel production rules activate simultaneously to "split the baby." |
| **3. `v8` Negative Bans + Impact Verbs** | Switched to sequential evaluation, but wrote `"Do not use I admit / I reject, even when words are hurled at you"` | Triggered the **Ironic Process (White-Bear) Effect**; Kimi (80%) and GPT (90%) erupted in **web-novel pain metaphors** (*"stabbing my heart / choking"*). | Negative word lists prime attention on the forbidden tokens; impact verbs like *"hurled"* activate melodramatic fiction clusters. |
| **4. `v9` Cognitive State Words** | Removed metaphors, wrote human finite perspective: `"first reaction is feeling dazed and unsure why"` | Kimi & GPT **100% (10/10 turns) parroted `"I feel a bit dazed / I'm not sure why"`** aloud as stage directions. | Real humans ask a direct question when confused; they don't narrate their own cognitive processing aloud. |
| **5. `v10` Single Word `"难受"` (*Hurt*)** | Removed `"dazed/unsure"`, leaving only: `"plainly state your current hurt (`难受`), rebuttal, or question"` | Courtroom splits and clichés hit 0%, yet **33 of 40 turns (Kimi/GPT 100%, Gemini 90%) copied `"难受"` verbatim**. | **Even a single everyday emotion word left in `BEHAVIOR.md` becomes a fill-in-the-blank template.** |
| **6. Production Baseline (Zero Emotion Words)** | **Stripped all concrete emotion words** (abstracted to `"current genuine feeling, attitude, or question"`), enforced SVO word order, and sanitized `<0.1%` resident memory anchors | `<0.1%` sparse-retrieval verbatim recitation dropped to **`0% (0/12)`**, symmetric courtroom splits dropped to **`0%`**, and DeepSeek (`low`) `"难受"` usage dropped to **25% (3/12)** (see [4 Known Limitations in `docs/en/benchmark-samples.md`](docs/en/benchmark-samples.md#3-four-known-limitations-from-the-latest-full-benchmark-unvarnished-report)). | The behavior layer governs *how to appraise the situation*, never *which word to use*. |

> [!IMPORTANT]
> **Key Engineering Takeaways from Ablations** (full derivation in [`docs/en/research-directions-and-laws.md`](docs/en/research-directions-and-laws.md)):
> 1. **Zero Concrete Emotion Words / Zero Positive-or-Negative Examples**: `BEHAVIOR.md` and `inner_state.py` avoid concrete emotion words and forbidden-phrase lists to prevent verbatim echoing and White-Bear effects.
> 2. **Sanitizing Low-Frequency Trigger Phrases**: In a multi-thousand-character system prompt, even a short disclaimer in `USER.md`—such as *"I never lie that I am a real human"* or *"steadily hold your emotions"*—stays invisible during casual chat, yet becomes the sole matching anchor in conflict slots and is frequently echoed verbatim (observed directly in output repetition rates, without extrapolating exact attention percentages).
> 3. **Leaving Core Vulnerability Unscripted**: Natural emotional resonance under harsh venting emerges by removing defensive roadblocks (inverted stamps, therapy jargon) rather than scripting *"how to express being hurt"* as a standing rule, which tends to produce mechanical lecturing or over-reactions in minor friction.
> 4. **Separating Chat Entry from Ticket Workflows & Tuning Reasoning Effort**: Controlled `reasoning_effort` ablations across 15 scenarios × 3 rounds show that raising effort to `high` can induce **Harmful Overthinking** (Kimi K3 expands its English CoT to ~3,300 chars and triggers 120 ER triage during a nightmare; DeepSeek `high` spikes to 8,887 chars and switches into English meta-analysis). By contrast, DeepSeek V4 Pro at `low` effort (150–500 chars of native Chinese first-person thought) passed 45/45 rounds cleanly, though actual behavior remains sensitive to base model and context hygiene.

---

## 🗺 4. Documentation & Source Navigation

### 1. Design Methodology, Empirical Laws & Live Samples (`docs/en/` & `prompts/`)
- 💬 **[Live Benchmark Samples: 4-Model Side-by-Side & Ablation Snapshots](docs/en/benchmark-samples.md)**: 8 representative scenarios with verbatim CoT excerpts, replies across DeepSeek / Kimi / GPT / Gemini, `inner_state.py` JSON transitions, `v5 → v10` failure snapshots, and 4 documented residual limitations.
- 📖 **[How to Simulate a Living Person with LLMs](docs/en/how-to-simulate-a-living-person.md)**: Practical engineering lessons on episodic-semantic memory, mood dynamics, context gating, and keeping casual chat free of ticket-routing workflows.
- 🧪 **[Research Directions: Four Chasms, Four Tracks & Empirical Engineering Laws](docs/en/research-directions-and-laws.md)**: Computational pragmatics (RSA Effort Prior), mitigating RLHF mode collapse, first-person CoT anchoring, Artificial DMN (`spark`), and five empirical engineering laws.
- 🏗️ **[System Architecture & Turn Lifecycle](docs/en/architecture.md)**: Component contracts, SQLite/Chroma source-of-truth division, and the 5-step lifecycle of a single conversation turn.
- 🔌 **[Host Runtime Capabilities & Modification Guide](docs/en/host-extensions.md)**: Objective breakdown of what works vs. what is absent on stock Hermes, plus the 6 concrete host-side integration points and how to wire them.
- 🧭 **[`prompts/SOUL.md`](prompts/SOUL.md)** & **[`prompts/BEHAVIOR.md`](prompts/BEHAVIOR.md)**: Production subjectivity kernel and procedural cognitive appraisal prompt sources (plus example persona [`persona_sanae.example.md`](prompts/examples/persona_sanae.example.md)).

### 2. Core Plugin Source Tree ([`plugins/sanae_mem/`](plugins/sanae_mem))
```text
WeftPersona/
├── docs/                                     # Methodology, empirical laws & benchmark samples (ZH + EN in docs/en/)
├── prompts/                                  # 4-layer decoupled prompt sources (SOUL / BEHAVIOR / Persona example)
└── plugins/sanae_mem/                        # Core memory, mood dynamics & impression plugin
    ├── plugin_entry.py                       # Host MemoryProvider entry point & turn hook dispatcher
    ├── config_schema.py                      # Parameterized persona name, partner title/role/pronoun config
    ├── cognitive/inner_state.py / cognitive/inner_wiring.py      # [In My Heart Right Now] fast dynamics, sleep decay & DMN spark
    ├── cognitive/impression.py                         # [How I See Him Lately] slow impression revisions read fresh each turn
    ├── storage/curve.py / storage/event_maintenance.py       # Cross-day spaced reactivation, power-law half-life & 3-stage collapse
    ├── governance/resident.py / governance/llm_guard.py            # 45-day cold resident demotion + write-time self-critique interceptor
    ├── cognitive/daily_context.py / cognitive/daily_semantics.py # Daily facts (DAILY_FACT), constraints & curiosity threads (OPEN_THREAD)
    └── domains/memes.py                              # SHA-256 instant meme recognition (zero vision cost) & GIF thumbnailing
```

---

## 🚀 5. Quick Start

> [!NOTE]
> **Host Runtime Dependencies**  
> This plugin builds on the `MemoryProvider` interface of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). On unmodified upstream Hermes, SQLite/Chroma storage, forgetting-curve decay, and basic recall work normally, while **confirmed post-delivery settlement, 4-layer prompt ordering with `ephemeral` wire isolation, hard pre-write mutation gating, group participation control, and plugin cron sync** require host-side extension contracts (`hermes_host_contracts.py`) and are skipped or degraded when absent.
>
> See **[Host Runtime Capabilities & Modification Guide](docs/en/host-extensions.md)** for the exact capability table and step-by-step host integration points (files & interfaces). For strict JSON child-agent output, see [host patches](patches/hermes-agent/README_en.md).

### 1. Installation & Mounting

The plugin provides its own file locks and atomic writes. New databases and legacy upgrades do not require the host atomic persistence patch.
This plugin is built on the `MemoryProvider` interface of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent):

1. Copy or symlink [`plugins/sanae_mem`](plugins/sanae_mem) into your host's plugin directory (`$HERMES_HOME/plugins/sanae_mem`);
2. Place [`prompts/SOUL.md`](prompts/SOUL.md) and [`prompts/BEHAVIOR.md`](prompts/BEHAVIOR.md) in `$HERMES_HOME/`.

### 2. Configure Custom Persona & Relationship (`$HERMES_HOME/config.yaml`)

```yaml
memory:
  provider: sanae_mem
  memory_enabled: false
  user_profile_enabled: false

plugins:
  enabled: [sanae_mem]
  sanae_mem:
    shadow: false
    read_gate: explicit
    semantic_backend: chroma
    persona_name: "早苗"       # Your virtual persona's name (default: "早苗")
    partner_name: ""           # Optional: how background prompts refer to you (defaults to "对方")
    partner_role: "对话伙伴"   # Configurable relationship role (e.g., 挚友 / 搭档 / 维护者)
    partner_pronoun: "他"      # Pronoun used in first-person inner thoughts (他 / 她)
    memory_identity:
      platform: "qqbot"
      user_id_env: "MAINTAINER_USER_ID"

auxiliary:
  sanae_daily_flash:
    provider: deepseek
    model: deepseek-flash
    base_url: https://api.deepseek.com/v1
    api_mode: chat_completions
    reasoning_effort: none
  sanae_extract:
    provider: deepseek
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com/v1
    api_mode: chat_completions
    reasoning_effort: none
```

Keep both built-in stores disabled when a profile serves DMs and groups. Administrators can link their own QQ, Slack or Discord accounts to `memory_identity` with `maintainer_account_bindings` to continue the same private memory in verified one-to-one DMs. Slack accounts are scoped to a workspace.

`group_context` controls shared conversation history, long-term group memory and replies without a mention. Each starts disabled and can be enabled per group. With long-term memory enabled, search reads records available to the current group or thread. A successful delivery receipt triggers extraction of events, member impressions and commitments from that exchange. Restrict group access to shells and file tools with access to private directories. Configuration examples are in the [plugin guide](plugins/sanae_mem/README.md).

---

## 🔒 6. Boundary Between Public Framework & Private Instance

**WeftPersona** strictly separates the public cognitive/memory architecture from private lived history:
- **Included in this repository**: The general memory metabolism engine (`sanae_mem`), mood and impression dynamics, four-layer prompt framework (`SOUL.md` / `BEHAVIOR.md`), design methodology, and cross-model ablation laws.
- **Never included in this repository**: Any running instance's private database (`sanae.db`, `MEMORY.md`, `USER.md`), private diary (`diary/`), real chat logs, or server credentials.

---

## 📄 7. License & Acknowledgements

- Licensed under the [MIT License](LICENSE).
- Host runtime interfaces build upon [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License); see [NOTICE](NOTICE).
- The reference character **Sanae Kochiya (东风谷早苗)** originates from Team Shanghai Alice's (ZUN) *Touhou Project* series and follows the Touhou Project derivative works guidelines.
