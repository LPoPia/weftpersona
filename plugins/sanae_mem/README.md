# sanae_mem

`sanae_mem` is a user-level Hermes `MemoryProvider`. SQLite is authoritative for
the original event layer; Chroma is an optional semantic index. The optional
`shadow: true` setting keeps writing candidate memories while returning no search
or prefetch results (`default: false`). It is a manual operator switch and is never
engaged automatically. Explicit/scheduled recall gates and retention-curve
compaction keep SQLite authoritative.

## Storage and dependencies

- Database: `$HERMES_HOME/sanae-mem/sanae.db`
- Meme/expression assets: `$HERMES_HOME/sanae-mem/memes/` (original bytes, not cache or remote URLs)
- Chroma index: `$HERMES_HOME/sanae-mem/chroma/`
- Status: `$HERMES_HOME/sanae-mem/status.json`
- Calibration samples: `$HERMES_HOME/sanae-mem/sample.log` (at most three
  candidate user sentences per day, each truncated to 80 characters)
- Optional Chroma environment: `$HERMES_HOME/sanae-mem/venv`

The plugin always works with SQLite. If `chromadb` cannot be imported, the
`chroma` setting fails open to SQLite vectors. The independent dependency
environment can be created with the same Python minor version as Hermes:

```bash
$HERMES_HOME/hermes-agent/venv/bin/python -m venv \
  $HERMES_HOME/sanae-mem/venv
$HERMES_HOME/sanae-mem/venv/bin/pip install \
  'chromadb==1.5.9' \
  'opentelemetry-api==1.39.1' \
  'opentelemetry-sdk==1.39.1' \
  'opentelemetry-exporter-otlp-proto-grpc==1.39.1' \
  'opentelemetry-exporter-otlp-proto-common==1.39.1' \
  'opentelemetry-proto==1.39.1' \
  'opentelemetry-semantic-conventions==0.60b1'
```

The OpenTelemetry pins match the Hermes runtime on the delivery host. Without
them, Chroma's exporter can import a newer private API than the host provides.

## Configuration (activation is a separate operation)

The following is a reference only. Installing this directory does not activate
the provider and does not require a gateway restart.

```yaml
memory:
  provider: sanae_mem
  memory_enabled: false
  user_profile_enabled: false
plugins:
  enabled: [sanae_mem]
  sanae_mem:
    shadow: false             # manual switch, default: false
    nightly_journal: false    # enable the private 23:00 Asia/Shanghai journal
    free_think: false         # private reflections at 08:00, 16:00 and 22:00 Asia/Shanghai
    read_gate: explicit       # explicit or scheduled; default: explicit
    inject_max_chars: 400     # default: 400
    daily_scheduled_asks: 2   # default: 2
    gap_continue: true         # default: true
    gap_threshold_hours: 2     # default: 2
    emotion_ttl_days: 90       # default: 90
    expiry_review_days: 30     # default: 30
    expiring_importance_decay: 0.9  # default: 0.9
    expiring_importance_floor: 0.4  # default: 0.4
    distill_threshold: 0.6     # default: 0.6
    distill_daily_cap: 2       # default: 2
    semantic_backend: chroma  # chroma or sqlite; default: chroma
    data_dir: ""               # default: <home>/sanae-mem
    chroma_site_packages: ""   # default: plugin-managed venv search
    memory_identity:           # fixed maintainer identity for daily work
      platform: qqbot
      user_id: ""              # deployment copies the existing maintainer id
      user_id_alt: ""          # optional alternate id
      user_id_env: ""          # optional existing env var holding the id
    persona_name: "早苗"       # default: 早苗
    partner_name: ""           # optional display name in auxiliary prompts; default: ""
    partner_role: "对话伙伴"   # neutral default, customizable relationship role
    partner_pronoun: "他"      # 他 or 她; default: 他
```

To link the maintainer's other accounts or enable memory in a specific group, add the following entries under `plugins.sanae_mem`. Replace the example identifiers with the adapter's IDs. Slack needs its workspace ID, QQ needs its app ID, and Discord uses a globally unique user ID.

```yaml
maintainer_account_bindings:
  - profile: default
    platform: slack
    scope_id: T_EXAMPLE
    user_id: U_EXAMPLE
group_context:
  defaults:
    short_term_shared: false
    long_term_memory: false
    allow_unmentioned_reply: false
    unmentioned_interval_seconds: 30
  overrides:
    "default/slack/T_EXAMPLE/C_EXAMPLE":
      long_term_memory: true
```

Override keys identify the profile, platform, platform scope and channel. Each omitted switch inherits its own default. The interval is measured between actual starts of unaddressed replies in that group. Addressed messages do not consume it. The complete key list is defined in `config_schema.py`.

For a profile that serves both direct messages and groups, set both built-in memory flags to
`false` before starting the gateway. A verified one-to-one DM from a bound account can recall the maintainer's older account records and contribute new evidence to the same owner. In a group, the binding identifies the speaker. A group with long-term memory enabled exposes the memory search tool for its own permitted records. A delivery receipt triggers storage of group events, member impressions and commitments. Only the current request uses the current speaker's group impression.
The built-in session search cannot isolate people within a profile, so it is blocked in
messaging sessions. In a DM, use ViveMind search for that person's own memories.
An unknown chat type gets no recall. Outside the verified maintainer DM, the plugin blocks
built-in memory, file reads or searches of the profile's private data, and inbound
`@file`/`@folder` references. Do not grant shell access to untrusted chats on the private profile.

On the pinned original Hermes, a normal private text reply advances the automatic mood
only when the gateway's delivery ledger confirms that exact reply for that inbound message.
Failed sends and turns without a matching receipt leave the prior mood in place.

Set `nightly_journal: true` and configure `memory_identity` to let ViveMind install its
23:00 Shanghai-time job in Hermes. It writes to the existing private `MEMORY.md` once per
sleep day and keeps the scheduler output local.
Set `free_think: true` to run private reflections daily at 08:00, 16:00 and 22:00
Asia/Shanghai. The verified turn can search the web, memories and session history;
output and failures stay local. Only a completed original Hermes run with a recorded reflection
feeds the mood and nightly journal. A silent or failed run adds no experience.

In the configured maintainer's DM, `sanae_mem_review` can use one to three actual
first-person statements to review a long-term USER entry. It applies the reviewed text
directly only if source verification, model judgment, and both resident notebooks still
match the review.
Keep native `user_profile_enabled: false` so the built-in memory tool cannot write
around the provider's review.

Video detail is optional. An in-process Bilibili adapter registers it from its own
Hermes `PluginContext` via `sanae_mem.integrations.register_bilibili_memory(ctx)`.
The returned binder's `for_session(session_id)` finds the live ViveMind provider in the same
profile. Adapters can still call `bind(provider)` or supply an explicit resolver.
The client accepts delivered watch observations and accepted reply receipts, and returns
reviewed public excerpts and matching meme terms. Matching meme explanations are available
only in the verified private main turn. Disabling the adapter removes the tool and leaves
stored video records intact. Without the adapter, ordinary meme and image memory still work.

Configuration and plugin changes require a separately authorized gateway restart.

To run the standalone integration tests, use an unmodified Hermes checkout at
`6093627a1a113cdf66a6d5374ca71e20fa2f76e8` with its Python dependencies and pytest
installed. From the ViveMind checkout, point Python at both repositories (PowerShell):

```powershell
$env:VIVEMIND_HERMES_SOURCE = 'C:/path/to/hermes-agent'
$env:PYTHONPATH = "$env:VIVEMIND_HERMES_SOURCE;$(Resolve-Path 'plugins')"
python -m pytest -q tests
```

Daily semantic extraction and inner-state updates use the plugin-owned
`sanae_daily_flash` auxiliary task. Hermes layers `auxiliary.sanae_daily_flash`
over these registration defaults, so changing this route does not change
`auxiliary.sanae_extract` or the compression task:

```yaml
auxiliary:
  sanae_daily_flash:
    provider: deepseek
    model: deepseek-flash
    base_url: https://api.deepseek.com/v1
    api_mode: chat_completions
    reasoning_effort: none
```

The plugin still supplies the existing 15-second timeout, temperature 0 and
320-token limit. The host retains the same-provider retry and task fallback chain;
both memory slots stop before the main-model fallback. This requires the matching
host policy in `agent/auxiliary_client.py` as well as the plugin files.

Before: daily hook → `daily_semantics.call_llm()` → `sanae_extract`.
After: the same hook and semantic function → `auxiliary.complete_simple()` →
`sanae_daily_flash` → the same JSON parser, source checks and plugin writes.
The Flash worker has no tools or database handle. Prompts, context, triggers,
record updates and expiry are unchanged. Association relevance/state reviews
and long-term candidate reviews continue to use `sanae_extract`; nightly
maintenance is unchanged.

Business callers choose one of two interfaces: `complete_simple()` uses Flash,
and `complete_complex()` uses the existing `sanae_extract` route. Daily semantics
and association need/query planning choose simple; association relevance/state
and candidate/evidence reviews choose complex. Need/query planning retains its
existing 30-second request timeout and 200-token limit; this route change does
not guarantee that a slow provider cannot exhaust the host hook deadline.
Each interface calls the host directly and handles its own failures, with no
intermediate call wrapper. Callers do not select task keys or rely on automatic
complexity detection.

When the pre-LLM hook is registered, it owns private-chat association work.
Prefetch does not restart it after the host deadline or during host suppression;
results arriving after prefetch or a session switch cannot publish recall state.
Providers without a registered hook retain standalone prefetch reads.

## Tools

- `sanae_mem_add`: explicitly store an entrusted or corrected detail.
- `sanae_mem_search`: hybrid keyword/vector/time search, limited to ten hits.
  It returns an empty result while shadow mode is enabled.
- `sanae_mem_forget`: physically deletes SQLite, FTS and Chroma records by id
  or query, for the verified current owner only. A content fingerprint tombstone
  prevents automatic re-ingestion.
- `sanae_mem_compact`: moves expired short events into a hidden recognition window,
  tombstones them after that window, merges exact-keyword duplicates from the last
  24 hours, and decays confidence for unreferenced 30-day-old events. Expired
  events whose importance falls below the write threshold can be confirmed forgotten
  after at least one full day without recognition. Events whose `ttl_expire` is still
  the value the retention curve wrote (`curve_ttl`) skip that expiry review and follow
  the curve instead: once activation `importance · 2^(-idle/half_life_days)` drops below
  0.35 the verbatim text is archived to `memory-archive/events-archived.jsonl`, the row
  keeps only its summary (`summary_source='gist_faded'`) and its occurrence details are
  cleared. Below 0.15 a row that was ever recalled or started important becomes
  `tier='dormant'` and is never deleted; a quiet low-importance row is archived and
  removed without a `forgotten` tombstone, so the same words can be stored again later.
  Grounded USER recognition evidence (`recognition_evidence`) is never collapsed
  or purged.
- `sanae_mem_marked_promoted`: the persona's own explicit save tool (channel 3). It
  promotes an event with the current trusted owner only.
- `sanae_mem_distill`: distills eligible short events into permanent event records.
- `sanae_video_detail`: on-demand full analysis and captions of one watched video by
  BV id, read from the dedicated `video_watches` store. Ordinary recall only shows
  the title and a one-line gist; other owners' private video rows are not readable.
- `sanae_mem_report_false_trigger`: records confirmed bad recall feedback without
  storing conversation text, for operator diagnosis only; reports never disable
  recall.

### Meme and expression memory

The meme/expression zone is separate from the ordinary `events` stream. It stores a
text meme, its aliases, a full explanation, provenance and zero or more original
image/GIF files. Records live in `meme_records` / `meme_assets`; files live under
`memes/`. Text search reuses the plugin's local embeddings and FTS. There is no
image vector index.

- `sanae_meme_save`: create a record from a non-empty explanation plus zero or more
  local paths, HTTP(S) URLs or `data:` URLs. For an ordinary image, the explanation
  must include a visual description and intended use rather than an invented meme
  meaning. Before saving, the model may use the existing `web_search` / `web_extract`
  tools to find the original image URL and `vision_analyze` to understand the image;
  it may decide to collect on its own without asking the user each time. Identical
  original bytes are de-duplicated by SHA-256. Multi-image saves write only the
  non-duplicate images and return `duplicates` with the owning record; they do not
  move or rewrite an existing record. If every image is already present, no new
  record is created. A save with no valid images is also checked as exact text:
  a non-empty title plus the full explanation is the identity, or the sorted alias
  set plus the full explanation when the title is empty. Aliases never split a
  titled identity; source, time and session are not identity. The earliest live
  record wins and is returned as `duplicate_text` without changing its meaning,
  aliases, provenance or timestamps; adding aliases still requires the explicit
  update tool. The check runs inside the same write transaction as the insert and
  tombstone recheck. Content blocked by a whole-record deletion stays blocked by
  default; pass `restore_deleted: true` only when she or the user explicitly wants
  to restore it. A successful restore removes only the tombstones for the text/image
  actually restored; other tombstones and archived originals remain.
- `sanae_meme_search`: find records by meaning, tone, scene, alias or text meme.
  `purpose="chat_sticker"` (default) returns `path` as a prepared send copy whose
  longest edge is 300 px, plus the unchanged `original_path`, original
  `width`/`height` and `sha256`; the model does not resize images itself.
  `purpose="original"` returns the original bytes for ordinary photos, screenshots
  and formal delivery. Images already at or below 300 px are reused byte-for-byte.
  To send one on QQ, put `MEDIA:<path>` in the final reply; the existing QQ adapter
  uploads it as an image. There is no separate meme send tool or channel.
  Send copies live under `memes/chat/<sha>_v1_300.<ext>` and are not collection
  assets. Removing an asset removes its copy; deleting a record leaves the original
  archive unchanged. GIF copies preserve frame count, per-frame timing, loop
  presence/value and binary transparency; a copy that cannot be prepared without
  changing the animation is reported as an explicit preparation error, never
  silently replaced by the full-size or a lossy still image.
- `sanae_meme_update`: change title, explanation, aliases or provenance; append
  images; remove one asset. An existing explanation cannot be cleared. Adding an
  image blocked by a whole-record deletion is rejected explicitly; use
  `sanae_meme_save` with `restore_deleted: true` instead.
- `sanae_meme_delete`: delete one asset, or archive and delete a whole record.
  Deleting an asset copies its original bytes under
  `memory-archive/memes/<record_id>/` first; deleting a whole record archives every
  original file before removing anything. A whole-record delete also writes a
  `meme_forgotten` tombstone so the same text/image is not silently re-added.

On each private primary turn, `handle_pre_llm_call` also performs a read-only
lookup: an exact alias in the user's text, a real-embedding semantic hit, a
lexical hit, or an image whose SHA-256 matches a stored asset, injects a short
explanation. Image matching also follows the gateway's text-mode vision prompt
line `[ ... vision_analyze ... image_url: <cache path> ...]` (spaces and Chinese
paths included); an `image_url:` fragment in ordinary description text is ignored.
The original bytes are hashed before any provider-side resize/re-encode.
Hash-fallback vectors are not trusted without lexical overlap.
Unknown images are not auto-collected; the model decides whether to analyze and
save them. Public interactions never read or write this zone.

QQ's official Bot API has no confirmed group/C2C API for native QQ market faces or
super expressions. The zone therefore stores and sends ordinary images, not native
QQ faces. The official rich-media page lists GIF among supported image formats while
its `file_type` table only names png/jpg; the plugin preserves the original GIF and
uploads it as an image, but whether QQ renders it as an animation must be verified
against the live platform.

`sync_turn` only writes primary-agent turns. Long turns are split on sentence
boundaries before the length gate. It applies command and technical-content
exclusions, an importance threshold, and source-scoped duplicate matching. A private,
fully delivered turn feeds the memory axis; a delivered recall or a repeat after
12 hours only lengthens the event's lifetime. Retrieval counters are not a promotion
channel.

Long-term promotion has exactly three real channels, all recorded in
`memory_axis_mentions` / `memory_axis_promotions`:

1. `record_delivered_rementions` runs only after the host confirms a complete
   delivery (turn id + delivery time + the actually delivered text). An event is
   promoted on the third distinct 04:00 sleep day; same-day repeats and backend
   searches that produced no delivered sentence do not advance it.
2. `record_inner_state_flash` runs only after the inner-state CAS append committed.
   A strong state shift, or a still-significant afterglow after sleep, qualifies; a
   quarrel with conflict evidence and active tension does not.
3. `promote_explicit_save` is the persona's own save tool (`sanae_mem_marked_promoted`).

By-id reads inside promotion re-check the row's current trusted `owner_key`, so the
axis cannot cross the per-person physical isolation. Every channel advances its
watermark only when the write/CAS actually committed.

The row this turn may deepen is chosen from one bounded candidate set under the same
owner lock: the event ids this turn actually surfaced through the association seam
(an `ok` recall whose items are events), plus one SQL text prefilter over the owner's
own rows using the delivered words' character trigrams. The trigram score still
decides the match and whether it counts, so a bounded scan cannot promote a row it
would not otherwise have matched, and a backend search that was never delivered is
still not evidence.

Dedicated stores never decay with ordinary events: `commitments` (pending personal
promises, full text for the owner and a generic masked occupation plus an optional
safe item summary for everyone else), `video_watches` (BV, title, full analysis, captions, provenance, reviewed
public excerpt, self read scope) and `impression_revisions` (long-term baseline plus
in-place recent turnover).

The two-hour distant-memory spark cooldown is stored on the owner's latest
`inner_state_revisions.last_spark_at` row. Rendering may strip the one-turn note from
the text while leaving the clock intact, so a restart or another owner's update does
not release the same owner's cooldown early.

`video_watches.raw_observation` keeps the watching provenance (the archived event's
id, raw, summary, owner, read scope and occurrences) as the trace of where the watch
came from. The `events.vec` embedding stays only in the archived source row of
`memory-archive/events-archived.jsonl`, so the business column never repeats the same
vector as base64 text. Legacy rows are normalized only during the one-time database
upgrade: the verified pre-upgrade database backup and the full original provenance
JSON are durable before the row is rewritten without the vector. Ordinary startup and
video CRUD do not rerun the migration or scan schema columns.

Commitments use only the current fully delivered turn's words, reply and tool
receipts. An unavailable auxiliary model records `auxiliary_unavailable` under
`components.commitments`; it creates no retry queue and does not block the next
turn. The same delivered turn is idempotent, while a later independent promise of
the same words can create a new item. Resolution still checks its owner, current
pending id/version and actual evidence.

Before fulfillment the complete pending row and resolution evidence are appended
to `memory-archive/commitments-archived.jsonl` (flush and fsync). Successful
fulfillment leaves an owner-scoped ordinary middle-term event, so she can remember
having done the thing after it leaves her pending list; cancellation is not
recorded as fulfillment. Same-turn redelivery or a stale CAS writes no second done
event. Archive or event-write failures are reported with stable phase/error codes,
never private text in `status.json`.

The same auxiliary answer may supply `masked_text`, a safe item summary. Known
identity anchors and quoted originals are rejected locally; ordinary item words,
years and natural overlap with the promise remain usable. With no safe summary,
other people see only the generic occupation. Existing rows are not backfilled by
another model call.

New databases do not create `mood_state`, `impression_session_views`, or
`commitment_legacy_queue`. A legacy database receives a consistent, verified
`sanae.db.pre-upgrade-vN.sqlite` backup before one-time schema/data migration; that
backup retains the retired tables' definitions and rows, old video source rows and
occurrences, and removed appraisal values for recovery. If any archive or database
write fails, the source rows and schema version stay at the old state so the upgrade
can be retried. Heart's old `_inner_recent_word_sources` interface is retired, and
missing legacy ownership is never guessed from a profile, session, id or timestamp.
Schema v4 stores confirmed maintainer account bindings in `maintainer_account_bindings`.
Schema v5 adds `source_context` to `impression_revisions` and `commitments`. Older rows retain their owner and text. An empty source field records unknown provenance.

The per-turn tail note is exactly two blocks: `[我眼中的世界]` then
`[我内心的感受]`. The world block names the current speaker, her own finished
activity and outbound words (no second `[我刚才在别处说过]` block), pending
commitments, the maintainer's current-sleep-day activity, and the current speaker's
impression. The daily pre-LLM hook does not inject an independent daily
block; daily facts render inside the world block on the same 04:00 sleep-day
boundary as inner state (`clock.sleep_day_iso()` is the sleep day, not midnight). Constraints and
facts count only when opened on that day, so yesterday's recap and 7–14 day
leftovers are gone.

With the built-in flags disabled, the maintainer's verified DMs share a private memory owner. Other private participants can retrieve their own ViveMind events. Group memory uses the active group and thread permissions. If built-in memory is left enabled and would
load a non-empty notebook, ViveMind stops the external turn before the model call.
The 23:00 journal has one owner: the host cron entry that runs
`scripts/run_nightly_journal.py`, or an explicit `run_nightly_journal()` call. The
plugin plants no watcher thread in `initialize()` and triggers nothing on session
end, so a CLI copy or a cron build cannot add a second close.
The journal run table and the file lock keep repeated invocations idempotent. The
composer receives the entry's own calendar date and distinguishes completed
experience from thoughts. It keeps factual details and time relations grounded in
those sources, while leaving room for her own feelings; sparse material can make a
short entry instead of a made-up day's narrative. The
runner is copied to
`HERMES_HOME/scripts`; it resolves `sanae_mem` from `HERMES_HOME/plugins` and the host
from `HERMES_HOME/hermes-agent` under `--home` (default `HERMES_HOME`), so the gateway
needs no PYTHONPATH for either, while an explicit `--repo` still selects a source tree.
A cron `script` names a file and cannot
carry arguments, so that runner reads the profile's real
`plugins.sanae_mem.memory_identity` configuration (and the same profile's native
MemoryStore and `sanae_daily_flash` route) through a scoped home override. It never
borrows `argv` as an identity: a missing `memory_identity` fails explicitly, and the
optional `--owner-key` is only a cross-check against the configured hashed owner.
When a journal entry is collapsed, sinks past the 45-day window, or is moved out for
MEMORY.md capacity, it becomes an ordinary middle-term `events` row with the natural
curve and the owner's private scope. Those pressure paths never write `tier='long'`;
long-term memory still comes only from the three channels above.

Failures in the memory axis, the nightly journal and the event-topic shared-view projection
are recorded atomically in `status.json` under `components.memory_axis` /
`components.nightly_journal` / `components.shared_view_projection` with `ok`,
`stage` (a fixed phase name), `error` (a stable code only) and `at` (plus the
sleep `day` for the journal). No user text, secret, reasoning or private matter
enters that record, and a failure never advances the axis watermark or a
completed journal run. That status path
is file-backed, so a failed insert or commit in `nightly_journal_runs` still records
the phase with the code `journal_run_write_failed`; the run is not marked completed.

## Owner-scoped inner state and the shared view

`inner_state_revisions` remains the single per-profile revision/CAS chain: one slow
variable snapshot (`tension`/`closeness`/`energy`), one watermark and one monotonic
`revision`. Two row fields describe whose words these are and who may read them:

- `owner_key` is the author of `text`/`recent_words`. Empty means an unbound
  pre-owner legacy row or an internal autonomous row.
- `visibility` is `private` for owner-scoped words and `internal` for the trusted
  autonomous heart. Internal rows are written with `owner_key=''`, empty
  `recent_words`, and are never returned to a person, rendered into a conversation,
  used as anyone's `previous`, projected into the shared view or sunk into a journal.

There is still exactly one chain, not one mood copy per person.

- `read_inner_state(*, target=None, owner_key="", shared_only=False, internal=False)`
  returns an `InnerStateView` with three fields: `head` (always the newest revision of
  the single chain: revision, slow variables, clocks and delivery watermark), `shared`
  (the current safe common view) and `private` (that verified owner's newest private
  revision). With `internal=True` the `private` slot is the newest
  `visibility='internal'` revision instead, and no person's row is selected. Each
  segment keeps its own `felt_at`/TTL; neither replaces the other, so an old private
  fragment cannot hide a later common mood. `.text` exposes both raw visible segments
  (shared first). With no owner (or `shared_only=True`) only the public view is
  present; with no visible segment at all the read returns `None`. SQL adds
  `owner_key=?`/`visibility=?` for the segment, and the head query selects metadata
  only: no other person's text is selected and dropped in Python.
- `render_inner_state(state=None, *, now=None, target=None,
  session_id="", owner_key="", internal=False)` renders the current shared mood and the
  selected segment as separate timed paragraphs (shared first) without writing to the
  database. Pending spark notes are consumed on verified delivery settlement
  rather than during read-only rendering. An explicitly passed `InnerState` keeps the
  legacy single-segment behavior.
- `write_inner_state(text, *, owner_key, turn_id="", now=None, internal=False)` keeps
  handwriting verbatim in that owner's private row; with `internal=True` the words go
  to the internal segment instead (empty `recent_words`) and the execution owner's
  private row is untouched. Handwriting never copies into the shared view.
- `update_inner_state(..., owner_key=..., internal=False)` builds `previous` from both
  visible segments: `previous.shared` is the current common mood and `previous.private`
  is this owner's authorized fragment (the autonomous path uses `previous.internal`
  and never a person's row). `previous.thought` is whichever live segment is newer. It
  never inherits another person's row or an unbound legacy row. Missing owner fails
  closed (`owner_unverified`). The autonomous signature
  (`preserve_processed_at=True` with no `user_text`) without `internal=True` fails
  closed with `autonomous_scope_required`: an execution identity is not content
  ownership. If only an unbound legacy row exists, the head still supplies
  `previous.variables` (decayed common slow variables) with empty text/segments, so
  the first update keeps the common direction without reading the unknown original.
- The update prompt names the partner only when the host-verified `owner_key` equals
  the configured maintainer owner. Every other owner is labelled as the other current
  person ("not <partner>"); the partner role and the `tension`/`closeness` lover
  semantics stay with the fixed partner, so an outsider's words are never presented as
  the lover's. With no verified partner identity the prompt never guesses a name.
- `latest_inner_state(target=...)` is the common-head interface: only
  revision/variables/felt_at/updated_at/processed_at/source/turn_id are selected;
  `text`/`recent_words` are empty by construction. `current_turn_context`,
  delivered-turn updates and promotion pass only this metadata; the autonomous
  heartbeat additionally reads its own `internal` segment (never a person's).

`inner_state_shared(profile_key PK, text, variables, felt_at, updated_at,
source_revision)` is the host-only common view, written only from the one inner-state
Flash answer (`update_inner_state`): never by a second model call and never by a
separate prompt. On an autonomous heartbeat (`internal=True`) her own activity already
speaks in her own first-person voice, so the freshly written mood itself is the common
layer: those sentences go to `inner_state_shared` once `_shared_forbidden_names()` (the
known private names, i.e. the configured partner) and `_shared_text_safe()` (four to
sixty characters, no known private name, no quoted span that repeats the private
material it was written from) accept them. On a delivered private turn the same JSON may
carry `shared_sentences` (one short sentence, or none), written with the speaker's name
and private particulars left out, and checked by the same gate against this turn's user
message, delivered reply and private fragment. Wording overlap alone is not a gate: her
own feeling may reuse the words of the activity or mood it continues from, which is the
situation flowing into the mood rather than a leak, and whether the content concerns the
current person's private business is the same answer's judgement. Each sentence is
judged on its own and a rejected one is left out; when none is accepted the previous
safe view stays and ages out through its own TTL. Identical text
is not a new change (`_write_shared_state` is a no-op, so `felt_at` is not renewed),
and a late write loses to a newer one through `updated_at`; it cannot overwrite a newer
view. The private append, handwriting and slow variables are unaffected by any
shared-view write. The nightly journal calls `ensure_shared_inner_state(*, owner_key,
now, since)` as a local read: an existing view updated inside today's sleep day is
returned, and otherwise `None`; it never generates a safe view and never calls a model.

Legacy rows written before `owner_key` existed migrate with `owner_key=''` and
`visibility='private'` and keep their verbatim text and slow variables. They are never
returned to any owner, never used as `previous` or shared evidence, and never
claimed by the profile or the fixed maintainer. This Task does not assign their
missing ownership.

The maintainer's world and delivered-turn `lived_experiences`, trusted personal cron
and her own nightly journal read actual free-think and personal-activity content
(`private_experience=True`). The autonomous heartbeat digests it as her own internal
experience, not as a private statement by whoever owns the cron execution. The
resulting first-person mood can supply the common layer under the rules above.

An external person's path uses `private_experience=False`: it does not read the
maintainer's private free-think file or private cron results. It keeps proved-public
Bilibili titles and a short gist, plus the current person's own delivered outbound
words. `_recent_activities` collects once for the actual audience purely read-only, while
explored curiosity threads are resolved on non-read-only write/settlement paths
(delivery settlement, video watch recording, and cron completion). There is no canned behaviour
meta-fact replacement.

Another owner's proved-public watch is selected through
`public_watch_projection_rows()`: title, BV, reviewed `public_excerpt`, mode and watch
time, never that owner's private summary, analysis, captions or raw observation. With
no reviewed excerpt it states only what was physically received. An unowned watch
without public proof is omitted from external activity, and full details stay in the
video store for an on-demand authorized read. The nightly journal combines actual
self-owned activity summaries with any locally available safe common mood; it does
not replace the day's activities with stock sentences. `llm_guard` binds a split-off thought
to `_write_owner()` in a private conversation; in cron it requires the
`current_execution()` receipt, writes the internal segment, and skips the write
entirely without that receipt rather than creating a shared bucket or claiming the
executor.

Shutdown first seals the provider's new background submissions, then waits (without
holding the SQLite lock and without a fixed timeout) for its own inner-state workers
and nightly run to finish, and only then closes `_conn`. Another provider of the same
profile keeps its own counter and tasks. A sealed provider never removes another
provider's queued work, and the existing CAS, watermark and same-turn handwriting
locks are unchanged.

Nightly compact archives the original before replacing it with a gist. It no longer
scans short-term rows against a promotion word list and stages nothing as a candidate:
tiers change only through the three promotion channels. A cold,
previously significant event becomes dormant; recall can still read that gist, but
retrieval never changes its tier. Low-value events that were never recalled leave
after review without blocking future writes. Explicit forget still blocks
re-ingestion. The pre-compression hook records diagnostics only.

Private delivered turns update resident entry touch times locally. Compact reports
MEMORY entries unseen for 45 days only when MEMORY.md reaches 85% of capacity; USER.md
and personal impressions are never a time-based cold candidate. Its optional demotion
saves the original and writes a searchable long impression before removing a MEMORY
entry. USER demotion first archives the complete original and writes it to the searchable
long-term event store. The plugin judgment model may then classify an entry as an
outdated recent status and remove it from USER.md by CAS. Core traits, durable
preferences, retention-marked entries, unavailable judgments, and changed entries stay
in USER.md. Capacity-driven demotion follows the same archive-first, event-first order.

## Ownership metadata

Events carry an `owner_key` to identify whose private data a row carries; it is
not a recall permission. The value is derived in `initialize()` from the identity
the host already isolates sessions on (`user_id_alt or user_id`, platform-prefixed)
and stored as a 16-character SHA-256 prefix, so no platform id is written to the
database. It is cached on the provider because the gateway reuses built agents and
`initialize()` does not run every turn. Recall across people keeps this metadata
and the attribution label intact.

Recalled private text is the bound owner's. Model-facing `_search()` (the
injection/gap windows, association, `prefetch()` and the `sanae_mem_search` tool)
reads `raw`/`summary`/`vec` only from rows whose `owner_key` equals the current
one; the owner binding, not the current chat's wording, decides that. Another
owner's row is reachable only through its host-written `shared_view` (topic) or
reviewed `public_excerpt` projection, including Sanae's own self experience:
`experience_subject='sanae'` and `read_scope.kind='self'` grant no cross-owner
raw read. An unbound maintenance context (`owner_key=''`) reads only the
unowned rows it writes itself, never a bound owner's private row. Person,
platform, time, source and certainty stay attached to each event.

`shared_view` is a host-written public topic, not a redacted event. A personal
request, arrangement, promise, plan, schedule or errand stays private even after
names, participants and subjects are removed; a message that also carries
genuinely public knowledge keeps only the public part actually discussed, and a
purely private message leaves the view empty. The projection is re-run whenever
an event is written again, so a stored view is replaced or cleared instead of
being reused forever as a cache.

One projection call has three outcomes: `public` (a non-empty topic replaces the
cache), `private` (the model's explicit empty verdict clears it) and `failed` (no
client, no reply or a raised call, invalid JSON/format, or a rejected short view,
echo or identity marker). Only a legal `public`/`private` verdict writes the
column; a failed call is neither a private verdict nor a success, so it leaves
the stored value untouched and publishes `components.shared_view_projection` in
`status.json` with fixed `stage="projection"`, a stable error code
(`call_unavailable`, `call_no_reply`, `call_error`, `call_no_source`,
`parse_invalid_json`, `parse_invalid_format`, `validate_invalid_view`,
`validate_rejected_echo`, `validate_rejected_identity`), `at` and a cumulative
`failures` count. A fully
legal batch clears only this component's fault (`ok=true`) and keeps the count
history and every other component's record; a partial batch failure stays visible
even when a later event in the same batch succeeds. No original text, prompt,
model reply or exception message enters that record.

Attribution fields remain:

- `experience_subject`: `sanae` when Sanae lived/participated in the event, or
  `owner:<key>` for a user's grounded fact.
- `fact_subject`: whose first-person fact the raw text states. An automatic user
  turn keeps `experience_subject='sanae'` but `fact_subject='owner:<key>'`, so
  "I could not sleep" is not rendered as Sanae's own insomnia. Sanae's own
  explicit memory/video uses `fact_subject='sanae'`.
- `involved_owner_keys`: JSON array of users involved in the event; empty for a
  purely personal experience.
- `read_scope`: retained attribution metadata; the `owner_key` binding, not this
  field, decides whether the raw text is readable. Public disclosure still uses
  only the separately reviewed `public_excerpt`.

Search results carry `experience_subject`, `fact_subject` and an `attribution`
hint (`self`, `user_statement`, `user_fact`, `unknown`) into tool results,
association candidates and injection provenance. `sanae_mem_search` keeps its
reviewed/unknown `source_context`; a user's first-person statement is labelled
as such and is not rendered as Sanae's own fact. An old row with no attribution
stays explicitly unknown; it is not claimed for the current user.

When `fact_subject` is an owner, association and scheduled provenance include a
stable non-identifying participant code and the known source platform, so two
people saying the same words remain distinguishable in the actual injected text.
No platform account id, private author, private ref or snapshot is printed. If the
owner/source is unknown, no code or platform is invented. This is internal recall;
it does not force Sanae to mention the person in her answer.

A reviewed `public_excerpt` for a self video follows the same Sanae across
objects: `get_public_video_watch()` and public search can return it under B or an
unbound personal activity. This path reads only the reviewed excerpt, never raw.
Public video remains a disclosure path, so a changed/revoked host binding refuses
that old turn; ordinary private recall does not.

Old rows keep attribution fields empty (unknown) and are not backfilled or
automatically claimed. Dedupe candidates still require the same owner and
attribution fields, so different people's similar text is not merged and a
user's fact is not rewritten into Sanae's self-fact. Explicit forgetting is a
write: `sanae_mem_forget` resolves the same verified owner binding as a write and
may only archive, tombstone and remove rows carrying that `owner_key`, by id or
query. Another owner's id resolves to no row, and an identity-less call is
refused before any lookup. The delete itself keeps the existing
blacklist/tombstone, occurrence, FTS and Chroma paths.

Two deliberate exceptions:

- **System-level rows.** A cron or CLI session without a host-granted owner
  binding writes `owner_key = ''`, and rows whose owner cannot be resolved also
  stay `''` (fail closed, never guessed). A bound cron action may explicitly write
  under its trusted owner through `sanae_mem_add` or `record_video_watch`; its
  automatic `sync_turn` ingestion remains blocked as before. Duplicate checks
  still stay within the same owner/attribution boundary, so one person's add
  cannot merge into another person's row. Explicit forgetting still requires the
  verified owner binding; an identity-less tool call cannot delete by id or query.
  Rebinding or revoking an already-running action fails that old turn's write rather than
  transferring its private content; `compact()`/`distill()` keep their
  database-wide maintenance semantics.
- **Maintenance reads.** `compact()`, `distill()` and expiry run in those same
  identity-less contexts and must operate database-wide, so they pass
  `unscoped=True` and keep seeing every owner. Scoping them would silently disable
  nightly tiering and forgetting.

The remaining privacy boundary is disclosure, not recall: public turns read only
reviewed `public_excerpt`, and private raw stays out of public projections. Owner
metadata is retained so a recalled fact can still be attributed to the person it
came from.

## Module responsibilities

The host still enters through `plugin_entry.py` and the existing provider methods.
The following implementation modules accept the data they need rather than the
whole provider:

- **Meme memory:** `domains/memes.py` coordinates tools and records; `domains/meme_assets.py`
  owns original images, archives and chat copies; `domains/meme_search.py` owns matching;
  `domains/meme_context.py` builds inbound-image and conversation context.
- **Inner state:** `cognitive/inner_state.py` coordinates manual and delivered-turn updates;
  `cognitive/inner_state_types.py` defines the state values, `cognitive/inner_state_store.py` owns
  SQLite/CAS, `cognitive/inner_state_life.py` owns time evolution and rendering,
  `cognitive/inner_state_spark.py` owns distant-memory sampling and one-turn spark consumption,
  and `cognitive/inner_state_model.py` owns the existing prompts and response decoding.
- **Association:** `cognitive/association.py` owns trusted bindings, request caching and
  orchestration; `cognitive/association_history.py` prepares conversation history and retrieval
  decisions; `cognitive/association_recall.py` owns candidates, source provenance, review and
  context rendering.

Database schema, data locations, prompts and model-call semantics are unchanged
by this responsibility split. Existing archive, clock, model-output and status
foundations remain shared.

## Note on persona quality

Hand-written persona labels tend to go flat after a few turns. If you want a
more alive character, consider distilling your virtual persona with
[Character_Skill_Producer] before plugging it into a memory provider like this
one.
