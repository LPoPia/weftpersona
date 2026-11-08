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
plugins:
  sanae_mem:
    shadow: false             # manual switch, default: false
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
    persona_name: "早苗"       # default: 早苗
    partner_name: ""           # optional display name in auxiliary prompts; default: ""
    partner_role: "挚友"       # customizable relationship role
    partner_pronoun: "他"      # 他 or 她; default: 他
```

The complete key list is defined in config_schema.py.

Configuration and plugin changes require a separately authorized gateway restart.

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
  or query. A content fingerprint tombstone prevents automatic re-ingestion.
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
  Pending USER-proposal evidence (`pending_grounded_evidence_ids` /
  `recognition_evidence`) is never collapsed or purged.
- `sanae_mem_marked_promoted`: confirms a pending event as long-term or important.
- `sanae_mem_distill`: distills eligible short events into permanent event records.
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
boundaries before the length gate. It applies command
and technical-content exclusions, an importance threshold, and source-scoped duplicate
matching. A private, delivered recall or a repeat after 12 hours strengthens the event;
three spaced recalls promote it to long. Scheduled and gap suggestions do not count.

Nightly compact archives the original before replacing it with a gist. A cold,
previously significant event becomes dormant; only a strong explicit reminder can
wake it. Low-value events that were never recalled leave after review without
blocking future writes. Explicit forget still blocks re-ingestion. The pre-compression
hook records diagnostics only.

Private delivered turns update resident entry touch times locally. Compact reports
entries unseen for 45 days when their file reaches 85% of capacity. Its optional
demotion saves the original and writes a searchable long impression before removing
a MEMORY entry; removal from USER requires approval and an exact read-back.

## Ownership metadata

Events carry an `owner_key` to identify whose private data a row carries; it is
not a recall permission. The value is derived in `initialize()` from the identity
the host already isolates sessions on (`user_id_alt or user_id`, platform-prefixed)
and stored as a 16-character SHA-256 prefix, so no platform id is written to the
database. It is cached on the provider because the gateway reuses built agents and
`initialize()` does not run every turn. Recall across people keeps this metadata
and the attribution label intact.

Recalled memory is Sanae's own. Model-facing `_search()` covers the whole event
store, including the injection/gap windows, association, `prefetch()` and the
`sanae_mem_search` tool, regardless of which person is currently chatting or
whether the activity is autonomously unbound. The current chat does not grant or
revoke her memory. Person, platform, time, source and certainty stay attached to
each event.

Attribution fields remain:

- `experience_subject`: `sanae` when Sanae lived/participated in the event, or
  `owner:<key>` for a user's grounded fact.
- `fact_subject`: whose first-person fact the raw text states. An automatic user
  turn keeps `experience_subject='sanae'` but `fact_subject='owner:<key>'`, so
  "I could not sleep" is not rendered as Sanae's own insomnia. Sanae's own
  explicit memory/video uses `fact_subject='sanae'`.
- `involved_owner_keys`: JSON array of users involved in the event; empty for a
  purely personal experience.
- `read_scope`: retained metadata; it no longer decides whether Sanae can recall
  an event. Public disclosure still uses only the separately reviewed
  `public_excerpt`.

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
user's fact is not rewritten into Sanae's self-fact. Normal forgetting is
maintenance of Sanae's own memory, not a deletion permission granted by the
current chat: it may select any matching stored event by id/query, while the
actual delete still uses the existing blacklist/tombstone, occurrence, FTS and
Chroma paths.

Two deliberate exceptions:

- **System-level rows.** A cron or CLI session without a host-granted owner
  binding writes `owner_key = ''`, and rows whose owner cannot be resolved also
  stay `''` (fail closed, never guessed). A bound cron action may explicitly write
  under its trusted owner through `sanae_mem_add` or `record_video_watch`; its
  automatic `sync_turn` ingestion remains blocked as before. Duplicate checks
  still stay within the same owner/attribution boundary, so one person's add
  cannot merge into another person's row. Explicit forgetting is Sanae's own
  maintenance and does not require the current chat's binding. Rebinding or
  revoking an already-running action fails that old turn's write rather than
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

## Note on persona quality

Hand-written persona labels tend to go flat after a few turns. If you want a
more alive character, consider distilling your virtual persona with
[Character_Skill_Producer] before plugging it into a memory provider like this
one.
