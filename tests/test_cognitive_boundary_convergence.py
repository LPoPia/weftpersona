"""Comprehensive verification of ViveMind cognitive-boundary convergence.

Verifies:
1. Zero custom/private Hermes fork imports, zero underscore calls into Hermes objects,
   and zero hardcoded private persona/maintainer literals across ``plugins/sanae_mem``.
2. Unified 3-tier ``SourceMaterial`` intake contract (``raw`` vs ``processed``, null-safe
   provenance, deduplication, ``T:`` isolation, and hard invariants barring ``processed``
   summaries/captions from ``USER.md`` and ``UserEvidence``).
3. ``ViveMindCognitivePromptPolicy`` (``Tier 0 / Tier 1 / Tier 2`` anti-recency ordering,
   surface gating, Chinese skill index, ``04:00`` sleep-day & speaker gap with verified
   cron sends, private compaction overrides & summary prompt).
4. ``ViveMindInboundMergePolicy`` & proactive cron hooks.
5. Verified ``DeliveryReceipt`` deduplicated handling.
"""

from __future__ import annotations

import ast
import importlib.util
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )

if HERMES_ROOT.is_dir() and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

if not HERMES_ROOT.is_dir():
    pytest.skip("Hermes host checkout unavailable for ViveMind boundary tests", allow_module_level=True)

from gateway import session_context as _session_context
import hermes_host_contracts as hc
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from sanae_mem import plugin_entry
from sanae_mem.runtime import provider as provider_module
from sanae_mem.runtime import hooks as runtime_hooks
from sanae_mem.infra.clock import SHANGHAI_TZ
from sanae_mem.governance.cognitive_context import (
    ViveMindCognitivePromptPolicy,
    ViveMindInboundMergePolicy,
    build_public_social_turn_policy,
    gate_proactive_cron_delivery,
    prepare_proactive_cron_prompt,
)
from sanae_mem.config_schema import DEFAULT_PERSONA_NAME
from sanae_mem.governance.memory_evidence import build_user_evidence
from sanae_mem.governance.source_intake import (
    SourceMaterial,
    ingest_source_material,
    submit_source_material,
)


FORBIDDEN_HERMES_MODULES = frozenset({
    "interaction_context",
    "gateway.interaction_context",
    "cron.proactive",
    "delivered_memory",
    "agent.delivered_memory",
    "gateway.qq_chat_archive",
    "response_filters",
    "gateway.response_filters",
    "memory_tool_store",
    "tools.memory_tool_store",
})

FORBIDDEN_UNDERSCORE_ATTRS = frozenset({
    "_entries_for",
    "_char_limit",
    "_locate_entry",
    "_execution_path",
})


@contextmanager
def _private_route(
    *,
    platform: str = "qqbot",
    actor: str = "maintainer-openid",
    session: str = "sess-dm-1",
    chat_type: str = "dm",
    chat_id: str = "chat-dm-1",
    message_id: str = "msg-1",
):
    _session_context.reset_session_vars()
    tokens = _session_context.set_session_vars(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=actor,
        session_id=session,
        session_key=f"agent:main:{platform}:{chat_type}:{chat_id}",
        message_id=message_id,
    )
    try:
        yield
    finally:
        _session_context.clear_session_vars(tokens)


# ──────────────────────────────────────────────────────────────────────
# 1. AST & Source Purity Audit
# ──────────────────────────────────────────────────────────────────────


def test_plugins_have_zero_private_hermes_couplings_and_zero_persona_literals():
    assert DEFAULT_PERSONA_NAME == "AI伙伴"

    package = PLUGIN_ROOT / "sanae_mem"
    py_files = sorted((
        *package.glob("*.py"),
        *package.glob("infra/*.py"),
        *package.glob("domains/*.py"),
        *package.glob("storage/*.py"),
        *package.glob("governance/*.py"),
        *package.glob("cognitive/*.py"),
        *package.glob("runtime/*.py"),
    ))
    assert len(py_files) >= 40

    violations: list[str] = []
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()

        for forbidden_literal in ("早苗", "\u98ce\u5439\u843d"):
            if forbidden_literal in source:
                violations.append(f"{rel}: contains hardcoded literal {forbidden_literal!r}")

        if "getattr(db, \"_conn\"" in source or "getattr(db, '_conn'" in source:
            violations.append(f"{rel}: accesses private SessionDB._conn")
        if "MemoryStore" in source and "_read_file" in source:
            violations.append(f"{rel}: calls private MemoryStore._read_file")

        tree = ast.parse(source, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_HERMES_MODULES:
                        violations.append(f"{rel}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in FORBIDDEN_HERMES_MODULES:
                    violations.append(f"{rel}:{node.lineno} imports from {mod}")
                for alias in node.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    if full in FORBIDDEN_HERMES_MODULES or alias.name in FORBIDDEN_UNDERSCORE_ATTRS:
                        violations.append(f"{rel}:{node.lineno} imports {full}")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_UNDERSCORE_ATTRS:
                    violations.append(f"{rel}:{node.lineno} accesses forbidden attr .{node.attr}")

    assert violations == []


# ──────────────────────────────────────────────────────────────────────
# 2. Unified 3-Tier SourceMaterial Intake & Hard Safety Invariants
# ──────────────────────────────────────────────────────────────────────


def test_source_intake_separates_raw_and_processed_and_blocks_user_md_pollution(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    cfg = {
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "qqbot", "user_id": "maintainer-openid"},
    }
    provider = plugin_entry.SanaeMemProvider(config=cfg)

    with _private_route():
        provider.initialize("sess-dm-1", hermes_home=str(home), platform="qqbot")
        try:
            # 1) Processed without processing_type fails closed
            bad_proc = SourceMaterial(
                source_plugin="external_web",
                platform="rss",
                material_kind="processed",
                content="某篇长文的初步摘要",
                privacy_scope="private",
            )
            bad_verdict = ingest_source_material(provider, bad_proc)
            assert bad_verdict.ok is False
            assert bad_verdict.reason == "processed_requires_processing_type"

            # 2) Processed summary with missing optional fields keeps None/"" (never invented)
            proc_mat = SourceMaterial(
                source_plugin="external_web",
                platform="web_digest",
                material_kind="processed",
                content="初步整理的外部文章摘要：提到用户喜欢喝冰美式。",
                privacy_scope="private",
                source_object_id="doc-42",
                processing_type="digest_summary",
                processing_version="v1.0",
                # author, occurred_at, collected_at, raw_locator intentionally omitted
            )
            v_proc = ingest_source_material(provider, proc_mat, create_memory_event=True)
            assert v_proc.ok is True
            assert v_proc.credibility == "derived_processed"
            assert v_proc.user_md_eligible is False
            assert v_proc.verbatim_eligible is False
            assert v_proc.linked_event_id is not None

            # Duplicate submission is deduplicated cleanly
            v_dup = ingest_source_material(provider, proc_mat, create_memory_event=True)
            assert v_dup.ok is True
            assert v_dup.deduplicated is True
            assert v_dup.material_id == v_proc.material_id

            # Verify stored provenance keeps NULLs instead of fabricated timestamps/authors
            row = provider._conn.execute(
                "SELECT author, occurred_at, collected_at, raw_locator, user_md_eligible, verbatim_eligible "
                "FROM source_materials WHERE id = ?",
                (v_proc.material_id,),
            ).fetchone()
            assert tuple(row) == (None, None, None, None, 0, 0)

            # 3) Raw external item preserves verbatim_eligible=True but user_md_eligible=False
            raw_mat = SourceMaterial(
                source_plugin="bilibili_public",
                platform="bilibili",
                material_kind="raw",
                content="UP主原评论：这段合成器音色太棒了",
                privacy_scope="public",
                source_object_id="BV1xx411c7mD#reply99",
                author="bili-user-88",
                occurred_at="2026-11-12T10:00:00+08:00",
                raw_locator="https://www.bilibili.com/video/BV1xx411c7mD/",
            )
            v_raw = ingest_source_material(provider, raw_mat, create_memory_event=True)
            assert v_raw.ok is True
            assert v_raw.credibility == "verified_raw"
            assert v_raw.user_md_eligible is False
            assert v_raw.verbatim_eligible is True
            assert v_raw.linked_event_id is not None

            # 4) HARD INVARIANT: processed/external events can NEVER be used as grounded USER.md evidence!
            proc_sub = SourceMaterial(
                source_plugin="bilibili_public",
                platform="bilibili",
                material_kind="processed",
                processing_type="subtitle_slice",
                processing_version="v1",
                content="字幕切片：他常喝冰美式",
                privacy_scope="private",
                source_object_id="BV1xx411c7mD#sub1",
            )
            v_proc_sub = ingest_source_material(provider, proc_sub, create_memory_event=True)
            assert v_proc_sub.ok is True and v_proc_sub.linked_event_id is not None

            # Layer 1: ingest_source_material never creates an event_occurrences row with UserEvidence
            review_attempt = provider.grounded_review.review_grounded_user(
                text="他喜欢喝冰美式",
                evidence_ids=[int(v_proc.linked_event_id), int(v_proc_sub.linked_event_id)],
                memory_dir=home / "memories",
            )
            assert review_attempt["ok"] is False
            assert review_attempt["reason"] == "evidence_event_missing"

            # Layer 2: even if an event_occurrences row with valid SHA256 exists, _qualified_occurrence rejects external/processed source_context
            import hashlib
            snap_json = '{"available": true, "user": "maintainer-openid", "platform": "qqbot"}'
            snap_sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()
            for eid in (int(v_proc.linked_event_id), int(v_proc_sub.linked_event_id)):
                ev_raw = provider._conn.execute("SELECT raw FROM events WHERE id = ?", (eid,)).fetchone()[0]
                provider._conn.execute(
                    "INSERT INTO event_occurrences(event_id, raw, observed_at, source_snapshot, source_snapshot_sha256) "
                    "VALUES (?, ?, '2026-11-12T10:00:00+08:00', ?, ?)",
                    (eid, ev_raw, snap_json, snap_sha),
                )
            provider._conn.commit()
            review_attempt_tampered = provider.grounded_review.review_grounded_user(
                text="他喜欢喝冰美式",
                evidence_ids=[int(v_proc.linked_event_id), int(v_proc_sub.linked_event_id)],
                memory_dir=home / "memories",
            )
            assert review_attempt_tampered["ok"] is False
            assert review_attempt_tampered["reason"] == "evidence_source_processed_or_external"

            # 5) HARD INVARIANT: build_user_evidence rejects processed/external source_context outright
            ev = build_user_evidence(
                [{"role": "user", "content": "喜欢喝冰美式"}],
                session="sess-dm-1",
                user="maintainer-openid",
                platform="qqbot",
                source_context={
                    "kind": "external_processed",
                    "author_id": "maintainer-openid",
                    "platform": "qqbot",
                    "certainty": "derived_processed",
                    "visibility": "private",
                    "material_kind": "processed",
                },
            )
            assert ev.available is False
        finally:
            provider.shutdown()

    # 6) T: test session is isolated from submit_source_material
    isolated_res = submit_source_material(
        {
            "source_plugin": "external_web",
            "platform": "rss",
            "material_kind": "raw",
            "content": "测试隔离内容",
        },
        session_id="T:probe-1",
        hermes_home_path=home,
    )
    assert isolated_res["ok"] is False
    assert isolated_res["reason"] == "test_session_isolated"


# ──────────────────────────────────────────────────────────────────────
# 3. ViveMindCognitivePromptPolicy (Tier 0/1/2, Surface Gating, Timing, Compaction)
# ──────────────────────────────────────────────────────────────────────


def test_cognitive_prompt_policy_anti_recency_tiering_and_surface_gating():
    cfg = {
        "persona_name": "早苗测试体",
        "memory_identity": {"platform": "qqbot", "user_id": "maintainer-1"},
    }
    policy = ViveMindCognitivePromptPolicy(lambda: cfg)
    bundle = {
        "identity_parts": ["<<SOUL_AND_BEHAVIOR_TAIL>>"],
        "help_guidance": "<<HELP_GUIDANCE>>",
        "guidance_parts": ["<<TOOL_OPERATIONAL_GUIDANCE>>"],
        "skills_prompt": "## 可用技能索引\n<available_skills></available_skills>",
        "context_files_parts": ["<<AGENTS_MD_WORKSPACE>>"],
        "core_memory_parts": ["<<PRIVATE_MEMORY_MD>>"],
        "user_profile_parts": ["<<PRIVATE_USER_MD>>"],
        "plugin_sections_after_memory": ["[我此刻心里]\n平静。"],
        "active_profile_line": "Active profile: default",
        "timestamp_line": "Current time: 2026-11-12",
        "runtime_environment_block": "<<HOST_SSH_AND_PATHS>>",
    }

    # Maintainer private DM: SOUL/BEHAVIOR at the tail of volatile context;
    # private workspace + MEMORY.md + USER.md allowed in Tier 1.
    maint_env = hc.TurnEnvelope(
        platform="qqbot",
        chat_type="dm",
        chat_id="c2c-maint",
        sender_id="maintainer-1",
        session_id="s-maint",
    )
    maint_tiers = policy.assemble_system_prompt_parts(None, bundle, None, maint_env)
    assert maint_tiers is not None
    assert "<<TOOL_OPERATIONAL_GUIDANCE>>" in maint_tiers["stable"]
    assert maint_tiers["volatile"].endswith("<<SOUL_AND_BEHAVIOR_TAIL>>")
    assert "<<PRIVATE_MEMORY_MD>>" in maint_tiers["context"]
    assert "<<PRIVATE_USER_MD>>" in maint_tiers["context"]
    assert "<<HOST_SSH_AND_PATHS>>" in maint_tiers["context"]

    # Private external DMs share named USER.md; workspace and private memory stay gated.
    ext_env = hc.TurnEnvelope(
        platform="qqbot",
        chat_type="dm",
        chat_id="c2c-other",
        sender_id="stranger-99",
        session_id="s-ext",
    )
    ext_tiers = policy.assemble_system_prompt_parts(None, bundle, None, ext_env)
    assert ext_tiers["volatile"].endswith("<<SOUL_AND_BEHAVIOR_TAIL>>")
    assert "<<PRIVATE_MEMORY_MD>>" not in ext_tiers["context"]
    assert "<<PRIVATE_USER_MD>>" in ext_tiers["context"]
    assert "<<AGENTS_MD_WORKSPACE>>" not in ext_tiers["context"]
    assert "<<HOST_SSH_AND_PATHS>>" not in ext_tiers["volatile"]

    # Restricted public surface: blocks skills, memory, user profile, plugin inner state, and host env
    pub_env = hc.TurnEnvelope(
        platform="bilibili",
        chat_type="public",
        chat_id="bv-1",
        sender_id="viewer-1",
        restricted_audience="bilibili_public",
    )
    pub_tiers = policy.assemble_system_prompt_parts(None, bundle, None, pub_env)
    assert "<<PRIVATE_MEMORY_MD>>" not in pub_tiers["context"]
    assert "<<PRIVATE_USER_MD>>" not in pub_tiers["context"]
    assert "可用技能索引" not in pub_tiers["context"]
    assert "[我此刻心里]" not in pub_tiers["volatile"]

    assert pub_tiers["volatile"].endswith("<<SOUL_AND_BEHAVIOR_TAIL>>")
    for tiers in (ext_tiers, pub_tiers):
        for marker in ("<<PRIVATE_MEMORY_MD>>", "<<AGENTS_MD_WORKSPACE>>", "<<HOST_SSH_AND_PATHS>>"):
            assert all(marker not in parts for parts in tiers.values())

    # Public social RestrictedTurnPolicy builder
    pub_policy = build_public_social_turn_policy(
        config=cfg,
        audience="bilibili_public",
        action_tool_name="bilibili_action",
    )
    assert pub_policy.isolate_history is True
    # The policy assembler gates workspace content while preserving persona files.
    assert pub_policy.skip_context_files is False
    assert "bilibili_action" in pub_policy.allowed_tools
    assert "terminal" not in pub_policy.allowed_tools


def test_skills_index_sleep_day_timestamp_and_compaction_policies(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = {
        "memory_identity": {"platform": "qqbot", "user_id": "maint-1"},
    }
    policy = ViveMindCognitivePromptPolicy(lambda: cfg)

    # 1) Chinese Skill index with surface context awareness
    rendered_skills = policy.format_skills_index_prompt(
        "default",
        {},
        {"skill_view"},
        None,
        index_lines=["- gdrive: Cloud archive sync"],
    )
    assert "## 可用技能索引" in rendered_skills
    assert "私聊、定时活动与公开场合各自可见的记忆" in rendered_skills
    assert "- gdrive: Cloud archive sync" in rendered_skills

    # 2) 04:00 sleep-day + verified proactive cron send reflected in 距我上次发言
    from sanae_mem.infra.textutil import _owner_key

    owner_key = _owner_key("qqbot", user_id="maint-1")
    event_dt = datetime(2026, 11, 12, 2, 30, 0, tzinfo=SHANGHAI_TZ)  # 02:30 -> belongs to 11-11 sleep day!
    event_epoch = event_dt.timestamp()
    prev_user_epoch = event_epoch - 3600 * 6  # 20:30 on 11-11 (same sleep day!)
    prev_reply_epoch = event_epoch - 3600 * 5
    cron_send_epoch = event_epoch - 900  # 15 minutes ago via verified cron send!

    hc.record_verified_cron_send(
        job_id="job-care-1",
        execution_id="exec-1",
        platform="qqbot",
        target="qqbot:dm:chat-maint",
        chat_id="chat-maint",
        bound_owner=owner_key,
        delivered_text="夜深啦，早点歇着。",
        delivered_at=cron_send_epoch,
    )

    env = hc.TurnEnvelope(
        platform="qqbot",
        chat_type="dm",
        chat_id="chat-maint",
        sender_id="maint-1",
        session_id="s-maint",
    )
    stamped = policy.format_inbound_turn_timestamp(
        "刚忙完手头的事",
        event_epoch,
        env,
        SHANGHAI_TZ,
        history=[
            {"role": "user", "content": "晚点聊", "timestamp": prev_user_epoch},
            {"role": "assistant", "content": "好呀", "timestamp": prev_reply_epoch},
        ],
    )
    assert stamped is not None
    assert "按04:00作息归属11-11深夜" in stamped
    assert "距对方上次发言6小时" in stamped
    assert "距我上次发言15分钟" in stamped
    assert stamped.endswith("\n刚忙完手头的事")

    # 3) Private compaction overrides & summary prompt vs CLI pass-through
    overrides = policy.compaction_config_overrides("s-maint", "qqbot", env, turns_count=8)
    assert overrides is not None
    assert overrides["omit_verbatim_user_section"] is True
    assert "## What I remember" in overrides["fallback_summary_body"]

    summary_prompt = policy.compaction_summary_prompt(
        "default prompt",
        "s-maint",
        "qqbot",
        env,
        content_to_summarize="user: 今天去看了展",
    )
    assert summary_prompt is not None
    assert "## What I remember" in summary_prompt
    assert "## Work I may need to pick up" in summary_prompt

    cli_env = hc.TurnEnvelope(platform="cli", source="cli", session_id="cli-1")
    assert policy.compaction_config_overrides("cli-1", "cli", cli_env) is None
    assert policy.compaction_summary_prompt("default", "cli-1", "cli", cli_env) is None


# ──────────────────────────────────────────────────────────────────────
# 4. ViveMindInboundMergePolicy & Proactive Cron Hooks
# ──────────────────────────────────────────────────────────────────────


def test_inbound_merge_policy_and_proactive_cron_hooks():
    merge = ViveMindInboundMergePolicy(qq_dm_window_seconds=5.0)
    assert merge.debounce_window_seconds("qqbot", "dm", busy=False) == 5.0
    assert merge.debounce_window_seconds("telegram", "dm", busy=False) is None

    plain_1 = SimpleNamespace(text="第一句", reply_to_message_id="", reply_to_text="", internal=False, message_type="text")
    plain_2 = SimpleNamespace(text="补充第二句", reply_to_message_id="", reply_to_text="", internal=False, message_type="text")
    quote_a1 = SimpleNamespace(text="回这句", reply_to_message_id="m-10", reply_to_text="旧话A", internal=False, message_type="text")
    quote_a2 = SimpleNamespace(text="再补一句", reply_to_message_id="m-10", reply_to_text="旧话A", internal=False, message_type="text")
    quote_b = SimpleNamespace(text="回另一句", reply_to_message_id="m-99", reply_to_text="旧话B", internal=False, message_type="text")
    slash_cmd = SimpleNamespace(text="/status", reply_to_message_id="", reply_to_text="", internal=False, message_type="text")
    test_msg = SimpleNamespace(text="T:隔离探针", reply_to_message_id="", reply_to_text="", internal=False, message_type="text")

    assert merge.can_merge_events(plain_1, plain_2, "qqbot", "dm", busy=False) is True
    assert merge.can_merge_events(plain_1, quote_a1, "qqbot", "dm", busy=False) is True
    assert merge.can_merge_events(quote_a1, quote_a2, "qqbot", "dm", busy=False) is True
    assert merge.can_merge_events(quote_a1, quote_b, "qqbot", "dm", busy=False) is False
    assert merge.can_merge_events(plain_1, slash_cmd, "qqbot", "dm", busy=False) is False
    assert merge.can_merge_events(plain_1, test_msg, "qqbot", "dm", busy=False) is False
    assert merge.merge_texts(["第一句", "补充第二句"], "qqbot", "dm") == "第一句\n补充第二句"

    # Cron proactive prompt preparer & silence delivery gate
    proactive_job = {"id": "job-1", "activity_class": "proactive"}
    enriched = prepare_proactive_cron_prompt(proactive_job, "看看要不要打个招呼")
    assert enriched is not None and "[SILENT]" in enriched

    suppressed = gate_proactive_cron_delivery(proactive_job, "[SILENT]")
    assert isinstance(suppressed, dict) and suppressed["allow"] is False
    assert gate_proactive_cron_delivery(proactive_job, "顺手分享一首刚听到的曲子") is None


# ──────────────────────────────────────────────────────────────────────
# 5. Verified DeliveryReceipt Deduplication
# ──────────────────────────────────────────────────────────────────────


def test_verified_delivery_receipt_updates_once_and_ignores_unconfirmed(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    hc.clear_all_contract_registrations()

    data_dir = tmp_path / "sanae-mem"
    db_path = data_dir / "sanae.db"
    monkeypatch.setattr(
        provider_module,
        "live_plugin_config",
        lambda *_args, **_kwargs: {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "memory_identity": {"platform": "qqbot", "user_id": "maint-1"},
        },
    )

    hook_manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), hook_manager)
    registered_providers = []
    orig_register_mp = ctx.register_memory_provider

    def _capture_provider(provider) -> None:
        registered_providers.append(provider)
        orig_register_mp(provider)

    inner_payloads: list[dict] = []

    def _complete(messages, *, purpose="", **_):
        if purpose == "inner_state_update":
            for message in reversed(messages):
                content = message.get("content") if isinstance(message, dict) else ""
                if isinstance(content, str) and content.lstrip().startswith("{"):
                    inner_payloads.append(json.loads(content))
                    break
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "sentences": ["今天忙完可以歇会儿了。"],
                        "variables": {"tension": 0.0, "closeness": 0.2, "energy": 0.0},
                        "shared_sentences": None,
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(text="{}")

    monkeypatch.setattr(ctx, "register_memory_provider", _capture_provider)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)
    try:
        plugin_entry.register(ctx)
        assert len(registered_providers) == 1
        registered_providers[0].attach_runtime_llm(SimpleNamespace(complete=_complete))

        def _watermark_rows() -> list[tuple[str, str, str, str, float]]:
            if not db_path.is_file():
                return []
            with sqlite3.connect(db_path) as conn:
                return list(
                    conn.execute(
                        "SELECT owner_key, session_id, platform, turn_id, delivered_at "
                        "FROM session_reply_watermarks WHERE session_id=?",
                        ("sess-receipt-1",),
                    ).fetchall()
                )

        # 1) Unconfirmed receipt MUST NOT trigger verified delivery update
        hc.emit_delivery_receipt(
            hc.DeliveryReceipt(
                platform="qqbot",
                chat_id="chat-maint",
                chat_type="dm",
                session_id="sess-receipt-1",
                turn_id="turn-101",
                message_id="msg-out-1",
                sender_id="maint-1",
                user_message="今天忙完啦",
                delivered_text="辛苦啦，歇会儿吧。",
                delivered_at=time.time(),
                transport_confirmed=False,
            )
        )
        time.sleep(0.15)
        assert _watermark_rows() == []
        assert inner_payloads == []

        # 2) Confirmed receipt triggers once, and duplicate emission for same turn_id is deduplicated
        delivered_at = time.time()
        confirmed = hc.DeliveryReceipt(
            platform="qqbot",
            chat_id="chat-maint",
            chat_type="dm",
            session_id="sess-receipt-1",
            turn_id="turn-101",
            message_id="msg-out-1",
            sender_id="maint-1",
            user_message="今天忙完啦",
            delivered_text="辛苦啦，歇会儿吧。",
            delivered_at=delivered_at,
            transport_confirmed=True,
        )
        duplicate = hc.DeliveryReceipt(
            platform="qqbot",
            chat_id="chat-maint",
            chat_type="dm",
            session_id="sess-receipt-1",
            turn_id="turn-101",
            message_id="msg-out-1",
            sender_id="maint-1",
            user_message="今天忙完啦",
            delivered_text="重复回执。",
            delivered_at=delivered_at + 10.0,
            transport_confirmed=True,
        )
        hc.emit_delivery_receipt(confirmed)
        hc.emit_delivery_receipt(duplicate)

        deadline = time.time() + 3.0
        rows = _watermark_rows()
        while (not rows or not inner_payloads) and time.time() < deadline:
            time.sleep(0.05)
            rows = _watermark_rows()
        time.sleep(0.15)
        rows = _watermark_rows()

        assert len(rows) == 1
        assert isinstance(rows[0][0], str) and rows[0][0]
        assert rows[0][1] == "sess-receipt-1"
        assert rows[0][2] == "qqbot"
        assert rows[0][3] == "turn-101"
        assert rows[0][4] == pytest.approx(delivered_at, abs=1e-3)
        assert len(inner_payloads) == 1
        assert inner_payloads[0]["user"] == "今天忙完啦"
        assert inner_payloads[0]["delivered_reply"] == "辛苦啦，歇会儿吧。"
    finally:
        hc.clear_all_contract_registrations()
