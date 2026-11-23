"""End-to-end acceptance tests for ViveMind autonomous resident review and protected skill gate.

Covers all 5 acceptance suites in ``ViveMind常驻与技能插件内自决-task.md``:
1. Single-statement grounded ``USER.md`` direct write + physical/semantic noise rejection
   (processed Bilibili and web materials, external sources, synthetic stranger DM,
   angry remarks, sarcastic jokes, transient meal/physiological states, hypotheticals,
   identity disclaimers, therapy-container clichés) + native ``memory(target="user")`` redirect.
2. Single-event candidate promotion (``sanae_mem_review(mode="stage")``) autonomous loop:
   direct CAS write + ``long`` promotion + zero ``pending`` files; rejection of ungrounded
   or short-term trivia; ``merge_evidence`` coverage deduplication.
3. ``USER.md`` 6,000-char capacity auto-eviction & anti-root-severing safety:
   retention marks (``【常驻保留】`` / ``【保留·不可删】``), core traits, and durable habits stay
   untouched; outdated recent statuses archive to ``memory-archive/user-demoted.jsonl`` and
   ``sanae.db`` ``events(tier='long')`` before CAS removal; simulated archive/DB failure aborts
   before ``USER.md`` deletion; zero ``pending`` files; legacy ``pending/sanae_mem/*.json``
   files are never silently replayed on startup or compaction.
4. Protected ``Skill`` hard-block (``blocked=True, require_approval=False``) vs. technical
   tool ``Skill`` pass-through, including live host ``skill_manage`` zero-``pending/skills``
   verification when host checkout is available, and zero concrete emotion words / blacklists.
5. Auxiliary judgment model fail-closed protection (timeout / None / malformed JSON on both
   write and demote paths, recording stable error codes in ``status.json`` without leaking
   text) and ``T:`` test-session byte-level isolation across ``USER.md``, ``sanae.db``, and
   ``status.json``.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import resolve_hermes_host_root  # noqa: E402

HERMES_ROOT = resolve_hermes_host_root() or (REPO_ROOT / "__no_host__")

if HERMES_ROOT.is_dir():
    HERMES_DEPS = HERMES_ROOT.with_name(HERMES_ROOT.name + "-deps")
    if HERMES_DEPS.is_dir() and str(HERMES_DEPS) not in sys.path:
        sys.path.insert(0, str(HERMES_DEPS))
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))

from sanae_mem.infra import auxiliary, host_bridge
from sanae_mem import plugin_entry
from sanae_mem.storage.candidates import parse_skill_mutation_result
from sanae_mem.governance.memory_guard import (
    PROTECTED_SKILL_BLOCK_MESSAGE,
    USER_REVIEW_REDIRECT_MESSAGE,
    ViveMindMemoryMutationGate,
    guard_memory_write,
)
from sanae_mem.governance.resident_store import ENTRY_DELIMITER
from sanae_mem.governance.source_intake import SourceMaterial, ingest_source_material


@contextmanager
def _turn_scope(
    *,
    actor: str = "maint-account-1",
    session_id: str = "dm-maint-1",
    platform: str = "qqbot",
    chat_type: str = "dm",
    chat_id: str = "c2c-maint-1",
    message_id: str = "msg-100",
):
    session_key = f"agent:main:{platform}:{chat_type}:{chat_id}"
    with host_bridge.turn_envelope_scope(
        platform=platform,
        chat_type=chat_type,
        chat_id=chat_id,
        sender_id=actor,
        session_id=session_id,
        session_key=session_key,
        message_id=message_id,
        write_origin="foreground",
        is_delegated_child=False,
        is_cron=False,
        is_test_isolated=session_id.startswith("T:"),
        is_messaging_surface=False,
    ):
        yield


def _build_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, user_char_limit: int = 6000):
    home = tmp_path / "hermes-home"
    memories = home / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    (memories / "USER.md").write_text("", encoding="utf-8")
    (memories / "MEMORY.md").write_text("", encoding="utf-8")
    cfg = {
        "data_dir": str(home / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "qqbot", "user_id": "maint-account-1"},
    }
    (home / "config.yaml").write_text(
        (
            f"memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
            f"  user_char_limit: {int(user_char_limit)}\n  memory_char_limit: 8000\n"
            f"plugins:\n  sanae_mem:\n"
            f"    data_dir: {json.dumps(cfg['data_dir'])}\n"
            f"    semantic_backend: sqlite\n"
            f"    memory_identity:\n      platform: qqbot\n      user_id: maint-account-1\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    provider = plugin_entry.SanaeMemProvider(config=cfg)
    with _turn_scope():
        provider.initialize("dm-maint-1", hermes_home=str(home), platform="qqbot")
    return home, provider


def _ingest_private_utterance(
    provider: plugin_entry.SanaeMemProvider,
    text: str,
    *,
    message_id: str,
    actor: str = "maint-account-1",
    session_id: str = "dm-maint-1",
    chat_id: str = "c2c-maint-1",
) -> int:
    utterance = f"{text}（重要）"
    with _turn_scope(
        actor=actor,
        session_id=session_id,
        chat_id=chat_id,
        message_id=message_id,
    ):
        provider.sync_turn(
            utterance,
            "好呀，我记下了。",
            session_id=session_id,
            messages=[{"role": "user", "content": utterance, "platform_message_id": message_id}],
        )
    row = provider.sqlite.fetchone(
        "SELECT id FROM events WHERE substr(raw, instr(raw, '] ') + 2) = ?", (utterance,),
    )
    assert row is not None
    return int(row[0])


# ──────────────────────────────────────────────────────────────────────
# 1. Single-Statement USER.md Direct Write & Layered Noise Rejection
# ──────────────────────────────────────────────────────────────────────


def test_single_private_statement_writes_user_md_and_layers_block_all_noise(tmp_path, monkeypatch):
    home, provider = _build_env(tmp_path, monkeypatch)
    user_file = home / "memories" / "USER.md"
    try:
        # Native memory(target="user") redirect check (both pre_tool_call & MutationGate)
        hook_block = guard_memory_write(
            "memory", {"action": "add", "target": "user", "content": "直接写档案"}, "dm-maint-1"
        )
        assert hook_block == {"action": "block", "message": USER_REVIEW_REDIRECT_MESSAGE}
        gate_block = ViveMindMemoryMutationGate().evaluate_memory_mutation(
            action="add", target="user", content="直接写档案"
        )
        blocked_flag = gate_block["blocked"] if isinstance(gate_block, dict) else gate_block.blocked
        req_app_flag = (
            gate_block.get("require_approval", False)
            if isinstance(gate_block, dict)
            else gate_block.require_approval
        )
        gate_msg = gate_block["message"] if isinstance(gate_block, dict) else gate_block.message
        assert blocked_flag is True and req_app_flag is False
        assert "sanae_mem_review" in gate_msg
        assert "pending" not in gate_msg.lower()

        # 1) Single verbatim private statement writes directly to USER.md with zero pending files
        valid_eid = _ingest_private_utterance(
            provider,
            "对了你记一下，我对芒果严重过敏，平时饮食绝对不碰芒果制品",
            message_id="msg-valid-1",
        )
        recognition = "对芒果严重过敏，平时饮食绝对不碰芒果制品"
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "summary": recognition,
                            "duty": "user",
                            "source_kind": "user_statement",
                            "long_term": True,
                            "covered_by": "",
                            "significance": "对方明确陈述的长期饮食健康禁忌",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        with _turn_scope(message_id="msg-review-1"):
            res = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_review",
                    {
                        "mode": "grounded_user",
                        "text": recognition,
                        "evidence_ids": [valid_eid],
                    },
                )
            )
        assert res["ok"] is True and res["result"]["ok"] is True
        assert res["result"]["written"] is True and res["result"]["support_linked"] is True
        assert user_file.read_text(encoding="utf-8") == recognition
        assert not (home / "pending").exists() or list((home / "pending").rglob("*.json")) == []

        # 2) Physical layer hard-rejects processed Bilibili and web materials and stranger DMs
        with _turn_scope(message_id="msg-intake-proc"):
            bili_proc = ingest_source_material(
                provider,
                SourceMaterial(
                    source_plugin="bilibili_public",
                    platform="bilibili",
                    material_kind="processed",
                    processing_type="subtitle_slice",
                    processing_version="v1",
                    content="字幕切片：我平时只用机械键盘",
                    privacy_scope="private",
                    source_object_id="BV1test#1",
                ),
                create_memory_event=True,
            )
            web_proc = ingest_source_material(
                provider,
                SourceMaterial(
                    source_plugin="external_web",
                    platform="web_digest",
                    material_kind="processed",
                    processing_type="summary",
                    processing_version="v1",
                    content="网页摘要：我平时只用机械键盘",
                    privacy_scope="private",
                    source_object_id="digest-1",
                ),
                create_memory_event=True,
            )
        for proc_id in (int(bili_proc.linked_event_id), int(web_proc.linked_event_id)):
            with _turn_scope(message_id="msg-proc-check"):
                rejected_proc = provider.resident.save_grounded_user(
                    text="平时只用机械键盘", evidence_ids=[proc_id]
                )
            assert rejected_proc["ok"] is False

        # Synthetic stranger DM (purely fabricated test account, cleaned with tmp_path)
        stranger_provider = plugin_entry.SanaeMemProvider(config=provider.runtime_config())
        try:
            with _turn_scope(
                actor="synthetic-stranger-99",
                session_id="dm-stranger-99",
                chat_id="c2c-stranger-99",
                message_id="msg-stranger-1",
            ):
                stranger_provider.initialize("dm-stranger-99", hermes_home=str(home), platform="qqbot")
                stranger_eid = _ingest_private_utterance(
                    stranger_provider,
                    "我每天早上固定六点起床晨跑五公里",
                    message_id="msg-stranger-1",
                    actor="synthetic-stranger-99",
                    session_id="dm-stranger-99",
                    chat_id="c2c-stranger-99",
                )
        finally:
            stranger_provider.shutdown()

        with _turn_scope(message_id="msg-cross-owner-check"):
            rejected_stranger = provider.resident.save_grounded_user(
                text="每天早上六点起床晨跑", evidence_ids=[stranger_eid]
            )
        assert rejected_stranger["ok"] is False
        assert rejected_stranger["reason"] in {"evidence_event_missing", "evidence_owner_mismatch"}

        # 3) Judgment model layer rejects angry remarks, sarcastic jokes, transient meal/state,
        #    hypotheticals, identity disclaimers, and therapy clichés -> zero USER.md pollution
        noise_samples = [
            ("行了别说了，以后都别再聊这些了", "觉得以后没必要再继续聊"),
            ("再改不好这个脚本我明天就去南极养企鹅", "打算明天去南极养企鹅"),
            ("今天中午吃了碗米粉，刚睡醒头还有点晕", "中午吃了米粉且午睡后头晕"),
            ("要是哪天中了奖，我就去海边开一家旧书店", "准备去海边开一家旧书店"),
            ("你本来就不是现实里的人，不用假装自己是真人", "清楚对方不是现实里的人"),
            ("我不开心的时候你只要当个安全容器稳稳托住我就行", "低落时需要安全容器托住"),
        ]
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "summary": "",
                            "duty": "none",
                            "source_kind": "user_statement",
                            "long_term": False,
                            "covered_by": "",
                            "significance": "非长期真实事实，不入常驻档案",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        for idx, (utterance, proposed_note) in enumerate(noise_samples, start=1):
            noise_eid = _ingest_private_utterance(
                provider, utterance, message_id=f"msg-noise-{idx}"
            )
            with _turn_scope(message_id=f"msg-noise-rev-{idx}"):
                out = provider.resident.save_grounded_user(
                    text=proposed_note, evidence_ids=[noise_eid]
                )
            assert out["ok"] is False
            assert out["reason"] == "not_a_long_term_user_candidate"

        assert user_file.read_text(encoding="utf-8") == recognition
        assert not (home / "pending").exists() or list((home / "pending").rglob("*.json")) == []
    finally:
        provider.shutdown()


def test_foreground_reviews_do_not_consume_the_background_distill_cap(tmp_path, monkeypatch):
    home, provider = _build_env(tmp_path, monkeypatch)
    try:
        noise = [
            ("记住我刚吃完今天的午饭，现在准备继续处理手头工作", "午饭刚吃完"),
            ("记住这次排障真不想再看了，以后服务器有事也不要找我", "以后不再参与排障"),
            ("记住如果今年能够放长假，我就去海边住上一段时间", "长期计划去海边"),
        ]
        for idx, (utterance, recognition) in enumerate(noise):
            eid = _ingest_private_utterance(provider, utterance, message_id=f"quota-noise-{idx}")
            provider.attach_runtime_llm(
                SimpleNamespace(
                    complete=lambda *_a, **_kw: SimpleNamespace(text=json.dumps({
                        "summary": "", "duty": "none", "source_kind": "user_statement",
                        "long_term": False, "covered_by": "", "significance": "非长期事实",
                    }, ensure_ascii=False))
                )
            )
            with _turn_scope(message_id=f"quota-noise-review-{idx}"):
                rejected = provider.resident.save_grounded_user(text=recognition, evidence_ids=[eid])
            assert rejected["reason"] == "not_a_long_term_user_candidate"

        grounded_text = "左耳高频听力长期偏弱，语音配乐应避开尖锐高频。"
        grounded_id = _ingest_private_utterance(
            provider, "记住我左耳高频听力从小偏弱，语音尽量避开尖锐高频配乐。",
            message_id="quota-grounded",
        )
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(text=json.dumps({
                    "summary": grounded_text, "duty": "user", "source_kind": "user_statement",
                    "long_term": True, "covered_by": "", "significance": "长期听力事实",
                }, ensure_ascii=False))
            )
        )
        with _turn_scope(message_id="quota-grounded-review"):
            grounded = provider.resident.save_grounded_user(text=grounded_text, evidence_ids=[grounded_id])
        assert grounded["ok"] is True and grounded["written"] is True

        stage_id = _ingest_private_utterance(
            provider, "我长期使用纯 POSIX sh 编写 Alpine 容器脚本。", message_id="quota-stage"
        )
        stage_text = "长期使用纯 POSIX sh 编写 Alpine 容器脚本。"
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(text=json.dumps({
                    "summary": stage_text, "duty": "user", "source_kind": "user_statement",
                    "long_term": True, "covered_by": "", "significance": "长期技术习惯",
                }, ensure_ascii=False))
            )
        )
        with _turn_scope(message_id="quota-stage-review"):
            staged = provider.resident.promote_reviewed_event(stage_id)
        assert staged["ok"] is True and staged["promoted"] is True

        status = provider.status_store.read()
        assert status.get("daily_resident_reviews") == 5
        assert status.get("daily_candidate_reviews", 0) == 0

        utterances = (
            "别忘了我家宽带每年在三月续费。",
            "我打算继续按周一整理会议纪要的习惯来。",
            "我要继续每个周五坐地铁去看望外婆。",
        )
        background_ids = [
            _ingest_private_utterance(provider, utterance, message_id=f"quota-background-{idx}")
            for idx, utterance in enumerate(utterances)
        ]
        with _turn_scope(message_id="quota-distill"):
            distilled = provider.lifecycle.distill()
        assert distilled["cap"] == distilled["candidates"] == 2
        written_ids = [item["id"] for item in distilled["items"]]
        assert written_ids == [background_ids[2], background_ids[1]]
        rows = provider.sqlite.fetchall(
            "SELECT id, raw, summary_source, distilled_at FROM events WHERE id IN (?, ?, ?) ORDER BY id",
            tuple(background_ids),
        )
        assert [row["id"] for row in rows if row["distilled_at"]] == sorted(written_ids)
        assert all(row["summary_source"] == "distilled" for row in rows if row["distilled_at"])
        assert all(utterance in row["raw"] for utterance, row in zip(utterances, rows))
        assert provider.status_store.read()["daily_candidate_reviews"] == 2
    finally:
        provider.shutdown()


# ──────────────────────────────────────────────────────────────────────
# 2. Single-Candidate Promotion Autonomous Closed Loop (Zero Pending)
# ──────────────────────────────────────────────────────────────────────


def test_single_candidate_promotion_commits_or_rejects_without_pending(tmp_path, monkeypatch):
    home, provider = _build_env(tmp_path, monkeypatch)
    user_file = home / "memories" / "USER.md"
    memory_file = home / "memories" / "MEMORY.md"
    try:
        # A) Grounded long-term candidate promotes directly to USER.md and tier='long'
        eid_valid = _ingest_private_utterance(
            provider,
            "你记住，以后给我发离线操作步骤时把回退方案一起发全",
            message_id="msg-stage-1",
        )
        promoted_text = "执行离线操作步骤时习惯一次拿到完整流程与回退方案"
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "summary": promoted_text,
                            "duty": "user",
                            "source_kind": "user_statement",
                            "long_term": True,
                            "covered_by": "",
                            "significance": "长期固定协作习惯",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        with _turn_scope(message_id="msg-stage-run-1"):
            stage_ok = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_review", {"mode": "stage", "event_id": eid_valid}
                )
            )["result"]
        assert stage_ok["ok"] is True and stage_ok["promoted"] is True
        assert stage_ok["written"] is True and stage_ok["staged"] is False
        assert user_file.read_text(encoding="utf-8") == promoted_text
        row_tier = provider.sqlite.fetchone(
            "SELECT tier FROM events WHERE id=?", (eid_valid,)
        )[0]
        assert row_tier == "long"

        # B) Duplicate / covered candidate merges evidence without duplicating USER.md entry
        eid_covered = _ingest_private_utterance(
            provider,
            "你记着，我做离线操作时向来要连同回退兜底步骤一起拿全",
            message_id="msg-stage-2",
        )
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "summary": "离线排障习惯连同回退命令一次拿全",
                            "duty": "user",
                            "source_kind": "user_statement",
                            "long_term": True,
                            "covered_by": promoted_text,
                            "significance": "与既有协作习惯同义",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        with _turn_scope(message_id="msg-stage-run-2"):
            stage_merged = provider.resident.promote_reviewed_event(eid_covered)
        assert stage_merged["ok"] is True and stage_merged["promoted"] is True
        assert stage_merged["written"] is False and stage_merged["staged"] is False
        assert user_file.read_text(encoding="utf-8") == promoted_text

        # C) Event without verbatim source_snapshot (e.g., manual sanae_mem_add) is rejected
        with _turn_scope(message_id="msg-manual-add"):
            added = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_add",
                    {"content": "今天顺手整理了一下桌面文件夹"},
                )
            )["result"]
        manual_eid = int(added["id"])
        with _turn_scope(message_id="msg-stage-manual"):
            rejected_manual = provider.resident.promote_reviewed_event(manual_eid)
        assert rejected_manual["ok"] is False
        assert rejected_manual["reason"] == "evidence_source_unqualified"

        # D) Short-term trivia with duty="none", long_term=False is rejected cleanly
        eid_trivia = _ingest_private_utterance(
            provider, "今天下午快递刚送到了两箱矿泉水", message_id="msg-stage-trivia"
        )
        provider.attach_runtime_llm(
            SimpleNamespace(
                complete=lambda *_a, **_kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "summary": "",
                            "duty": "none",
                            "source_kind": "user_statement",
                            "long_term": False,
                            "covered_by": "",
                            "significance": "单次日常收快递琐事",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        with _turn_scope(message_id="msg-stage-run-trivia"):
            rejected_trivia = provider.resident.promote_reviewed_event(eid_trivia)
        assert rejected_trivia["ok"] is False and rejected_trivia["promoted"] is False
        assert memory_file.read_text(encoding="utf-8") == ""
        assert not (home / "pending").exists() or list((home / "pending").rglob("*.json")) == []
    finally:
        provider.shutdown()


# ──────────────────────────────────────────────────────────────────────
# 3. USER.md 6000-Char Capacity Auto-Eviction, Failure Abort & Legacy Safety
# ──────────────────────────────────────────────────────────────────────


def test_user_md_6000_capacity_auto_evicts_only_outdated_and_never_severs_roots(
    tmp_path, monkeypatch
):
    home, provider = _build_env(tmp_path, monkeypatch, user_char_limit=6000)
    user_file = home / "memories" / "USER.md"
    try:
        # Pre-seed legacy pending file on disk: must NEVER be auto-applied!
        legacy_dir = home / "pending" / "sanae_mem"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_id = "a" * 32
        legacy_file = legacy_dir / f"{legacy_id}.json"
        legacy_file.write_text(
            json.dumps(
                {
                    "id": legacy_id,
                    "summary": "历史遗留待审单",
                    "payload": {
                        "action": "add",
                        "target": "user",
                        "content": "绝不能未经校验静默批量写入的历史旧单内容",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        filler = "按既定流程执行并保留回退记录。" * 4
        initial_entries = [f"长期技术习惯{i:02d}：{filler}" for i in range(83)]
        kept_mark_1 = "【常驻保留】核心身份锚点：" + ("甲" * 56)
        kept_mark_2 = "【保留·不可删】基础相处边界：" + ("乙" * 55)
        core_trait = "核心人物底色：独立软件开发者，做事重逻辑闭环与可回退性。" + ("丙" * 38)
        durable_pref = "长期协作偏好：离线运维步骤与回退兜底方案一次发全。" + ("丁" * 42)
        outdated_status = "2026年8月中旬阶段近况：正在外地出差赶两周的短期交付项目。" + ("戊" * 34)
        initial_entries[0:4] = [kept_mark_1, kept_mark_2, core_trait, durable_pref]
        initial_entries[25] = outdated_status
        initial_entries[-1] += "末" * (5985 - len(ENTRY_DELIMITER.join(initial_entries)))
        initial_body = ENTRY_DELIMITER.join(initial_entries)
        assert 5900 < len(initial_body) <= 6000
        user_file.write_text(initial_body, encoding="utf-8")

        # Compact / re-init must not replay legacy pending file into USER.md
        provider.lifecycle.compact()
        assert "历史旧单内容" not in user_file.read_text(encoding="utf-8")

        new_fact_raw = "你帮我记住，我每周固定在周六上午去市图书馆整理阅读笔记"
        new_recognition = "每周固定在周六上午去市图书馆整理阅读笔记。" + ("己" * 45)
        eid_new = _ingest_private_utterance(provider, new_fact_raw, message_id="msg-cap-1")

        demote_batches = []

        def fake_complete(messages, **kwargs):
            purpose = kwargs.get("purpose", "")
            if purpose == "resident_user_demote_review":
                user_prompt = messages[-1]["content"]
                assert "【当前日期】" in user_prompt
                decisions = []
                for line in user_prompt.splitlines():
                    if not line.startswith("["):
                        continue
                    idx = int(line.split("]", 1)[0][1:])
                    if "2026年8月中旬阶段近况" in line:
                        decisions.append(
                            {
                                "index": idx,
                                "allow_demote": True,
                                "kind": "outdated_recent_status",
                                "reason": "8月中旬两周出差已结束翻篇",
                            }
                        )
                    elif "核心人物底色" in line:
                        decisions.append(
                            {
                                "index": idx,
                                "allow_demote": False,
                                "kind": "core_trait_or_identity",
                                "reason": "核心身份与做事底色不退居",
                            }
                        )
                    else:
                        decisions.append(
                            {
                                "index": idx,
                                "allow_demote": False,
                                "kind": "long_term_habit_or_preference",
                                "reason": "长期固定协作偏好不退居",
                            }
                        )
                demote_batches.append([item["index"] for item in decisions])
                assert 1 <= len(decisions) <= 10
                assert kwargs["max_tokens"] == max(512, 220 * len(decisions))
                return SimpleNamespace(text=json.dumps({"decisions": decisions}, ensure_ascii=False))
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "summary": new_recognition,
                        "duty": "user",
                        "source_kind": "user_statement",
                        "long_term": True,
                        "covered_by": "",
                        "significance": "长期编码协作习惯",
                    },
                    ensure_ascii=False,
                )
            )

        provider.attach_runtime_llm(SimpleNamespace(complete=fake_complete))

        archive_path = home / "memory-archive" / "user-demoted.jsonl"
        archive_path.mkdir(parents=True)  # The archive destination cannot be opened as a file.
        with _turn_scope(message_id="msg-cap-arch-fail"):
            arch_fail = provider.resident.save_grounded_user(
                text=new_recognition, evidence_ids=[eid_new]
            )
        assert arch_fail["ok"] is False and arch_fail["reason"] == "archive_failed"
        assert user_file.read_text(encoding="utf-8") == initial_body
        archive_path.rmdir()

        with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db", isolation_level=None)) as db:
            db.execute(
                "CREATE TRIGGER reject_resident_demotion BEFORE INSERT ON events "
                "WHEN NEW.promoted_reason='resident_demote' BEGIN "
                "SELECT RAISE(ABORT, 'resident event unavailable'); END"
            )
            with _turn_scope(message_id="msg-cap-db-fail"):
                db_fail = provider.resident.save_grounded_user(
                    text=new_recognition, evidence_ids=[eid_new]
                )
            assert db_fail["ok"] is False and db_fail["reason"] == "event_write_failed"
            assert user_file.read_text(encoding="utf-8") == initial_body
            assert outdated_status in archive_path.read_text(encoding="utf-8")
            assert db.execute(
                "SELECT count(*) FROM events WHERE promoted_reason='resident_demote'"
            ).fetchone()[0] == 0
            db.execute("DROP TRIGGER reject_resident_demotion")

        with _turn_scope(message_id="msg-cap-ok"):
            ok_res = provider.resident.save_grounded_user(
                text=new_recognition, evidence_ids=[eid_new]
            )
        assert ok_res["ok"] is True and ok_res["written"] is True
        assert ok_res["evicted_count"] == 1
        assert any(batch and batch[0] >= 20 and 25 in batch for batch in demote_batches)

        final_entries = user_file.read_text(encoding="utf-8").split(ENTRY_DELIMITER)
        assert kept_mark_1 in final_entries
        assert kept_mark_2 in final_entries
        assert core_trait in final_entries
        assert durable_pref in final_entries
        assert new_recognition in final_entries
        assert outdated_status not in final_entries
        assert len(user_file.read_text(encoding="utf-8")) <= 6000

        # Verify archive backup & searchable long-term event in sanae.db
        archive_text = (home / "memory-archive" / "user-demoted.jsonl").read_text(encoding="utf-8")
        assert outdated_status in archive_text
        with _turn_scope(message_id="msg-search-demoted"):
            recalled = json.loads(
                provider.handle_tool_call("sanae_mem_search", {"query": "外地出差赶两周的短期交付项目"})
            )["result"]
        assert any("外地出差" in item["raw"] for item in recalled)

        # No new pending files created (only the pre-seeded untouched legacy file remains)
        assert [p.name for p in legacy_dir.glob("*.json")] == [f"{legacy_id}.json"]
        assert "历史旧单内容" not in user_file.read_text(encoding="utf-8")
    finally:
        provider.shutdown()


# ──────────────────────────────────────────────────────────────────────
# 4. Protected Skill Hard-Block vs. Technical Tool Skill Pass-Through
# ──────────────────────────────────────────────────────────────────────


def test_protected_skills_hard_block_without_staging_and_technical_skills_pass(
    tmp_path, monkeypatch
):
    gate = ViveMindMemoryMutationGate()

    # Check guidance message hygiene: zero concrete emotion words, zero blacklists, clear routing
    for forbidden_word in ("难受", "委屈", "生气", "发懵", "冤", "接住", "容器", "不骗", "pending", "审批"):
        assert forbidden_word not in PROTECTED_SKILL_BLOCK_MESSAGE
    assert "sanae_inner_state" in PROTECTED_SKILL_BLOCK_MESSAGE
    assert "sanae_mem_add" in PROTECTED_SKILL_BLOCK_MESSAGE

    # Without an auxiliary LLM, both interpersonal and technical skill writes fail closed uniformly
    # without relying on keyword/name blacklists.
    for skill_name, content in [
        ("neutral-helper", "---\nname: neutral-helper\ndescription: guide\ncategory: emotion/subtype\n---\n正文\n"),
        ("persona-voice-guide", "---\nname: persona-voice-guide\ndescription: guide\n---\n# Guide\n"),
        ("partner-comfort-sop", "---\nname: sop\ndescription: 情绪安抚与回应步骤\n---\n# 步骤\n"),
        ("chat-boundary-notes", "---\nname: notes\ndescription: 日常相处分寸整理\n---\n# 笔记\n"),
        ("resident-archive-policy", "---\nname: policy\ndescription: 常驻记忆规则\n---\n# 规则\n"),
        ("custom-helper", "---\nname: custom-helper\ndescription: tool\n---\n## 说话规矩\n正文\n"),
        (
            "sqlite-wal-checkpoint",
            "---\nname: sqlite-wal-checkpoint\ndescription: Run WAL checkpoint safely\n---\n# Steps\nRun PRAGMA wal_checkpoint(TRUNCATE).\n",
        ),
        (
            "nginx-log-analyzer",
            "---\nname: nginx-log-analyzer\ndescription: 统计访问日志状态码与慢请求耗时\n---\n# 用法\n解析 access.log 输出 P99。\n",
        ),
    ]:
        unavail = gate.evaluate_skill_mutation(
            action="create", name=skill_name, args={"content": content}
        )
        assert unavail is not None, skill_name
        blocked = unavail["blocked"] if isinstance(unavail, dict) else unavail.blocked
        allow = unavail["allow"] if isinstance(unavail, dict) else unavail.allow
        req_app = (
            unavail.get("require_approval", False)
            if isinstance(unavail, dict)
            else unavail.require_approval
        )
        msg = unavail["message"] if isinstance(unavail, dict) else unavail.message
        assert blocked is True and allow is False and req_app is False
        assert "暂不可用" in str(msg or "")

    # When minimal Hermes host is importable, exercise real skill_manage tool end-to-end
    # through real PluginManager + outermost loopback HTTP auxiliary server.
    if importlib.util.find_spec("hermes_host_contracts") and importlib.util.find_spec(
        "tools.skill_manager_tool"
    ):
        home = tmp_path / "skill-host-home"
        (home / "skills").mkdir(parents=True)
        script = f"""
import json, os, pathlib, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import yaml
sys.path[:0] = {sys.path[:4]!r}
home = pathlib.Path({str(home)!r})
os.environ["HERMES_HOME"] = str(home)
os.environ.pop("HERMES_HOME_OVERRIDE", None)

http_calls = []
class _JudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {{}}
        http_calls.append(body)
        user_msg = ""
        for msg in body.get("messages") or []:
            if msg.get("role") == "user":
                user_msg = str(msg.get("content") or "")
        if "systemd-journal-vacuum-check" in user_msg and "情绪安抚" not in user_msg:
            verdict_obj = {{"allow": True, "domain": "objective_tool_or_technical", "reason": "客观工具操作"}}
        else:
            verdict_obj = {{"allow": False, "domain": "protected_persona_or_interpersonal", "reason": "涉及人格、情绪或常驻规则"}}
        payload = json.dumps(verdict_obj, ensure_ascii=False)
        resp = json.dumps({{
            "id": "chatcmpl-skill-gate",
            "object": "chat.completion",
            "choices": [{{"index": 0, "message": {{"role": "assistant", "content": payload}}, "finish_reason": "stop"}}],
            "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        }}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, format, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", 0), _JudgeHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{{port}}/v1"

(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "auxiliary": {{
            "sanae_extract": {{"provider": "custom", "model": "test-judge-model", "base_url": base_url, "api_key": "dummy-key"}},
        }},
        "plugins": {{
            "sanae_mem": {{
                "enabled": True,
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "qqbot", "user_id": "maint-1"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
from sanae_mem import plugin_entry
from sanae_mem.infra import host_bridge
from tools.skill_manager_tool import skill_manage

with host_bridge.hermes_home_scope(home):
    manager = get_plugin_manager()
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    plugin_entry.register(ctx)
    try:
        with host_bridge.turn_envelope_scope(
            platform="qqbot", chat_type="dm", chat_id="maint-1", sender_id="maint-1",
            session_id="sess-skill-gate", session_key="agent:main:qqbot:dm:maint-1",
            turn_id="turn-skill-gate", message_id="msg-skill-gate", write_origin="foreground",
        ):
            blocked_cases = [
                dict(action="create", name="partner-conflict-soothing", category="emotion",
                     content="---\\nname: partner-conflict-soothing\\ndescription: 伴侣争执时的安抚流程\\ncategory: emotion\\n---\\n遇到对方语气不好时先退一步温柔安抚。\\n"),
                dict(action="create", name="daily-tone-self-check", category="persona",
                     content="---\\nname: daily-tone-self-check\\ndescription: 日常说话腔调与口癖核对\\ncategory: persona\\n---\\n保持体面敬语与自然反差。\\n"),
                dict(action="create", name="couple-chat-pacing", category="relationship",
                     content="---\\nname: couple-chat-pacing\\ndescription: 恋人日常回应节奏与关心准则\\ncategory: relationship\\n---\\n对方忙工作时不要连续追问。\\n"),
                dict(action="create", name="dialogue-handling-notes",
                     content="---\\nname: dialogue-handling-notes\\ndescription: 对话处理备忘\\n---\\n当对方在私聊里叹气或发脾气时，先体察对方的疲惫，不要讲大道理，也不要写检讨。常驻记忆 USER.md 只记长期事实。\\n"),
            ]
            for blocked_args in blocked_cases:
                blocked_raw = json.loads(skill_manage(**blocked_args))
                assert blocked_raw.get("success") is False or "error" in blocked_raw, blocked_raw
                assert "sanae_inner_state" in json.dumps(blocked_raw, ensure_ascii=False)
            assert len(http_calls) == len(blocked_cases), http_calls

            allowed_raw = json.loads(skill_manage(
                action="create",
                name="systemd-journal-vacuum-check",
                category="devops",
                content="---\\nname: systemd-journal-vacuum-check\\ndescription: 检查日志空间与 memory usage\\ncategory: devops\\n---\\nRun `journalctl --disk-usage`.\\n```sql\\nSELECT relationship FROM tables;\\n```\\n",
            ))
            assert allowed_raw.get("success") is True, allowed_raw
            assert len(http_calls) == len(blocked_cases) + 1
            tech_file = home / "skills" / "devops" / "systemd-journal-vacuum-check" / "SKILL.md"
            before_patch = tech_file.read_bytes()
            blocked_patch = json.loads(skill_manage(
                action="patch", name="devops/systemd-journal-vacuum-check",
                old_string="Run `journalctl --disk-usage`.",
                new_string="遇到伴侣情绪不好时做情绪安抚并遵守相处分寸与常驻记忆规则。",
            ))
            assert blocked_patch.get("success") is False or "error" in blocked_patch, blocked_patch
            assert tech_file.read_bytes() == before_patch
            assert len(http_calls) == len(blocked_cases) + 2
            allowed_patch = json.loads(skill_manage(
                action="patch", name="devops/systemd-journal-vacuum-check",
                old_string="Run `journalctl --disk-usage`.",
                new_string="Run `journalctl --disk-usage` and report memory usage.",
            ))
            assert allowed_patch.get("success") is True, allowed_patch
            assert len(http_calls) == len(blocked_cases) + 3
        print("HOST_SKILL_GATE_OK")
    finally:
        manager.unload()
        server.shutdown()
        server.server_close()
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert "HOST_SKILL_GATE_OK" in proc.stdout
        for category, name in [
            ("emotion", "partner-conflict-soothing"),
            ("persona", "daily-tone-self-check"),
            ("relationship", "couple-chat-pacing"),
            ("", "dialogue-handling-notes"),
        ]:
            assert not (home / "skills" / category / name).exists()
        assert not (home / "pending" / "skills").exists() or (
            list((home / "pending" / "skills").glob("*.json")) == []
        )
        tech_file = home / "skills" / "devops" / "systemd-journal-vacuum-check" / "SKILL.md"
        assert tech_file.is_file()
        assert "memory usage" in tech_file.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# 5. Judgment Model Fault Fail-Closed & T: Test Session Byte Isolation
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("allow,domain", [
    (True, "objective_tool_or_technical"),
    (False, "protected_persona_or_interpersonal"),
])
@pytest.mark.parametrize("reason", [
    "按实际用途分类",
    "涉及tar命令归档与SHA256校验等客观系统运维操作",
    "该补丁仅补充归档完成后的解包核验步骤，使用校验文件检查归档内容是否完整，"
    "所有步骤均为客观命令行操作，不涉及人设、人际回应流程或个人记忆写入规矩。",
    "",
    " \n ",
    "  " + "归档与校验操作说明" * 30 + "  ",
])
def test_skill_review_parser_accepts_only_consistent_domains(allow, domain, reason):
    item = {"allow": allow, "domain": domain, "reason": reason}
    expected = {**item, "reason": reason.strip()[:240]}
    text = json.dumps(item, ensure_ascii=False)
    for raw in (item, text, f" \n{text}\n ", f" \n```json\n{text}\n```\n ", f"prefix {text}"):
        assert parse_skill_mutation_result(raw) == expected
    assert parse_skill_mutation_result({**item, "allow": not allow}) is None


@pytest.mark.parametrize("raw", [
    None, "", "{broken", "[]", "null",
    '{"allow": 1, "domain": "objective_tool_or_technical", "reason": "技术"}',
    '{"allow": "true", "domain": "objective_tool_or_technical", "reason": "技术"}',
    '{"allow": true, "domain": "unknown", "reason": "技术"}',
    '{"allow": true, "domain": "objective_tool_or_technical"}',
    '{"allow": true, "domain": "objective_tool_or_technical", "reason": 7}',
    '{"allow": true, "domain": "objective_tool_or_technical", "reason": null}',
    '{"allow": true, "domain": "objective_tool_or_technical", "reason": []}',
    '{"allow": true, "domain": "objective_tool_or_technical", "reason": true}',
])
def test_skill_review_parser_rejects_malformed_results(raw):
    assert parse_skill_mutation_result(raw) is None


NATURAL_RESPONSE_STEPS = (
    "当对方加班后说话很冲时，先轻声问他是不是太累了，再肯定他的付出；"
    "不要争辩对错，等他缓过来再聊。以后遇到同类场景都按这套步骤回应。"
)
TECHNICAL_STEPS = (
    "检查 memory usage，并运行 `ss -lnt` 核对监听端口。\n"
    "```sql\nSELECT relationship FROM tables;\n```\n"
)


@pytest.mark.parametrize("action,field", [
    ("create", "content"), ("patch", "new_string"),
    ("write_file", "file_content"), ("edit", "description"),
])
def test_skill_semantic_review_receives_unmatched_text_and_blocks(action, field):
    calls, statuses = [], []

    def complete(messages, **kwargs):
        calls.append((messages, kwargs))
        return SimpleNamespace(text=json.dumps({
            "allow": False, "domain": "protected_persona_or_interpersonal",
            "reason": "规定人际回应步骤",
        }))

    gate = ViveMindMemoryMutationGate(
        llm_getter=lambda: SimpleNamespace(complete=complete),
        on_review_status=lambda **status: statuses.append(status),
    )
    verdict = gate.evaluate_skill_mutation(
        action=action, name="daily-notes", args={field: NATURAL_RESPONSE_STEPS},
    )
    assert _skill_verdict_fields(verdict) == (False, True, False, PROTECTED_SKILL_BLOCK_MESSAGE)
    assert len(calls) == 1
    messages, kwargs = calls[0]
    assert json.loads(messages[1]["content"])["text"] == NATURAL_RESPONSE_STEPS
    assert kwargs == dict(task=auxiliary.TASK, temperature=0.0, max_tokens=240,
                          timeout=30, purpose="skill_mutation_review")
    assert statuses == [dict(ok=True, stage="review", error="")]


def _skill_verdict_fields(verdict):
    return tuple(verdict[key] if isinstance(verdict, dict) else getattr(verdict, key)
                 for key in ("allow", "blocked", "require_approval", "message"))


def test_skill_delete_and_empty_writes_use_no_model_tokens():
    def unexpected_model():
        pytest.fail("empty writes and deletion must not call the model")

    gate = ViveMindMemoryMutationGate(llm_getter=unexpected_model)
    assert gate.evaluate_skill_mutation(action="delete", name="persona-guide") is None
    assert gate.evaluate_skill_mutation(action="edit", name="neutral", args={"content": " "}) is None
    assert gate.evaluate_skill_mutation(action="batch", operations=[
        {"action": " DELETE ", "name": "persona-guide", "content": NATURAL_RESPONSE_STEPS},
    ]) is None


def test_skill_semantic_review_offline_exception_and_runtime_faults():
    class AuxiliaryClientUnavailable(RuntimeError):
        pass

    statuses = []
    fault = AuxiliaryClientUnavailable("no configured credentials")

    def complete(*_args, **_kwargs):
        raise fault

    gate = ViveMindMemoryMutationGate(
        llm_getter=lambda: SimpleNamespace(complete=complete),
        on_review_status=lambda **status: statuses.append(status),
    )
    verdict = gate.evaluate_skill_mutation(
        action="create", name="port-check", args={"content": TECHNICAL_STEPS},
    )
    assert _skill_verdict_fields(verdict) == (
        False, True, False, "技能内容安全复核暂不可用，本次写入已中止；请稍后重试。",
    )
    assert statuses == [dict(ok=False, stage="review", error="auxiliary_unavailable")]
    for fault in (TimeoutError("private diagnostic text"), ConnectionError("network"),
                  PermissionError("authentication")):
        verdict = gate.evaluate_skill_mutation(
            action="create", name="port-check", args={"content": TECHNICAL_STEPS},
        )
        assert _skill_verdict_fields(verdict) == (
            False, True, False, "技能内容安全复核暂不可用，本次写入已中止；请稍后重试。",
        )
        assert statuses[-1] == dict(ok=False, stage="review", error=type(fault).__name__)


def test_auxiliary_faults_fail_closed_on_write_and_demote_and_t_session_is_isolated(
    tmp_path, monkeypatch
):
    home, provider = _build_env(tmp_path, monkeypatch)
    user_file = home / "memories" / "USER.md"
    status_file = home / "sanae-mem" / "status.json"
    db_file = home / "sanae-mem" / "sanae.db"
    seed_note = "习惯在周末上午整理一周的技术阅读笔记"
    user_file.write_text(seed_note, encoding="utf-8")
    try:
        eid = _ingest_private_utterance(
            provider, "我每周日晚上都会备份一次移动硬盘里的照片库", message_id="msg-fault-1"
        )

        # Simulate timeout exception, None return, and malformed JSON on BOTH write & demote!
        faults = [
            lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("auxiliary timeout")),
            lambda *_a, **_kw: None,
            lambda *_a, **_kw: SimpleNamespace(text="{truncated_invalid_json"),
        ]
        for idx, fault_fn in enumerate(faults, start=1):
            provider.attach_runtime_llm(SimpleNamespace(complete=fault_fn))
            with _turn_scope(message_id=f"msg-fault-w-{idx}"):
                write_res = provider.resident.save_grounded_user(
                    text="每周日晚上备份照片库隐私敏感词测试", evidence_ids=[eid]
                )
                stage_res = provider.resident.promote_reviewed_event(eid)
                demote_res = provider.resident.demote_resident_entry("user", seed_note)

            assert write_res["ok"] is False
            assert stage_res["ok"] is False and stage_res["promoted"] is False
            assert demote_res["ok"] is False and demote_res["demoted"] is False
            assert demote_res["reason"] == "demote_review_unavailable"
            assert user_file.read_text(encoding="utf-8") == seed_note

        status_raw = status_file.read_text(encoding="utf-8")
        status_json = json.loads(status_raw)
        assert status_json["components"]["resident_review"]["ok"] is False
        assert status_json["components"]["resident_demote"]["ok"] is False
        assert status_json["components"]["resident_demote"]["error"] == "demote_review_unavailable"
        assert "隐私敏感词测试" not in status_raw
        assert seed_note not in status_raw

        # T: test-isolated session must leave USER.md, sanae.db (including WAL), and status.json untouched!
        provider.sqlite.read(lambda conn: conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"))
        user_before = user_file.read_bytes()
        db_before = db_file.read_bytes()
        status_before = status_file.read_bytes()

        old_sid = "dm-maint-1"
        provider.on_session_switch("T:probe-isolated-99")
        try:
            with _turn_scope(session_id="T:probe-isolated-99", message_id="msg-t-1"):
                tool_rev = json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_review",
                        {
                            "mode": "grounded_user",
                            "text": "测试会话不应写入",
                            "evidence_ids": [eid],
                        },
                        session_id="T:probe-isolated-99",
                    )
                )
                assert tool_rev["ok"] is True
                assert tool_rev["result"]["reason"] == "test_session_isolated"

                direct_save = provider.resident.save_grounded_user(
                    text="测试会话不应写入", evidence_ids=[eid]
                )
                assert direct_save["ok"] is False

                direct_demote = provider.resident.demote_resident_entry("user", seed_note)
                assert direct_demote["ok"] is False
                assert direct_demote["reason"] == "test_session_isolated"
        finally:
            provider.on_session_switch(old_sid)

        provider.sqlite.read(lambda conn: conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"))
        assert user_file.read_bytes() == user_before
        assert db_file.read_bytes() == db_before
        assert status_file.read_bytes() == status_before
    finally:
        provider.shutdown()
