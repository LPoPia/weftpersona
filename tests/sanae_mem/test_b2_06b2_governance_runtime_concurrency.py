"""Multi-threaded concurrency stress tests for B2-06B-2:
governance/ + runtime/ SQLite access converged onto SqliteFacade.

Uses real file-backed SQLite (WAL mode) + real host plugin discovery
(``discover_plugins`` / ``load_memory_provider`` / ``emit_delivery_receipt``)
+ real loopback OpenAI-compatible ``ThreadingHTTPServer`` (outermost HTTP
boundary only). Zero internal mocks, zero custom LLM shims, and zero private
``._xxx`` backdoors.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
from contextlib import closing
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Iterator, Optional, cast
from zoneinfo import ZoneInfo

import pytest
import yaml

if TYPE_CHECKING:
    from sanae_mem.provider import SanaeMemProvider

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import (  # noqa: E402
    LoopbackLlmServer as _LoopbackLLMServer,
    install_plugin_copy,
    resolve_hermes_host_root,
)

SANAE_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

HERMES_ROOT = resolve_hermes_host_root(require_full_host=True)
if HERMES_ROOT is None:
    pytest.skip("full hermes-agent host not available", allow_module_level=True)
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

import hermes_constants as _hc  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.platforms.base import BasePlatformAdapter, SendResult  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.platforms.event import MessageEvent, MessageType  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.platform_registry import PlatformEntry, platform_registry  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.run import GatewayRunner  # noqa: E402  # pyright: ignore[reportMissingImports]
from gateway.session import SessionSource  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_cli.plugins import discover_plugins, get_plugin_manager  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_host_contracts import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    TurnEnvelope,
    clear_all_contract_registrations,
    turn_envelope_scope,
)
from plugins.memory import load_memory_provider  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_all_contract_registrations()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback-fixture")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    yield
    clear_all_contract_registrations()


@contextlib.contextmanager
def _home_override(home: Path) -> Iterator[Path]:
    token = _hc.set_hermes_home_override(home)
    try:
        yield home
    finally:
        _hc.reset_hermes_home_override(token)


@contextlib.contextmanager
def _discover_plugin_in_home(home: Path) -> Iterator[None]:
    """Mount sanae_mem through the real host discover_plugins() pipeline."""
    target_dir = home / "plugins" / "sanae_mem"
    if not target_dir.exists():
        install_plugin_copy(home, SANAE_PLUGIN_DIR.parents[1], plugin_name="sanae_mem")
    with _home_override(home):
        discover_plugins(force=True)
    try:
        yield
    finally:
        with _home_override(home):
            get_plugin_manager().unload("sanae_mem")


def _write_home_config(
    home: Path,
    *,
    base_url: str,
    group_enabled: bool = True,
    memory_entries: Optional[list[str]] = None,
) -> tuple[Path, dict[str, Any]]:
    home.mkdir(parents=True, exist_ok=True)
    data_dir = (home / "sanae-mem").resolve()
    (home / ".env").write_text("OPENAI_API_KEY=loopback-fixture\n", encoding="utf-8")
    memories_dir = home / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    (memories_dir / "USER.md").write_text(
        "我平时习惯早睡早起\n§\n我喜欢用纯黑美式咖啡开始一天",
        encoding="utf-8",
    )
    mem_text = "\n§\n".join(memory_entries) if memory_entries else "长期记忆基线"
    (memories_dir / "MEMORY.md").write_text(mem_text, encoding="utf-8")
    plugin_cfg: dict[str, Any] = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "resident_review_daily_cap": 100,
        "distill_daily_cap": 100,
        "memory_identity": {"platform": "telegram", "user_id": "owner-1"},
        "group_context": {
            "defaults": {
                "short_term_shared": group_enabled,
                "long_term_memory": group_enabled,
                "allow_unmentioned_reply": True,
            },
        },
    }
    cfg = {
        "model": {"provider": "custom", "base_url": base_url, "default": "vm-flash"},
        "auxiliary": {
            "sanae_daily_flash": {
                "provider": "custom",
                "base_url": base_url,
                "model": "vm-flash",
                "api_mode": "chat_completions",
            },
            "sanae_extract": {
                "provider": "custom",
                "base_url": base_url,
                "model": "vm-extract",
                "api_mode": "chat_completions",
            },
        },
        "memory": {
            "provider": "sanae_mem",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": plugin_cfg,
        },
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return data_dir, plugin_cfg


def _count_rows(db_path: Path, table: str) -> int:
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row is not None else 0


@pytest.mark.parametrize("first", ("review", "write"))
def test_daily_counters_roll_together_before_first_action(tmp_path: Path, first: str) -> None:
    from sanae_mem.plugin_entry import resolve_active_provider

    home = tmp_path / first
    envelope = TurnEnvelope(
        platform="telegram", chat_type="dm", chat_id="dm-1", sender_id="owner-1",
        session_id="daily-counters", session_key="agent:main:telegram:dm:dm-1",
        message_id="daily-counters",
    )
    with contextlib.ExitStack() as stack:
        server = _LoopbackLLMServer(lambda _payload, _index: (200, "invalid candidate"))
        stack.callback(server.close)
        data_dir, _ = _write_home_config(home, base_url=server.base_url, group_enabled=False)
        stack.enter_context(_home_override(home))
        stack.enter_context(_discover_plugin_in_home(home))
        stack.enter_context(turn_envelope_scope(envelope))
        worker = load_memory_provider("sanae_mem", register_skills=False)
        assert worker is not None
        stack.callback(worker.shutdown)
        worker.initialize(
            envelope.session_id, hermes_home=str(home), platform="telegram", agent_context="primary",
        )
        provider = resolve_active_provider(home, envelope.session_id)
        assert provider is not None and provider is worker

        def write_event(content: str) -> int:
            response = json.loads(provider.handle_tool_call(
                "sanae_mem_add", {"content": content, "importance": 0.85}
            ))
            result = response["result"]
            assert result.get("stored") is True, response
            event_id = result.get("id")
            assert isinstance(event_id, int), response
            return event_id

        event_id = write_event("我习惯每周整理一次旧笔记。")
        now = datetime.now().astimezone()
        today = now.date().isoformat()
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        provider.inner_state.record_inner_state_failure("flash_unavailable", now=now.timestamp())
        status_path = data_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("components", {})["archive"] = {
            "ok": False, "stage": "append", "error": "OSError", "at": yesterday,
        }
        status.update(
            date=yesterday, daily_writes=8, daily_scheduled_asks=2,
            daily_scheduled_attempts=3, daily_candidate_reviews=4,
            daily_compactor_runs=5, distill_errors_today=6,
            pre_compress_date=yesterday, pre_compress_count_today=7,
        )
        status_path.write_text(json.dumps(status), encoding="utf-8")

        def review_event() -> None:
            review = provider.promotion_review.review_promotion_candidate(event_id, now)
            assert review["reason"] == "candidate_parse_failed", review
            assert review["parsed"] is False

        if first == "review":
            review_event()
            write_event("傍晚去旧书店淘到了一本园艺手册。")
        else:
            write_event("傍晚去旧书店淘到了一本园艺手册。")
            review_event()

        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["date"] == today
        assert status["daily_writes"] == status["daily_candidate_reviews"] == 1
        assert status["daily_scheduled_asks"] == 0
        assert status["daily_scheduled_attempts"] == 0
        assert status["daily_compactor_runs"] == 0
        assert status["distill_errors_today"] == 0
        assert status["pre_compress_date"] == yesterday
        assert status["pre_compress_count_today"] == 7
        assert status["components"]["archive"]["error"] == "OSError"
        assert status["inner_state"]["last_failure"] == "flash_unavailable"

        review_event()
        write_event("今天完成了图书馆的志愿者培训。")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["daily_writes"] == status["daily_candidate_reviews"] == 2
        assert status["components"]["archive"]["error"] == "OSError"
        assert status["inner_state"]["last_failure"] == "flash_unavailable"
        review_requests = [request for request in server.requests if request["model"] == "vm-extract"]
        assert len(review_requests) == 2


def test_concurrent_governance_and_runtime_sqlite_stress(tmp_path: Path) -> None:
    """Concurrent writes and reads across memory_axis, resident, source_intake,
    read_gate, and lifecycle on a real host-loaded ``SanaeMemProvider``.

    Asserts zero OperationalError/SQLITE_MISUSE and exact SQLite + file consistency.
    """
    home = tmp_path / "gov-runtime-home"
    demotable_entries = [
        f"阶段性常驻条目编号{i}记录了一次普通散步经历" for i in range(6)
    ]

    def responder(_payload: dict[str, Any], _idx: int) -> tuple[int, Any]:
        return 200, {
            "summary": "并发蒸馏后的事件摘要",
            "action": "keep_short",
            "target": "",
            "text": "并发蒸馏后的事件摘要",
        }

    llm_server = _LoopbackLLMServer(responder)
    try:
        data_dir, plugin_cfg = _write_home_config(
            home,
            base_url=llm_server.base_url,
            group_enabled=False,
            memory_entries=["长期记忆基线", *demotable_entries],
        )
        db_path = data_dir / "sanae.db"
        private_envelope = TurnEnvelope(
            platform="telegram",
            chat_type="dm",
            chat_id="dm-1",
            sender_id="owner-1",
            session_id="sess-priv-concurrent",
            session_key="agent:main:telegram:dm:dm-1",
            message_id="msg-init",
        )
        with _discover_plugin_in_home(home):
            with _home_override(home), turn_envelope_scope(private_envelope):
                provider = cast("SanaeMemProvider", load_memory_provider("sanae_mem"))
                provider.initialize(
                    "sess-priv-concurrent",
                    hermes_home=str(home),
                    platform="telegram",
                    agent_context="primary",
                )
                try:
                    assert provider.sqlite.is_open
                    distinct_topics = [
                        "今天傍晚在河边散步看到了漂亮的晚霞和白鹭",
                        "周末去旧书店淘到一本绝版手绘植物图鉴",
                        "昨晚第一次尝试自己煮日式咖喱饭味道很香",
                        "上午修好了阳台那盏接触不良的复古台灯",
                        "下午在音乐厅听了大提琴独奏会特别动人",
                        "刚学会用手冲壶控制水温泡出柑橘香咖啡",
                        "朋友送来一盆多肉植物摆在书桌阳光角落",
                        "清晨骑自行车穿过梧桐大道吹到清凉微风",
                    ]
                    seeded_ids: list[int] = []
                    for topic_text in distinct_topics:
                        res = provider.handle_tool_call(
                            "sanae_mem_add",
                            {"content": topic_text, "importance": 0.85},
                        )
                        parsed = json.loads(res)
                        result_obj = parsed.get("result") if isinstance(parsed, dict) else None
                        assert isinstance(result_obj, dict) and result_obj.get("stored") is True, (
                            f"sanae_mem_add failed: {parsed!r}"
                        )
                        seeded_ids.append(int(result_obj["id"]))
                    assert len(set(seeded_ids)) == len(distinct_topics)

                    with closing(sqlite3.connect(str(db_path))) as conn:
                        owner_row = conn.execute(
                            "SELECT owner_key FROM events WHERE id=?", (seeded_ids[0],)
                        ).fetchone()
                    assert owner_row is not None and str(owner_row[0] or "").strip()
                    owner_key = str(owner_row[0])
                    now = datetime.now(_SHANGHAI_TZ)
                    base_delivered = now - timedelta(minutes=20)
                    errors: list[BaseException] = []
                    err_lock = threading.Lock()
                    barrier = threading.Barrier(6)

                    def _record_error(exc: BaseException) -> None:
                        with err_lock:
                            errors.append(exc)

                    def worker_memory_axis(worker_idx: int) -> list[dict[str, Any]]:
                        out: list[dict[str, Any]] = []
                        with _home_override(home), turn_envelope_scope(private_envelope):
                            barrier.wait()
                            try:
                                for step in range(6):
                                    turn_id = f"axis-w{worker_idx}-s{step}"
                                    res = provider.memory_axis.record_delivered_rementions(
                                        owner_key=owner_key,
                                        user_text=distinct_topics[step],
                                        assistant_text="我也记得那天的事。",
                                        turn_id=turn_id,
                                        delivered_at=base_delivered + timedelta(seconds=worker_idx * 10 + step),
                                    )
                                    out.append(res)
                                for eid in seeded_ids[6 + worker_idx : 8 : 2]:
                                    out.append(
                                        provider.memory_axis.promote_explicit_save(
                                            eid, owner_key=owner_key, now=now
                                        )
                                    )
                            except BaseException as exc:  # pragma: no cover
                                _record_error(exc)
                        return out

                    def worker_resident_demote() -> list[dict[str, Any]]:
                        out: list[dict[str, Any]] = []
                        with _home_override(home), turn_envelope_scope(private_envelope):
                            barrier.wait()
                            try:
                                for entry in demotable_entries:
                                    out.append(
                                        provider.resident.demote_resident_entry("memory", entry, now=now)
                                    )
                            except BaseException as exc:  # pragma: no cover
                                _record_error(exc)
                        return out

                    def worker_source_intake() -> list[dict[str, Any]]:
                        out: list[dict[str, Any]] = []
                        with _home_override(home), turn_envelope_scope(private_envelope):
                            barrier.wait()
                            try:
                                for idx in range(8):
                                    mat = {
                                        "source_plugin": "rss_reader",
                                        "platform": "rss",
                                        "material_kind": "raw",
                                        "content": f"并发外部素材正文编号{idx}：记录了一段公开阅读笔记",
                                        "privacy_scope": "public",
                                        "source_object_id": f"item-concurrent-{idx}",
                                        "title": f"阅读笔记{idx}",
                                    }
                                    import importlib
                                    ingest_source_material = importlib.import_module(
                                        "sanae_mem.governance.source_intake"
                                    ).ingest_source_material

                                    out.append(
                                        ingest_source_material(
                                            provider, mat, create_memory_event=True, importance=0.65
                                        ).to_dict()
                                    )
                            except BaseException as exc:  # pragma: no cover
                                _record_error(exc)
                        return out

                    def worker_lifecycle() -> tuple[list[bool], dict[str, Any], dict[str, Any]]:
                        distill_out: list[bool] = []
                        marked_out: dict[str, Any] = {}
                        compact_out: dict[str, Any] = {}
                        with _home_override(home), turn_envelope_scope(private_envelope):
                            barrier.wait()
                            try:
                                for eid in seeded_ids[:4]:
                                    distill_out.append(
                                        provider.lifecycle.distill_promote(
                                            eid, f"事件{eid}的并发蒸馏摘要内容", now=now
                                        )
                                    )
                                marked_out = provider.lifecycle.marked_promoted(seeded_ids[4:6], "long")
                                compact_out = provider.lifecycle.compact(now=now)
                            except BaseException as exc:  # pragma: no cover
                                _record_error(exc)
                        return distill_out, marked_out, compact_out

                    def worker_read_gate() -> list[str]:
                        out: list[str] = []
                        with _home_override(home), turn_envelope_scope(private_envelope):
                            barrier.wait()
                            try:
                                for step in range(len(distinct_topics)):
                                    out.append(
                                        provider.prefetch(
                                            distinct_topics[step],
                                            session_id=f"sess-priv-concurrent-{step}",
                                        )
                                    )
                            except BaseException as exc:  # pragma: no cover
                                _record_error(exc)
                        return out

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        f_axis_0 = pool.submit(worker_memory_axis, 0)
                        f_axis_1 = pool.submit(worker_memory_axis, 1)
                        f_demote = pool.submit(worker_resident_demote)
                        f_intake = pool.submit(worker_source_intake)
                        f_life = pool.submit(worker_lifecycle)
                        f_read = pool.submit(worker_read_gate)

                        axis_0 = f_axis_0.result()
                        axis_1 = f_axis_1.result()
                        demote_results = f_demote.result()
                        intake_results = f_intake.result()
                        distill_results, marked_result, compact_result = f_life.result()
                        read_results = f_read.result()

                    assert not errors, f"unexpected concurrent SQLite errors: {errors!r}"
                    assert all(r.get("ok") is True for r in axis_0), f"axis_0 failures: {axis_0!r}"
                    assert all(r.get("ok") is True for r in axis_1), f"axis_1 failures: {axis_1!r}"
                    assert all(
                        r.get("ok") is True and r.get("demoted") is True for r in demote_results
                    ), f"demote failures: {demote_results!r}"
                    assert all(r.get("ok") is True for r in intake_results), (
                        f"intake failures: {intake_results!r}"
                    )
                    assert len(distill_results) == 4 and all(ok is True for ok in distill_results), (
                        f"distill_promote failures: {distill_results!r}"
                    )
                    assert marked_result.get("updated") == len(seeded_ids[4:6]), (
                        f"marked_promoted unexpected result: {marked_result!r}"
                    )
                    assert isinstance(compact_result, dict)
                    assert len(read_results) == len(distinct_topics)
                    non_empty_reads = [r for r in read_results if isinstance(r, str) and r.strip()]
                    assert len(non_empty_reads) >= 1, (
                        f"prefetch returned no memories: {read_results!r}"
                    )
                    expected_keywords = (
                        "晚霞", "散步", "图鉴", "咖喱", "台灯", "大提琴", "咖啡", "多肉", "梧桐",
                        "并发蒸馏摘要内容",
                    )
                    assert all(
                        any(kw in r for kw in expected_keywords) for r in non_empty_reads
                    ), f"prefetch returned unexpected content: {non_empty_reads!r}"

                    assert _count_rows(db_path, "source_materials") == 8
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        conn.row_factory = sqlite3.Row
                        distilled_rows = conn.execute(
                            "SELECT id, summary, distilled_at, tier FROM events WHERE id IN (%s) ORDER BY id"
                            % ",".join("?" for _ in seeded_ids[:4]),
                            seeded_ids[:4],
                        ).fetchall()
                        explicit_long_rows = conn.execute(
                            "SELECT id, summary, distilled_at, tier FROM events WHERE id IN (%s) ORDER BY id"
                            % ",".join("?" for _ in seeded_ids[4:6]),
                            seeded_ids[4:6],
                        ).fetchall()
                        promoted_count = int(
                            conn.execute(
                                "SELECT count(*) FROM events WHERE id IN (%s) AND tier='long'"
                                % ",".join("?" for _ in seeded_ids),
                                seeded_ids,
                            ).fetchone()[0]
                        )
                        demoted_events = int(
                            conn.execute(
                                "SELECT count(*) FROM events WHERE promoted_reason='resident_demote'"
                            ).fetchone()[0]
                        )
                        mention_count = int(
                            conn.execute("SELECT count(*) FROM memory_axis_mentions").fetchone()[0]
                        )
                        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])

                    assert len(distilled_rows) == 4
                    for row in distilled_rows:
                        eid = int(row["id"])
                        assert row["summary"] == f"事件{eid}的并发蒸馏摘要内容"
                        assert row["distilled_at"] is not None
                    assert len(explicit_long_rows) == len(seeded_ids[4:6])
                    for row in explicit_long_rows:
                        assert row["tier"] == "long"
                        assert row["distilled_at"] is None
                    assert promoted_count == len(seeded_ids[4:])
                    assert demoted_events == len(demotable_entries)
                    assert mention_count >= 1
                    assert integrity == "ok"

                    remaining_memory = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
                    assert "长期记忆基线" in remaining_memory
                    for entry in demotable_entries:
                        assert entry not in remaining_memory

                    # Concurrent duplicate source_intake submissions must serialize via
                    # provider.sqlite.locked() so event_occurrences and daily_writes
                    # each increment by exactly 1.
                    status_file = data_dir / "status.json"
                    status_before = json.loads(status_file.read_text(encoding="utf-8"))
                    writes_before = int(status_before.get("daily_writes", 0))
                    dup_barrier = threading.Barrier(2)
                    dup_material = {
                        "source_plugin": "rss_reader",
                        "platform": "rss",
                        "material_kind": "raw",
                        "content": "并发重复投递同一篇公开阅读素材正文，验证去重与计数一致性",
                        "privacy_scope": "public",
                        "source_object_id": "item-dup-race-001",
                        "title": "并发重复素材",
                    }

                    def submit_dup() -> dict[str, Any]:
                        import importlib
                        ingest_source_material = importlib.import_module(
                            "sanae_mem.governance.source_intake"
                        ).ingest_source_material

                        with _home_override(home), turn_envelope_scope(private_envelope):
                            dup_barrier.wait()
                            return ingest_source_material(
                                provider, dup_material, create_memory_event=True, importance=0.65
                            ).to_dict()

                    with ThreadPoolExecutor(max_workers=2) as dup_pool:
                        f_dup_a = dup_pool.submit(submit_dup)
                        f_dup_b = dup_pool.submit(submit_dup)
                        res_a = f_dup_a.result()
                        res_b = f_dup_b.result()

                    assert res_a.get("ok") is True and res_b.get("ok") is True
                    assert {bool(res_a.get("deduplicated")), bool(res_b.get("deduplicated"))} == {
                        False,
                        True,
                    }
                    linked_eid = int(res_a["linked_event_id"])
                    assert int(res_b["linked_event_id"]) == linked_eid
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        occ_count = int(
                            conn.execute(
                                "SELECT count(*) FROM event_occurrences WHERE event_id=?",
                                (linked_eid,),
                            ).fetchone()[0]
                        )
                    assert occ_count == 1
                    status_after = json.loads(status_file.read_text(encoding="utf-8"))
                    assert int(status_after.get("daily_writes", 0)) == writes_before + 1
                finally:
                    provider.shutdown()
    finally:
        llm_server.close()


class _GroupHostAdapter(BasePlatformAdapter):
    """Minimal real host BasePlatformAdapter subclass for concurrent group delivery."""

    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        super().__init__(config or PlatformConfig(enabled=True), Platform.QQBOT)
        self.sent: list[tuple[str, str]] = []
        self._send_lock = threading.Lock()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"chat_id": chat_id}

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Any = None
    ) -> SendResult:
        with self._send_lock:
            self.sent.append((chat_id, content))
            msg_num = len(self.sent)
        return SendResult(success=True, message_id=f"host-grp-out-{msg_num}")


def test_concurrent_submit_source_material_without_active_provider(tmp_path: Path) -> None:
    """When no active provider is initialized, concurrent calls to the public
    ``submit_source_material()`` API create separate temporary worker providers;
    ``_intake_lock_for_db`` must serialize them so duplicate submissions yield
    exactly 1 source_material, 1 event, 1 event_occurrence, and daily_writes == 1.
    """
    home = tmp_path / "submit-no-active-home"
    llm_server = _LoopbackLLMServer(lambda _p, _i: (200, {"ok": True}))
    try:
        data_dir, _plugin_cfg = _write_home_config(
            home, base_url=llm_server.base_url, group_enabled=False
        )
        db_path = data_dir / "sanae.db"
        private_envelope = TurnEnvelope(
            platform="telegram",
            chat_type="dm",
            chat_id="dm-no-active",
            sender_id="owner-1",
            session_id="sess-no-active-submit",
            session_key="agent:main:telegram:dm:dm-no-active",
            message_id="msg-no-active",
        )
        with _discover_plugin_in_home(home):
            from sanae_mem import submit_source_material  # pyright: ignore[reportMissingImports]

            dup_material = {
                "source_plugin": "rss_reader",
                "platform": "rss",
                "material_kind": "raw",
                "content": "无活跃provider时双线程并发投递同一公开素材验证跨worker去重",
                "privacy_scope": "public",
                "source_object_id": "item-no-active-dup-001",
                "title": "跨worker并发去重素材",
            }
            barrier = threading.Barrier(2)

            def worker_submit() -> dict[str, Any]:
                with _home_override(home), turn_envelope_scope(private_envelope):
                    barrier.wait()
                    return submit_source_material(
                        dup_material, create_memory_event=True, importance=0.65
                    )

            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(worker_submit)
                fb = pool.submit(worker_submit)
                res_a = fa.result()
                res_b = fb.result()

            assert res_a.get("ok") is True and res_b.get("ok") is True
            assert {bool(res_a.get("deduplicated")), bool(res_b.get("deduplicated"))} == {
                False,
                True,
            }
            assert int(res_a["linked_event_id"]) == int(res_b["linked_event_id"])
            assert _count_rows(db_path, "source_materials") == 1
            assert _count_rows(db_path, "events") == 1
            assert _count_rows(db_path, "event_occurrences") == 1
            with closing(sqlite3.connect(str(db_path))) as conn:
                assert str(conn.execute("PRAGMA integrity_check").fetchone()[0]) == "ok"
            status_obj = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
            assert int(status_obj.get("daily_writes", 0)) == 1
    finally:
        llm_server.close()


def test_concurrent_host_group_receipts_and_lifecycle_on_real_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent real host ``GatewayRunner`` + ``BasePlatformAdapter.handle_message``
    group turns + concurrent cron lifecycle ``compact()`` against the same file-backed
    SQLite DB and loopback HTTP server.
    """
    home = tmp_path / "host-group-concurrent-home"
    n_receipts = 6
    first_stream_seen = threading.Event()
    compact_first_done = threading.Event()
    all_delivered = threading.Event()
    wire_lock = threading.Lock()
    group_wire_messages: dict[int, list[dict[str, Any]]] = {}
    pre_delivery_hits: dict[int, int] = {}

    def _msg_text(msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        return str(content or "")

    def responder(payload: dict[str, Any], _idx: int) -> tuple[int, Any]:
        messages = payload.get("messages") or []
        if payload.get("stream"):
            first_stream_seen.set()
            joined = "\n".join(_msg_text(m) for m in messages if isinstance(m, dict))
            for idx in range(n_receipts):
                marker = f"宿主并发群消息编号{idx}一起打羽毛球"
                if marker in joined:
                    hits = 0
                    if db_path.exists():
                        with closing(sqlite3.connect(str(db_path))) as conn:
                            has_events = conn.execute(
                                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
                            ).fetchone()
                            if has_events is not None:
                                hits = int(
                                    conn.execute(
                                        "SELECT count(*) FROM events WHERE raw LIKE ? OR summary LIKE ?",
                                        (f"%{marker}%", f"%{marker}%"),
                                    ).fetchone()[0]
                                )
                    with wire_lock:
                        group_wire_messages.setdefault(idx, list(messages))
                        pre_delivery_hits.setdefault(idx, hits)
                    break
            assert compact_first_done.wait(timeout=15.0), "compact() did not overlap in-flight group turn"
            return 200, "没问题，周六见。"
        user_msg = str(messages[-1].get("content") or "") if messages else ""
        for line in user_msg.splitlines():
            if "宿主并发群消息编号" in line:
                idx_pos = line.find("宿主并发群消息编号")
                quote = line[idx_pos:].strip()
                return 200, {
                    "events": [{"text": f"宿主摘要：{quote}", "quote": quote}],
                    "impression": None,
                    "new_commitments": [],
                    "commitment_updates": [],
                }
        return 200, {
            "summary": "群聊周期蒸馏摘要",
            "events": [],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [],
        }

    llm_server = _LoopbackLLMServer(responder)
    try:
        data_dir, _plugin_cfg = _write_home_config(
            home, base_url=llm_server.base_url, group_enabled=True
        )
        db_path = data_dir / "sanae.db"
        plugins_dir = home / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            SANAE_PLUGIN_DIR,
            plugins_dir / "sanae_mem",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
        )
        allowed_users = ",".join(f"member-{i}" for i in range(n_receipts))
        monkeypatch.setenv("QQ_ALLOWED_USERS", allowed_users)
        base_ts = time.time()

        async def run_gateway_and_compact() -> tuple[int, dict[str, Any]]:
            with _home_override(home):
                platform_scope = platform_registry.current_scope_key()
                platform_registry.register(
                    PlatformEntry(
                        name=Platform.QQBOT.value,
                        label="QQ test transport",
                        adapter_factory=lambda config: _GroupHostAdapter(config),
                        check_fn=lambda: True,
                        allowed_users_env="QQ_ALLOWED_USERS",
                    ),
                    scope=platform_scope,
                )
                gateway = GatewayRunner(
                    GatewayConfig(
                        platforms={Platform.QQBOT: PlatformConfig(enabled=True)},
                        multiplex_profiles=False,
                    )
                )
                assert await gateway.start()
                try:
                    transport = cast(_GroupHostAdapter, gateway.adapters[Platform.QQBOT])

                    def run_cron_compact_loop() -> tuple[int, dict[str, Any]]:
                        assert first_stream_seen.wait(timeout=20.0)
                        with _home_override(home):
                            cron_provider = cast("SanaeMemProvider", load_memory_provider("sanae_mem"))
                            cron_provider.initialize(
                                "cron-group-compact",
                                hermes_home=str(home),
                                platform="cron",
                                agent_context="cron",
                            )
                            try:
                                runs = 0
                                last_res: dict[str, Any] = {}
                                while not all_delivered.is_set():
                                    last_res = cron_provider.lifecycle.compact(now=datetime.now(_SHANGHAI_TZ))
                                    runs += 1
                                    compact_first_done.set()
                                    if not all_delivered.is_set():
                                        time.sleep(0.01)
                                return runs, last_res
                            finally:
                                compact_first_done.set()
                                cron_provider.shutdown()

                    compact_task = asyncio.create_task(asyncio.to_thread(run_cron_compact_loop))

                    async def send_group_turn(idx: int) -> None:
                        user_msg = f"宿主并发群消息编号{idx}一起打羽毛球"
                        in_msg_id = f"host-grp-in-{idx}-{int(base_ts * 1000)}"
                        source = SessionSource(
                            platform=Platform.QQBOT,
                            chat_id=f"room-host-stress-{idx}",
                            chat_type="group",
                            user_id=f"member-{idx}",
                            scope_id="default",
                            message_id=in_msg_id,
                        )
                        await transport.handle_message(
                            MessageEvent(
                                text=user_msg,
                                authored_text=user_msg,
                                message_type=MessageType.TEXT,
                                message_id=in_msg_id,
                                source=source,
                                reply_expected=True,
                            )
                        )

                    await asyncio.gather(*(send_group_turn(idx) for idx in range(n_receipts)))

                    def _events_ready() -> bool:
                        if not db_path.exists():
                            return False
                        with closing(sqlite3.connect(str(db_path))) as conn:
                            has_events = conn.execute(
                                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
                            ).fetchone()
                            if has_events is None:
                                return False
                            row = conn.execute("SELECT count(*) FROM events").fetchone()
                            return int(row[0]) >= n_receipts if row is not None else False

                    deadline = time.monotonic() + 30.0
                    while time.monotonic() < deadline:
                        if len(transport.sent) >= n_receipts and _events_ready():
                            break
                        await asyncio.sleep(0.05)

                    all_delivered.set()
                    return await compact_task
                finally:
                    all_delivered.set()
                    await gateway.stop()
                    platform_registry.unregister(Platform.QQBOT.value, scope=platform_scope)
                    get_plugin_manager().unload("sanae_mem")

        with _home_override(home):
            compact_runs, compact_res = asyncio.run(run_gateway_and_compact())

        assert compact_runs >= 1
        assert isinstance(compact_res, dict)
        assert set(group_wire_messages.keys()) == set(range(n_receipts))
        assert set(pre_delivery_hits.keys()) == set(range(n_receipts))
        for idx in range(n_receipts):
            assert pre_delivery_hits[idx] == 0, f"group {idx} leaked event before delivery"
            msgs = group_wire_messages[idx]
            system_text = "\n".join(
                _msg_text(m) for m in msgs if isinstance(m, dict) and m.get("role") == "system"
            )
            user_text = "\n".join(
                _msg_text(m) for m in msgs if isinstance(m, dict) and m.get("role") == "user"
            )
            assert f"宿主并发群消息编号{idx}一起打羽毛球" in user_text
            assert system_text.strip()
            for other in range(n_receipts):
                if other != idx:
                    assert f"宿主并发群消息编号{other}一起打羽毛球" not in user_text
                    assert f"宿主并发群消息编号{other}一起打羽毛球" not in system_text

        assert _count_rows(db_path, "events") == n_receipts
        with closing(sqlite3.connect(str(db_path))) as conn:
            assert str(conn.execute("PRAGMA integrity_check").fetchone()[0]) == "ok"
            summaries = [
                str(r[0] or "")
                for r in conn.execute("SELECT summary FROM events ORDER BY id").fetchall()
            ]
        assert len(summaries) == n_receipts
        observed_indices: set[int] = set()
        for summary in summaries:
            assert summary.startswith("宿主摘要：宿主并发群消息编号")
            for idx in range(n_receipts):
                if summary == f"宿主摘要：宿主并发群消息编号{idx}一起打羽毛球":
                    observed_indices.add(idx)
                    break
        assert observed_indices == set(range(n_receipts))
    finally:
        llm_server.close()


def test_submit_source_material_multiprocess_concurrent_dedup_exact_counts(
    tmp_path: Path,
) -> None:
    """4 independent OS processes submitting the same 8 public materials concurrently
    must deduplicate cleanly across processes and produce exact 8/8/8/8 counts.
    """
    home = tmp_path / "mp-intake-home"
    llm_server = _LoopbackLLMServer(lambda _p, _i: (200, {"ok": True}))
    try:
        data_dir, _ = _write_home_config(
            home, base_url=llm_server.base_url, group_enabled=False
        )
        db_path = data_dir / "sanae.db"
        # Ensure plugins/sanae_mem is installed into home once before spawning workers
        # so workers do not race on shutil.copytree of the plugin directory itself.
        with _discover_plugin_in_home(home):
            pass

        plugin_home_parent = home / "plugins"
        ready_dir = tmp_path / "mp-ready"
        ready_dir.mkdir(parents=True, exist_ok=True)
        go_flag = tmp_path / "mp-go.flag"

        worker_script = "\n".join(
            [
                "import json, os, sys, time",
                "from pathlib import Path",
                "worker_idx = int(sys.argv[1])",
                "home = Path(sys.argv[2])",
                "hermes_root = Path(sys.argv[3])",
                "plugin_parent = Path(sys.argv[4])",
                "ready_file = Path(sys.argv[5])",
                "go_flag = Path(sys.argv[6])",
                "sys.path.insert(0, str(hermes_root))",
                "sys.path.insert(0, str(plugin_parent))",
                "import hermes_constants as hc",
                "from hermes_cli.plugins import discover_plugins, get_plugin_manager",
                "from hermes_host_contracts import TurnEnvelope, turn_envelope_scope",
                "token = hc.set_hermes_home_override(home)",
                "try:",
                "    discover_plugins()",
                "    from sanae_mem import submit_source_material",
                "    ready_file.write_text('ok', encoding='utf-8')",
                "    deadline = time.monotonic() + 20.0",
                "    while not go_flag.exists():",
                "        if time.monotonic() > deadline:",
                "            raise TimeoutError('go_flag timeout')",
                "        time.sleep(0.01)",
                "    results = []",
                "    for item_idx in range(8):",
                "        env = TurnEnvelope(",
                "            platform='telegram',",
                "            chat_type='dm',",
                "            chat_id='dm-mp',",
                "            sender_id='owner-1',",
                "            session_id=f'sess-mp-{worker_idx}-{item_idx}',",
                "            session_key='agent:main:telegram:dm:dm-mp',",
                "            message_id=f'msg-mp-{worker_idx}-{item_idx}',",
                "        )",
                "        mat = {",
                "            'source_plugin': 'rss_reader',",
                "            'platform': 'rss',",
                "            'material_kind': 'raw',",
                "            'content': f'跨进程并发素材正文编号{item_idx}用于验证精确去重',",
                "            'privacy_scope': 'public',",
                "            'source_object_id': f'mp-obj-{item_idx:03d}',",
                "            'title': f'跨进程素材{item_idx}',",
                "        }",
                "        with turn_envelope_scope(env):",
                "            res = submit_source_material(",
                "                mat,",
                "                session_id=f'sess-mp-{worker_idx}-{item_idx}',",
                "                create_memory_event=True,",
                "                importance=0.65,",
                "                hermes_home_path=home,",
                "            )",
                "        results.append(res)",
                "    print(json.dumps(results, ensure_ascii=False))",
                "finally:",
                "    get_plugin_manager().unload('sanae_mem')",
                "    hc.reset_hermes_home_override(token)",
            ]
        )

        env = os.environ.copy()
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        env["OPENAI_API_KEY"] = "loopback-fixture"
        env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
        env.pop("HERMES_HOME", None)

        procs: list[subprocess.Popen[str]] = []
        for idx in range(4):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    worker_script,
                    str(idx),
                    str(home),
                    str(HERMES_ROOT),
                    str(plugin_home_parent),
                    str(ready_dir / f"w{idx}.ready"),
                    str(go_flag),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
            )
            procs.append(proc)

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if all((ready_dir / f"w{i}.ready").exists() for i in range(4)):
                break
            time.sleep(0.02)
        assert all((ready_dir / f"w{i}.ready").exists() for i in range(4))
        go_flag.write_text("go", encoding="utf-8")

        all_results: list[dict[str, Any]] = []
        for proc in procs:
            stdout, stderr = proc.communicate(timeout=45)
            assert proc.returncode == 0, f"worker failed: {stderr}"
            all_results.extend(json.loads(stdout.strip().splitlines()[-1]))

        assert len(all_results) == 32
        assert all(item.get("ok") is True for item in all_results)
        stored_count = sum(1 for item in all_results if not item.get("deduplicated"))
        dedup_count = sum(1 for item in all_results if item.get("deduplicated"))
        assert stored_count == 8
        assert dedup_count == 24

        assert _count_rows(db_path, "source_materials") == 8
        assert _count_rows(db_path, "events") == 8
        assert _count_rows(db_path, "event_occurrences") == 8
        with closing(sqlite3.connect(str(db_path))) as conn:
            assert str(conn.execute("PRAGMA integrity_check").fetchone()[0]) == "ok"
        status_obj = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
        assert int(status_obj.get("daily_writes", 0)) == 8
    finally:
        llm_server.close()


def test_submit_source_material_insert_failure_rollback_and_retry_no_double_count(
    tmp_path: Path,
) -> None:
    """When inserting into ``source_materials`` fails via a real SQLite trigger,
    events/event_occurrences/daily_writes must roll back completely so a subsequent
    retry produces exact 1/1/1/1 counts instead of double-counting.
    """
    home = tmp_path / "retry-rollback-home"
    llm_server = _LoopbackLLMServer(lambda _p, _i: (200, {"ok": True}))
    try:
        data_dir, _ = _write_home_config(
            home, base_url=llm_server.base_url, group_enabled=False
        )
        db_path = data_dir / "sanae.db"
        private_envelope = TurnEnvelope(
            platform="telegram",
            chat_type="dm",
            chat_id="dm-retry",
            sender_id="owner-1",
            session_id="sess-retry-intake",
            session_key="agent:main:telegram:dm:dm-retry",
            message_id="msg-retry",
        )
        with _discover_plugin_in_home(home):
            from sanae_mem import submit_source_material  # pyright: ignore[reportMissingImports]

            mat_seed = {
                "source_plugin": "rss_reader",
                "platform": "rss",
                "material_kind": "raw",
                "content": "基准素材初始化数据库结构与触发器宿主状态",
                "privacy_scope": "public",
                "source_object_id": "seed-001",
                "title": "基准素材",
            }
            with _home_override(home), turn_envelope_scope(private_envelope):
                seed_res = submit_source_material(
                    mat_seed, create_memory_event=True, importance=0.65
                )
            assert seed_res.get("ok") is True and seed_res.get("deduplicated") is False
            assert _count_rows(db_path, "source_materials") == 1
            assert _count_rows(db_path, "events") == 1
            assert _count_rows(db_path, "event_occurrences") == 1
            status_before = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
            assert int(status_before.get("daily_writes", 0)) == 1

            with closing(sqlite3.connect(str(db_path))) as conn:
                conn.execute(
                    """CREATE TRIGGER trg_inject_material_insert_fail
                       BEFORE INSERT ON source_materials
                       BEGIN
                           SELECT RAISE(ABORT, 'injected_material_insert_failure');
                       END"""
                )
                conn.commit()

            mat_retry = {
                "source_plugin": "rss_reader",
                "platform": "rss",
                "material_kind": "raw",
                "content": "临时故障回滚后重试不重复记账的公开素材",
                "privacy_scope": "public",
                "source_object_id": "retry-002",
                "title": "重试素材",
            }
            with _home_override(home), turn_envelope_scope(private_envelope):
                with pytest.raises(sqlite3.IntegrityError, match="injected_material_insert_failure"):
                    submit_source_material(
                        mat_retry, create_memory_event=True, importance=0.65
                    )

            assert _count_rows(db_path, "source_materials") == 1
            assert _count_rows(db_path, "events") == 1
            assert _count_rows(db_path, "event_occurrences") == 1
            status_after_fail = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
            assert int(status_after_fail.get("daily_writes", 0)) == 1

            with closing(sqlite3.connect(str(db_path))) as conn:
                conn.execute("DROP TRIGGER trg_inject_material_insert_fail")
                conn.commit()

            with _home_override(home), turn_envelope_scope(private_envelope):
                retry_res = submit_source_material(
                    mat_retry, create_memory_event=True, importance=0.65
                )
            assert retry_res.get("ok") is True and retry_res.get("deduplicated") is False
            assert _count_rows(db_path, "source_materials") == 2
            assert _count_rows(db_path, "events") == 2
            assert _count_rows(db_path, "event_occurrences") == 2
            status_after_retry = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
            assert int(status_after_retry.get("daily_writes", 0)) == 2
    finally:
        llm_server.close()
