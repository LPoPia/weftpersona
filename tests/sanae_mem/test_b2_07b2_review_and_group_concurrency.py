"""Concurrency and atomicity tests for B2-07B-2:
1. Single-transaction atomic double-write for grounded_review and promotion_review
   (zero orphan rows in event_reviews and no event promotion when
   recognition_review_targets write fails).
2. Group delivery receipt failure handling (apply_group_delivery ok=False logs warning
   and persists failure reason to status.json without dropping receipts, plus explicit
   SQLite before/after diffs).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from contextlib import closing
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Iterator, cast

import pytest
import yaml

if TYPE_CHECKING:
    from sanae_mem.provider import SanaeMemProvider

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
SANAE_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import (  # noqa: E402
    LoopbackLlmServer as _LoopbackLLMServer,
    install_plugin_copy,
    resolve_hermes_host_root,
)

HERMES_ROOT = resolve_hermes_host_root()
if HERMES_ROOT is None:
    pytest.skip("hermes_host_contracts not available", allow_module_level=True)
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

import hermes_constants as _hc  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_cli.plugins import discover_plugins, get_plugin_manager  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_host_contracts import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DeliveryReceipt,
    clear_all_contract_registrations,
    emit_delivery_receipt,
    turn_envelope_scope,
)
from plugins.memory import load_memory_provider  # noqa: E402  # pyright: ignore[reportMissingImports]


def _wait_all_pending_updates(timeout: float = 30.0) -> None:
    """Drain background queues across all loaded sanae_mem module instances."""
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "wait_for_pending_updates", None) if mod is not None else None
        if callable(fn):
            assert fn(timeout=timeout)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_all_contract_registrations()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback-fixture")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    yield
    _wait_all_pending_updates(timeout=10.0)
    clear_all_contract_registrations()


@contextlib.contextmanager
def _home_override(home: Path) -> Iterator[Path]:
    token = _hc.set_hermes_home_override(home)
    try:
        yield home
    finally:
        _hc.reset_hermes_home_override(token)


def _write_home_config(
    home: Path,
    *,
    base_url: str,
    group_enabled: bool = False,
    resident_review_daily_cap: int = 100,
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
    (memories_dir / "MEMORY.md").write_text("长期记忆基线", encoding="utf-8")
    plugin_cfg: dict[str, Any] = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "resident_review_daily_cap": resident_review_daily_cap,
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


@contextlib.contextmanager
def _discover_plugin_in_home(home: Path) -> Iterator[None]:
    """Mount sanae_mem through the real host discover_plugins() pipeline without overrides."""
    target_dir = home / "plugins" / "sanae_mem"
    if not target_dir.exists():
        install_plugin_copy(home, SANAE_PLUGIN_DIR.parents[1], plugin_name="sanae_mem")
    with _home_override(home):
        discover_plugins(force=True)
    try:
        yield
    finally:
        _wait_all_pending_updates(timeout=10.0)
        with _home_override(home):
            get_plugin_manager().unload("sanae_mem")


def _private_turn(session_id: str, message_id: str) -> Any:
    return turn_envelope_scope(
        platform="telegram",
        chat_type="dm",
        chat_id="owner-1",
        sender_id="owner-1",
        session_id=session_id,
        session_key="agent:main:telegram:dm:owner-1",
        message_id=message_id,
        is_messaging_surface=True,
    )


def _seed_grounded_events(home: Path, count: int) -> list[int]:
    """Seed real grounded private events through public emit_delivery_receipt."""
    base_ts = time.time()
    for idx in range(count):
        receipt = DeliveryReceipt(
            session_id="tg:dm:owner-1",
            platform="telegram",
            chat_type="dm",
            chat_id="owner-1",
            scope_id="default",
            sender_id="owner-1",
            turn_id=f"seed-turn-{idx}",
            message_id=f"seed-out-{idx}",
            incoming_message_id=f"seed-in-{idx}",
            session_key="agent:main:telegram:dm:owner-1",
            user_message=f"我答应以后每周{idx + 1}都会去河边慢跑五公里，这对我非常重要",
            delivered_text="收到，记下了。",
            clean_text="收到，记下了。",
            delivered_at=base_ts + idx,
            transport_confirmed=True,
            is_bot=False,
        )
        with _home_override(home):
            emit_delivery_receipt(receipt)
    data_dir = (home / "sanae-mem").resolve()
    _wait_all_pending_updates(timeout=30.0)
    with closing(sqlite3.connect(str(data_dir / "sanae.db"))) as conn:
        rows = conn.execute(
            "SELECT id FROM events WHERE blacklist=0 AND source_snapshot<>'' ORDER BY id"
        ).fetchall()
    return [int(r[0]) for r in rows]


def _event_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    with closing(sqlite3.connect(str(db_path))) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])


def test_grounded_review_concurrent_and_target_failure_rolls_back_atomically(tmp_path: Path) -> None:
    """Concurrent grounded reviews + SQLite TRIGGER failure on recognition_review_targets leaves 0 orphan rows."""
    home = tmp_path / "grounded-home"
    covered_entry = "我喜欢用纯黑美式咖啡开始一天"
    summary_text = "每天早上都会坚持喝美式并晨跑"
    now = datetime.now().astimezone()
    daily_cap = 13

    def responder(payload: dict[str, Any], _idx: int) -> tuple[int, dict[str, Any]]:
        user_blob = json.dumps(payload.get("messages", []), ensure_ascii=False)
        if "认识正文" in user_blob or "事实证据" in user_blob:
            return 200, {
                "duty": "user",
                "source_kind": "user_statement",
                "long_term": True,
                "summary": summary_text,
                "covered_by": covered_entry,
                "significance": "稳定习惯",
            }
        return 200, {"events": [], "commitments": [], "updates": []}

    llm_server = _LoopbackLLMServer(responder)
    try:
        data_dir, _ = _write_home_config(
            home, base_url=llm_server.base_url, resident_review_daily_cap=daily_cap,
        )
        with _discover_plugin_in_home(home):
            event_ids = _seed_grounded_events(home, 6)
            assert len(event_ids) == 6

            with _home_override(home), _private_turn("sess-review", "msg-review"):
                raw_worker = load_memory_provider("sanae_mem", register_skills=False)
                assert raw_worker is not None
                worker = cast("SanaeMemProvider", raw_worker)
                worker.initialize(
                    "sess-review",
                    hermes_home=home,
                    platform="telegram",
                    agent_context="primary",
                )
                try:
                    db_path = data_dir / "sanae.db"
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        conn.execute(
                            "CREATE TRIGGER block_grounded_target_insert "
                            "BEFORE INSERT ON recognition_review_targets "
                            "BEGIN SELECT RAISE(ABORT, 'injected_target_failure'); END"
                        )
                        conn.commit()

                    def run_review(eid: int) -> dict[str, Any]:
                        with _home_override(home), _private_turn("sess-review", f"msg-r-{eid}"):
                            return worker.grounded_review.review_grounded_user(
                                text=summary_text,
                                evidence_ids=[eid],
                                memory_dir=home / "memories",
                                now=now,
                            )

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        failed_results = list(pool.map(run_review, event_ids))

                    assert all(
                        res.get("ok") is False and res.get("reason") == "review_coverage_binding_failed"
                        for res in failed_results
                    ), failed_results

                    with closing(sqlite3.connect(str(db_path))) as conn:
                        review_count = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
                        target_count = conn.execute(
                            "SELECT COUNT(*) FROM recognition_review_targets"
                        ).fetchone()[0]
                    assert review_count == 0, f"expected 0 orphan event_reviews rows, got {review_count}"
                    assert target_count == 0

                    with closing(sqlite3.connect(str(db_path))) as conn:
                        conn.execute("DROP TRIGGER IF EXISTS block_grounded_target_insert")
                        conn.commit()

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        ok_results = list(pool.map(run_review, event_ids))

                    assert all(res.get("ok") is True for res in ok_results), ok_results
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        review_count = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
                        target_count = conn.execute(
                            "SELECT COUNT(*) FROM recognition_review_targets"
                        ).fetchone()[0]
                    assert review_count == 6
                    assert target_count == 6

                    request_count = len(llm_server.requests)
                    with ThreadPoolExecutor(max_workers=6) as pool:
                        final_results = list(pool.map(run_review, event_ids))
                    assert sum(result["ok"] is True for result in final_results) == 1
                    assert all(
                        result["reason"] == "grounded_review_unavailable"
                        for result in final_results if not result["ok"]
                    )
                    assert len(llm_server.requests) == request_count + 1

                    promotion = worker.promotion_review.review_promotion_candidate(
                        event_ids[0], now=now, enforce_daily_cap=False,
                    )
                    assert promotion["review_skip_reason"] == "quota_exhausted"
                    assert len(llm_server.requests) == request_count + 1
                    status = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
                    assert status["daily_resident_reviews"] == daily_cap
                finally:
                    worker.shutdown()
    finally:
        llm_server.close()


def test_promotion_review_concurrent_and_target_failure_rolls_back_atomically(tmp_path: Path) -> None:
    """Concurrent promotion reviews + SQLite TRIGGER failure leaves 0 orphan rows and blocks stage promotion."""
    home = tmp_path / "promo-home"
    covered_entry = "我平时习惯早睡早起"

    def responder(_payload: dict[str, Any], _idx: int) -> tuple[int, dict[str, Any]]:
        return 200, {
            "duty": "user",
            "long_term": True,
            "summary": "保持规律作息与河边慢跑习惯",
            "covered_by": covered_entry,
            "significance": "长期习惯",
        }

    llm_server = _LoopbackLLMServer(responder)
    try:
        data_dir, plugin_cfg = _write_home_config(home, base_url=llm_server.base_url)
        with _discover_plugin_in_home(home):
            event_ids = _seed_grounded_events(home, 6)
            assert len(event_ids) == 6

            with _home_override(home), _private_turn("sess-promo", "msg-promo"):
                raw_worker = load_memory_provider("sanae_mem", register_skills=False)
                assert raw_worker is not None
                worker = cast("SanaeMemProvider", raw_worker)
                worker.initialize(
                    "sess-promo",
                    hermes_home=home,
                    platform="telegram",
                    agent_context="primary",
                )
                try:
                    db_path = data_dir / "sanae.db"
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        conn.execute(
                            "CREATE TRIGGER block_promo_target_insert "
                            "BEFORE INSERT ON recognition_review_targets "
                            "BEGIN SELECT RAISE(ABORT, 'injected_promo_target_failure'); END"
                        )
                        conn.commit()

                    def run_promo(eid: int) -> dict[str, Any]:
                        with _home_override(home), _private_turn("sess-promo", f"msg-p-{eid}"):
                            return worker.promotion_review.review_promotion_candidate(
                                eid,
                                now=datetime.now().astimezone(),
                                enforce_daily_cap=False,
                            )

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        failed_results = list(pool.map(run_promo, event_ids))

                    assert all(
                        res.get("review_sha256") == ""
                        and res.get("action") == "none"
                        and res.get("reason") == "review_storage_failed"
                        for res in failed_results
                    ), failed_results

                    # Public tool sanae_mem_review(mode="stage") must NOT promote the event when
                    # the atomic review transaction fails and rolls back.
                    stage_raw = worker.handle_tool_call(
                        "sanae_mem_review",
                        {"mode": "stage", "event_id": event_ids[0]},
                    )
                    stage_envelope = json.loads(stage_raw)
                    stage_res = stage_envelope.get("result") if isinstance(stage_envelope.get("result"), dict) else stage_envelope
                    assert stage_res.get("ok") is False, stage_envelope
                    assert stage_res.get("promoted") is False, stage_envelope
                    assert stage_res.get("reason") == "review_storage_failed", stage_envelope

                    with closing(sqlite3.connect(str(db_path))) as conn:
                        tier_row = conn.execute(
                            "SELECT tier FROM events WHERE id=?", (event_ids[0],)
                        ).fetchone()
                        review_count = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
                        target_count = conn.execute(
                            "SELECT COUNT(*) FROM recognition_review_targets"
                        ).fetchone()[0]
                        evidence_count = conn.execute(
                            "SELECT COUNT(*) FROM recognition_evidence"
                        ).fetchone()[0]
                    assert tier_row is not None and str(tier_row[0]) == "short"
                    assert review_count == 0, f"expected 0 orphan event_reviews rows, got {review_count}"
                    assert target_count == 0
                    assert evidence_count == 0

                    with closing(sqlite3.connect(str(db_path))) as conn:
                        conn.execute("DROP TRIGGER IF EXISTS block_promo_target_insert")
                        conn.commit()

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        ok_results = list(pool.map(run_promo, event_ids))

                    assert all(bool(res.get("review_sha256")) for res in ok_results), ok_results
                    with closing(sqlite3.connect(str(db_path))) as conn:
                        review_count = conn.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0]
                        target_count = conn.execute(
                            "SELECT COUNT(*) FROM recognition_review_targets"
                        ).fetchone()[0]
                    assert review_count == 6
                    assert target_count == 6
                finally:
                    worker.shutdown()
    finally:
        llm_server.close()


def test_concurrent_group_delivery_receipts_and_failure_status_recording(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Concurrent emit_delivery_receipt for group turns records ok=False reason in status.json and asserts DB diff."""
    home = tmp_path / "group-home"
    mode_lock = threading.Lock()
    state = {"fail": True}

    def responder(_payload: dict[str, Any], _idx: int) -> tuple[int, Any]:
        with mode_lock:
            should_fail = state["fail"]
        if should_fail:
            return 200, "not-valid-json"
        return 200, {
            "events": [
                {"text": "群友提到周末一起去爬山", "quote": "周末一起去爬山吧"},
            ],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [],
        }

    llm_server = _LoopbackLLMServer(responder)
    canary_msg = "CANARY_GROUP_MSG_DO_NOT_LEAK"
    try:
        data_dir, _plugin_cfg = _write_home_config(home, base_url=llm_server.base_url, group_enabled=True)
        db_path = data_dir / "sanae.db"
        with _discover_plugin_in_home(home):
            base_ts = time.time()
            assert _event_count(db_path) == 0
            assert len(llm_server.requests) == 0

            def emit_one(idx: int, *, text: str) -> None:
                receipt = DeliveryReceipt(
                    session_id="qq:group:room-1",
                    platform="qq",
                    chat_type="group",
                    chat_id="room-1",
                    scope_id="default",
                    sender_id=f"member-{idx}",
                    turn_id=f"grp-turn-{idx}-{int(base_ts * 1000)}",
                    message_id=f"grp-out-{idx}-{int(base_ts * 1000)}",
                    incoming_message_id=f"grp-in-{idx}-{int(base_ts * 1000)}",
                    session_key="agent:main:qq:group:room-1",
                    user_message=text,
                    delivered_text="好呀。",
                    clean_text="好呀。",
                    delivered_at=base_ts + idx,
                    transport_confirmed=True,
                    user_message_authored=True,
                )
                with _home_override(home):
                    emit_delivery_receipt(receipt)

            with caplog.at_level(logging.WARNING):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(emit_one, i, text=f"{canary_msg} 第{i}条消息")
                        for i in range(4)
                    ]
                    for fut in futures:
                        fut.result()

                _wait_all_pending_updates(timeout=15.0)

            # Captured HTTP requests == 4, SQLite events diff == 0 on failed deliveries.
            assert len(llm_server.requests) == 4
            assert _event_count(db_path) == 0

            status_path = data_dir / "status.json"
            assert status_path.is_file()
            raw_status = status_path.read_text(encoding="utf-8")
            assert canary_msg not in raw_status
            assert canary_msg not in caplog.text
            assert "group delivery failed" in caplog.text

            status = json.loads(raw_status)
            gd = status["components"]["group_delivery"]
            assert gd["ok"] is False
            assert gd["stage"] == "apply_group_delivery"
            assert gd["error"] == "invalid_json"
            assert gd["failure_count"] == 4

            # Subsequent valid group delivery writes exactly 1 group event and recovers status.json.
            with mode_lock:
                state["fail"] = False
            emit_one(99, text="周末一起去爬山吧")
            _wait_all_pending_updates(timeout=15.0)
            assert len(llm_server.requests) == 5
            assert _event_count(db_path) == 1
            with closing(sqlite3.connect(str(db_path))) as conn:
                saved_raw = str(conn.execute("SELECT raw FROM events LIMIT 1").fetchone()[0])
            assert "周末一起去爬山吧" in saved_raw

            recovered = json.loads(status_path.read_text(encoding="utf-8"))
            gd_ok = recovered["components"]["group_delivery"]
            assert gd_ok["ok"] is True
            assert gd_ok["failure_count"] == 0
    finally:
        llm_server.close()


def test_group_commitment_archive_failure_preserves_pending_and_increments_failure_count(
    tmp_path: Path,
) -> None:
    """When archiving a group commitment fails, the pending/v1 row must stay intact
    and ``group_delivery`` in status.json must record ok=False with incremented
    failure_count (never reset to ok=True, failure_count=0).
    """
    home = tmp_path / "group-archive-fail-home"
    step_lock = threading.Lock()
    state: dict[str, Any] = {"step": "create", "commitment_id": None}

    def responder(_payload: dict[str, Any], _idx: int) -> tuple[int, Any]:
        with step_lock:
            step = state["step"]
            cid = state["commitment_id"]
        if step == "create":
            return 200, {
                "events": [],
                "impression": None,
                "new_commitments": [
                    {"text": "周末帮小李看代码", "quote": "周末我帮你看代码"},
                ],
                "commitment_updates": [],
            }
        if step == "bad_json":
            return 200, "invalid-json-payload"
        assert cid is not None
        return 200, {
            "events": [],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [
                {
                    "id": int(cid),
                    "action": "fulfilled",
                    "basis": "user_confirmed",
                    "quote": "谢谢，已经看完了",
                }
            ],
        }

    llm_server = _LoopbackLLMServer(responder)
    try:
        data_dir, _ = _write_home_config(
            home, base_url=llm_server.base_url, group_enabled=True
        )
        db_path = data_dir / "sanae.db"
        archive_path = home / "memory-archive" / "commitments-archived.jsonl"
        status_path = data_dir / "status.json"

        with _discover_plugin_in_home(home):
            base_ts = time.time()

            def emit_turn(idx: int, *, user_text: str, delivered_text: str) -> None:
                receipt = DeliveryReceipt(
                    session_id="qq:group:room-archive",
                    platform="qq",
                    chat_type="group",
                    chat_id="room-archive",
                    scope_id="default",
                    sender_id="member-1",
                    turn_id=f"grp-arch-{idx}-{int(base_ts * 1000)}",
                    message_id=f"grp-arch-out-{idx}",
                    incoming_message_id=f"grp-arch-in-{idx}",
                    session_key="agent:main:qq:group:room-archive",
                    user_message=user_text,
                    delivered_text=delivered_text,
                    clean_text=delivered_text,
                    delivered_at=base_ts + idx,
                    transport_confirmed=True,
                    user_message_authored=True,
                )
                with _home_override(home):
                    emit_delivery_receipt(receipt)

            # 1. Create pending/v1 commitment
            emit_turn(1, user_text="周末有空能帮我看下代码吗", delivered_text="好呀，周末我帮你看代码。")
            _wait_all_pending_updates(timeout=15.0)
            with closing(sqlite3.connect(str(db_path))) as conn:
                row = conn.execute(
                    "SELECT id, status, version FROM commitments"
                ).fetchone()
            assert row is not None
            cid, status_val, ver = int(row[0]), str(row[1]), int(row[2])
            assert status_val == "pending" and ver == 1
            with step_lock:
                state["commitment_id"] = cid
                state["step"] = "bad_json"

            # 2. Trigger an initial failure so failure_count == 1
            emit_turn(2, user_text="今天天气真好", delivered_text="是呀。")
            _wait_all_pending_updates(timeout=15.0)
            st_after_bad = json.loads(status_path.read_text(encoding="utf-8"))
            gd_bad = st_after_bad["components"]["group_delivery"]
            assert gd_bad["ok"] is False
            assert gd_bad["failure_count"] == 1

            # 3. Make archive path unwritable by placing a directory at archive_path
            archive_path.mkdir(parents=True, exist_ok=True)
            with step_lock:
                state["step"] = "fulfill"
            emit_turn(3, user_text="谢谢，已经看完了", delivered_text="不客气！")
            _wait_all_pending_updates(timeout=15.0)

            with closing(sqlite3.connect(str(db_path))) as conn:
                row_after_fail = conn.execute(
                    "SELECT status, version FROM commitments WHERE id = ?",
                    (cid,),
                ).fetchone()
            assert row_after_fail is not None
            assert str(row_after_fail[0]) == "pending"
            assert int(row_after_fail[1]) == 1
            assert _event_count(db_path) == 0

            st_after_archive_fail = json.loads(status_path.read_text(encoding="utf-8"))
            gd_archive_fail = st_after_archive_fail["components"]["group_delivery"]
            assert gd_archive_fail["ok"] is False
            assert gd_archive_fail["error"] == "archive_failed"
            assert gd_archive_fail["failure_count"] == 2

            # 4. Remove the blocking directory and retry confirmation -> fulfilled/v2 & ok=True
            archive_path.rmdir()
            emit_turn(4, user_text="谢谢，已经看完了", delivered_text="收到！")
            _wait_all_pending_updates(timeout=15.0)

            with closing(sqlite3.connect(str(db_path))) as conn:
                row_after_recover = conn.execute(
                    "SELECT status, version FROM commitments WHERE id = ?",
                    (cid,),
                ).fetchone()
            assert row_after_recover is not None
            assert str(row_after_recover[0]) == "fulfilled"
            assert int(row_after_recover[1]) == 2
            assert archive_path.is_file()
            archived_lines = [
                line for line in archive_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            assert len(archived_lines) == 1
            assert json.loads(archived_lines[0])["id"] == cid

            st_recovered = json.loads(status_path.read_text(encoding="utf-8"))
            gd_recovered = st_recovered["components"]["group_delivery"]
            assert gd_recovered["ok"] is True
            assert gd_recovered["failure_count"] == 0
    finally:
        llm_server.close()
