"""Multi-threaded concurrency stress tests for B2-06B-1 SqliteFacade migration."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402


def _write_home_config(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config_yaml = "\n".join([
        "plugins:",
        "  sanae_mem:",
        "    memory_identity:",
        "      platform: discord",
        "      user_id: owner-a",
        "    group_context:",
        "      defaults:",
        "        long_term_memory: false",
        "        short_term_shared: false",
        "      overrides:",
        '        "default/discord/dom-a/guild-a":',
        "          long_term_memory: true",
        "          short_term_shared: true",
    ]) + "\n"
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")


def test_b2_06b1_concurrent_storage_domains_and_provider_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes-home"
    _write_home_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))

    data_dir = tmp_path / "sanae-mem"
    provider = SanaeMemProvider({
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "discord", "user_id": "owner-a"},
        "partner_name": "PartnerA",
    })
    session_id = "conc-priv-session"
    with host_bridge.turn_envelope_scope(
        platform="discord",
        chat_type="direct",
        chat_id="dm-owner-a",
        sender_id="owner-a",
        session_id=session_id,
        session_key=session_id,
        message_id="init-msg",
    ):
        provider.initialize(
            session_id,
            hermes_home=str(home),
            platform="discord",
            agent_context="primary",
        )

    errors: List[BaseException] = []
    errors_lock = threading.Lock()
    saved_meme_titles: List[str] = []
    recorded_commitment_ids: List[int] = []
    fulfilled_commitment_ids: List[int] = []
    cancelled_commitment_ids: List[int] = []
    added_event_markers: List[str] = []
    state_lock = threading.Lock()

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def _record_error(exc: BaseException) -> None:
        with errors_lock:
            errors.append(exc)

    def _private_scope(msg_id: str):
        return host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="direct",
            chat_id="dm-owner-a",
            sender_id="owner-a",
            session_id=session_id,
            session_key=session_id,
            message_id=msg_id,
        )

    def _group_scope(msg_id: str):
        return host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="group",
            chat_id="guild-a",
            scope_id="dom-a",
            sender_id="member-1",
            session_id="conc-group-session",
            session_key="conc-group-session",
            message_id=msg_id,
        )

    def meme_writer(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(12):
                title = f"梗词并发{worker_idx}_{i}"
                with _private_scope(f"mw-{worker_idx}-{i}"):
                    raw = provider.handle_tool_call("sanae_meme_save", {
                        "title": title,
                        "meaning": f"并发测试梗解释 {worker_idx}-{i}",
                        "aliases": [f"别名{worker_idx}_{i}"],
                    }, session_id=session_id)
                resp = json.loads(raw)
                assert resp["ok"] is True, resp
                assert resp["result"]["stored"] is True, resp
                with state_lock:
                    saved_meme_titles.append(title)
        except BaseException as exc:
            _record_error(exc)

    def meme_reader(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(25):
                with _private_scope(f"mr-{worker_idx}-{i}"):
                    raw = provider.handle_tool_call("sanae_meme_search", {
                        "query": "梗词并发",
                        "top_k": 5,
                    }, session_id=session_id)
                    resp = json.loads(raw)
                    assert resp["ok"] is True, resp
                    _ = provider.meme_context_for_turn("刚才聊到梗词并发")
                    _ = provider.known_meme_terms(["梗词并发0_0"])
        except BaseException as exc:
            _record_error(exc)

    def commitment_worker(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            owner_key = "discord:owner-a"
            for i in range(12):
                text = f"并发约定-{worker_idx}-{i}"
                quote = f"我答应你完成并发约定-{worker_idx}-{i}"
                cid = provider.record_commitment(
                    owner_key,
                    text,
                    source_quote=quote,
                    turn_id=f"turn-c-{worker_idx}-{i}",
                )
                assert cid is not None
                with state_lock:
                    recorded_commitment_ids.append(cid)
                if i % 3 == 0:
                    ok = provider.commitments.resolve_commitment(
                        cid,
                        "fulfilled",
                        evidence=f"done-{worker_idx}-{i}",
                        expected_version=1,
                    )
                    assert ok is True
                    with state_lock:
                        fulfilled_commitment_ids.append(cid)
                elif i % 3 == 1:
                    ok = provider.commitments.resolve_commitment(
                        cid,
                        "cancelled",
                        evidence=f"cancel-{worker_idx}-{i}",
                        expected_version=1,
                    )
                    assert ok is True
                    with state_lock:
                        cancelled_commitment_ids.append(cid)
                _ = provider.commitments.render_pending_commitments(owner_key)
                _ = provider.list_commitments(owner_key=owner_key)
        except BaseException as exc:
            _record_error(exc)

    def event_writer(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(12):
                marker = f"并发记忆事件_{worker_idx}_{i}"
                with _private_scope(f"ew-{worker_idx}-{i}"):
                    raw = provider.handle_tool_call("sanae_mem_add", {
                        "content": f"今天一起测试了{marker}，一切顺利。",
                        "topic": "日常",
                        "importance": 0.6,
                    }, session_id=session_id)
                resp = json.loads(raw)
                assert resp["ok"] is True, resp
                with state_lock:
                    added_event_markers.append(marker)
        except BaseException as exc:
            _record_error(exc)

    def retrieval_and_maintenance_reader(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            now_iso = datetime.now().astimezone().isoformat()
            for i in range(20):
                with _private_scope(f"rr-{worker_idx}-{i}"):
                    raw = provider.handle_tool_call("sanae_mem_search", {
                        "query": "并发记忆事件",
                        "top_k": 5,
                    }, session_id=session_id)
                    resp = json.loads(raw)
                    assert resp["ok"] is True, resp
                    _ = provider.event_maintenance.expire_to_review(now_iso)
                    _ = provider.event_maintenance.decay_expiring_importance(0.9, 0.4, now_iso)
        except BaseException as exc:
            _record_error(exc)

    def provider_identity_and_group_reader(worker_idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(25):
                assert provider.is_active_session(session_id) is True
                assert provider.is_active_session("other-session") is False
                with _group_scope(f"gr-{worker_idx}-{i}"):
                    assert provider.permits_group_delivery() is True
                with _private_scope(f"pr-{worker_idx}-{i}"):
                    assert provider.permits_group_delivery() is False
                    res = provider.handle_pre_llm_call(
                        session_id=session_id,
                        user_message="聊聊梗词并发",
                        turn_id=f"pre-{worker_idx}-{i}",
                    )
                    assert isinstance(res, dict)
        except BaseException as exc:
            _record_error(exc)

    threads = [
        threading.Thread(target=meme_writer, args=(0,)),
        threading.Thread(target=meme_writer, args=(1,)),
        threading.Thread(target=meme_reader, args=(2,)),
        threading.Thread(target=meme_reader, args=(3,)),
        threading.Thread(target=commitment_worker, args=(4,)),
        threading.Thread(target=commitment_worker, args=(5,)),
        threading.Thread(target=event_writer, args=(6,)),
        threading.Thread(target=retrieval_and_maintenance_reader, args=(7,)),
        threading.Thread(target=provider_identity_and_group_reader, args=(8,)),
        threading.Thread(target=provider_identity_and_group_reader, args=(9,)),
    ]

    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive(), "worker thread timed out"

        assert errors == [], f"Unexpected concurrency errors: {errors!r}"

        # Verify data consistency via public APIs and fresh read-only DB inspection.
        with _private_scope("verify-memes"):
            search_raw = provider.handle_tool_call("sanae_meme_search", {
                "query": "梗词并发0_0",
                "top_k": 5,
            }, session_id=session_id)
            search_resp = json.loads(search_raw)
            assert search_resp["ok"] is True
            assert len(saved_meme_titles) == 24

        all_commitments = provider.list_commitments(owner_key="discord:owner-a")
        assert len(all_commitments) == len(recorded_commitment_ids) == 24
        by_id = {int(c["id"]): c for c in all_commitments}
        for cid in fulfilled_commitment_ids:
            assert by_id[cid]["status"] == "fulfilled"
            assert int(by_id[cid]["version"]) == 2
        for cid in cancelled_commitment_ids:
            assert by_id[cid]["status"] == "cancelled"
            assert int(by_id[cid]["version"]) == 2

        archive_path = home / "memory-archive" / "commitments-archived.jsonl"
        assert archive_path.is_file(), "commitments-archived.jsonl must be written before resolve"
        archived_entries = [
            json.loads(line)
            for line in archive_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(archived_entries) == len(fulfilled_commitment_ids) + len(cancelled_commitment_ids)
        for cid in fulfilled_commitment_ids:
            matches = [entry for entry in archived_entries if int(entry["id"]) == cid]
            assert len(matches) == 1
            assert matches[0]["archive_reason"] == "commitment_fulfilled"
            assert matches[0]["resolution"]["action"] == "fulfilled"
        for cid in cancelled_commitment_ids:
            matches = [entry for entry in archived_entries if int(entry["id"]) == cid]
            assert len(matches) == 1
            assert matches[0]["archive_reason"] == "commitment_cancelled"
            assert matches[0]["resolution"]["action"] == "cancelled"

        db_path = data_dir / "sanae.db"
        ro_conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            meme_count = ro_conn.execute("SELECT COUNT(*) FROM meme_records").fetchone()[0]
            assert meme_count == 24
            event_count = ro_conn.execute(
                "SELECT COUNT(*) FROM events WHERE raw LIKE '%并发记忆事件_%'"
            ).fetchone()[0]
            assert event_count == len(added_event_markers) == 12
        finally:
            ro_conn.close()
    finally:
        provider.shutdown()


def test_b2_06b1_concurrent_same_commitment_resolve_archives_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent fulfilled vs cancelled on the same commitment archives exactly once."""
    home = tmp_path / "hermes-home"
    _write_home_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))

    data_dir = tmp_path / "sanae-mem"
    provider = SanaeMemProvider({
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "discord", "user_id": "owner-a"},
        "partner_name": "PartnerA",
    })
    session_id = "conc-race-session"
    with host_bridge.turn_envelope_scope(
        platform="discord",
        chat_type="direct",
        chat_id="dm-owner-a",
        sender_id="owner-a",
        session_id=session_id,
        session_key=session_id,
        message_id="race-init-msg",
    ):
        provider.initialize(
            session_id,
            hermes_home=str(home),
            platform="discord",
            agent_context="primary",
        )

    try:
        owner_key = "discord:owner-a"
        race_ids: List[int] = []
        winners_by_id: dict[int, str] = {}
        actions = ("fulfilled", "cancelled", "fulfilled", "cancelled")

        for round_idx in range(6):
            cid = provider.record_commitment(
                owner_key,
                f"同一待办并发竞态测试_{round_idx}",
                source_quote=f"答应并发竞态_{round_idx}",
                turn_id=f"race-turn-{round_idx}",
                now=datetime(2026, 6, 1, 12, round_idx, 0).astimezone(),
            )
            assert cid is not None
            race_ids.append(cid)

            barrier = threading.Barrier(len(actions))
            outcomes: List[tuple[str, bool]] = []
            errors: List[BaseException] = []
            outcomes_lock = threading.Lock()

            def _racer(action: str, idx: int, target_id: int = cid) -> None:
                try:
                    barrier.wait(timeout=10)
                    ok = provider.commitments.resolve_commitment(
                        target_id,
                        action,
                        evidence=f"race-{round_idx}-{idx}-{action}",
                        expected_version=1,
                    )
                    with outcomes_lock:
                        outcomes.append((action, ok))
                except BaseException as exc:
                    with outcomes_lock:
                        errors.append(exc)

            threads = [
                threading.Thread(target=_racer, args=(act, idx))
                for idx, act in enumerate(actions)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
                assert not t.is_alive(), "race worker thread timed out"

            assert errors == [], f"Unexpected race errors: {errors!r}"
            succeeded = [act for act, ok in outcomes if ok]
            failed = [act for act, ok in outcomes if not ok]
            assert len(succeeded) == 1, f"Expected exactly 1 winner for cid={cid}, got {outcomes!r}"
            assert len(failed) == len(actions) - 1
            winners_by_id[cid] = succeeded[0]

        all_commitments = provider.list_commitments(owner_key=owner_key)
        by_id = {int(c["id"]): c for c in all_commitments}
        archive_path = home / "memory-archive" / "commitments-archived.jsonl"
        assert archive_path.is_file()
        archived_lines = [
            json.loads(line)
            for line in archive_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(archived_lines) == len(race_ids)

        for cid in race_ids:
            winning_action = winners_by_id[cid]
            assert by_id[cid]["status"] == winning_action
            assert int(by_id[cid]["version"]) == 2
            matching_rows = [entry for entry in archived_lines if int(entry["id"]) == cid]
            assert len(matching_rows) == 1, (
                f"Expected exactly 1 archive entry for cid={cid}, got {matching_rows!r}"
            )
            expected_reason = (
                "commitment_cancelled" if winning_action == "cancelled" else "commitment_fulfilled"
            )
            assert matching_rows[0]["archive_reason"] == expected_reason
            assert matching_rows[0]["resolution"]["action"] == winning_action
    finally:
        provider.shutdown()
