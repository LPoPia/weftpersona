"""Regression tests for Task 01: memory_axis prefilter trigram cap."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any, Tuple

import pytest

from sanae_mem import SanaeMemProvider
from sanae_mem.infra.clock import shanghai_datetime
from sanae_mem.infra.host_bridge import hermes_home_scope, turn_envelope_scope
from sanae_mem.governance.memory_axis import (
    _MAX_PREFILTER_TERMS,
    _MIN_TEXT,
    _like_pattern,
    _prefilter_terms,
    _trigrams,
)


def _distinct_cjk_text(length: int, *, start: int = 0x4E00) -> str:
    """Return ``length`` distinct CJK characters producing ``length - 2`` unique trigrams."""
    return "".join(chr(start + offset) for offset in range(length))


def _create_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_id: str = "task01-user",
) -> Tuple[SanaeMemProvider, Path, str]:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    for env_key in ("HERMES_HOME", "HERMES_HOME_OVERRIDE"):
        monkeypatch.setenv(env_key, str(home))
    data_dir = home / "sanae-mem"
    provider = SanaeMemProvider(
        {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "memory_identity": {"platform": "telegram", "user_id": user_id},
        }
    )
    with hermes_home_scope(home), turn_envelope_scope(
        inherit_context=False,
        platform="telegram",
        chat_type="dm",
        chat_id=user_id,
        sender_id=user_id,
        session_id="task01-dm-session",
        session_key=f"agent:main:telegram:dm:{user_id}",
        message_id="init-1",
        turn_id="turn-init-1",
        is_messaging_surface=False,
        write_origin="foreground",
    ):
        provider.initialize("task01-dm-session", platform="telegram", hermes_home=str(home))
    return provider, data_dir / "sanae.db", home.as_posix()


def _seed_event_via_tool(
    provider: SanaeMemProvider,
    db_path: Path,
    *,
    home: Path,
    user_id: str,
    content: str,
    created_at: datetime,
) -> Tuple[int, str]:
    """Seed an event through the public tool API and backdate its timestamp in SQLite."""
    with hermes_home_scope(home), turn_envelope_scope(
        inherit_context=False,
        platform="telegram",
        chat_type="dm",
        chat_id=user_id,
        sender_id=user_id,
        session_id="task01-dm-session",
        session_key=f"agent:main:telegram:dm:{user_id}",
        message_id="seed-msg",
        turn_id="turn-seed",
        is_messaging_surface=False,
        write_origin="foreground",
    ):
        raw_reply = provider.handle_tool_call(
            "sanae_mem_add",
            {"content": content, "importance": 0.7},
            session_id="task01-dm-session",
        )
    payload: dict[str, Any] = json.loads(raw_reply)
    assert payload.get("ok") is True, payload
    result = payload.get("result")
    assert isinstance(result, dict) and result.get("stored") is True, payload
    event_id = int(result["id"])

    stamp = created_at.isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "UPDATE events SET created_at=?, updated_at=?, tier='short' WHERE id=?",
            (stamp, stamp, event_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT owner_key, tier FROM events WHERE id=?",
            (event_id,),
        ).fetchone()
    assert row is not None
    assert str(row["tier"]) == "short"
    owner_key = str(row["owner_key"])
    assert owner_key
    return event_id, owner_key


def test_prefilter_terms_caps_long_cjk_deterministically_with_full_span_coverage() -> None:
    assert 0 < _MAX_PREFILTER_TERMS <= 128

    long_text = _distinct_cjk_text(650)
    full_trigrams = _trigrams(long_text)
    assert len(full_trigrams) == 648
    assert len(full_trigrams) > 500

    first_pass = _prefilter_terms(long_text)
    second_pass = _prefilter_terms(long_text)
    assert first_pass == second_pass
    assert len(first_pass) == _MAX_PREFILTER_TERMS
    assert len(set(first_pass)) == _MAX_PREFILTER_TERMS
    assert set(first_pass).issubset(full_trigrams)

    head_grams = _trigrams(long_text[:20])
    middle_grams = _trigrams(long_text[315:335])
    tail_grams = _trigrams(long_text[-20:])
    assert set(first_pass) & head_grams
    assert set(first_pass) & middle_grams
    assert set(first_pass) & tail_grams

    # ASCII prefix must not starve CJK tail terms (unlike lexicographical truncation).
    ascii_prefix = "".join(f"A{idx:03d}" for idx in range(70))  # 280 chars
    cjk_tail = _distinct_cjk_text(420, start=0x5200)
    mixed_terms = _prefilter_terms(ascii_prefix + cjk_tail)
    assert len(mixed_terms) == _MAX_PREFILTER_TERMS
    assert set(mixed_terms) & _trigrams(cjk_tail[-24:])

    # Repeated filler prefix must be deduplicated before uniform sampling.
    padded = ("哈" * 300) + _distinct_cjk_text(620, start=0x5600)
    padded_terms = _prefilter_terms(padded)
    assert len(padded_terms) == _MAX_PREFILTER_TERMS
    assert set(padded_terms) & _trigrams(padded[-24:])


def test_prefilter_terms_short_text_unchanged_and_validation() -> None:
    assert _prefilter_terms("") == ()
    assert _prefilter_terms("太短了") == ()
    assert _prefilter_terms("   1234567   ") == ()
    assert len("一二三四五六七") < _MIN_TEXT
    assert _prefilter_terms("一二三四五六七") == ()

    short_text = "那把旧吉他的琴颈有一道细裂纹，我想等天暖了再拿去修"
    expected = tuple(sorted(_trigrams(short_text)))
    assert len(expected) <= _MAX_PREFILTER_TERMS
    assert _prefilter_terms(short_text) == expected

    exact_cap_text = _distinct_cjk_text(_MAX_PREFILTER_TERMS + 2, start=0x5A00)
    assert len(_trigrams(exact_cap_text)) == _MAX_PREFILTER_TERMS
    assert _prefilter_terms(exact_cap_text) == tuple(sorted(_trigrams(exact_cap_text)))

    over_by_one_text = _distinct_cjk_text(_MAX_PREFILTER_TERMS + 3, start=0x5C00)
    over_terms = _prefilter_terms(over_by_one_text)
    assert len(over_terms) == _MAX_PREFILTER_TERMS
    assert over_by_one_text[:3] in over_terms
    assert over_by_one_text[-3:] in over_terms

    single = _prefilter_terms(short_text, max_terms=1)
    assert len(single) == 1
    assert single[0] in _trigrams(short_text)

    with pytest.raises(ValueError):
        _prefilter_terms(short_text, max_terms=0)
    with pytest.raises(ValueError):
        _prefilter_terms(short_text, max_terms=-4)


def test_sqlite_prefilter_query_depth_stays_safe_for_650_char_cjk_message() -> None:
    long_text = _distinct_cjk_text(650, start=0x6000)
    uncapped_terms = tuple(sorted(_trigrams(long_text)))
    capped_terms = _prefilter_terms(long_text)
    assert len(uncapped_terms) >= 600
    assert len(capped_terms) <= _MAX_PREFILTER_TERMS

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(
            "CREATE TABLE events ("
            "id INTEGER PRIMARY KEY, owner_key TEXT, raw TEXT, summary TEXT, "
            "blacklist INTEGER DEFAULT 0, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO events (owner_key, raw, summary, blacklist, created_at) "
            "VALUES (?, ?, ?, 0, ?)",
            ("owner-1", long_text, long_text[:150], "2026-11-10T12:00:00+08:00"),
        )
        conn.commit()

        # Uncapped 600+ trigrams (1200+ OR-ed LIKE clauses) exceed SQLite's 1000 depth limit.
        uncapped_pattern = " OR ".join(
            ["raw LIKE ? ESCAPE '\\'", "summary LIKE ? ESCAPE '\\'"] * len(uncapped_terms)
        )
        uncapped_params: list[Any] = ["owner-1", "2026-01-01T00:00:00+08:00"]
        for term in uncapped_terms:
            uncapped_params.extend([_like_pattern(term)] * 2)
        uncapped_params.append(120)
        with pytest.raises(sqlite3.OperationalError, match="Expression tree is too large"):
            conn.execute(
                "SELECT id FROM events "
                "WHERE owner_key IN (?) AND blacklist=0 AND created_at>=? "
                f"AND ({uncapped_pattern}) ORDER BY id DESC LIMIT ?",
                uncapped_params,
            ).fetchall()

        # Capped terms (<= 256 OR-ed LIKE clauses) execute cleanly and hit the target row.
        capped_pattern = " OR ".join(
            ["raw LIKE ? ESCAPE '\\'", "summary LIKE ? ESCAPE '\\'"] * len(capped_terms)
        )
        capped_params: list[Any] = ["owner-1", "2026-01-01T00:00:00+08:00"]
        for term in capped_terms:
            capped_params.extend([_like_pattern(term)] * 2)
        capped_params.append(120)
        matched = conn.execute(
            "SELECT id FROM events "
            "WHERE owner_key IN (?) AND blacklist=0 AND created_at>=? "
            f"AND ({capped_pattern}) ORDER BY id DESC LIMIT ?",
            capped_params,
        ).fetchall()
        assert [int(row[0]) for row in matched] == [1]


def test_long_650_char_cjk_remention_hits_and_promotes_on_real_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "task01-long-remention"
    provider, db_path, home_str = _create_provider(tmp_path, monkeypatch, user_id=user_id)
    home = Path(home_str)
    base_noon = shanghai_datetime().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - timedelta(days=5)

    long_event_text = _distinct_cjk_text(650, start=0x6400)
    assert len(_trigrams(long_event_text)) >= 600
    unrelated_text = _distinct_cjk_text(40, start=0x7800)

    try:
        long_event_id, owner_key = _seed_event_via_tool(
            provider,
            db_path,
            home=home,
            user_id=user_id,
            content=long_event_text,
            created_at=base_noon,
        )
        unrelated_id, _ = _seed_event_via_tool(
            provider,
            db_path,
            home=home,
            user_id=user_id,
            content=unrelated_text,
            created_at=base_noon,
        )

        # Replay 650+ unique-character Chinese messages across 3 distinct sleep days.
        expected_stages = (
            ("d1", 1, "instant", False),
            ("d2", 2, "medium", False),
            ("d3", 3, "long", True),
        )
        for label, day_offset, expected_stage, expected_promoted in expected_stages:
            # Slightly varied 660-char message preserving high overlap with the long event.
            user_message = long_event_text + _distinct_cjk_text(12, start=0x7C00 + day_offset * 16)
            assert len(_trigrams(user_message)) >= 650
            result = provider.memory_axis.record_delivered_rementions(
                owner_key=owner_key,
                user_text=user_message,
                assistant_text="我都认真听进去了，这件事还一直记着。",
                turn_id=f"task01-long-{label}",
                delivered_at=base_noon + timedelta(days=day_offset, minutes=10),
            )
            assert result["ok"] is True, result
            assert result["recorded"] is True, result
            assert result["event_id"] == long_event_id, result
            assert result["stage"] == expected_stage, result
            assert result["promoted"] is expected_promoted, result
    finally:
        provider.shutdown()

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        long_row = conn.execute(
            "SELECT tier, promoted_reason FROM events WHERE id=?",
            (long_event_id,),
        ).fetchone()
        unrelated_row = conn.execute(
            "SELECT tier FROM events WHERE id=?",
            (unrelated_id,),
        ).fetchone()
        mention_rows = conn.execute(
            "SELECT event_id, stage FROM memory_axis_mentions "
            "WHERE owner_key=? AND channel='delivered_remention' ORDER BY id",
            (owner_key,),
        ).fetchall()
        promo_rows = conn.execute(
            "SELECT event_id, channel, previous_tier FROM memory_axis_promotions "
            "WHERE owner_key=?",
            (owner_key,),
        ).fetchall()
        watermark = conn.execute(
            "SELECT last_error, failed_count FROM memory_axis_watermarks "
            "WHERE owner_key=? AND channel='delivered_remention'",
            (owner_key,),
        ).fetchone()

    assert long_row is not None
    assert str(long_row["tier"]) == "long"
    assert str(long_row["promoted_reason"]) == "memory_axis:delivered_remention"
    assert unrelated_row is not None and str(unrelated_row["tier"]) == "short"
    assert [(int(r["event_id"]), str(r["stage"])) for r in mention_rows] == [
        (long_event_id, "instant"),
        (long_event_id, "medium"),
        (long_event_id, "long"),
    ]
    assert [(int(r["event_id"]), str(r["channel"]), str(r["previous_tier"])) for r in promo_rows] == [
        (long_event_id, "delivered_remention", "short"),
    ]
    assert watermark is not None
    assert str(watermark["last_error"]) == ""
    assert int(watermark["failed_count"]) == 0


def test_long_650_char_message_recalls_tail_short_event_and_flash_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "task01-long-flash"
    provider, db_path, home_str = _create_provider(tmp_path, monkeypatch, user_id=user_id)
    home = Path(home_str)
    base_noon = shanghai_datetime().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - timedelta(days=3)
    short_event_text = "那把旧吉他的琴颈有一道细裂纹，我想等天暖了再拿去修"
    long_message_with_tail_event = _distinct_cjk_text(630, start=0x6A00) + short_event_text
    assert len(_trigrams(long_message_with_tail_event)) >= 640

    try:
        short_event_id, owner_key = _seed_event_via_tool(
            provider,
            db_path,
            home=home,
            user_id=user_id,
            content=short_event_text,
            created_at=base_noon,
        )
        flash_time = base_noon + timedelta(days=1, hours=2)
        prev_ts = (flash_time - timedelta(minutes=20)).timestamp()
        curr_ts = flash_time.timestamp()
        flash_result = provider.memory_axis.record_inner_state_flash(
            owner_key=owner_key,
            user_text=long_message_with_tail_event,
            assistant_text="听到你又提起那把旧吉他，心里一下子软了下来。",
            turn_id="task01-long-flash-1",
            previous_state={
                "variables": {"tension": 0.0, "closeness": 0.1, "energy": 0.0, "mood": 0.0},
                "felt_at": prev_ts,
            },
            current_state={
                "variables": {"tension": 0.0, "closeness": 0.85, "energy": 0.0, "mood": 0.3},
                "felt_at": curr_ts,
            },
            now=flash_time,
        )
        assert flash_result["ok"] is True, flash_result
        assert flash_result["recorded"] is True, flash_result
        assert flash_result["event_id"] == short_event_id, flash_result
        assert flash_result["promoted"] is True, flash_result
    finally:
        provider.shutdown()

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT tier, promoted_reason FROM events WHERE id=?",
            (short_event_id,),
        ).fetchone()
        watermark = conn.execute(
            "SELECT last_error, failed_count FROM memory_axis_watermarks "
            "WHERE owner_key=? AND channel='inner_state_flash'",
            (owner_key,),
        ).fetchone()

    assert row is not None
    assert str(row["tier"]) == "long"
    assert str(row["promoted_reason"]) == "memory_axis:inner_state_flash"
    assert watermark is not None
    assert str(watermark["last_error"]) == ""
    assert int(watermark["failed_count"]) == 0


def test_long_650_char_partial_middle_overlap_still_recalls_on_real_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 650-char message overlapping a prior event only in its middle (~32%) still hits."""
    user_id = "task01-middle-overlap"
    provider, db_path, home_str = _create_provider(tmp_path, monkeypatch, user_id=user_id)
    home = Path(home_str)
    base_noon = shanghai_datetime().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - timedelta(days=3)

    shared_middle = _distinct_cjk_text(210, start=0x7000)
    prior_event = _distinct_cjk_text(200, start=0x7200) + shared_middle + _distinct_cjk_text(200, start=0x7400)
    new_long_message = _distinct_cjk_text(220, start=0x7600) + shared_middle + _distinct_cjk_text(220, start=0x7800)
    assert len(_trigrams(new_long_message)) >= 640

    try:
        event_id, owner_key = _seed_event_via_tool(
            provider,
            db_path,
            home=home,
            user_id=user_id,
            content=prior_event,
            created_at=base_noon,
        )
        res = provider.memory_axis.record_delivered_rementions(
            owner_key=owner_key,
            user_text=new_long_message,
            assistant_text="那段中间提到的事情我也还记得。",
            turn_id="task01-mid-1",
            delivered_at=base_noon + timedelta(days=1, minutes=5),
        )
        assert res["ok"] is True, res
        assert res["recorded"] is True, res
        assert res["event_id"] == event_id, res
    finally:
        provider.shutdown()


def test_short_message_existing_recall_and_3day_promotion_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "task01-short-remention"
    provider, db_path, home_str = _create_provider(tmp_path, monkeypatch, user_id=user_id)
    home = Path(home_str)
    base_noon = shanghai_datetime().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - timedelta(days=5)
    short_event_text = "周末想去植物园温室看新开的热带兰花展览"

    try:
        event_id, owner_key = _seed_event_via_tool(
            provider,
            db_path,
            home=home,
            user_id=user_id,
            content=short_event_text,
            created_at=base_noon,
        )
        turns = (
            ("s1", 1, "周末想去植物园温室看新开的热带兰花展览", "instant", False),
            ("s2", 2, "我这周末还是想去植物园温室看热带兰花展览", "medium", False),
            ("s3", 3, "今天又想起去植物园温室看新开的热带兰花展览", "long", True),
        )
        for label, day_offset, query_text, expected_stage, expected_promoted in turns:
            res = provider.memory_axis.record_delivered_rementions(
                owner_key=owner_key,
                user_text=query_text,
                assistant_text="到时候一起去看看吧。",
                turn_id=f"task01-short-{label}",
                delivered_at=base_noon + timedelta(days=day_offset, minutes=5),
            )
            assert res["ok"] is True, res
            assert res["recorded"] is True, res
            assert res["event_id"] == event_id, res
            assert res["stage"] == expected_stage, res
            assert res["promoted"] is expected_promoted, res
    finally:
        provider.shutdown()

    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT tier, promoted_reason FROM events WHERE id=?",
            (event_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "long"
    assert row[1] == "memory_axis:delivered_remention"
