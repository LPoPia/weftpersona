"""隔离沙箱里的时间快进推演。

合成库验证曲线机制；线上真实快照（DB含WAL、常驻文件、Chroma）验证存量迁移后
与新增永久结构在 365 天内不被误删。真实库某类无存量时用例会告警，并在同一 /tmp
副本补受控不变量夹具，绝不写生产，也不把纯合成库当作线上实测。
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import time
import warnings
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from test_sanae_mem import make_provider, _trusted_turns, _enter_test_turn
from sanae_mem import SanaeMemProvider
from sanae_mem.infra.textutil import _embed_text, _owner_key

PROD_HOME = Path(os.environ.get("HERMES_PROD_HOME") or (Path.home() / ".hermes"))


def _fixture_owner() -> str:
    """明确的沙箱身份，经生产同一 _owner_key 派生，不拿 raw userid 冒充 hash owner。"""
    return _owner_key("predeploy-sandbox", user_id="timelapse-fixture")


@pytest.fixture(autouse=True)
def _isolated_embedding_runtime():
    from sanae_mem.infra import textutil

    textutil.reset_embedding_state()
    try:
        yield
    finally:
        textutil.reset_embedding_state()


def test_synthetic_365d_timelapse_mechanism(tmp_path, monkeypatch):
    for name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE", "HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    db = provider._conn

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        "第%02d条旧记录：那年秋天在旧书店买过一本绝版诗集，老板还送了一张手写书签，一直夹在扉页里没动过。" % i
        for i in range(1, 45)
    ]
    (mem_dir / "MEMORY.md").write_text("\n§\n".join(entries), encoding="utf-8")
    (mem_dir / "USER.md").write_text("", encoding="utf-8")
    provider.resident.update_touch("今天去旧书店翻了那本绝版诗集和手写书签", "嗯", session_id="s1", now=now)

    r_trivial = json.loads(provider.handle_tool_call(
        "sanae_mem_add", {"content": "顺手在便利店买了一瓶常温矿泉水", "importance": 0.5}))["result"]["id"]
    r_high = json.loads(provider.handle_tool_call(
        "sanae_mem_add", {"content": "约好了等冬天第一场雪的时候一起去拍老城区的夜景", "importance": 0.75}))["result"]["id"]
    r_bili = json.loads(provider.handle_tool_call(
        "sanae_mem_add", {"content": "早苗在B站阅读了该视频的字幕：《测试视频》", "importance": 0.5}))["result"]["id"]
    db.execute("UPDATE events SET ttl_expire=NULL, curve_ttl='' WHERE id=?", (r_bili,))

    old_iso = (now - timedelta(days=10)).isoformat()
    db.execute(
        "INSERT INTO events (ts, raw, summary, summary_source, keywords, importance, topic, "
        "ttl_expire, created_at, tier, recall_count, last_active_at, half_life_days, curve_ttl) "
        "VALUES ('leg-l', '[2026-10-28] 存量长期记忆曾经聊过无数次的海边旅行计划', '', '', '海边 旅行', "
        "0.8, '关系', NULL, ?, 'long', 3, ?, 405.0, '')",
        (old_iso, old_iso),
    )
    r_legacy_long = db.execute("SELECT id FROM events WHERE ts='leg-l'").fetchone()[0]

    db.execute(
        "INSERT INTO events (ts, raw, summary, summary_source, keywords, importance, topic, "
        "ttl_expire, created_at, tier, recall_count, last_active_at, half_life_days, curve_ttl) "
        "VALUES ('pin-l', '[2026-10-28] 绑定了证据锁的长期记忆原话不可坍缩', '绑定了证据锁的长期记忆', 'distilled', '证据锁', "
        "0.8, '其他', NULL, ?, 'long', 3, ?, 60.0, '')",
        (old_iso, old_iso),
    )
    r_pinned = db.execute("SELECT id FROM events WHERE ts='pin-l'").fetchone()[0]
    db.execute(
        "INSERT INTO recognition_evidence (recognition_sha256, event_id, review_sha256, linked_at) "
        "VALUES ('rec-sha', ?, 'rev-sha', ?)",
        (r_pinned, old_iso),
    )
    db.execute("PRAGMA user_version=0")
    db.commit()
    provider.shutdown()

    provider = make_provider(tmp_path)
    db = provider._conn

    hl_min, hl_max = db.execute("SELECT MIN(half_life_days), MAX(half_life_days) FROM events").fetchone()
    assert hl_min >= 9.0 and hl_max <= 120.0
    assert db.execute("SELECT COUNT(*) FROM events WHERE trim(COALESCE(summary,''))=''").fetchone()[0] == 0

    for d in (0, 5, 7):
        res = provider.lifecycle.compact(now + timedelta(days=d))
        assert res["faded"] == 0 and res["dormant"] == 0, f"day {d}: {res}"

    trivial_hl = db.execute("SELECT half_life_days FROM events WHERE id=?", (r_trivial,)).fetchone()[0]
    expire_day = int(math.ceil(trivial_hl * math.log2(0.5 / 0.15))) + 1
    res_exp = provider.lifecycle.compact(now + timedelta(days=expire_day))
    assert db.execute("SELECT expired_at FROM events WHERE id=?", (r_trivial,)).fetchone()[0] is not None
    assert provider.lifecycle.compact(now + timedelta(days=expire_day + 1))["forgotten"] == 0
    assert provider.lifecycle.compact(now + timedelta(days=expire_day + 2))["forgotten"] == 0

    res_46 = provider.lifecycle.compact(now + timedelta(days=max(46, expire_day + 31)))
    assert len(res_46["cold_resident_entries"]) > 0
    assert db.execute("SELECT COUNT(*) FROM events WHERE id=?", (r_trivial,)).fetchone()[0] == 0

    provider.lifecycle.compact(now + timedelta(days=365))
    curve_short = db.execute(
        "SELECT COUNT(*) FROM events WHERE tier IN ('short','pending') AND ttl_expire=curve_ttl"
    ).fetchone()[0]
    assert curve_short == 0
    assert db.execute("SELECT tier, summary_source FROM events WHERE id=?", (r_high,)).fetchone()[:] == ("dormant", "gist_faded")
    assert db.execute("SELECT tier, summary_source FROM events WHERE id=?", (r_legacy_long,)).fetchone()[:] == ("dormant", "gist_faded")
    assert db.execute("SELECT tier, summary_source FROM events WHERE id=?", (r_pinned,)).fetchone()[:] == ("long", "distilled")
    assert db.execute("SELECT tier, ttl_expire FROM events WHERE id=?", (r_bili,)).fetchone()[:] == ("short", None)


@pytest.mark.skipif(not (PROD_HOME / "sanae-mem/sanae.db").is_file(), reason="HERMES_PROD_HOME snapshot unavailable")
def test_production_snapshot_365d_timelapse(tmp_path, monkeypatch):
    import yaml

    backend = "chroma" if importlib.util.find_spec("chromadb") is not None else "sqlite"

    home = tmp_path / "home"
    data = home / "sanae-mem"
    data.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(home))

    with sqlite3.connect((PROD_HOME / "sanae-mem/sanae.db").as_uri() + "?mode=ro", uri=True) as src, \
            sqlite3.connect(data / "sanae.db") as dst:
        src.backup(dst)
    if (PROD_HOME / "sanae-mem/chroma").is_dir():
        shutil.copytree(PROD_HOME / "sanae-mem/chroma", data / "chroma")
    for name in ("memories/MEMORY.md", "memories/USER.md", "sanae-mem/status.json", "sanae-mem/resident_touch.json"):
        if (PROD_HOME / name).is_file():
            (home / name).parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(PROD_HOME / name, home / name)
    if (PROD_HOME / "sanae-mem/models").is_dir():
        (data / "models").symlink_to(PROD_HOME / "sanae-mem/models", target_is_directory=True)

    cfg = (yaml.safe_load((PROD_HOME / "config.yaml").read_text(encoding="utf-8")) or {}).get("memory") or {}
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": {
        k: cfg[k] for k in ("memory_char_limit", "user_char_limit", "memory_enabled", "user_profile_enabled") if k in cfg
    }}), encoding="utf-8")

    now = datetime.now().astimezone()
    provider = SanaeMemProvider({"data_dir": str(data), "semantic_backend": backend, "shadow": False})
    _enter_test_turn("sandbox-replay", {"user_id": "sandbox-reader"})
    provider.initialize("sandbox-replay", hermes_home=str(home), platform="test", user_id="sandbox-reader")
    db = provider._conn
    collection = provider.storage.current_collection()
    if backend == "chroma":
        assert collection is not None and collection.count() > 0
    else:
        assert collection is None

    before = {r["id"]: dict(r) for r in db.execute("SELECT * FROM events")}
    pins = {
        r[0] for r in db.execute("SELECT DISTINCT event_id FROM recognition_evidence")
    }
    init_empty_sum = db.execute(
        "SELECT COUNT(*) FROM events WHERE blacklist=0 AND trim(COALESCE(summary,''))=''"
    ).fetchone()[0]
    hl_min, hl_max = db.execute("SELECT MIN(half_life_days), MAX(half_life_days) FROM events").fetchone()
    short_hl_max = db.execute("SELECT MAX(half_life_days) FROM events WHERE tier IN ('short','pending')").fetchone()[0]
    assert sum(not r["last_active_at"] for r in before.values()) == 0
    assert init_empty_sum == 0
    assert hl_min >= 9.0 and hl_max <= 120.0 and short_hl_max <= 60.0

    provider.lifecycle.compact(now)
    after = {r["id"]: dict(r) for r in db.execute("SELECT * FROM events")}
    assert sum(i not in after or before[i]["raw"] != after[i]["raw"] for i in pins) == 0
    assert sum(
        i not in after or r["raw"] != after[i]["raw"]
        for i, r in before.items()
        if r["importance"] >= 0.6 and r["last_active_at"] >= (now - timedelta(days=7)).isoformat()
    ) == 0

    eligible = [
        r for r in after.values()
        if r["id"] not in pins and r["tier"] == "short" and r["importance"] >= 0.6
        and r["curve_ttl"] and r["ttl_expire"] == r["curve_ttl"] and not r["expired_at"]
        and r["summary_source"] != "gist_faded" and 8 <= len(r["summary"] or "") <= 120
    ]
    assert eligible
    sample = min(eligible, key=lambda r: r["half_life_days"])
    eid, gist = sample["id"], sample["summary"]
    db.execute("UPDATE events SET owner_key=? WHERE id=?", (provider._owner_key, eid))
    db.commit()
    anchor = datetime.fromisoformat(sample["last_active_at"])
    half = sample["half_life_days"]

    def clock_at(activation):
        return anchor + timedelta(days=half * math.log2(sample["importance"] / activation))

    provider.lifecycle.compact(clock_at(0.25))
    row_faded = db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    assert row_faded["summary_source"] == "gist_faded" and row_faded["tier"] == "short"

    provider.lifecycle.compact(clock_at(0.13))
    row_dormant = db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    assert row_dormant["tier"] == "dormant" and row_dormant["summary_source"] == "gist_faded"
    assert not any(r["id"] == eid for r in provider.retrieval.search({"query": gist, "gap_continue": True}))
    assert _embed_text(gist)[1]

    sync_path = PROD_HOME / "memory-sync/sync.py"
    if sync_path.is_file():
        spec = importlib.util.spec_from_file_location("sync_candidate", sync_path)
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        out = tmp_path / "stable.json"
        assert sync.export_stable_db(data / "sanae.db", out)
        assert any(r["id"] == eid for r in json.loads(out.read_text(encoding="utf-8"))["events"])

    class ReplayDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            t = clock_at(0.12)
            return t.astimezone(tz) if tz is not None else t.astimezone().replace(tzinfo=None)

    question = "还记得以前" + gist + "那件事吗？"
    with ExitStack() as stack:
        for mod in ("storage.event_store", "storage.retrieval", "governance.read_gate",
                    "cognitive.association", "cognitive.association_recall", "cognitive.association_history"):
            m = importlib.import_module("sanae_mem." + mod)
            if hasattr(m, "datetime"):
                stack.enter_context(patch.object(m, "datetime", ReplayDateTime))
        context = provider.prefetch(question, session_id="sandbox-replay")
    assert "久远记忆" in context
    assert db.execute("SELECT tier FROM events WHERE id=?", (eid,)).fetchone()[0] == "dormant", \
        "检索只读取久远印象，不再把它唤醒成长久层"
    provider.shutdown()

    home_tl = tmp_path / "home_tl"
    data_tl = home_tl / "sanae-mem"
    data_tl.mkdir(parents=True)
    with sqlite3.connect((PROD_HOME / "sanae-mem/sanae.db").as_uri() + "?mode=ro", uri=True) as src, \
            sqlite3.connect(data_tl / "sanae.db") as dst:
        src.backup(dst)
    if (PROD_HOME / "sanae-mem/chroma").is_dir():
        shutil.copytree(PROD_HOME / "sanae-mem/chroma", data_tl / "chroma")
    for name in ("memories/MEMORY.md", "memories/USER.md", "sanae-mem/status.json", "sanae-mem/resident_touch.json"):
        if (PROD_HOME / name).is_file():
            (home_tl / name).parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(PROD_HOME / name, home_tl / name)
    if (PROD_HOME / "sanae-mem/models").is_dir():
        (data_tl / "models").symlink_to(PROD_HOME / "sanae-mem/models", target_is_directory=True)
    (home_tl / "config.yaml").write_text(yaml.safe_dump({"memory": {
        "memory_char_limit": 1800, "user_char_limit": 2000,
        "memory_enabled": True, "user_profile_enabled": True,
    }}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_tl))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home_tl))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(home_tl))

    p_tl = SanaeMemProvider({"data_dir": str(data_tl), "semantic_backend": backend, "shadow": False})
    _enter_test_turn("cron_nightly", {}, context="cron")
    p_tl.initialize("cron_nightly", hermes_home=str(home_tl), platform="qqbot", agent_context="cron",
                    user_id=os.environ.get("HERMES_PROD_USER_ID", "sandbox-owner"))
    db_tl = p_tl._conn
    # cron provider 的 _owner_key 合法为空；受控夹具改用明确的沙箱 hash 身份，
    # 不猜生产 owner，也不认领空 owner 的历史行。
    fixture_owner = _fixture_owner()
    assert fixture_owner and fixture_owner != "timelapse-fixture"

    # 存量迁移：带 BV 的视频观看行已从 events 拔出，专库接住完整分析/字幕；365 天不再衰减。
    leftover = db_tl.execute(
        "SELECT COUNT(*) FROM events WHERE "
        "(raw LIKE '早苗在B站收到该视频的抽样画面分析%' "
        " OR raw LIKE '早苗在B站阅读了该视频的字幕%') "
        "AND (raw GLOB '*BV??????????*' OR source_context GLOB '*BV??????????*')"
    ).fetchone()[0]
    assert leftover == 0, "存量视频迁移未完成；不得跳过或忽略迁移错误"

    stock = {
        "video_watches": db_tl.execute("SELECT COUNT(*) FROM video_watches").fetchone()[0],
        "meme_records": db_tl.execute("SELECT COUNT(*) FROM meme_records").fetchone()[0],
        "commitments": db_tl.execute("SELECT COUNT(*) FROM commitments").fetchone()[0],
        "impression_revisions": db_tl.execute("SELECT COUNT(*) FROM impression_revisions").fetchone()[0],
    }
    empty_stock = [name for name, count in stock.items() if not count]
    if empty_stock:
        warnings.warn(
            "真实库以下类别无存量：%s；在同一真实副本补受控不变量夹具，不把纯合成库冒充线上数据"
            % "、".join(empty_stock), stacklevel=1)
    print("production stock after migration: %s" % stock)

    # 受控不变量夹具只写入 /tmp 真实副本，覆盖“新增永久结构”一路。
    fixture_bv = "BVfixture001"
    fixture_analysis = "受控夹具：完整画面分析正文，含逐段观察与音轨未核验说明。" * 12
    fixture_captions = json.dumps(
        [{"text": "受控夹具字幕一"}, {"text": "受控夹具字幕二"}], ensure_ascii=False)
    old_stamp = (now - timedelta(days=30)).isoformat()
    db_tl.execute(
        "INSERT INTO video_watches (bvid, title, cid, mode, analysis, captions, summary, "
        "source_context, raw_observation, read_scope, public_excerpt, owner_key, session_id, "
        "watched_at, created_at, updated_at) "
        "VALUES (?, ?, 'fixture-cid', 'video_analysis', ?, ?, ?, '', '', ?, ?, ?, 'fixture', ?, ?, ?)",
        (fixture_bv, "受控夹具：365 天视频专库保留", fixture_analysis, fixture_captions,
         "受控夹具：视频一句话大概", '{"kind":"self"}', "受控夹具公开摘要", fixture_owner,
         old_stamp, old_stamp, old_stamp),
    )
    fixture_commitment = "受控夹具：答应帮对方在冬天前把旧相机修好"
    cursor = db_tl.execute(
        "INSERT INTO commitments (owner_key, text, masked_text, source_quote, turn_id, status, "
        "version, created_at, updated_at) VALUES (?, ?, ?, ?, 'fixture-turn', 'pending', 1, ?, ?)",
        (fixture_owner, fixture_commitment, "受控夹具：还答应了另一位朋友一件事待办",
         "冬天前把旧相机修好", old_stamp, old_stamp),
    )
    fixture_commitment_id = int(cursor.lastrowid)
    db_tl.execute(
        "INSERT INTO meme_records (title, meaning, aliases, source, source_ref, source_time, "
        "created_at, updated_at, owner_key, session_id) "
        "VALUES (?, ?, '[]', 'fixture', '', ?, ?, ?, ?, 'fixture')",
        ("受控夹具：梗图标题", "受控夹具：梗图含义不随年份删除",
         old_stamp, old_stamp, old_stamp, fixture_owner),
    )

    # write_impression 走独立连接；夹具必须先提交，否则它不可见，关闭 provider 时还会回滚。
    db_tl.commit()
    with sqlite3.connect((data_tl / "sanae.db").as_uri() + "?mode=ro", uri=True) as committed:
        assert committed.execute(
            "SELECT COUNT(*) FROM video_watches WHERE bvid=?", (fixture_bv,)).fetchone()[0] == 1
        assert committed.execute(
            "SELECT COUNT(*) FROM commitments WHERE id=?", (fixture_commitment_id,)).fetchone()[0] == 1
        assert committed.execute(
            "SELECT COUNT(*) FROM meme_records WHERE title=?",
            ("受控夹具：梗图标题",)).fetchone()[0] == 1

    from sanae_mem.cognitive.impression import apply_impression, read_impression, write_impression

    impression_target = (data_tl, "fixture-profile", fixture_owner)
    impression_base = write_impression(
        impression_target,
        "【长期底色】受控夹具：慢热、记好也记仇，遇事先自己扛。\n【阶段近况】最近在整理旧照片。",
        turn_id="fixture-impression-init",
    )
    new_quote = "我下周要去外地出差半个月"
    assert apply_impression(
        impression_target, impression_base,
        {
            "text": "【长期底色】受控夹具：慢热、记好也记仇，遇事先自己扛。\n"
                    "【阶段近况】下周要去外地出差半个月。",
            "baseline": "受控夹具：慢热、记好也记仇，遇事先自己扛。",
            "recent": "下周要去外地出差半个月。",
            "speaker": "user",
            "quote": new_quote,
        },
        turn_id="fixture-impression-flip", now=time.time(),
        user="我下周要去外地出差半个月，家里的事先放一放。", assistant="路上注意安全。",
    )
    flipped = read_impression(impression_target)
    assert flipped is not None and flipped.baseline == impression_base.baseline
    assert flipped.recent == "下周要去外地出差半个月。"
    assert "整理旧照片" not in flipped.recent

    video_before = {int(r["id"]): dict(r) for r in db_tl.execute("SELECT * FROM video_watches")}
    meme_before = {
        int(r["id"]): (r["title"], r["meaning"], r["aliases"])
        for r in db_tl.execute("SELECT * FROM meme_records")
    }
    commitment_before = {
        int(r["id"]): (r["status"], r["updated_at"])
        for r in db_tl.execute("SELECT * FROM commitments")
    }
    impression_sql = (
        "SELECT profile_key, owner_key, revision, text, speaker, evidence, turn_id, updated_at "
        "FROM impression_revisions ORDER BY profile_key, owner_key, revision"
    )
    impression_before = [tuple(r) for r in db_tl.execute(impression_sql)]
    user_md = home_tl / "memories/USER.md"
    user_before = user_md.read_text(encoding="utf-8") if user_md.is_file() else None

    timeline = {}
    for day in (0, 1, 2, 7, 14, 30, 46, 90, 180, 270, 365):
        r_tl = p_tl.lifecycle.compact(now + timedelta(days=day))
        tiers = dict(Counter(r[0] for r in db_tl.execute("SELECT tier FROM events WHERE blacklist=0").fetchall()))
        curve_short = db_tl.execute(
            "SELECT COUNT(*) FROM events WHERE blacklist=0 AND tier IN ('short','pending') AND ttl_expire=curve_ttl"
        ).fetchone()[0]
        timeline[day] = {
            "tiers": tiers, "curve_short": curve_short,
            "faded": r_tl["faded"], "dormant": r_tl["dormant"],
            "forgotten": r_tl["forgotten"], "cold": len(r_tl["cold_resident_entries"]),
        }
    assert timeline[1]["forgotten"] == 0 and timeline[2]["forgotten"] == 0
    assert timeline[46]["cold"] > 0
    assert timeline[365]["curve_short"] == 0
    assert timeline[365]["tiers"].get("dormant", 0) >= 50

    video_after = {int(r["id"]): dict(r) for r in db_tl.execute("SELECT * FROM video_watches")}
    assert set(video_before) <= set(video_after)
    for vid, row in video_before.items():
        assert video_after[vid]["analysis"] == row["analysis"]
        assert video_after[vid]["captions"] == row["captions"]
        assert video_after[vid]["summary"] == row["summary"]
    # 搜索与详情按明确 owner 读取；缺 owner 返回空，不再默认全库。
    detail = p_tl.video_memory.get_video_watch_detail(fixture_bv, owner_key=fixture_owner)
    assert detail is not None
    assert detail["analysis"] == fixture_analysis and detail["captions"] == fixture_captions
    assert any(row["bvid"] == fixture_bv
               for row in p_tl.video_memory.search_video_watches("受控夹具", owner_key=fixture_owner))
    assert any(row["bvid"] == fixture_bv
               for row in p_tl.video_memory.list_video_watches(owner_key=fixture_owner))

    meme_after = {
        int(r["id"]): (r["title"], r["meaning"], r["aliases"])
        for r in db_tl.execute("SELECT * FROM meme_records")
    }
    assert set(meme_before) <= set(meme_after)
    for mid, value in meme_before.items():
        assert meme_after[mid] == value

    commitment_after = {
        int(r["id"]): (r["status"], r["updated_at"])
        for r in db_tl.execute("SELECT * FROM commitments")
    }
    assert set(commitment_before) <= set(commitment_after)
    for cid, value in commitment_before.items():
        assert commitment_after[cid] == value, "承诺状态不应被时间代谢"
    assert commitment_after[fixture_commitment_id] == ("pending", old_stamp)
    assert fixture_commitment in p_tl.commitments.render_pending_commitments(fixture_owner)

    assert [tuple(r) for r in db_tl.execute(impression_sql)] == impression_before
    final_impression = read_impression(impression_target)
    assert final_impression is not None
    assert final_impression.baseline == impression_base.baseline
    assert final_impression.recent == "下周要去外地出差半个月。"
    if user_before is not None:
        assert user_md.read_text(encoding="utf-8") == user_before, "USER.md 长期底色不应冷退居"
    p_tl.shutdown()


def test_cron_shape_keeps_fixture_ownership_explicit(tmp_path, monkeypatch):
    """cron provider 的 owner 合法为空；受控夹具用明确沙箱 hash 身份，本人视图一致。"""
    for name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE", "HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    provider = make_provider(tmp_path, context="cron", user_id="scheduler")
    db_path = provider._data_dir / "sanae.db"
    try:
        fixture_owner = _fixture_owner()
        assert fixture_owner and fixture_owner != "timelapse-fixture"

        stamp = datetime.now().astimezone().isoformat()
        analysis = "受控夹具：cron 形态下明确 owner 的完整视频分析。"
        captions = json.dumps([{"text": "受控夹具字幕"}], ensure_ascii=False)
        db = provider._conn
        db.execute(
            "INSERT INTO video_watches (bvid, title, cid, mode, analysis, captions, summary, "
            "source_context, raw_observation, read_scope, public_excerpt, owner_key, session_id, "
            "watched_at, created_at, updated_at) "
            "VALUES (?, ?, 'fixture-cid', 'video_analysis', ?, ?, ?, '', '', ?, ?, ?, 'fixture', ?, ?, ?)",
            ("BVfixture001", "受控夹具：cron 形态视频", analysis, captions,
             "受控夹具：cron 形态一句话", '{"kind":"self"}', "受控夹具公开摘要", fixture_owner,
             stamp, stamp, stamp),
        )
        db.execute(
            "INSERT INTO commitments (owner_key, text, masked_text, source_quote, turn_id, status, "
            "version, created_at, updated_at) VALUES (?, ?, ?, ?, 'fixture-turn', 'pending', 1, ?, ?)",
            (fixture_owner, "受控夹具：cron 形态承诺",
             "受控夹具：还答应了另一位朋友一件事待办", "cron 形态承诺", stamp, stamp),
        )
        db.execute(
            "INSERT INTO meme_records (title, meaning, aliases, source, source_ref, source_time, "
            "created_at, updated_at, owner_key, session_id) "
            "VALUES (?, ?, '[]', 'fixture', '', ?, ?, ?, ?, 'fixture')",
            ("受控夹具：cron 形态梗图", "受控夹具：梗图随提交保留",
             stamp, stamp, stamp, fixture_owner),
        )
        db.commit()
        # 独立连接先确认夹具已提交，随后 write_impression 才用另一个连接写入。
        with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as committed:
            assert committed.execute(
                "SELECT COUNT(*) FROM video_watches WHERE bvid=?", ("BVfixture001",)).fetchone()[0] == 1
            assert committed.execute(
                "SELECT COUNT(*) FROM commitments WHERE owner_key=?", (fixture_owner,)).fetchone()[0] == 1
            assert committed.execute(
                "SELECT COUNT(*) FROM meme_records WHERE title=?",
                ("受控夹具：cron 形态梗图",)).fetchone()[0] == 1

        assert any(row["bvid"] == "BVfixture001"
                   for row in provider.video_memory.list_video_watches(owner_key=fixture_owner))
        assert any(row["bvid"] == "BVfixture001"
                   for row in provider.video_memory.search_video_watches("受控夹具", owner_key=fixture_owner))
        detail = provider.video_memory.get_video_watch_detail("BVfixture001", owner_key=fixture_owner)
        assert detail is not None and detail["analysis"] == analysis
        assert "受控夹具：cron 形态承诺" in provider.commitments.render_pending_commitments(fixture_owner)

        from sanae_mem.cognitive.impression import read_impression, write_impression

        target = (provider._data_dir, "fixture-profile", fixture_owner)
        write_impression(target, "【长期底色】受控夹具：cron 形态也不认领空 owner。",
                         turn_id="fixture-cron")
        assert read_impression(target).baseline == "受控夹具：cron 形态也不认领空 owner。"
    finally:
        provider.shutdown()

    # provider 关闭后仍能读到夹具，确认提交没有随连接关闭回滚。
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as reopened:
        assert reopened.execute(
            "SELECT COUNT(*) FROM video_watches WHERE bvid=?", ("BVfixture001",)).fetchone()[0] == 1
        assert reopened.execute(
            "SELECT COUNT(*) FROM commitments WHERE owner_key=?", (fixture_owner,)).fetchone()[0] == 1
        assert reopened.execute(
            "SELECT COUNT(*) FROM meme_records WHERE title=?",
            ("受控夹具：cron 形态梗图",)).fetchone()[0] == 1
    assert read_impression((db_path.parent, "fixture-profile", fixture_owner)).baseline == \
        "受控夹具：cron 形态也不认领空 owner。"


def _seed_curve_rows(provider, now):
    db = provider._conn
    old = (now - timedelta(days=10)).isoformat()
    rows = (
        ("cont-emotion", "[2026-10-28] 持续运行对照情绪行", "持续运行对照情绪行", "original", 0.5, "情绪", 60.0),
        ("cont-trivial", "[2026-10-28] 持续运行对照琐事", "持续运行对照琐事", "original", 0.5, "购物", 13.5),
        ("cont-pinned", "[2026-10-28] 持续运行对照证据行", "持续运行对照证据行", "distilled", 0.8, "其他", 120.0),
        ("cont-long", "[2026-10-28] 持续运行对照长期行", "持续运行对照长期行", "distilled", 0.8, "关系", 120.0),
    )
    ids = {}
    for key, raw, summary, source, importance, topic, half in rows:
        db.execute(
            "INSERT INTO events (ts, raw, summary, summary_source, keywords, importance, topic, "
            "ttl_expire, created_at, tier, recall_count, last_active_at, half_life_days, curve_ttl) "
            "VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, 3, ?, ?, ?)",
            (key, raw, summary, source, importance, topic,
             (now + timedelta(days=half * 2)).isoformat(), old,
             "long" if key in ("cont-long", "cont-pinned") else "short",
             old, half, (now + timedelta(days=half * 2)).isoformat()))
        ids[key] = int(db.execute("SELECT id FROM events WHERE ts=?", (key,)).fetchone()[0])
    db.execute(
        "INSERT INTO recognition_evidence (recognition_sha256, event_id, review_sha256, linked_at) "
        "VALUES ('sha-cont', ?, 'rev-cont', ?)", (ids["cont-pinned"], old))
    db.commit()
    return ids


def _curve_snapshot(provider):
    return {
        row[1]: (row[0], row[2] is not None, row[3], row[4])
        for row in provider._conn.execute(
            "SELECT tier, ts, expired_at, blacklist, summary_source FROM events WHERE ts LIKE 'cont-%'"
        ).fetchall()
    }


def test_365d_axis_promotion_and_person_impression_survive_capacity_checks(tmp_path, monkeypatch):
    for name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE", "HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    provider = make_provider(tmp_path, user_id="axis-owner")
    owner = provider._owner_key
    now = datetime.now().astimezone().replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(days=3)
    text = "那把旧吉他的琴颈有一道细裂纹，我想等天暖了再拿去修"
    with provider._lock:
        cursor = provider._conn.execute(
            """INSERT INTO events
               (ts, raw, summary, summary_source, keywords, importance, confidence, topic,
                ttl_expire, created_at, updated_at, session_id, tier, owner_key,
                experience_subject, fact_subject, shared_view, half_life_days, curve_ttl)
               VALUES (?, ?, ?, 'original', '', 0.7, 0.6, '生活', NULL, ?, ?, 's1',
                       'short', ?, 'sanae', 'sanae', '公共：旧乐器修复', 14.0, '')""",
            (now.isoformat(), f"[{now.isoformat()}] {text}", text[:150],
             now.isoformat(), now.isoformat(), owner))
        provider._conn.commit()
    event_id = int(cursor.lastrowid)
    for turn, offset in (("d1", 0), ("d2", 1), ("d3", 2)):
        result = provider.memory_axis.record_delivered_rementions(
            owner_key=owner, user_text=text, assistant_text="我记得那道裂纹。",
            turn_id=f"axis-{turn}", delivered_at=now + timedelta(days=offset, minutes=5))
        assert result["ok"], result
    row = provider._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    assert row["tier"] == "long" and row["owner_key"] == owner
    assert row["raw"].endswith(text) and row["shared_view"] == "公共：旧乐器修复"

    memories = tmp_path / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    (memories / "MEMORY.md").write_text(
        "【2026-10-01 · 今日小记】那天我把旧吉他擦了一遍。", encoding="utf-8")
    (memories / "USER.md").write_text(
        "他喜欢旧乐器，遇到喜欢的东西会慢慢修好，不会随手丢掉。", encoding="utf-8")
    provider.resident.update_touch("这句和常驻条目完全无关", "", session_id="s1",
                                   now=now + timedelta(days=20))
    result = provider.lifecycle.compact(now + timedelta(days=365))
    assert all(item["target"] != "user" for item in result["cold_resident_entries"])
    assert "旧乐器" in (memories / "USER.md").read_text(encoding="utf-8")
    assert "2026-10-01 · 今日小记" in (memories / "MEMORY.md").read_text(encoding="utf-8")
    kept = provider._conn.execute(
        "SELECT tier, owner_key, shared_view FROM events WHERE id=?", (event_id,)).fetchone()
    assert kept is not None and kept["owner_key"] == owner
    assert kept["shared_view"] == "公共：旧乐器修复"
    provider.shutdown()


def test_continuous_run_and_reopen_produce_the_same_lifetime(tmp_path, monkeypatch):
    """同一批记忆：一路连续运行、一路每天重开，层级/到期复核/遗忘结果必须一致。"""
    for name in ("HERMES_HOME", "HERMES_HOME_OVERRIDE", "HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    now = datetime.now().astimezone()
    continuous = make_provider(tmp_path / "continuous")
    reopened = make_provider(tmp_path / "reopened")
    _seed_curve_rows(continuous, now)
    _seed_curve_rows(reopened, now)

    for day in (0, 7, 30, 90, 180, 365):
        continuous.lifecycle.compact(now + timedelta(days=day))
        if day:
            reopened.shutdown()
            reopened = make_provider(tmp_path / "reopened")
        reopened.lifecycle.compact(now + timedelta(days=day))
        assert _curve_snapshot(continuous) == _curve_snapshot(reopened), f"day {day}"
    continuous.shutdown()
    reopened.shutdown()
