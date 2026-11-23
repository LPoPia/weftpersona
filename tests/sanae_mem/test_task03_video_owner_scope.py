"""Task 03 regression tests: strict owner scoping for video detail and video search.

All assertions exercise public provider methods, public tool entrypoints
(``sanae_video_detail`` via ``tools.registry`` and ``sanae_mem_search`` via
``provider.handle_tool_call``), and ``public_watch_projection_rows`` without
touching private ``._xxx`` attributes or helper methods.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (
                path
                for path in (
                    REPO_ROOT.parent / "hermes-minimal-host",
                    REPO_ROOT.parent / "hermes-minimal-host-20261112",
                )
                if path.is_dir()
            ),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
DEPENDENCIES = HERMES_ROOT.with_name(HERMES_ROOT.name + "-deps")
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source is unavailable", allow_module_level=True)
for _import_root in (DEPENDENCIES, HERMES_ROOT, PLUGIN_ROOT):
    if _import_root.is_dir() and str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

_session_ctx: Any = importlib.import_module("gateway.session_context")
_hermes_plugins: Any = importlib.import_module("hermes_cli.plugins")
_memory_manager_mod: Any = importlib.import_module("agent.memory_manager")
_tools_registry_mod: Any = importlib.import_module("tools.registry")

from sanae_mem.plugin_entry import SanaeMemProvider  # noqa: E402
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity  # noqa: E402
from sanae_mem.domains.video_memory import public_watch_projection_rows  # noqa: E402


VIDEO_DETAIL_TOOL = "sanae_video_detail"
TEST_RECORD_TOOL = "task03_test_bili_record"


@contextmanager
def _chat_route(
    *,
    platform: str = "telegram",
    chat_type: str = "dm",
    chat_id: str | None = None,
    user_id: str,
    session_id: str,
    scope_id: str = "",
    session_key: str | None = None,
    message_id: str = "msg-1",
) -> Iterator[None]:
    _session_ctx.reset_session_vars()
    resolved_chat_id = chat_id or f"chat-{user_id}"
    resolved_session_key = (
        session_key or f"agent:main:{platform}:{chat_type}:{resolved_chat_id}"
    )
    facts: dict[str, Any] = {
        "platform": platform,
        "chat_type": chat_type,
        "chat_id": resolved_chat_id,
        "user_id": user_id,
        "session_id": session_id,
        "session_key": resolved_session_key,
        "message_id": message_id,
    }
    if scope_id:
        facts["scope_id"] = scope_id
    tokens = _session_ctx.set_session_vars(**facts)
    try:
        yield
    finally:
        _session_ctx.clear_session_vars(tokens)


def _install_minimal_bilibili_plugin(home: Path) -> None:
    plugin_dir = home / "plugins" / "bilibili_public"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        'name: bilibili_public\nversion: 0.1.0\ndescription: "Task 03 test adapter"\nhooks: []\n',
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from sanae_mem.integrations import register_bilibili_memory\n\n"
        "def register(ctx):\n"
        "    binder = register_bilibili_memory(ctx)\n"
        "    def record_handler(args, *, session_id=''):\n"
        "        target_session = str(args.get('session_id') or session_id)\n"
        "        client = binder.for_session(target_session)\n"
        "        if client is None:\n"
        "            return json.dumps({'ok': False, 'error': 'no active session'})\n"
        "        approve = bool(args.get('approve_public_excerpt', False))\n"
        "        stored = client.record_video_watch(\n"
        "            dict(args['observation']),\n"
        "            delivered=True,\n"
        "            public=True,\n"
        "            _watch_receipt=str(args['receipt']),\n"
        "            review_public_excerpt=(lambda _text: approve),\n"
        "        )\n"
        "        return json.dumps({'ok': bool(stored)})\n"
        "    ctx.register_tool(\n"
        f"        name={TEST_RECORD_TOOL!r},\n"
        "        toolset='bilibili_public',\n"
        "        schema={\n"
        f"            'name': {TEST_RECORD_TOOL!r},\n"
        "            'description': 'Record a delivered Bilibili observation in tests.',\n"
        "            'parameters': {'type': 'object', 'properties': {}},\n"
        "        },\n"
        "        handler=record_handler,\n"
        "        check_fn=lambda: True,\n"
        "    )\n",
        encoding="utf-8",
    )


@contextmanager
def _isolated_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_bilibili_plugin: bool = True,
) -> Iterator[tuple[Path, Path, str]]:
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    empty_bundled = tmp_path / "empty-bundled-plugins"
    empty_bundled.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - bilibili_public\n  isolation: in_process\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))

    data_dir = tmp_path / "sanae-mem"
    if load_bilibili_plugin:
        _install_minimal_bilibili_plugin(home)
        manager = _hermes_plugins.PluginManager(scope_key=str(home))
        manager.discover_and_load(force=True)
        try:
            yield home, data_dir, str(manager.scope_key)
        finally:
            manager.unload()
    else:
        yield home, data_dir, str(home)


def _open_provider(
    stack: ExitStack,
    *,
    home: Path,
    config: dict[str, Any],
    platform: str,
    user_id: str,
    session_id: str,
    scope_id: str = "",
) -> tuple[SanaeMemProvider, str]:
    provider = SanaeMemProvider(dict(config))
    memory_manager = _memory_manager_mod.MemoryManager()
    stack.callback(memory_manager.shutdown_all)
    with _chat_route(
        platform=platform,
        user_id=user_id,
        session_id=session_id,
        scope_id=scope_id,
    ):
        memory_manager.add_provider(provider)
        memory_manager.initialize_all(
            session_id=session_id,
            hermes_home=str(home),
            platform=platform,
            agent_context="primary",
        )
        identity = resolve_chat_identity(current_chat_scope(), config)
        owner_key = identity.private_owner_key or current_chat_scope().owner
    return provider, owner_key


def _dispatch_video_detail(scope_key: str, bvid: str, *, session_id: str = "") -> dict[str, Any]:
    raw = _tools_registry_mod.registry.dispatch(
        VIDEO_DETAIL_TOOL,
        {"bvid": bvid},
        scope=scope_key,
        session_id=session_id,
    )
    return json.loads(raw)


def _dispatch_record_watch(
    scope_key: str,
    *,
    session_id: str,
    observation: dict[str, Any],
    receipt: str,
    approve_public_excerpt: bool = False,
) -> dict[str, Any]:
    raw = _tools_registry_mod.registry.dispatch(
        TEST_RECORD_TOOL,
        {
            "session_id": session_id,
            "observation": observation,
            "receipt": receipt,
            "approve_public_excerpt": approve_public_excerpt,
        },
        scope=scope_key,
        session_id=session_id,
    )
    return json.loads(raw)


def _search_memory_tool(provider: SanaeMemProvider, query: str) -> list[dict[str, Any]]:
    payload = json.loads(
        provider.handle_tool_call("sanae_mem_search", {"query": query, "top_k": 10})
    )
    assert payload["ok"] is True, payload
    return list(payload["result"])


def _insert_video_watch(
    db_path: Path,
    *,
    bvid: str,
    platform: str = "bilibili",
    source_object_id: str | None = None,
    title: str,
    cid: str = "cid-1",
    mode: str = "video_analysis+captions_only",
    analysis: str,
    captions: str = '[{"text": "DEFAULT-CAPTION"}]',
    summary: str,
    source_context: str,
    read_scope: str,
    public_excerpt: str = "",
    owner_key: str,
    watched_at: str | None = None,
) -> None:
    stamp = watched_at or datetime.now().astimezone().isoformat()
    obj_id = bvid if source_object_id is None else source_object_id
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            """
            INSERT INTO video_watches (
                bvid, platform, source_object_id, title, cid, mode,
                analysis, captions, summary, source_context, raw_observation,
                read_scope, public_excerpt, owner_key, session_id,
                watched_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?, '', ?, ?, ?)
            """,
            (
                bvid,
                platform,
                obj_id,
                title,
                cid,
                mode,
                analysis,
                captions,
                summary,
                source_context,
                read_scope,
                public_excerpt,
                owner_key,
                stamp,
                stamp,
                stamp,
            ),
        )
        conn.commit()


def test_other_owner_self_scoped_public_bilibili_watch_is_blocked_across_detail_and_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private Bilibili watches written with visibility='public' and read_scope='self' must not leak across owners."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider_a, owner_a = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        provider_b, owner_b = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-B", session_id="dm-B"
        )
        assert owner_a and owner_b and owner_a != owner_b

        bvid_a = "BV1OwnerA001"
        now_ts = int(time.time())
        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            rec_analysis = _dispatch_record_watch(
                scope_key,
                session_id="dm-A",
                observation={
                    "status": "ok",
                    "mode": "video_analysis",
                    "bvid": bvid_a,
                    "cid": "cid-a1",
                    "title": "星尘望远镜校准记录",
                    "analysis": "OWNER-A-SECRET-VISUAL-ANALYSIS",
                },
                receipt="a123456789abcdef0123456789abcdef",
                approve_public_excerpt=True,
            )
            assert rec_analysis["ok"] is True
            rec_captions = _dispatch_record_watch(
                scope_key,
                session_id="dm-A",
                observation={
                    "status": "ok",
                    "mode": "captions_only",
                    "bvid": bvid_a,
                    "cid": "cid-a1",
                    "title": "星尘望远镜校准记录",
                    "lines": [{"text": "OWNER-A-SECRET-CAPTION-LINE"}],
                },
                receipt="a223456789abcdef0123456789abcdef",
                approve_public_excerpt=True,
            )
            assert rec_captions["ok"] is True

        db_path = data_dir / "sanae.db"
        # Verify the real write path actually stored visibility='public' + read_scope='{"kind":"self"}' + owner_a.
        with closing(sqlite3.connect(str(db_path))) as conn:
            stored = conn.execute(
                "SELECT owner_key, read_scope, source_context, public_excerpt FROM video_watches WHERE bvid=?",
                (bvid_a,),
            ).fetchone()
            assert stored is not None
            assert stored[0] == owner_a
            assert json.loads(stored[1]) == {"kind": "self"}
            assert json.loads(stored[2])["visibility"] == "public"
            assert stored[3] != ""

        # Also seed explicit 'someone-else' rows: one public+self (like BV1OtherMx01) and one private+self.
        bvid_other_public = "BV1OtherMx01"
        _insert_video_watch(
            db_path,
            bvid=bvid_other_public,
            title="他人公开标记时钟维修",
            analysis="SOMEONE-ELSE-PUBLIC-ANALYSIS",
            captions='[{"text": "SOMEONE-ELSE-PUBLIC-CAPTION"}]',
            summary="他人公开标记时钟维修总结：SOMEONE-ELSE-PUBLIC-SUMMARY",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": f"https://www.bilibili.com/video/{bvid_other_public}/",
                }
            ),
            read_scope='{"kind":"self"}',
            public_excerpt="脱敏公开摘要：时钟维修",
            owner_key="someone-else",
        )
        bvid_other_private = "BV1OtherPv01"
        _insert_video_watch(
            db_path,
            bvid=bvid_other_private,
            title="他人私密标记乐谱整理",
            analysis="SOMEONE-ELSE-PRIVATE-ANALYSIS",
            captions='[{"text": "SOMEONE-ELSE-PRIVATE-CAPTION"}]',
            summary="他人私密标记乐谱整理总结：SOMEONE-ELSE-PRIVATE-SUMMARY",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "private",
                    "ref": f"https://www.bilibili.com/video/{bvid_other_private}/",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key="someone-else",
        )

        # Account A reads its own video via all public paths, but never someone-else's rows.
        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            own_detail = provider_a.video_memory.get_video_watch_detail(bvid_a, owner_key=owner_a)
            assert own_detail is not None
            assert "OWNER-A-SECRET-VISUAL-ANALYSIS" in own_detail["analysis"]
            assert "OWNER-A-SECRET-CAPTION-LINE" in own_detail["captions"]

            own_tool = _dispatch_video_detail(scope_key, bvid_a, session_id="dm-A")
            assert own_tool["ok"] is True and own_tool["result"]["found"] is True
            assert "OWNER-A-SECRET-VISUAL-ANALYSIS" in own_tool["result"]["analysis"]

            own_search_bv = _search_memory_tool(provider_a, bvid_a)
            assert any(item.get("bvid") == bvid_a for item in own_search_bv)
            own_search_kw = _search_memory_tool(provider_a, "星尘望远镜 校准记录")
            assert any(item.get("bvid") == bvid_a for item in own_search_kw)
            assert any(
                row["bvid"] == bvid_a
                for row in provider_a.video_memory.search_video_watches("星尘望远镜", owner_key=owner_a)
            )

            for foreign_bvid, foreign_kw in (
                (bvid_other_public, "时钟维修"),
                (bvid_other_private, "乐谱整理"),
            ):
                assert provider_a.video_memory.get_video_watch_detail(foreign_bvid, owner_key=owner_a) is None
                foreign_tool = _dispatch_video_detail(scope_key, foreign_bvid, session_id="dm-A")
                assert foreign_tool["ok"] is True and foreign_tool["result"]["found"] is False
                assert provider_a.video_memory.search_video_watches(foreign_kw, owner_key=owner_a) == []
                assert all(
                    item.get("bvid") != foreign_bvid
                    for item in _search_memory_tool(provider_a, foreign_bvid)
                )
                assert all(
                    item.get("bvid") != foreign_bvid
                    for item in _search_memory_tool(provider_a, f"{foreign_kw} 标记")
                )

        # Account B cannot read Account A's or someone-else's full detail or video search hits.
        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-B", session_id="dm-B"
        ):
            for forbidden_bvid, forbidden_kw in (
                (bvid_a, "星尘望远镜"),
                (bvid_other_public, "时钟维修"),
                (bvid_other_private, "乐谱整理"),
            ):
                assert provider_b.video_memory.get_video_watch_detail(forbidden_bvid, owner_key=owner_b) is None
                tool_resp = _dispatch_video_detail(scope_key, forbidden_bvid, session_id="dm-B")
                assert tool_resp["ok"] is True and tool_resp["result"]["found"] is False
                assert provider_b.video_memory.search_video_watches(forbidden_kw, owner_key=owner_b) == []
                bv_hits = _search_memory_tool(provider_b, forbidden_bvid)
                kw_hits = _search_memory_tool(provider_b, f"{forbidden_kw} 记录")
                assert all(item.get("bvid") != forbidden_bvid for item in bv_hits)
                assert all(item.get("bvid") != forbidden_bvid for item in kw_hits)
                serialized = json.dumps(bv_hits + kw_hits, ensure_ascii=False)
                assert "OWNER-A-SECRET" not in serialized
                assert "SOMEONE-ELSE-" not in serialized

        # Cross-owner sanitized public_excerpt projection via public_watch_projection_rows stays intact.
        with closing(sqlite3.connect(str(db_path))) as conn:
            projected = public_watch_projection_rows(
                conn,
                owner_key=owner_b,
                since=now_ts - 3600,
                through=now_ts + 3600,
            )
        projected_by_bv = {row["bvid"]: row for row in projected}
        assert bvid_a in projected_by_bv
        assert bvid_other_public in projected_by_bv
        assert bvid_other_private not in projected_by_bv
        for row in projected:
            assert set(row.keys()) == {"watched_at", "title", "bvid", "mode", "public_excerpt"}


def test_same_bvid_watched_by_two_owners_does_not_merge_foreign_analysis_or_captions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When two owners watch the same BVID, detail merging must stay strictly within the caller's rows."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider_a, owner_a = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        provider_b, owner_b = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-B", session_id="dm-B"
        )

        shared_bvid = "BV1SharedBv1"
        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            assert _dispatch_record_watch(
                scope_key,
                session_id="dm-A",
                observation={
                    "status": "ok",
                    "mode": "video_analysis",
                    "bvid": shared_bvid,
                    "cid": "cid-shared",
                    "title": "同一公开视频",
                    "analysis": "ONLY-ALPHA-VISUAL-NOTES",
                },
                receipt="b123456789abcdef0123456789abcdef",
            )["ok"] is True

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-B", session_id="dm-B"
        ):
            assert _dispatch_record_watch(
                scope_key,
                session_id="dm-B",
                observation={
                    "status": "ok",
                    "mode": "captions_only",
                    "bvid": shared_bvid,
                    "cid": "cid-shared",
                    "title": "同一公开视频",
                    "lines": [{"text": "ONLY-BETA-CAPTIONS-TEXT"}],
                },
                receipt="b223456789abcdef0123456789abcdef",
            )["ok"] is True

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            detail_a = provider_a.video_memory.get_video_watch_detail(shared_bvid, owner_key=owner_a)
            assert detail_a is not None
            assert "ONLY-ALPHA-VISUAL-NOTES" in detail_a["analysis"]
            assert "ONLY-BETA-CAPTIONS-TEXT" not in detail_a["captions"]
            tool_a = _dispatch_video_detail(scope_key, shared_bvid, session_id="dm-A")["result"]
            assert tool_a["found"] is True
            assert "ONLY-ALPHA-VISUAL-NOTES" in tool_a["analysis"]
            assert tool_a["captions"] == []

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-B", session_id="dm-B"
        ):
            detail_b = provider_b.video_memory.get_video_watch_detail(shared_bvid, owner_key=owner_b)
            assert detail_b is not None
            assert detail_b["analysis"] == ""
            assert "ONLY-ALPHA-VISUAL-NOTES" not in detail_b["analysis"]
            assert "ONLY-BETA-CAPTIONS-TEXT" in detail_b["captions"]
            tool_b = _dispatch_video_detail(scope_key, shared_bvid, session_id="dm-B")["result"]
            assert tool_b["found"] is True
            assert tool_b["analysis"] == ""
            assert [c["text"] for c in tool_b["captions"]] == ["ONLY-BETA-CAPTIONS-TEXT"]


def test_unowned_legacy_public_self_row_allowed_only_with_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only unowned (owner_key='') Bilibili rows with complete observed public evidence remain readable."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider, owner = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        db_path = data_dir / "sanae.db"

        # 1. Valid unowned legacy self-scoped public Bilibili row.
        legacy_ok = "BV1LegacyOk1"
        _insert_video_watch(
            db_path,
            bvid=legacy_ok,
            title="遗留公开木雕工艺",
            analysis="LEGACY-PUBLIC-SELF-ANALYSIS",
            captions='[{"text": "LEGACY-PUBLIC-SELF-CAPTION"}]',
            summary="遗留公开木雕工艺：细刻花纹的完整演示",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": f"https://www.bilibili.com/video/{legacy_ok}/",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key="",
        )

        # 2. Nine unowned rows missing or forging one required piece of evidence (including malformed JSON).
        invalid_cases: list[tuple[str, str, str, str, str]] = [
            (
                "BV1BadVis001",
                "bilibili",
                "伪造非公开风铃样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "private",
                        "ref": "https://www.bilibili.com/video/BV1BadVis001/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1NoVis0001",
                "bilibili",
                "缺可见度陶艺样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "ref": "https://www.bilibili.com/video/BV1NoVis0001/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1BadKind01",
                "bilibili",
                "非自看来源漆器样本",
                json.dumps(
                    {
                        "kind": "video",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": "https://www.bilibili.com/video/BV1BadKind01/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1BadCert01",
                "bilibili",
                "非实测确定性铜器样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "inferred",
                        "visibility": "public",
                        "ref": "https://www.bilibili.com/video/BV1BadCert01/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1BadRef001",
                "bilibili",
                "伪造外链琉璃样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": "https://evil.example/video/BV1BadRef001/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1NoRef0001",
                "bilibili",
                "缺链接折扇样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1BadScop01",
                "bilibili",
                "非自读域青瓷样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": "https://www.bilibili.com/video/BV1BadScop01/",
                    }
                ),
                '{"kind":"owner"}',
            ),
            (
                "BV1BadPlat01",
                "douyin",
                "非B站平台竹编样本",
                json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": "https://www.bilibili.com/video/BV1BadPlat01/",
                    }
                ),
                '{"kind":"self"}',
            ),
            (
                "BV1BadJson01",
                "bilibili",
                "损坏JSON砚台样本",
                "{not-valid-source-json",
                "{not-valid-scope-json",
            ),
        ]
        for bvid, platform, title, src_ctx, r_scope in invalid_cases:
            _insert_video_watch(
                db_path,
                bvid=bvid,
                platform=platform,
                title=title,
                analysis=f"MUST-NOT-LEAK-{bvid}",
                captions=f'[{{"text":"CAPTION-{bvid}"}}]',
                summary=f"{title}完整正文：{bvid}",
                source_context=src_ctx,
                read_scope=r_scope,
                owner_key="",
            )

        # Empty owner must always refuse detail reads, even for valid legacy public rows.
        assert provider.video_memory.get_video_watch_detail(legacy_ok, owner_key="") is None
        assert provider.video_memory.get_video_watch_detail(legacy_ok, owner_key=None) is None

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            # Valid unowned legacy self row stays readable via detail, tool, and search.
            detail = provider.video_memory.get_video_watch_detail(legacy_ok, owner_key=owner)
            assert detail is not None
            assert detail["analysis"] == "LEGACY-PUBLIC-SELF-ANALYSIS"
            tool_ok = _dispatch_video_detail(scope_key, legacy_ok, session_id="dm-A")
            assert tool_ok["ok"] is True and tool_ok["result"]["found"] is True
            assert tool_ok["result"]["analysis"] == "LEGACY-PUBLIC-SELF-ANALYSIS"
            assert any(
                item.get("bvid") == legacy_ok for item in _search_memory_tool(provider, legacy_ok)
            )
            assert any(
                item.get("bvid") == legacy_ok
                for item in _search_memory_tool(provider, "遗留公开 木雕工艺")
            )

            # Every incomplete/forged/malformed unowned row is refused without crashing SQLite.
            for bvid, _platform, title, _src_ctx, _r_scope in invalid_cases:
                assert provider.video_memory.get_video_watch_detail(bvid, owner_key=owner) is None, bvid
                tool_bad = _dispatch_video_detail(scope_key, bvid, session_id="dm-A")
                assert tool_bad["ok"] is True and tool_bad["result"]["found"] is False, bvid
                bv_hits = _search_memory_tool(provider, bvid)
                kw_hits = _search_memory_tool(provider, f"{title} 完整正文")
                assert all(item.get("bvid") != bvid for item in bv_hits), bvid
                assert all(item.get("bvid") != bvid for item in kw_hits), bvid


def test_unowned_row_with_bare_bv_prefix_ref_is_rejected_in_detail_and_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unowned rows whose source_context.ref is only the bare 'BV' prefix or short bvid must be rejected."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider, owner = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        db_path = data_dir / "sanae.db"

        bvid_prefix_ref = "BV1Prefx0001"
        _insert_video_watch(
            db_path,
            bvid=bvid_prefix_ref,
            title="残缺前缀引用玉笛样本",
            analysis="MUST-NOT-LEAK-BARE-BV-PREFIX",
            captions='[{"text": "CAPTION-BARE-BV-PREFIX"}]',
            summary="残缺前缀引用玉笛样本完整正文：仅含BV前缀",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": "https://www.bilibili.com/video/BV",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key="",
        )
        _insert_video_watch(
            db_path,
            bvid="BV",
            title="超短号空壳铜镜样本",
            analysis="MUST-NOT-LEAK-SHORT-BV",
            captions='[{"text": "CAPTION-SHORT-BV"}]',
            summary="超短号空壳铜镜样本完整正文：仅两字符BV",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": "https://www.bilibili.com/video/BV/",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key="",
        )

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            assert provider.video_memory.get_video_watch_detail(bvid_prefix_ref, owner_key=owner) is None
            tool_resp = _dispatch_video_detail(scope_key, bvid_prefix_ref, session_id="dm-A")
            assert tool_resp["ok"] is True and tool_resp["result"]["found"] is False
            assert all(
                item.get("bvid") != bvid_prefix_ref
                for item in _search_memory_tool(provider, bvid_prefix_ref)
            )
            kw_prefix_hits = _search_memory_tool(provider, "残缺前缀引用 玉笛样本")
            assert all(item.get("bvid") != bvid_prefix_ref for item in kw_prefix_hits)
            assert "玉笛样本" not in json.dumps(kw_prefix_hits, ensure_ascii=False)

            assert provider.video_memory.get_video_watch_detail("BV", owner_key=owner) is None
            kw_short_hits = _search_memory_tool(provider, "超短号空壳 铜镜样本")
            assert all(item.get("bvid") != "BV" for item in kw_short_hits)
            assert "铜镜样本" not in json.dumps(kw_short_hits, ensure_ascii=False)


def test_unowned_row_with_mismatched_other_bvid_ref_is_rejected_in_detail_and_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unowned rows whose source_context.ref points to a different BVID must be rejected."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider, owner = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        db_path = data_dir / "sanae.db"

        target_bvid = "BV1Target001"
        other_ref_bvid = "BV1Other0001"
        _insert_video_watch(
            db_path,
            bvid=target_bvid,
            title="错配引用古琴调音样本",
            analysis="MUST-NOT-LEAK-MISMATCHED-REF",
            captions='[{"text": "CAPTION-MISMATCHED-REF"}]',
            summary="错配引用古琴调音样本完整正文：引用指向另一视频",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": f"https://www.bilibili.com/video/{other_ref_bvid}/",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key="",
        )

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            assert provider.video_memory.get_video_watch_detail(target_bvid, owner_key=owner) is None
            tool_resp = _dispatch_video_detail(scope_key, target_bvid, session_id="dm-A")
            assert tool_resp["ok"] is True and tool_resp["result"]["found"] is False
            for query in (target_bvid, other_ref_bvid, "错配引用 古琴调音"):
                hits = _search_memory_tool(provider, query)
                assert all(item.get("bvid") != target_bvid for item in hits), query
                assert "古琴调音" not in json.dumps(hits, ensure_ascii=False), query


def test_unowned_row_with_non_standard_bvid_length_or_chars_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unowned rows whose bvid is not exact 12-char BV[0-9A-Za-z]{10} must be rejected even with matching ref."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider, owner = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        db_path = data_dir / "sanae.db"

        malformed_bvids: list[tuple[str, str, str]] = [
            ("BV1ShortId1", "十一位短号竹笛样本", "十一位短号 竹笛样本"),
            ("BV1TooLongId1", "十三位长号紫砂样本", "十三位长号 紫砂样本"),
            ("BV1TooLongId01", "十四位超长端砚样本", "十四位超长 端砚样本"),
            ("BV1Bad_Char1", "含下划线玉佩样本", "含下划线 玉佩样本"),
            ("bv1LowerCase", "小写前缀宣纸样本", "小写前缀 宣纸样本"),
        ]
        for bad_bvid, title, _kw_query in malformed_bvids:
            _insert_video_watch(
                db_path,
                bvid=bad_bvid,
                title=title,
                analysis=f"MUST-NOT-LEAK-{bad_bvid}",
                captions=f'[{{"text":"CAPTION-{bad_bvid}"}}]',
                summary=f"{title}完整正文：{bad_bvid}",
                source_context=json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": f"https://www.bilibili.com/video/{bad_bvid}/",
                    }
                ),
                read_scope='{"kind":"self"}',
                owner_key="",
            )

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            for bad_bvid, title, kw_query in malformed_bvids:
                assert provider.video_memory.get_video_watch_detail(bad_bvid, owner_key=owner) is None, bad_bvid
                tool_resp = _dispatch_video_detail(scope_key, bad_bvid, session_id="dm-A")
                assert not (
                    isinstance(tool_resp.get("result"), dict)
                    and tool_resp["result"].get("found") is True
                ), (bad_bvid, tool_resp)
                for query in (bad_bvid, kw_query):
                    hits = _search_memory_tool(provider, query)
                    assert all(item.get("bvid") != bad_bvid for item in hits), (bad_bvid, query)
                    assert title not in json.dumps(hits, ensure_ascii=False), (bad_bvid, query)


def test_unowned_row_with_nul_byte_in_bvid_or_ref_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded U+0000 NUL bytes in bvid or source_context.ref must not bypass SQLite length/GLOB checks."""
    with _isolated_env(tmp_path, monkeypatch) as (home, data_dir, scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        }
        provider, owner = _open_provider(
            stack, home=home, config=config, platform="telegram", user_id="account-A", session_id="dm-A"
        )
        db_path = data_dir / "sanae.db"

        nul_cases: list[tuple[str, str, str, str]] = [
            (
                "BV1LegacyOk1\x00evil",
                "https://www.bilibili.com/video/BV1LegacyOk1\x00evil/",
                "尾随空字节玉箫样本",
                "尾随空字节 玉箫样本",
            ),
            (
                "BV1LegacyOk\x00",
                "https://www.bilibili.com/video/BV1LegacyOk\x00/",
                "末尾空字节铜铃样本",
                "末尾空字节 铜铃样本",
            ),
            (
                "BV1NulRef001",
                "https://www.bilibili.com/video/BV1NulRef001/\x00evil",
                "引用空字节青铜样本",
                "引用空字节 青铜样本",
            ),
        ]
        for bad_bvid, bad_ref, title, _kw_query in nul_cases:
            _insert_video_watch(
                db_path,
                bvid=bad_bvid,
                title=title,
                analysis=f"MUST-NOT-LEAK-NUL-{title}",
                captions=f'[{{"text":"CAPTION-NUL-{title}"}}]',
                summary=f"{title}完整正文：含NUL控制字符",
                source_context=json.dumps(
                    {
                        "kind": "self",
                        "platform": "bilibili",
                        "certainty": "observed",
                        "visibility": "public",
                        "ref": bad_ref,
                    }
                ),
                read_scope='{"kind":"self"}',
                owner_key="",
            )

        with _tools_registry_mod.session_tool_scope(scope_key), _chat_route(
            user_id="account-A", session_id="dm-A"
        ):
            for bad_bvid, _bad_ref, title, kw_query in nul_cases:
                assert provider.video_memory.get_video_watch_detail(bad_bvid, owner_key=owner) is None, repr(bad_bvid)
                tool_resp = _dispatch_video_detail(scope_key, bad_bvid, session_id="dm-A")
                assert not (
                    isinstance(tool_resp.get("result"), dict)
                    and tool_resp["result"].get("found") is True
                ), (repr(bad_bvid), tool_resp)
                for query in (bad_bvid, "BV1LegacyOk1", "BV1NulRef001", kw_query):
                    hits = _search_memory_tool(provider, query)
                    assert all(item.get("bvid") != bad_bvid for item in hits), (repr(bad_bvid), repr(query))
                    assert title not in json.dumps(hits, ensure_ascii=False), (repr(bad_bvid), repr(query))


def test_bound_multi_account_owners_keep_own_reads_while_stranger_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound maintainer accounts can read across their own owner keys, while a stranger reads none."""
    with _isolated_env(
        tmp_path, monkeypatch, load_bilibili_plugin=False
    ) as (home, data_dir, _scope_key), ExitStack() as stack:
        config: dict[str, Any] = {
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "qqbot", "user_id": "qq-openid-1"},
            "maintainer_account_bindings": [
                {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_WORK"},
            ],
        }
        provider_qq, anchor_owner = _open_provider(
            stack,
            home=home,
            config=config,
            platform="qqbot",
            user_id="qq-openid-1",
            session_id="qq-dm-1",
            scope_id="102000001",
        )
        provider_slack, slack_bound_owner = _open_provider(
            stack,
            home=home,
            config=config,
            platform="slack",
            user_id="U_MAINT",
            session_id="slack-dm-1",
            scope_id="T_WORK",
        )
        provider_stranger, stranger_owner = _open_provider(
            stack,
            home=home,
            config=config,
            platform="slack",
            user_id="U_STRANGER",
            session_id="slack-dm-stranger",
            scope_id="T_WORK",
        )
        assert slack_bound_owner == anchor_owner
        assert stranger_owner and stranger_owner != anchor_owner

        # Resolve the legacy pre-binding Slack owner key via public identity resolution.
        with _chat_route(
            platform="slack",
            user_id="U_MAINT",
            session_id="slack-dm-1",
            scope_id="T_WORK",
        ):
            bound_identity = resolve_chat_identity(current_chat_scope(), config)
            legacy_owners = [
                key for key in bound_identity.private_read_owner_keys if key != anchor_owner
            ]
        assert len(legacy_owners) == 1
        slack_legacy_owner = legacy_owners[0]

        db_path = data_dir / "sanae.db"
        bvid_legacy_acct = "BV1BoundSlk1"
        _insert_video_watch(
            db_path,
            bvid=bvid_legacy_acct,
            title="旧账号胶片暗房冲洗",
            analysis="BOUND-LEGACY-SLACK-ANALYSIS",
            captions='[{"text": "BOUND-LEGACY-CAPTION"}]',
            summary="旧账号胶片暗房冲洗：黑白定影液配比记录",
            source_context="",
            read_scope='{"kind":"owner"}',
            owner_key=slack_legacy_owner,
        )
        bvid_anchor = "BV1BoundQqa1"
        _insert_video_watch(
            db_path,
            bvid=bvid_anchor,
            title="主账号天文台延时摄影",
            analysis="BOUND-ANCHOR-QQ-ANALYSIS",
            captions='[{"text": "BOUND-ANCHOR-CAPTION"}]',
            summary="主账号天文台延时摄影：冬季猎户座星轨记录",
            source_context=json.dumps(
                {
                    "kind": "self",
                    "platform": "bilibili",
                    "certainty": "observed",
                    "visibility": "public",
                    "ref": f"https://www.bilibili.com/video/{bvid_anchor}/",
                }
            ),
            read_scope='{"kind":"self"}',
            owner_key=anchor_owner,
        )
        # Non-Bilibili own watch under legacy bound account stays searchable for the bound maintainer.
        _insert_video_watch(
            db_path,
            bvid="douyin:dy-bound-01",
            platform="douyin",
            source_object_id="dy-bound-01",
            title="旧账号手冲手磨咖啡笔记",
            analysis="BOUND-DOUYIN-ANALYSIS",
            summary="旧账号手冲手磨咖啡笔记：水温九十二度萃取",
            source_context=json.dumps({"kind": "video", "platform": "douyin", "visibility": "private"}),
            read_scope=json.dumps({"kind": "owner", "owner": slack_legacy_owner}),
            owner_key=slack_legacy_owner,
        )

        # Bound Slack DM and QQ DM both read across anchor_owner + slack_legacy_owner.
        for active_provider, plat, uid, sid, sc_id in (
            (provider_slack, "slack", "U_MAINT", "slack-dm-1", "T_WORK"),
            (provider_qq, "qqbot", "qq-openid-1", "qq-dm-1", "102000001"),
        ):
            with _chat_route(platform=plat, user_id=uid, session_id=sid, scope_id=sc_id):
                detail_legacy = active_provider.video_memory.get_video_watch_detail(
                    bvid_legacy_acct, owner_key=anchor_owner
                )
                assert detail_legacy is not None
                assert detail_legacy["analysis"] == "BOUND-LEGACY-SLACK-ANALYSIS"
                detail_anchor = active_provider.video_memory.get_video_watch_detail(
                    bvid_anchor, owner_key=anchor_owner
                )
                assert detail_anchor is not None
                assert detail_anchor["analysis"] == "BOUND-ANCHOR-QQ-ANALYSIS"

                assert any(
                    row["bvid"] == bvid_legacy_acct
                    for row in active_provider.video_memory.search_video_watches("暗房冲洗", owner_key=anchor_owner)
                )
                assert any(
                    item.get("bvid") == bvid_legacy_acct
                    for item in _search_memory_tool(active_provider, bvid_legacy_acct)
                )
                assert any(
                    item.get("bvid") == bvid_anchor
                    for item in _search_memory_tool(active_provider, "天文台 延时摄影")
                )
                assert any(
                    item.get("bvid") == "douyin:dy-bound-01"
                    for item in _search_memory_tool(active_provider, "手冲 手磨咖啡")
                )

        # Same-workspace stranger cannot read any of the maintainer's bound rows (including public+self anchor row).
        with _chat_route(
            platform="slack",
            user_id="U_STRANGER",
            session_id="slack-dm-stranger",
            scope_id="T_WORK",
        ):
            for bv, kw in (
                (bvid_legacy_acct, "暗房冲洗"),
                (bvid_anchor, "天文台"),
                ("douyin:dy-bound-01", "手磨咖啡"),
            ):
                assert provider_stranger.video_memory.get_video_watch_detail(bv, owner_key=stranger_owner) is None
                assert provider_stranger.video_memory.search_video_watches(kw, owner_key=stranger_owner) == []
                assert all(
                    item.get("bvid") != bv for item in _search_memory_tool(provider_stranger, bv)
                )
                assert all(
                    item.get("bvid") != bv
                    for item in _search_memory_tool(provider_stranger, f"{kw} 记录")
                )
            assert provider_stranger.video_memory.list_video_watches(owner_key=stranger_owner) == []
