"""Exercise ordinary image asset writes and reads through the memory tools."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

from test_original_hermes import _route
from sanae_mem import SanaeMemProvider


def _png(color: tuple[int, int, int]) -> bytes:
    image = BytesIO()
    Image.new("RGB", (4, 3), color).save(image, format="PNG")
    return image.getvalue()


def _tool(provider, name, args):
    return json.loads(provider.handle_tool_call(name, args))


def _initialized_provider(data_dir: Path, home: Path, session: str) -> SanaeMemProvider:
    provider = SanaeMemProvider({
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
    })
    with _route(actor="fixture-account", session=session):
        provider.initialize(
            session, hermes_home=str(home), platform="telegram", agent_context="primary",
        )
    return provider


def _call(provider, session, name, args):
    with _route(actor="fixture-account", session=session):
        return _tool(provider, name, args)


def test_url_and_local_images_are_stored_and_read_as_original_assets(tmp_path, monkeypatch):
    from sanae_mem.domains import meme_assets
    import ipaddress

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    url_bytes = _png((220, 40, 30))
    local_bytes = _png((20, 80, 210))
    local_path = tmp_path / "SYNTHETIC-LOCAL-IMAGE.png"
    local_path.write_bytes(local_bytes)

    class _ImageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(url_bytes)))
            self.end_headers()
            self.wfile.write(url_bytes)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Keep the real HTTP connection and parser. Only classify this local fixture as public.
    real_address = ipaddress.ip_address
    monkeypatch.setattr(meme_assets.ipaddress, "ip_address",
                        lambda value: real_address("8.8.8.8") if value == "127.0.0.1" else real_address(value))

    provider = SanaeMemProvider({
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
    })
    try:
        with _route(actor="fixture-account", session="fixture-images"):
            provider.initialize(
                "fixture-images", hermes_home=str(home),
                platform="telegram", agent_context="primary",
            )
            url = f"http://127.0.0.1:{server.server_port}/fixture.png"
            url_save = _tool(provider, "sanae_meme_save", {
                "title": "SYNTHETIC URL ASSET",
                "meaning": "A small synthetic fixture image fetched by the standalone HTTP reader.",
                "images": [url],
            })
            assert url_save["ok"] is True, url_save
            url_result = url_save["result"]
            assert url_result["stored"] is True
            url_asset = url_result["assets"][0]
            assert url_asset["mime"] == "image/png"
            assert (url_asset["width"], url_asset["height"]) == (4, 3)
            assert url_asset["sha256"] == hashlib.sha256(url_bytes).hexdigest()
            assert Path(url_asset["path"]).read_bytes() == url_bytes

            local_save = _tool(provider, "sanae_meme_save", {
                "title": "SYNTHETIC LOCAL ASSET",
                "meaning": "A small synthetic fixture image read from a local file.",
                "images": [str(local_path)],
            })
            assert local_save["ok"] is True, local_save
            local_result = local_save["result"]
            assert local_result["stored"] is True
            local_asset = local_result["assets"][0]
            assert local_asset["mime"] == "image/png"
            assert local_asset["sha256"] == hashlib.sha256(local_bytes).hexdigest()
            assert Path(local_asset["path"]).read_bytes() == local_bytes

            for query, expected_title, expected_bytes in (
                ("synthetic URL asset", "SYNTHETIC URL ASSET", url_bytes),
                ("synthetic local asset", "SYNTHETIC LOCAL ASSET", local_bytes),
            ):
                found = _tool(provider, "sanae_meme_search", {
                    "query": query, "purpose": "original",
                })
                assert found["ok"] is True, found
                record = next(item for item in found["result"] if item["title"] == expected_title)
                asset = record["assets"][0]
                assert Path(asset["original_path"]).read_bytes() == expected_bytes
                assert asset["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_text_meme_tombstone_tracks_identity_across_alias_updates(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    provider = _initialized_provider(tmp_path / "sanae-mem", home, "fixture-text")
    try:
        saved = _call(provider, "fixture-text", "sanae_meme_save", {
            "title": "SYNTHETIC ALIAS CASE",
            "meaning": "A synthetic phrase with a stable explanation.",
            "aliases": ["alias-old"],
        })["result"]
        assert saved["stored"] is True
        record_id = saved["record_id"]
        updated = _call(provider, "fixture-text", "sanae_meme_update", {
            "record_id": record_id,
            "aliases": ["alias-new-1", "alias-new-2"],
        })
        assert updated["ok"] is True, updated
        deleted = _call(provider, "fixture-text", "sanae_meme_delete", {
            "record_id": record_id,
        })
        assert deleted["ok"] is True, deleted

        for aliases in (["alias-old"], ["alias-new-1", "alias-new-2"]):
            rejected = _call(provider, "fixture-text", "sanae_meme_save", {
                "title": "SYNTHETIC ALIAS CASE",
                "meaning": "A synthetic phrase with a stable explanation.",
                "aliases": list(aliases),
            })["result"]
            assert rejected["stored"] is False
            assert rejected["reason"] == "forgotten"

        distinct = _call(provider, "fixture-text", "sanae_meme_save", {
            "title": "SYNTHETIC ALIAS CASE",
            "meaning": "A different explanation remains a distinct identity.",
            "aliases": ["alias-unrelated"],
        })["result"]
        assert distinct["stored"] is True
        for aliases in (["alias-old"], ["alias-new-1", "alias-new-2"]):
            still_forgotten = _call(provider, "fixture-text", "sanae_meme_save", {
                "title": "SYNTHETIC ALIAS CASE",
                "meaning": "A synthetic phrase with a stable explanation.",
                "aliases": list(aliases),
            })["result"]
            assert still_forgotten["stored"] is False
            assert still_forgotten["reason"] == "forgotten"

        restored = _call(provider, "fixture-text", "sanae_meme_save", {
            "title": "SYNTHETIC ALIAS CASE",
            "meaning": "A synthetic phrase with a stable explanation.",
            "aliases": ["alias-old"],
            "restore_deleted": True,
        })["result"]
        assert restored["stored"] is True
        assert restored["restored"] is True
        restored_id = restored["record_id"]
        assert restored["restored_fingerprints"]

        for aliases in (["alias-old"], ["alias-new-1", "alias-new-2"]):
            duplicate = _call(provider, "fixture-text", "sanae_meme_save", {
                "title": "SYNTHETIC ALIAS CASE",
                "meaning": "A synthetic phrase with a stable explanation.",
                "aliases": list(aliases),
            })["result"]
            assert duplicate["stored"] is False
            assert duplicate["reason"] == "duplicate_text"
            assert duplicate["duplicate_record_id"] == restored_id

        found = _call(provider, "fixture-text", "sanae_meme_search", {
            "query": "stable", "top_k": 10, "purpose": "original",
        })["result"]
        assert [item["record_id"] for item in found] == [restored_id]
    finally:
        provider.shutdown()


def test_empty_title_tombstone_uses_order_independent_alias_set(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    provider = _initialized_provider(tmp_path / "sanae-mem", home, "fixture-empty-title")
    try:
        saved = _call(provider, "fixture-empty-title", "sanae_meme_save", {
            "meaning": "A synthetic explanation without a title.",
            "aliases": ["alias-alpha", "alias-beta"],
        })["result"]
        record_id = saved["record_id"]
        assert _call(provider, "fixture-empty-title", "sanae_meme_delete", {
            "record_id": record_id,
        })["ok"] is True

        reordered = _call(provider, "fixture-empty-title", "sanae_meme_save", {
            "meaning": "A synthetic explanation without a title.",
            "aliases": ["alias-beta", "alias-alpha"],
        })["result"]
        assert reordered["stored"] is False
        assert reordered["reason"] == "forgotten"

        distinct = _call(provider, "fixture-empty-title", "sanae_meme_save", {
            "meaning": "A synthetic explanation without a title.",
            "aliases": ["alias-alpha", "alias-gamma"],
        })["result"]
        assert distinct["stored"] is True
    finally:
        provider.shutdown()


def test_title_only_private_meme_matches_turn_without_alias(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    provider = _initialized_provider(tmp_path / "sanae-mem", home, "fixture-title-only")
    try:
        saved = _call(provider, "fixture-title-only", "sanae_meme_save", {
            "title": "SYNTHETIC TITLE ONLY",
            "meaning": "A title-only text meme with a synthetic explanation.",
        })["result"]
        assert saved["stored"] is True
        with _route(actor="fixture-account", session="fixture-title-only"):
            context = provider.meme_context_for_turn("SYNTHETIC TITLE ONLY")
        assert "A title-only text meme" in context
    finally:
        provider.shutdown()


def test_two_connections_restore_same_text_tombstone_once(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    data_dir = tmp_path / "sanae-mem"
    first = _initialized_provider(data_dir, home, "fixture-restore-a")
    second = _initialized_provider(data_dir, home, "fixture-restore-b")
    try:
        seed = _call(first, "fixture-restore-a", "sanae_meme_save", {
            "title": "SYNTHETIC CONCURRENT RESTORE",
            "meaning": "A synthetic record restored by two connections.",
            "aliases": ["alias-concurrent"],
        })["result"]
        _call(first, "fixture-restore-a", "sanae_meme_delete", {
            "record_id": seed["record_id"],
        })

        barrier = threading.Barrier(2)

        def restore(provider, session):
            barrier.wait(timeout=10)
            return _call(provider, session, "sanae_meme_save", {
                "title": "SYNTHETIC CONCURRENT RESTORE",
                "meaning": "A synthetic record restored by two connections.",
                "aliases": ["alias-concurrent"],
                "restore_deleted": True,
            })["result"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(
                lambda item: restore(*item),
                [(first, "fixture-restore-a"), (second, "fixture-restore-b")],
            ))
        assert sorted(item["reason"] for item in outcomes) == ["created", "duplicate_text"]
        assert first._conn.execute("SELECT COUNT(*) FROM meme_records").fetchone()[0] == 1
    finally:
        first.shutdown()
        second.shutdown()


def test_two_connections_delete_and_legacy_alias_save_cannot_resurrect(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    data_dir = tmp_path / "sanae-mem"
    first = _initialized_provider(data_dir, home, "fixture-delete-a")
    second = _initialized_provider(data_dir, home, "fixture-delete-b")
    try:
        seed = _call(first, "fixture-delete-a", "sanae_meme_save", {
            "title": "SYNTHETIC CONCURRENT DELETE",
            "meaning": "A synthetic record deleted while an old alias is saved.",
            "aliases": ["alias-old"],
        })["result"]
        record_id = seed["record_id"]
        _call(first, "fixture-delete-a", "sanae_meme_update", {
            "record_id": record_id,
            "aliases": ["alias-new"],
        })

        barrier = threading.Barrier(2)

        def delete():
            barrier.wait(timeout=10)
            return _call(first, "fixture-delete-a", "sanae_meme_delete", {
                "record_id": record_id,
            })

        def save_old_alias():
            barrier.wait(timeout=10)
            return _call(second, "fixture-delete-b", "sanae_meme_save", {
                "title": "SYNTHETIC CONCURRENT DELETE",
                "meaning": "A synthetic record deleted while an old alias is saved.",
                "aliases": ["alias-old"],
            })

        with ThreadPoolExecutor(max_workers=2) as executor:
            delete_future = executor.submit(delete)
            save_future = executor.submit(save_old_alias)
            deleted, saved = delete_future.result(timeout=20), save_future.result(timeout=20)
        assert deleted["ok"] is True
        assert saved["ok"] is True
        assert first._conn.execute("SELECT COUNT(*) FROM meme_records").fetchone()[0] == 0
        assert saved["result"].get("reason") in {"forgotten", "duplicate_text"}
        replayed = _call(second, "fixture-delete-b", "sanae_meme_save", {
            "title": "SYNTHETIC CONCURRENT DELETE",
            "meaning": "A synthetic record deleted while an old alias is saved.",
            "aliases": ["alias-never-saved"],
        })
        assert replayed["ok"] is True, replayed
        assert replayed["result"]["stored"] is False
        assert replayed["result"]["reason"] == "forgotten"
    finally:
        first.shutdown()
        second.shutdown()
