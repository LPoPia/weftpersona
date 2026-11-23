"""Exercise turn sync through Hermes with a loopback model and an isolated database."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from sanae_mem.infra import auxiliary
from sanae_mem.provider import SanaeMemProvider


def test_host_sync_keeps_delivery_gates_and_reports_sql_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    host = pytest.importorskip("hermes_host_contracts")
    memory = pytest.importorskip("agent.memory_manager")
    plugins = pytest.importorskip("hermes_cli.plugins")
    prompts: list[str] = []

    class LoopbackModel(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = request["messages"][-1]["content"]
            prompts.append(prompt)
            content = "invalid model reply" if "海边" in prompt else '{"shared_view":""}'
            body = json.dumps({
                "id": "sync-test", "object": "chat.completion", "model": request["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    home = tmp_path / "home"
    home.mkdir()
    data = home / "sanae-mem"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    with ThreadingHTTPServer(("127.0.0.1", 0), LoopbackModel) as server:
        model = {"provider": "custom", "model": "loopback", "api_key": "dummy-key",
                 "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        (home / "config.yaml").write_text(json.dumps({
            "model": model, "auxiliary": {"sanae_daily_flash": model, "sanae_extract": model},
        }), encoding="utf-8")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        manager = memory.MemoryManager()
        plugin_manager = plugins.get_plugin_manager()
        try:
            ctx = plugins.PluginContext(plugins.PluginManifest(name="sanae_mem", source="user"), plugin_manager)
            provider = SanaeMemProvider({
                "data_dir": str(data), "semantic_backend": "sqlite",
                "memory_identity": {"platform": "telegram", "user_id": "1001"},
            })
            provider.attach_runtime_llm(auxiliary.register(ctx))
            auxiliary.register_daily(ctx)
            manager.add_provider(provider)
            private = host.TurnEnvelope(
                platform="telegram", chat_type="dm", chat_id="1001", sender_id="1001",
                session_id="sync-dm", session_key="agent:main:telegram:dm:1001", message_id="dm-1",
                write_origin="foreground",
            )
            with host.turn_envelope_scope(private):
                manager.initialize_all("sync-dm", hermes_home=str(home), platform="telegram")

            def sync(user: str, *, session_id: str = "", message_id: str = "dm-1", reply: str = "好的。") -> None:
                manager.sync_all(user, reply, session_id=session_id,
                                 messages=[{"role": "user", "content": user, "platform_message_id": message_id}])
                assert manager.flush_pending(timeout=15)

            with closing(sqlite3.connect(data / "sanae.db")) as observer:
                before = list(observer.iterdump())
                # The manager can dispatch without a receipt expectation. The plugin still gates by the envelope.
                signal = "我很信任你，请记住下周陪我去天文馆这件重要的事"
                for envelope in (replace(private, is_messaging_surface=True),
                                 replace(private, chat_type="group", chat_id="room"),
                                 replace(private, sender_id="1002", chat_id="1002"),
                                 replace(private, is_test_isolated=True), host.TurnEnvelope()):
                    with host.turn_envelope_scope(envelope):
                        sync(signal)
                    assert list(observer.iterdump()) == before
                assert prompts == []

                with host.turn_envelope_scope(private):
                    sync(signal)
                row = observer.execute(
                    "SELECT raw, session_id, owner_key, read_scope, source_snapshot FROM events",
                ).fetchone()
                assert row is not None
                raw, session_id, owner_key, read_scope, snapshot = row
                assert raw.endswith(" " + signal) and session_id == "sync-dm"
                assert owner_key and json.loads(read_scope) == {"kind": "owner", "owner": owner_key}
                evidence = json.loads(snapshot)
                assert evidence["turn"] == "dm-1"
                assert evidence["source_context"]["author_id"] == "1001"
                assert evidence["source_messages"] == [{"role": "user", "content": signal}]
                assert len(prompts) == 1 and signal in prompts[0]

                observer.execute("CREATE TRIGGER fail_sync_insert BEFORE INSERT ON events "
                                 "BEGIN SELECT RAISE(ABORT, 'isolated event insert failed'); END")
                fault_snapshot = list(observer.iterdump())
                retry_text = "我很信任你，请记住明年陪我去海边拍照这件重要的事"
                with host.turn_envelope_scope(replace(private, message_id="dm-2")):
                    sync(retry_text, message_id="dm-2")
                assert "Memory provider 'sanae_mem' sync_turn failed: isolated event insert failed" in caplog.text
                assert list(observer.iterdump()) == fault_snapshot
                assert len(prompts) == 1

                observer.execute("DROP TRIGGER fail_sync_insert")
                with host.turn_envelope_scope(replace(private, message_id="dm-2")):
                    sync(retry_text, message_id="dm-2")
                rows = observer.execute("SELECT raw, shared_view FROM events ORDER BY id").fetchall()
                assert len(rows) == 2 and rows[1][0].endswith(" " + retry_text)
                assert not rows[1][1]
                assert len(prompts) == 2 and retry_text in prompts[1]
                status = json.loads((data / "status.json").read_text(encoding="utf-8"))
                projection = status["components"]["shared_view_projection"]
                assert projection["ok"] is False and projection["error"] == "parse_invalid_format"

                cron = host.TurnEnvelope(platform="cron", session_id="cron-sync", is_cron=True)
                before = list(observer.iterdump())
                with host.turn_envelope_scope(cron):
                    sync("继续探索天文现象", session_id="cron-sync", reply="查到了一条天文资料。")
                assert list(observer.iterdump()) == before
                assert "no such table: daily_context" not in caplog.text

                observer.execute("ALTER TABLE daily_context RENAME TO injected_missing_daily_context")
                fault_snapshot = list(observer.iterdump())
                with host.turn_envelope_scope(cron):
                    sync("继续探索天文现象", session_id="cron-sync", reply="查到了一条天文资料。")
                assert "Memory provider 'sanae_mem' sync_turn failed: no such table: daily_context" in caplog.text
                assert list(observer.iterdump()) == fault_snapshot
                observer.execute("ALTER TABLE injected_missing_daily_context RENAME TO daily_context")
        finally:
            manager.shutdown_all()
            plugin_manager.unload()
            server.shutdown()
            thread.join(timeout=5)
