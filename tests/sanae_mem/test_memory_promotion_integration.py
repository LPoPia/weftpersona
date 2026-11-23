from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import resolve_hermes_host_root  # noqa: E402

HERMES_ROOT = resolve_hermes_host_root(require_full_host=True)
if HERMES_ROOT is None:
    pytest.skip("full hermes-agent host not available", allow_module_level=True)
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))
PLUGIN_ROOT = TESTS_ROOT.parent / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import plugin_entry  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]
from sanae_mem.infra import host_bridge  # noqa: E402  # pyright: ignore[reportMissingImports]


@contextmanager
def _private_turn(
    home: Path,
    message_id: str,
    *,
    actor: str = "account-A",
    session: str = "private-A",
    chat_type: str = "dm",
) -> Iterator[None]:
    with (
        host_bridge.hermes_home_scope(home),
        host_bridge.turn_envelope_scope(
            platform="telegram",
            chat_type=chat_type,
            chat_id="chat-test",
            sender_id=actor,
            session_id=session,
            session_key=f"agent:main:telegram:{chat_type}:test",
            message_id=message_id,
            is_messaging_surface=False,
        ),
    ):
        yield


@pytest.fixture
def promotion_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    MemoryManager = importlib.import_module("agent.memory_manager").MemoryManager
    plugins_mod = importlib.import_module("hermes_cli.plugins")
    PluginContext = plugins_mod.PluginContext
    PluginManager = plugins_mod.PluginManager
    PluginManifest = plugins_mod.PluginManifest

    home = tmp_path / "hermes-home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    (memories / "USER.md").write_text("A_PROFILE_PRIVATE_NOTE", encoding="utf-8")
    (memories / "MEMORY.md").write_text("PLUGIN_MEMORY_PRIVATE_NOTE", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    data_dir = tmp_path / "sanae-mem"
    config = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        "partner_name": "测试账号 A",
    }
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "memory": {"memory_enabled": False, "user_profile_enabled": False},
                "plugins": {"sanae_mem": config},
            }
        ),
        encoding="utf-8",
    )

    class _ObservedPluginContext(PluginContext):  # type: ignore[misc,valid-type]
        def __init__(self, manifest: Any, hook_manager: Any) -> None:
            super().__init__(manifest, hook_manager)
            self.registered_provider: Any = None
            self.registered_commands: list[str] = []

        def register_memory_provider(self, provider: Any) -> None:
            self.registered_provider = provider
            super().register_memory_provider(provider)

        def register_command(self, name: str, *args: Any, **kwargs: Any) -> None:
            self.registered_commands.append(name)
            super().register_command(name, *args, **kwargs)

    hook_manager = PluginManager(scope_key=str(home))
    ctx = _ObservedPluginContext(PluginManifest(name="sanae_mem", source="user"), hook_manager)
    with host_bridge.hermes_home_scope(home):
        plugin_entry.register(ctx)
    provider = ctx.registered_provider
    assert provider is not None

    memory_manager = MemoryManager()
    with _private_turn(home, "init-turn"):
        memory_manager.add_provider(provider)
        assert memory_manager.has_tool("sanae_mem_add")
        assert memory_manager.has_tool("sanae_mem_search")
        memory_manager.initialize_all(
            session_id="private-A",
            hermes_home=str(home),
            agent_context="primary",
            platform="telegram",
        )

    try:
        yield {
            "home": home,
            "provider_a": provider,
            "memory_manager_a": memory_manager,
            "registered_commands": ctx.registered_commands,
        }
    finally:
        provider.shutdown()
        with host_bridge.hermes_home_scope(home):
            hook_manager.unload()


def test_original_hermes_review_checks_evidence_and_both_files_before_user_write(
    promotion_world: dict[str, Any],
) -> None:
    home = promotion_world["home"]
    provider = promotion_world["provider_a"]
    manager = promotion_world["memory_manager_a"]
    user = home / "memories" / "USER.md"
    memory = home / "memories" / "MEMORY.md"
    original_user = user.read_bytes()
    recognition = "偏好长期参与有固定节奏的户外活动"

    for index, text in enumerate(("我每周都参加社区公园的慢跑活动，这个约定对我很重要",), 1):
        with _private_turn(home, f"incoming-{index}"):
            manager.sync_all(text, "收到。", session_id="private-A", messages=[
                {"role": "user", "content": text, "platform_message_id": f"incoming-{index}"},
            ])
            manager.flush_pending(timeout=5)
    evidence_ids = [row[0] for row in provider.sqlite.fetchall(
        "SELECT id FROM events WHERE source_snapshot<>'' ORDER BY id")]
    assert len(evidence_ids) == 1
    assert manager.has_tool("sanae_mem_review")

    def call(ids: list[int]) -> dict[str, Any]:
        return json.loads(manager.handle_tool_call("sanae_mem_review", {
            "mode": "grounded_user", "text": recognition, "evidence_ids": ids,
        }))

    with _private_turn(home, "review"):
        assert call([])["result"]["reason"] == "evidence_ids_requires_one_to_three"
        assert user.read_bytes() == original_user

        def edit_during_review(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            memory.write_text("A concurrent notebook edit.", encoding="utf-8")
            return verdict(True)

        def verdict(accepted: bool) -> SimpleNamespace:
            return SimpleNamespace(text=json.dumps({
                "duty": "user" if accepted else "none", "long_term": accepted,
                "summary": recognition, "covered_by": "", "source_kind": "user_statement",
                "significance": "两条事实来自不同活动。",
            }, ensure_ascii=False))

        provider.attach_runtime_llm(SimpleNamespace(complete=lambda *a, **kw: verdict(False)))
        assert not call(evidence_ids)["result"]["ok"]
        assert user.read_bytes() == original_user
        provider.attach_runtime_llm(SimpleNamespace(complete=edit_during_review))
        assert call(evidence_ids)["result"]["reason"] == "resident_baseline_changed"
        assert user.read_bytes() == original_user
        assert memory.read_text(encoding="utf-8") == "A concurrent notebook edit."
        provider.attach_runtime_llm(SimpleNamespace(complete=lambda *a, **kw: verdict(True)))
        saved = call(evidence_ids)
        assert saved["result"]["written"] and saved["result"]["support_linked"], saved
        assert user.read_text(encoding="utf-8").split("\n§\n") == [
            original_user.decode("utf-8"), recognition,
        ]
        assert [row[0] for row in provider.sqlite.fetchall(
            "SELECT event_id FROM recognition_evidence ORDER BY event_id")] == evidence_ids
    with _private_turn(home, "group", chat_type="group"):
        assert not call(evidence_ids)["ok"]


def test_user_demote_is_autonomous_and_registers_no_command(
    promotion_world: dict[str, Any],
) -> None:
    home = promotion_world["home"]
    provider = promotion_world["provider_a"]
    user = home / "memories" / "USER.md"
    original = user.read_text(encoding="utf-8")
    assert "sanae-memory" not in promotion_world["registered_commands"]

    def model_verdict(messages: Any, *, purpose: str, task: Any, temperature: float, max_tokens: int, timeout: float) -> SimpleNamespace:
        assert purpose == "resident_user_demote_review"
        assert f"[0] {original}" in messages[-1]["content"]
        return SimpleNamespace(text=json.dumps({"decisions": [{
            "index": 0, "allow_demote": allow_demote,
            "kind": "outdated_recent_status" if allow_demote else "core_trait_or_identity",
        }]}))

    allow_demote = True
    provider.attach_runtime_llm(SimpleNamespace(complete=model_verdict))

    with _private_turn(home, "demote"):
        demoted = provider.resident.demote_resident_entry("user", original)
        assert demoted["demoted"] and not demoted["staged"], demoted
    assert not user.read_text(encoding="utf-8")
    assert not (home / "pending").exists()
    archive = (home / "memory-archive" / "user-demoted.jsonl").read_text(encoding="utf-8")
    assert original in archive
    row = provider.sqlite.fetchone(
        "SELECT raw FROM events WHERE id=? AND tier='long'", (demoted["event_id"],),
    )
    assert row is not None and row[0]

    user.write_text(original, encoding="utf-8")
    allow_demote = False
    with _private_turn(home, "demote-again"):
        kept = provider.resident.demote_resident_entry("user", original)
    assert kept["reason"] == "user_entry_not_outdated"
    assert user.read_text(encoding="utf-8") == original
