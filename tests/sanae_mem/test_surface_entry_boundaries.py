from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

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
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
if not HERMES_ROOT.is_dir():
    pytest.skip("candidate host is required", allow_module_level=True)
sys.path.insert(0, str(HERMES_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins"))

import hermes_host_contracts as hc
from sanae_mem import SanaeMemProvider
from sanae_mem.governance import memory_guard, source_context, source_intake
from sanae_mem.governance.cognitive_context import ViveMindCognitivePromptPolicy


@pytest.fixture(autouse=True)
def isolated_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(tmp_path / "home"))
    hc.clear_all_contract_registrations()
    yield
    hc.clear_all_contract_registrations()


def test_public_provider_block_keeps_behavior_without_private_notes(tmp_path):
    home = tmp_path / "home"
    notes = home / "memories"
    notes.mkdir(parents=True)
    (notes / "USER.md").write_text("PRIVATE_USER", encoding="utf-8")
    (notes / "MEMORY.md").write_text("PRIVATE_MEMORY", encoding="utf-8")
    (home / "BEHAVIOR.md").write_text(
        "<!-- RENDER:system_prompt_block:BEGIN -->\n"
        "PROCEDURAL_BEHAVIOR\n"
        "<!-- RENDER:system_prompt_block:END -->\n",
        encoding="utf-8",
    )
    provider = SanaeMemProvider({"semantic_backend": "sqlite", "shadow": False})
    with hc.turn_envelope_scope(hc.TurnEnvelope(
        platform="telegram", chat_type="group", chat_id="group", sender_id="visitor",
        session_id="group-session", session_key="group-route",
    )):
        provider.initialize("group-session", hermes_home=str(home), platform="telegram")
        assert provider.system_prompt_block() == "PROCEDURAL_BEHAVIOR"
        provider.shutdown()


@pytest.mark.parametrize("chat_type", ["group", "dm"])
def test_gateway_admission_uses_active_policy_and_keeps_local_reference_guard(monkeypatch, chat_type):
    monkeypatch.setattr(memory_guard, "_has_enabled_builtin_notes", lambda: True)
    monkeypatch.setattr(memory_guard, "_is_maintainer", lambda *args: False)
    source = {"platform": "telegram", "chat_type": chat_type, "chat_id": "external", "user_id": "visitor"}
    policy = ViveMindCognitivePromptPolicy(lambda: {})
    hc.register_cognitive_prompt_policy(policy)
    assert memory_guard.guard_public_local_references(source=source, text="今天下雨了。") is None
    assert memory_guard.guard_public_local_references(source=source, text="@file:notes.txt")["action"] == "handled"
    assert memory_guard.guard_public_local_references(source=source, text="@folder:notes")["action"] == "handled"
    hc.unregister_cognitive_prompt_policy(policy)
    assert memory_guard.guard_public_local_references(source=source, text="今天下雨了。")["action"] == "handled"


def test_ambient_session_variables_cannot_supply_a_private_owner(monkeypatch):
    values = {"PLATFORM": "telegram", "CHAT_TYPE": "dm", "CHAT_ID": "private",
              "USER_ID": "maintainer", "KEY": "private-route", "ID": "private-session"}
    for key, value in values.items():
        monkeypatch.setenv("HERMES_SESSION_" + key, value)
    with hc.turn_envelope_scope(hc.TurnEnvelope()):
        scope = source_context.current_chat_scope()
        assert (scope.visibility, scope.owner) == ("unknown", "")
        provider = SimpleNamespace(
            sqlite=SimpleNamespace(is_open=True),
            source_intake_context=lambda: source_intake.SourceIntakeContext(
                session_id="",
                agent_context="primary",
                shadow=False,
                daily_owner_key="configured-owner",
                daily_owner_platform="telegram",
                daily_profile_key="",
                private_chat_owner="configured-owner",
                is_configured_private_owner=True,
            ),
        )
        assert source_intake._resolve_intake_owner(provider, allow_configured_owner_fallback=True) == ""
    with hc.turn_envelope_scope(hc.TurnEnvelope(
        platform="telegram", chat_type="dm", chat_id="real-chat", sender_id="real-sender",
        session_id="real-session", session_key="real-route",
    )):
        scope = source_context.current_chat_scope()
        assert (scope.visibility, scope.actor, scope.session_id) == ("private", "real-sender", "real-session")


def test_ingest_source_material_preserves_test_session_isolation_after_shutdown(tmp_path):
    provider = SanaeMemProvider({
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "maint-1"},
    })
    provider.initialize("T:pr4-intake-isolation", hermes_home=str(tmp_path / "home"), platform="telegram")
    provider.shutdown()
    verdict = source_intake.ingest_source_material(
        provider,
        {
            "source_plugin": "bilibili_public",
            "platform": "bilibili",
            "material_kind": "raw",
            "privacy_scope": "public",
            "content": "Synthetic intake payload after shutdown.",
            "source_object_id": "BV1xx411c7mD",
            "raw_locator": "https://www.bilibili.com/video/BV1xx411c7mD",
        },
        allow_configured_owner_fallback=True,
    )
    assert (verdict.ok, verdict.reason) == (False, "test_session_isolated")
