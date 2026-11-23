from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
    pytest.skip("pinned Hermes source is unavailable", allow_module_level=True)
@pytest.mark.parametrize("loader", ["memory", "directory"])
def test_installed_provider_keeps_public_imports_and_persists_intake(tmp_path, loader):
    home = tmp_path / "hermes"
    provider_dir = home / "plugins" / "sanae_mem"
    provider_dir.parent.mkdir(parents=True)
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem",
        provider_dir,
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"),
    )
    (home / "config.yaml").write_text(
        "memory:\n  provider: sanae_mem\n  memory_enabled: false\n  user_profile_enabled: false\n"
        "plugins:\n  enabled: [sanae_mem]\n  sanae_mem:\n"
        "    semantic_backend: sqlite\n"
        "    memory_identity: {platform: qq, user_id: '10001'}\n",
        encoding="utf-8",
    )
    code = r"""
import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from hermes_cli.plugins import PluginManager
from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
from plugins.memory import load_memory_provider

home, loader = Path(sys.argv[1]), sys.argv[2]
assert importlib.util.find_spec("sanae_mem") is None
original_finders = sys.meta_path[:]
if loader == "memory":
    provider = load_memory_provider("sanae_mem", register_skills=False)
    assert provider is not None
    assert type(provider).__module__.startswith("_hermes_user_memory.sanae_mem__source_")
else:
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    assert any(item["name"] == "sanae_mem" and item["enabled"] for item in manager.list_plugins())
    assert "hermes_plugins.sanae_mem.plugin_entry" in sys.modules

import sanae_mem
from sanae_mem import integrations, plugin_entry, submit_source_material
from sanae_mem import source_intake as public_source_intake
from sanae_mem.governance import source_intake

assert Path(sanae_mem.__file__).resolve().parent == home / "plugins" / "sanae_mem"
assert integrations.__name__ == "sanae_mem.integrations"
assert plugin_entry.SanaeMemProvider is sanae_mem.SanaeMemProvider
assert submit_source_material is source_intake.submit_source_material
assert importlib.import_module("sanae_mem.source_intake") is public_source_intake
assert public_source_intake.submit_source_material is source_intake.submit_source_material
assert not {home / "plugins", home / "plugins" / "sanae_mem"} & {
    Path(entry).resolve() for entry in sys.path
}
assert sys.meta_path == original_finders
envelope = TurnEnvelope(platform="qq", chat_type="dm", chat_id="10001", sender_id="10001",
                        session_id="public-intake", session_key="agent:main:qq:dm:10001")
with turn_envelope_scope(envelope):
    if loader == "memory":
        provider.initialize("public-intake", platform="qq", hermes_home=str(home))
        # A wrong lookup would open the config fallback worker in a different database.
        config = home / "config.yaml"
        text = config.read_text(encoding="utf-8")
        config.write_text(text.replace("semantic_backend: sqlite",
                                       "semantic_backend: sqlite\n    data_dir: different-db"),
                          encoding="utf-8")
    verdict = submit_source_material(
        source_intake.SourceMaterial(
            source_plugin="bilibili_public", platform="bilibili", material_kind="processed",
            processing_type="summary", content="cold install intake", source_object_id="cold-1",
        ),
        session_id="public-intake",
    )
    assert verdict["ok"], verdict
with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db")) as conn:
    assert conn.execute("SELECT session_id, content FROM source_materials").fetchall() == [
        ("public-intake", "cold install intake"),
    ]
with turn_envelope_scope(envelope):
    evidence = submit_source_material({
        "source_plugin": "bilibili_public", "platform": "bilibili", "material_kind": "processed",
        "processing_type": "video_meme_evidence", "privacy_scope": "public", "title": "paper stars",
        "content": "paper stars", "source_object_id": "BV5678901234",
        "raw_locator": "https://www.bilibili.com/video/BV5678901234",
        "extra_metadata": {
            "occurrence_kind": "comment", "occurrence_quote": "paper stars", "meaning": "folded stars",
            "verification_url": "https://example.test/definition",
            "verification_text": "paper stars means folded stars",
            "verification_quote": "paper stars means folded stars",
        },
    }, session_id="public-intake")
    assert evidence["ok"], evidence
with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db")) as conn:
    assert conn.execute("SELECT title FROM meme_records").fetchall() == [("paper stars",)]
    assert conn.execute("SELECT COUNT(*) FROM video_watches").fetchone() == (0,)
if loader == "memory":
    provider.shutdown()
else:
    assert manager.unload("sanae_mem")
    assert importlib.import_module("sanae_mem.governance.source_intake") is source_intake
    reloaded_public_source = importlib.import_module("sanae_mem.source_intake")
    assert reloaded_public_source is public_source_intake
    assert reloaded_public_source.submit_source_material is submit_source_material
"""
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(HERMES_ROOT),
           "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1"}
    env.pop("HERMES_HOME_OVERRIDE", None)
    run = subprocess.run([sys.executable, "-c", code, str(home), loader], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
