from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem.storage.database import open_database
from sanae_mem.domains.meme_context import context_for_public_material, context_for_turn
from sanae_mem import SanaeMemProvider
from sanae_mem.infra.host_bridge import turn_envelope_scope


def _record(conn, *, title, meaning, owner="", tier="common", visibility="private",
            source_ref="", evidence="uncertain"):
    now = datetime.now().astimezone().isoformat()
    return conn.execute(
        """INSERT INTO meme_records
           (title, meaning, aliases, source, source_ref, source_time, created_at,
            updated_at, owner_key, session_id, tier, visibility_scope, evidence_status)
           VALUES (?, ?, '[]', '', ?, '', ?, ?, ?, '', ?, ?, ?)""",
        (title, meaning, source_ref, now, now, owner, tier, visibility, evidence),
    ).lastrowid


def test_turn_context_selects_scope_before_matching_and_keeps_short_reminders(tmp_path):
    conn = open_database(tmp_path / "sanae.db")
    try:
        _record(conn, title="SHARED PHRASE", meaning="Public explanation one.",
                tier="eye_familiar", visibility="public", evidence="confirmed",
                source_ref="https://example.test/meme")
        _record(conn, title="SHARED PHRASE", meaning="Public explanation two.",
                tier="eye_familiar", visibility="public", evidence="confirmed")
        _record(conn, title="SHARED PHRASE", meaning="Owner A explanation.", owner="owner-A")
        _record(conn, title="PRIVATE A PHRASE", meaning="Owner A short reminder.", owner="owner-A")
        _record(conn, title="PRIVATE B PHRASE", meaning="Owner B secret.", owner="owner-B")
        conn.commit()

        public = context_for_turn(conn, "SHARED PHRASE", shadow=False, public=True)
        assert "Public explanation one." in public
        assert "Public explanation two." in public
        assert "Owner A explanation." not in public
        assert "PRIVATE A PHRASE" not in public

        private = context_for_turn(conn, "PRIVATE A PHRASE", shadow=False, public=False,
                                   owner_key="owner-A")
        assert "Owner A short reminder." in private
        assert "Owner B secret." not in private

        public_material = context_for_public_material(
            conn, ["SHARED PHRASE"], shadow=False, public=True, public_only=True,
        )
        assert "https://example.test/meme" in public_material
        assert "Public explanation one." in public_material
        assert "verification_text" not in public_material
        assert "Owner A explanation." not in public_material
    finally:
        conn.close()


def test_private_meme_tools_cannot_read_or_mutate_other_owner_or_public_rows(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    data_dir = tmp_path / "sanae-mem"
    provider = SanaeMemProvider({"data_dir": str(data_dir), "semantic_backend": "sqlite"})
    with ExitStack() as stack:
        stack.enter_context(turn_envelope_scope(
            inherit_context=False, platform="telegram", chat_type="dm", chat_id="account-A",
            sender_id="account-A", session_id="private-A", session_key="dm:account-A",
            message_id="meme-tool-message",
        ))
        stack.callback(provider.shutdown)
        provider.initialize("private-A", hermes_home=str(home), platform="telegram",
                            agent_context="primary")
        other_id = _record(provider._conn, title="SYNTHETIC SHARED TITLE",
                           meaning="Other owner secret.", owner="owner-B")
        public_id = _record(provider._conn, title="SYNTHETIC PUBLIC TITLE",
                            meaning="Confirmed public evidence.", tier="eye_familiar",
                            visibility="public", evidence="confirmed")
        provider._conn.commit()

        duplicate = json.loads(provider.handle_tool_call("sanae_meme_save", {
            "title": "SYNTHETIC SHARED TITLE", "meaning": "Other owner secret.",
        }))
        assert duplicate["ok"] is True, duplicate
        assert duplicate["result"]["stored"] is True
        assert duplicate["result"]["record_id"] not in {other_id, public_id}
        assert "duplicate_record_id" not in duplicate["result"]

        for tool, args in (
            ("sanae_meme_update", {"record_id": other_id, "meaning": "Changed."}),
            ("sanae_meme_update", {"record_id": public_id, "meaning": "Changed."}),
            ("sanae_meme_delete", {"record_id": other_id}),
            ("sanae_meme_delete", {"record_id": public_id}),
        ):
            result = json.loads(provider.handle_tool_call(tool, args))
            assert result["ok"] is False

        assert provider._conn.execute(
            "SELECT meaning FROM meme_records WHERE id=?", (other_id,)
        ).fetchone()[0] == "Other owner secret."
        assert provider._conn.execute(
            "SELECT meaning FROM meme_records WHERE id=?", (public_id,)
        ).fetchone()[0] == "Confirmed public evidence."
