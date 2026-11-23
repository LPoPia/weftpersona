
from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import meme_assets, meme_context, meme_search
from ..infra.archive import _append_fsynced_jsonl
from .meme_assets import MemeChatCopyError
from ..storage.sqlite_facade import SqliteFacade
from ..infra.textutil import (
    _cosine,
    _embed_text,
    _fingerprint,
    _pack_vector,
    _tokens,
    _unpack_vector,
)

def known_inbound_images(paths: list[str]) -> dict[str, str]:
    """Resolve exact known images without opening a writable store or invoking a model."""
    from ..infra.host_bridge import get_hermes_home

    from ..governance.source_context import current_chat_scope, in_public_interaction, resolve_chat_identity
    from ..infra.textutil import _bool, _load_plugin_config
    config = _load_plugin_config()
    if in_public_interaction() or _bool(config.get("shadow"), False):
        return {}
    root = Path(config.get("data_dir") or get_hermes_home() / "sanae-mem").expanduser()
    scope = current_chat_scope()
    if scope.visibility == "private":
        identity = resolve_chat_identity(scope, config)
        owner_key = identity.private_owner_key
        read_owners = identity.private_read_owner_keys
        allow_legacy_blank = bool(identity.maintainer and owner_key)
    else:
        owner_key = ""
        read_owners = ()
        allow_legacy_blank = False
    return meme_context.known_inbound_images(
        paths, root / "sanae.db", owner_key=owner_key,
        owner_keys=read_owners, allow_legacy_blank_owner=allow_legacy_blank,
    )


def _meme_now() -> str:
    return datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class MemeRuntime:
    """Resolve the current profile and trusted identity when a meme operation needs them."""

    data_dir: Callable[[], Path]
    archive_home: Callable[[], Path]
    session_id: Callable[[], str]
    shadow_enabled: Callable[[], bool]
    private_chat_owner: Callable[[], str]
    association_read_owner: Callable[[], str]
    write_owner: Callable[[], str]
    read_owner_keys: Callable[[str], Sequence[str]]
    is_configured_private_owner: Callable[[], bool]


class MemeStore:
    def __init__(self, sqlite: SqliteFacade, runtime: MemeRuntime) -> None:
        self._sqlite = sqlite
        self._runtime = runtime

    @staticmethod
    def _meme_aliases(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raise ValueError("aliases must be a string or a list of strings")
        aliases: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in aliases:
                aliases.append(text)
        return aliases


    @staticmethod
    def _meme_text_fingerprint(title: str, aliases: List[str], meaning: str) -> str:
        text = meme_search.record_text(title, aliases, meaning)
        return _fingerprint(text) if text else ""

    @staticmethod
    def _meme_text_field(value: Any) -> str:
        return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @classmethod
    def _meme_text_identity(
        cls, title: str, aliases: List[str], meaning: str
    ) -> tuple[str, str | tuple[str, ...], str]:
        title = cls._meme_text_field(title)
        meaning = cls._meme_text_field(meaning)
        if title:
            return ("title", title, meaning)
        alias_set = tuple(sorted({
            cleaned for alias in aliases
            if (cleaned := cls._meme_text_field(alias))
        }))
        return ("aliases", alias_set, meaning)

    @classmethod
    def identity_fingerprint(cls, title: str, aliases: List[str], meaning: str) -> str:
        identity = cls._meme_text_identity(title, aliases, meaning)
        if not cls._meme_text_field(title) and not cls._meme_text_field(meaning) and not any(
            cls._meme_text_field(alias) for alias in aliases
        ):
            return ""
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "text-id:" + _fingerprint(encoded)

    def _meme_insert_record(
        self,
        *,
        title: str,
        meaning: str,
        aliases: List[str],
        source: str,
        source_ref: str,
        source_time: str,
        vector: Optional[bytes],
        owner_key: str,
        now: Optional[str] = None,
    ) -> Tuple[int, str]:
        now = now or _meme_now()
        cursor = self._sqlite.execute(
            "INSERT INTO meme_records "
            "(title, meaning, aliases, source, source_ref, source_time, "
            " created_at, updated_at, vec, owner_key, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title, meaning, json.dumps(aliases, ensure_ascii=False),
                source, source_ref, source_time, now, now, vector,
                owner_key, self._runtime.session_id(),
            ),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid), now

    def _meme_find_text_duplicate(
        self, title: str, aliases: List[str], meaning: str, *, owner_key: str,
        allow_legacy_blank_owner: bool = False,
    ) -> Optional[int]:
        if not self._sqlite.is_open:
            return None
        wanted = self._meme_text_identity(title, aliases, meaning)
        owner_clause = "owner_key=?"
        params: tuple[Any, ...] = (owner_key,)
        if allow_legacy_blank_owner:
            owner_clause = "(owner_key=? OR owner_key='')"
        rows = self._sqlite.fetchall(
            "SELECT id, title, aliases, meaning FROM meme_records WHERE "
            "visibility_scope='private' AND tier='common' AND " + owner_clause + " ORDER BY id",
            params,
        )
        for row in rows:
            if self._meme_text_identity(
                row["title"], meme_search.parse_aliases(row["aliases"]), row["meaning"]
            ) == wanted:
                return int(row["id"])
        return None

    def _meme_archived_deleted_fingerprints(
        self, title: str, aliases: List[str], meaning: str
    ) -> List[str]:
        wanted = self._meme_text_identity(title, aliases, meaning)
        try:
            path = self._meme_archive_root() / "memes-archived.jsonl"
            with path.open("rb") as archive:
                lines = archive.readlines()
        except OSError:
            return []

        fingerprints = set()
        for line in lines:
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or payload.get("kind") != "record_deleted":
                continue
            record = payload.get("record")
            if not isinstance(record, dict):
                continue
            archived_title = str(record.get("title") or "")
            archived_meaning = str(record.get("meaning") or "")
            archived_aliases = meme_search.parse_aliases(record.get("aliases"))
            if self._meme_text_identity(
                archived_title, archived_aliases, archived_meaning
            ) == wanted:
                fingerprint = self._meme_text_fingerprint(
                    archived_title, archived_aliases, archived_meaning
                )
                if fingerprint:
                    fingerprints.add(fingerprint)
        return sorted(fingerprints)

    def _meme_save_text_only(
        self,
        *,
        title: str,
        meaning: str,
        aliases: List[str],
        source: str,
        source_ref: str,
        source_time: str,
        text_fp: str,
        restore_deleted: bool,
        owner_key: str,
    ) -> Dict[str, Any]:
        text = meme_search.record_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text, data_dir=self._runtime.data_dir())[0])
        outcome = "forgotten"
        record_id: Optional[int] = None
        restored = False
        restored_fps: List[str] = []
        identity_fp = self.identity_fingerprint(title, aliases, meaning)
        row = None
        if self._sqlite.tx_depth > 0:
            raise RuntimeError("meme save must start its own write transaction")
        with self._sqlite.transaction():
            text_forgotten = self.is_forgotten(text_fp) or self.is_forgotten(identity_fp)
            if not (text_forgotten and not restore_deleted):
                duplicate_id = self._meme_find_text_duplicate(
                    title, aliases, meaning, owner_key=owner_key,
                    allow_legacy_blank_owner=bool(self._runtime.is_configured_private_owner()))
                if duplicate_id is not None:
                    outcome, record_id = "duplicate", duplicate_id
                else:
                    outcome = "created"
                    record_id, _ = self._meme_insert_record(
                        title=title,
                        meaning=meaning,
                        aliases=aliases,
                        source=source,
                        source_ref=source_ref,
                        source_time=source_time,
                        vector=vector,
                        owner_key=owner_key,
                    )
                if text_forgotten and restore_deleted:
                    candidates = {
                        fingerprint
                        for fingerprint in [
                            text_fp,
                            identity_fp,
                            *self._meme_archived_deleted_fingerprints(
                                title, aliases, meaning
                            ),
                        ]
                        if fingerprint
                    }
                    restored_fps = sorted(
                        fingerprint for fingerprint in candidates
                        if self.is_forgotten(fingerprint)
                    )
                    self._meme_clear_forgotten(restored_fps)
                    restored = True
            row = self._meme_record_row(record_id or 0) if record_id is not None else None
            if outcome != "forgotten" and row is None:
                raise RuntimeError("meme text save points to a missing record")
        if outcome == "forgotten":
            return {
                "stored": False,
                "reason": "forgotten",
                "message": (
                    "内容曾被整条删除；默认不写回。若确实要重新收藏/恢复，"
                    "调用 sanae_meme_save 并传 restore_deleted=true。"
                ),
                "text_forgotten": True,
            }
        if row is None:
            raise RuntimeError("meme text save points to a missing record")
        payload = self._meme_payload(row)
        if outcome == "duplicate":
            payload.update({
                "stored": False,
                "reason": "duplicate_text",
                "duplicate_record_id": record_id,
                "message": (
                    "完全相同的文字梗已存在；未新建，也未改写原记录。"
                    f"如需增补别名，请用 sanae_meme_update 明确更新记录 #{record_id}。"
                ),
            })
            if restored:
                payload["restored"] = True
                payload["restored_fingerprints"] = restored_fps
            return payload
        payload.update({"stored": True, "reason": "created"})
        if restored:
            payload["restored"] = True
            payload["restored_fingerprints"] = restored_fps
        return payload

    def is_forgotten(self, fingerprint: str) -> bool:
        if not fingerprint or not self._sqlite.is_open:
            return False
        row = self._sqlite.fetchone(
            "SELECT 1 FROM meme_forgotten WHERE fingerprint=?", (fingerprint,)
        )
        return row is not None

    def _meme_mark_forgotten(self, fingerprints: List[str]) -> None:
        now = _meme_now()
        with self._sqlite.transaction() as conn:
            for fingerprint in fingerprints:
                if fingerprint:
                    conn.execute(
                        "INSERT OR IGNORE INTO meme_forgotten(fingerprint, created_at) VALUES (?, ?)",
                        (fingerprint, now),
                    )

    def _meme_clear_forgotten(self, fingerprints: List[str]) -> None:
        with self._sqlite.transaction() as conn:
            for fingerprint in sorted({item for item in fingerprints if item}):
                conn.execute(
                    "DELETE FROM meme_forgotten WHERE fingerprint=?", (fingerprint,)
                )

    def _meme_archive_root(self) -> Path:
        root = self._runtime.archive_home() / "memory-archive"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _meme_append_archive(self, payload: Dict[str, Any]) -> None:
        path = self._meme_archive_root() / "memes-archived.jsonl"
        _append_fsynced_jsonl(path, (json.dumps(payload, ensure_ascii=False),))


    def _meme_archive(self, row: sqlite3.Row, assets: List[sqlite3.Row]) -> Dict[str, Any]:
        payload = {
            "archived_at": _meme_now(),
            "kind": "record_deleted",
            "record": {
                "id": int(row["id"]),
                "title": str(row["title"] or ""),
                "meaning": str(row["meaning"] or ""),
                "aliases": meme_search.parse_aliases(row["aliases"]),
                "source": str(row["source"] or ""),
                "source_ref": str(row["source_ref"] or ""),
                "source_time": str(row["source_time"] or ""),
                "created_at": str(row["created_at"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            },
            "assets": [
                meme_assets.copy_asset(
                    self._runtime.data_dir(), self._meme_archive_root(), int(row["id"]), asset)
                for asset in assets
            ],
        }
        self._meme_append_archive(payload)
        return payload

    def _meme_archive_asset(self, record_id: int, asset: sqlite3.Row) -> Dict[str, Any]:
        archived = meme_assets.copy_asset(
            self._runtime.data_dir(), self._meme_archive_root(), int(record_id), asset)
        self._meme_append_archive({
            "archived_at": _meme_now(),
            "kind": "asset_deleted",
            "record_id": int(record_id),
            "asset": archived,
        })
        return archived


    def _meme_record_row(self, record_id: int) -> Optional[sqlite3.Row]:
        if not self._sqlite.is_open:
            return None
        return self._sqlite.fetchone(
            "SELECT * FROM meme_records WHERE id=?", (int(record_id),)
        )

    def _meme_owned_record_row(self, record_id: int, owner_key: str) -> Optional[sqlite3.Row]:
        if not self._sqlite.is_open or not owner_key:
            return None
        allow_legacy = bool(self._runtime.is_configured_private_owner())
        # Mutation stays on the current write anchor; historical accounts are read-only.
        owner_clause = "owner_key=?"
        params: tuple[Any, ...] = (int(record_id), owner_key)
        if allow_legacy:
            owner_clause = "(owner_key=? OR owner_key='')"
        return self._sqlite.fetchone(
            "SELECT * FROM meme_records WHERE id=? AND visibility_scope='private' "
            "AND tier='common' AND " + owner_clause,
            params,
        )

    def _meme_tool_owner(self) -> str:
        from ..governance.source_context import current_chat_scope
        scope = current_chat_scope()
        owner = self._runtime.private_chat_owner() if scope.visibility == "private" else ""
        if not owner or self._runtime.association_read_owner() != owner:
            return ""
        try:
            write_owner = self._runtime.write_owner()
            return write_owner if write_owner == owner else ""
        except PermissionError:
            return ""

    def _meme_asset_rows(self, record_id: int) -> List[sqlite3.Row]:
        if not self._sqlite.is_open:
            return []
        return self._sqlite.fetchall(
            "SELECT * FROM meme_assets WHERE record_id=? ORDER BY id", (int(record_id),)
        )

    def _meme_asset_by_sha(self, sha256: str) -> Optional[sqlite3.Row]:
        if not self._sqlite.is_open:
            return None
        return self._sqlite.fetchone(
            "SELECT * FROM meme_assets WHERE sha256=?", (str(sha256),)
        )

    def _meme_asset_payload(self, row: sqlite3.Row, *, purpose: str = "original") -> Dict[str, Any]:
        original = str(meme_assets.asset_path(self._runtime.data_dir(), str(row["path"])))
        payload: Dict[str, Any] = {
            "asset_id": int(row["id"]),
            "path": original,
            "original_path": original,
            "sha256": str(row["sha256"]),
            "mime": str(row["mime"]),
            "bytes": int(row["bytes"] or 0),
            "width": int(row["width"] or 0),
            "height": int(row["height"] or 0),
        }
        if purpose == "original":
            return payload
        if purpose != "chat_sticker":
            raise ValueError(f"unsupported meme purpose: {purpose}")
        try:
            chat_path, send_width, send_height = meme_assets.chat_copy(self._runtime.data_dir(), row)
        except MemeChatCopyError as exc:
            raise MemeChatCopyError(
                f"cannot prepare chat_sticker copy for asset {int(row['id'])}: {exc}. "
                "Use purpose='original' or fix/replace the source image."
            ) from exc
        payload.update({
            "path": str(chat_path),
            "chat_path": str(chat_path),
            "send_width": int(send_width),
            "send_height": int(send_height),
            "send_bytes": int(chat_path.stat().st_size),
        })
        return payload

    def _meme_payload(
        self, row: sqlite3.Row, assets: Optional[List[sqlite3.Row]] = None, *,
        score: Optional[float] = None, match: str = "", purpose: str = "original",
    ) -> Dict[str, Any]:
        aliases = meme_search.parse_aliases(row["aliases"])
        payload: Dict[str, Any] = {
            "record_id": int(row["id"]),
            "title": str(row["title"] or ""),
            "meaning": str(row["meaning"] or ""),
            "aliases": aliases,
            "source": str(row["source"] or ""),
            "source_ref": str(row["source_ref"] or ""),
            "source_time": str(row["source_time"] or ""),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
            "purpose": purpose,
            "assets": [
                self._meme_asset_payload(asset, purpose=purpose)
                for asset in (assets if assets is not None else self._meme_asset_rows(row["id"]))
            ],
        }
        if score is not None:
            payload["score"] = round(float(score), 4)
            payload["match"] = match
        return payload


    def save(self, args: Dict[str, Any]) -> Dict[str, Any]:
        owner_key = self._meme_tool_owner()
        if not owner_key:
            raise PermissionError("meme save requires the verified private owner")
        title = str(args.get("title") or "").strip()
        meaning = str(args.get("meaning") or "").strip()
        aliases = self._meme_aliases(args.get("aliases"))
        raw_images = args.get("images")
        if raw_images is None:
            raw_images = []
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        if not isinstance(raw_images, (list, tuple)):
            raise ValueError("images must be a list of paths or URLs")
        if not meaning:
            raise ValueError(
                "meaning is required: give the full explanation, including a visual "
                "description/use for an ordinary image"
            )

        source = str(args.get("source") or "unknown").strip() or "unknown"
        source_ref = str(args.get("source_ref") or "").strip()
        source_time = str(args.get("source_time") or "").strip() or _meme_now()

        prepared: List[Dict[str, Any]] = []
        for item in raw_images:
            src = str(item or "").strip()
            if not src:
                continue
            data = meme_assets.read_source(src)
            meta = meme_assets.validate_image(data)
            sha256 = hashlib.sha256(data).hexdigest()
            if any(entry["sha256"] == sha256 for entry in prepared):
                continue
            prepared.append(
                {"data": data, "sha256": sha256, **meta, "source_ref": src}
            )

        restore_deleted = args.get("restore_deleted") is True
        forgotten_images = [
            entry for entry in prepared
            if self.is_forgotten(f"image:{entry['sha256']}")
        ]
        text_fp = self._meme_text_fingerprint(title, aliases, meaning)
        if not prepared:
            return self._meme_save_text_only(
                title=title,
                meaning=meaning,
                aliases=aliases,
                source=source,
                source_ref=source_ref,
                source_time=source_time,
                text_fp=text_fp,
                restore_deleted=restore_deleted,
                owner_key=owner_key,
            )
        text_forgotten = bool(text_fp and self.is_forgotten(text_fp))
        if (forgotten_images or text_forgotten) and not restore_deleted:
            payload = {
                "stored": False,
                "reason": "forgotten",
                "message": (
                    "内容曾被整条删除；默认不写回。若确实要重新收藏/恢复，"
                    "调用 sanae_meme_save 并传 restore_deleted=true。"
                ),
            }
            if forgotten_images:
                payload["sha256"] = forgotten_images[0]["sha256"]
            if text_forgotten:
                payload["text_forgotten"] = True
            return payload

        duplicate_rows: List[Dict[str, Any]] = []
        new_prepared: List[Dict[str, Any]] = []
        for entry in prepared:
            existing_asset = self._meme_asset_by_sha(entry["sha256"])
            if existing_asset is None:
                new_prepared.append(entry)
                continue
            row = self._meme_record_row(int(existing_asset["record_id"]))
            if row is None:
                raise RuntimeError("meme asset points to a missing record")
            if not (row["visibility_scope"] == "private" and row["tier"] == "common"
                    and (row["owner_key"] == owner_key or
                         (not row["owner_key"] and self._runtime.is_configured_private_owner()))):
                return {"stored": False, "reason": "asset_in_other_scope"}
            duplicate_rows.append({
                "asset_id": int(existing_asset["id"]),
                "sha256": str(entry["sha256"]),
                "record_id": int(row["id"]),
                "title": str(row["title"] or ""),
                "meaning": str(row["meaning"] or ""),
            })

        duplicate_records: List[Dict[str, Any]] = []
        seen_records: set[int] = set()
        for item in duplicate_rows:
            if item["record_id"] in seen_records:
                continue
            seen_records.add(item["record_id"])
            duplicate_records.append({
                "record_id": item["record_id"],
                "title": item["title"],
                "meaning": item["meaning"],
            })

        if not new_prepared and duplicate_rows:
            primary_id = duplicate_rows[0]["record_id"]
            row = self._meme_record_row(primary_id)
            if row is None:
                raise RuntimeError("meme asset points to a missing record")
            payload = self._meme_payload(row)
            payload.update({
                "stored": False,
                "reason": "duplicate_image",
                "duplicate_record_id": primary_id,
                "duplicates": duplicate_rows,
                "duplicate_records": duplicate_records,
                "message": "全部图片已在梗与表情记忆区；未新建记录，也未改写原有解释。",
            })
            if text_forgotten:
                payload["text_forgotten"] = True
            return payload

        now = _meme_now()
        text = meme_search.record_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text, data_dir=self._runtime.data_dir())[0])
        forgotten_fps = {f"image:{entry['sha256']}" for entry in forgotten_images}
        restored_fps = [
            f"image:{entry['sha256']}" for entry in new_prepared
            if f"image:{entry['sha256']}" in forgotten_fps
        ]
        if text_forgotten:
            restored_fps.append(text_fp)
        created_files: List[str] = []
        try:
            with self._sqlite.transaction() as conn:
                record_id, now = self._meme_insert_record(
                    title=title,
                    meaning=meaning,
                    aliases=aliases,
                    source=source,
                    source_ref=source_ref,
                    source_time=source_time,
                    vector=vector,
                    owner_key=owner_key,
                    now=now,
                )
                for entry in new_prepared:
                    rel, was_created = meme_assets.store_file(
                        meme_assets.asset_root(self._runtime.data_dir()), entry["data"],
                        entry["sha256"], entry["ext"])
                    if was_created:
                        created_files.append(rel)
                    conn.execute(
                        "INSERT INTO meme_assets "
                        "(record_id, path, sha256, mime, bytes, width, height, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            record_id, rel, entry["sha256"], entry["mime"],
                            len(entry["data"]), entry["width"], entry["height"], now,
                        ),
                    )
                if restored_fps:
                    self._meme_clear_forgotten(restored_fps)
        except Exception:
            for rel in created_files:
                meme_assets.remove_file(self._runtime.data_dir(), rel)
            raise

        row = self._meme_record_row(record_id)
        payload = self._meme_payload(row) if row is not None else {}
        payload.update({"stored": True, "reason": "created"})
        if duplicate_rows:
            payload["duplicates"] = duplicate_rows
            payload["duplicate_records"] = duplicate_records
            payload["duplicate_record_id"] = duplicate_records[0]["record_id"]
        if restored_fps:
            payload["restored"] = True
            payload["restored_fingerprints"] = restored_fps
        return payload

    def update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        owner_key = self._meme_tool_owner()
        if not owner_key:
            raise PermissionError("meme update requires the verified private owner")
        record_id = int(args.get("record_id") or 0)
        row = self._meme_owned_record_row(record_id, owner_key)
        if row is None:
            raise ValueError("meme record not found or not editable in this private scope")

        title = str(row["title"] or "")
        meaning = str(row["meaning"] or "")
        aliases = meme_search.parse_aliases(row["aliases"])
        if "title" in args:
            title = str(args.get("title") or "").strip()
        if "meaning" in args:
            updated_meaning = str(args.get("meaning") or "").strip()
            if not updated_meaning:
                raise ValueError("meaning cannot be cleared; omit it to keep the current explanation")
            meaning = updated_meaning
        if "aliases" in args:
            aliases = self._meme_aliases(args.get("aliases"))

        remove_ids: List[int] = []
        raw_remove = args.get("remove_asset_ids") or []
        if not isinstance(raw_remove, (list, tuple)):
            raise ValueError("remove_asset_ids must be a list")
        for value in raw_remove:
            asset_id = int(value)
            asset = self._sqlite.fetchone(
                "SELECT * FROM meme_assets WHERE id=? AND record_id=?",
                (asset_id, record_id),
            )
            if asset is None:
                raise ValueError(f"asset {asset_id} does not belong to record {record_id}")
            remove_ids.append(asset_id)

        raw_add = args.get("add_images") or []
        if isinstance(raw_add, str):
            raw_add = [raw_add]
        if not isinstance(raw_add, (list, tuple)):
            raise ValueError("add_images must be a list of paths or URLs")

        prepared: List[Dict[str, Any]] = []
        forgotten_images: List[str] = []
        for item in raw_add:
            src = str(item or "").strip()
            if not src:
                continue
            data = meme_assets.read_source(src)
            meta = meme_assets.validate_image(data)
            sha256 = hashlib.sha256(data).hexdigest()
            if any(entry["sha256"] == sha256 for entry in prepared):
                continue
            if self.is_forgotten(f"image:{sha256}"):
                forgotten_images.append(sha256)
                continue
            existing_asset = self._meme_asset_by_sha(sha256)
            if existing_asset is not None:
                if int(existing_asset["record_id"]) == record_id:
                    continue
                if int(existing_asset["id"]) in remove_ids:
                    pass
                else:
                    raise ValueError("image already belongs to another meme record")
            prepared.append(
                {"data": data, "sha256": sha256, **meta, "source_ref": src}
            )
        if forgotten_images:
            raise ValueError(
                "image was deleted and is tombstoned; update does not silently restore it. "
                "Use sanae_meme_save with restore_deleted=true and the original image + explanation."
            )

        source = str(row["source"] or "")
        source_ref = str(row["source_ref"] or "")
        source_time = str(row["source_time"] or "")
        if "source" in args:
            source = str(args.get("source") or "").strip()
        if "source_ref" in args:
            source_ref = str(args.get("source_ref") or "").strip()
        if "source_time" in args:
            source_time = str(args.get("source_time") or "").strip()

        now = _meme_now()
        text = meme_search.record_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text, data_dir=self._runtime.data_dir())[0])
        removed_rels: List[str] = []
        removed_assets: List[sqlite3.Row] = []
        archived_assets: List[str] = []
        created_files: List[str] = []
        try:
            with self._sqlite.transaction() as conn:
                allow_legacy = bool(self._runtime.is_configured_private_owner())
                owner_clause = "owner_key=?"
                owner_params: tuple[Any, ...] = (owner_key,)
                if allow_legacy:
                    owner_clause = "(owner_key=? OR owner_key='')"
                changed = conn.execute(
                    "UPDATE meme_records SET title=?, meaning=?, aliases=?, source=?, "
                    "source_ref=?, source_time=?, updated_at=?, vec=? WHERE id=? "
                    "AND visibility_scope='private' AND tier='common' AND " + owner_clause,
                    (
                        title, meaning, json.dumps(aliases, ensure_ascii=False),
                        source, source_ref, source_time, now, vector, record_id, *owner_params,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError("meme record not found or not editable in this private scope")
                for asset_id in remove_ids:
                    asset = conn.execute(
                        "SELECT * FROM meme_assets WHERE id=? AND record_id=?",
                        (asset_id, record_id),
                    ).fetchone()
                    if asset is not None:
                        archived = self._meme_archive_asset(record_id, asset)
                        archived_assets.append(archived["archived_path"])
                        removed_rels.append(str(asset["path"]))
                        removed_assets.append(asset)
                        conn.execute(
                            "DELETE FROM meme_assets WHERE id=? AND record_id=?",
                            (asset_id, record_id),
                        )
                for entry in prepared:
                    rel, was_created = meme_assets.store_file(
                        meme_assets.asset_root(self._runtime.data_dir()), entry["data"],
                        entry["sha256"], entry["ext"])
                    if was_created:
                        created_files.append(rel)
                    conn.execute(
                        "INSERT INTO meme_assets "
                        "(record_id, path, sha256, mime, bytes, width, height, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            record_id, rel, entry["sha256"], entry["mime"],
                            len(entry["data"]), entry["width"], entry["height"], now,
                        ),
                    )
        except Exception:
            for rel in created_files:
                meme_assets.remove_file(self._runtime.data_dir(), rel)
            raise

        for rel in removed_rels:
            meme_assets.remove_file(self._runtime.data_dir(), rel)
        for asset in removed_assets:
            meme_assets.remove_chat_copy(self._runtime.data_dir(), asset)
        row = self._meme_record_row(record_id)
        payload = self._meme_payload(row) if row is not None else {}
        payload.update({
            "stored": True,
            "reason": "updated",
            "archived_assets": archived_assets,
        })
        return payload

    def delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        owner_key = self._meme_tool_owner()
        if not owner_key:
            raise PermissionError("meme delete requires the verified private owner")
        asset_id = args.get("asset_id")
        record_id = args.get("record_id")
        if asset_id not in (None, ""):
            asset = self._sqlite.fetchone(
                "SELECT a.* FROM meme_assets a JOIN meme_records r ON r.id=a.record_id "
                "WHERE a.id=? AND r.visibility_scope='private' AND r.tier='common' "
                "AND (r.owner_key=? OR (? AND r.owner_key=''))",
                (int(asset_id), owner_key, int(bool(self._runtime.is_configured_private_owner()))),
            )
            if asset is None:
                raise ValueError("meme asset not found in this private scope")
            if record_id not in (None, "") and int(record_id) != int(asset["record_id"]):
                raise ValueError("asset does not belong to the given record")
            rel = str(asset["path"])
            archived = self._meme_archive_asset(int(asset["record_id"]), asset)
            self._sqlite.execute(
                "DELETE FROM meme_assets WHERE id=? AND record_id=? "
                "AND EXISTS (SELECT 1 FROM meme_records r WHERE r.id=? "
                "AND r.visibility_scope='private' AND r.tier='common' "
                "AND (r.owner_key=? OR (? AND r.owner_key='')))",
                (int(asset_id), int(asset["record_id"]), int(asset["record_id"]),
                 owner_key, int(bool(self._runtime.is_configured_private_owner()))),
            )
            meme_assets.remove_chat_copy(self._runtime.data_dir(), asset)
            meme_assets.remove_file(self._runtime.data_dir(), rel)
            return {
                "deleted_asset": int(asset_id),
                "record_id": int(asset["record_id"]),
                "archived_path": archived["archived_path"],
                "reason": "deleted_asset",
            }

        if record_id in (None, ""):
            raise ValueError("record_id or asset_id is required")
        record_id = int(record_id)
        row = self._meme_owned_record_row(record_id, owner_key)
        if row is None:
            raise ValueError("meme record not found or not deletable in this private scope")
        assets = self._meme_asset_rows(record_id)
        archived = self._meme_archive(row, assets)
        fingerprints = [
            self._meme_text_fingerprint(
                str(row["title"] or ""),
                meme_search.parse_aliases(row["aliases"]),
                str(row["meaning"] or ""),
            )
        ]
        if not assets:
            fingerprints.append(
                self.identity_fingerprint(
                    str(row["title"] or ""),
                    meme_search.parse_aliases(row["aliases"]),
                    str(row["meaning"] or ""),
                )
            )
        fingerprints.extend(f"image:{asset['sha256']}" for asset in assets)
        with self._sqlite.transaction() as conn:
            self._meme_mark_forgotten(fingerprints)
            owner_clause = "owner_key=?"
            owner_params: tuple[Any, ...] = (owner_key,)
            if self._runtime.is_configured_private_owner():
                owner_clause = "(owner_key=? OR owner_key='')"
            changed = conn.execute(
                "DELETE FROM meme_records WHERE id=? AND visibility_scope='private' "
                "AND tier='common' AND " + owner_clause,
                (record_id, *owner_params),
            )
            if changed.rowcount != 1:
                raise ValueError("meme record not found or not deletable in this private scope")
        for asset in assets:
            meme_assets.remove_chat_copy(self._runtime.data_dir(), asset)
            meme_assets.remove_file(self._runtime.data_dir(), str(asset["path"]))
        return {
            "deleted_record": record_id,
            "deleted_assets": len(assets),
            "archived_assets": [asset["archived_path"] for asset in archived["assets"]],
            "reason": "deleted_record",
        }


    def available_sticker_summary(self) -> str:
        from ..governance.source_context import current_chat_scope, in_public_interaction
        scope = current_chat_scope()
        owner = self._runtime.private_chat_owner() if scope.visibility == "private" else ""
        if not owner or self._runtime.association_read_owner() != owner:
            owner = ""
        return self._sqlite.read_optional(
            lambda conn: meme_context.available_sticker_summary(
                conn, shadow=self._runtime.shadow_enabled(), public=in_public_interaction(),
                owner_key=owner, owner_keys=self._runtime.read_owner_keys(owner) if owner else (),
                allow_legacy_blank_owner=bool(owner and self._runtime.is_configured_private_owner()),
            )
        )

    def known_meme_terms(self, parts: List[str]) -> List[str]:
        return self._sqlite.read_optional(
            lambda conn: meme_context.known_meme_terms(conn, parts, shadow=self._runtime.shadow_enabled())
        )

    def public_meme_senses(self, parts: List[str]) -> List[Dict[str, Any]]:
        return self._sqlite.read_optional(
            lambda conn: meme_context.public_meme_senses(conn, parts, shadow=self._runtime.shadow_enabled())
        )

    def meme_context_for_public_material(self, parts: List[str], *, public_only: bool = False) -> str:
        from ..governance.source_context import current_chat_scope, in_public_interaction
        scope = current_chat_scope()
        owner = self._runtime.private_chat_owner() if scope.visibility == "private" else ""
        if owner and self._runtime.association_read_owner() != owner:
            owner = ""
        return self._sqlite.read_optional(
            lambda conn: meme_context.context_for_public_material(
                conn, parts, shadow=self._runtime.shadow_enabled(),
                public=in_public_interaction(), public_only=public_only,
                owner_key=owner, owner_keys=self._runtime.read_owner_keys(owner) if owner else (),
                allow_legacy_blank_owner=bool(owner and self._runtime.is_configured_private_owner()),
            )
        )

    def meme_context_for_turn(self, user_message: Any, *, known_context: Any = None) -> str:
        from ..governance.source_context import current_chat_scope, in_public_interaction
        scope = current_chat_scope()
        owner = self._runtime.private_chat_owner() if scope.visibility == "private" else ""
        if owner and self._runtime.association_read_owner() != owner:
            owner = ""
        return self._sqlite.read_optional(
            lambda conn: meme_context.context_for_turn(
                conn, user_message, known_context=known_context,
                shadow=self._runtime.shadow_enabled(), public=in_public_interaction(),
                owner_key=owner, owner_keys=self._runtime.read_owner_keys(owner) if owner else (),
                allow_legacy_blank_owner=bool(owner and self._runtime.is_configured_private_owner()),
            )
        )

    def search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._runtime.shadow_enabled() or not self._sqlite.is_open:
            return []
        query = str(args.get("query") or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 5))))
        purpose = str(args.get("purpose") or "chat_sticker").strip().lower()
        if purpose not in {"chat_sticker", "original"}:
            raise ValueError("purpose must be 'chat_sticker' or 'original'")
        from ..governance.source_context import current_chat_scope
        scope = current_chat_scope()
        owner = self._runtime.private_chat_owner() if scope.visibility == "private" else ""
        if owner and self._runtime.association_read_owner() != owner:
            owner = ""
        if owner:
            owners = self._runtime.read_owner_keys(owner)
            owner_clause = "owner_key IN (%s)" % ",".join("?" for _ in owners)
            if self._runtime.is_configured_private_owner():
                owner_clause = f"({owner_clause} OR owner_key='')"
            sql = (
                "SELECT * FROM meme_records WHERE "
                "(visibility_scope='public' AND tier='eye_familiar' AND evidence_status='confirmed') "
                "OR (visibility_scope='private' AND tier='common' AND " + owner_clause + ")"
            )
            params: Tuple[Any, ...] = tuple(owners)
        else:
            sql = (
                "SELECT * FROM meme_records WHERE visibility_scope='public' "
                "AND tier='eye_familiar' AND evidence_status='confirmed'"
            )
            params = ()

        def _snapshot(conn: sqlite3.Connection) -> Tuple[List[sqlite3.Row], set[int]]:
            snapshot_rows = conn.execute(sql, params).fetchall()
            fts_hits = meme_search.fts_ids(conn, query) if query else set()
            return snapshot_rows, fts_hits

        rows, matched_fts_ids = self._sqlite.read(_snapshot)
        ranked = meme_search.rank_records(
            None, query, top_k,
            embed_text=functools.partial(_embed_text, data_dir=self._runtime.data_dir()),
            tokens=_tokens,
            cosine=_cosine, unpack_vector=_unpack_vector,
            rows=rows, matched_fts_ids=matched_fts_ids,
        )
        return [
            self._meme_payload(
                item["row"], score=item["score"], match=item["match"], purpose=purpose,
            )
            for item in ranked
        ]
