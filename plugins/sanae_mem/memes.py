"""她的梗与表情记忆区。

独立于普通 ``events``：文字梗/黑话、解释、别名、来源与 0..n 个素材文件存在
``meme_records`` / ``meme_assets``。这里只复用 ``textutil`` 的文字嵌入与 SQLite/FTS，
不建图片向量，也不把梗写进普通事件流。

设计边界：
- 保存是显式动作（模型工具或用户要求），不自动把聊天图片收进来。
- 解释必须非空；普通图片也要写视觉描述和用途，不为了凑梗强行编造含义。
- 完全相同图按原始字节 SHA-256 去重；同图再次保存返回原记录。
- 删除先把原图字节复制到 ``memory-archive/memes/<record_id>/``，再删行/文件；
  整条删除另写 ``meme_forgotten`` 防止同内容被自动写回。单图删除同样归档原文件。
- 素材文件放在 ``<data_dir>/memes/``，不引用入站 cache 或临时 URL。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sqlite3
import io
import json
import logging
import math
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .textutil import (
        _cosine,
        _embed_text,
        _fingerprint,
        _pack_vector,
        _tokens,
        _unpack_vector,
        provider_home,
    )
except ImportError:
    from textutil import (
        _cosine,
        _embed_text,
        _fingerprint,
        _pack_vector,
        _tokens,
        _unpack_vector,
        provider_home,
    )

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

logger = logging.getLogger(__name__)

CHAT_STICKER_MAX_EDGE = 300
CHAT_STICKER_COPY_VERSION = "v1"
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_FORMAT_TO_MIME = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "GIF": ("image/gif", ".gif"),
    "WEBP": ("image/webp", ".webp"),
    "BMP": ("image/bmp", ".bmp"),
}
_MEME_RECENCY_DAYS = 180.0
_IMAGE_HINT_RE = re.compile(r"\[Image attached at:\s*([^\]]+)\]")
_VISION_IMAGE_URL_RE = re.compile(
    r"^\[[^\r\n]*\bvision_analyze\b[^\r\n]*\bimage_url:\s*([^\r\n]+?)\s*(?:~\s*)?\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class MemeChatCopyError(RuntimeError):
    """A chat-sticker send copy cannot be prepared without violating fidelity."""


def known_inbound_images(paths: list[str]) -> dict[str, str]:
    """Resolve exact known images without opening a writable store or invoking a model."""
    from contextlib import closing
    from hermes_constants import get_hermes_home

    try:
        from .source_context import in_public_interaction
        from .textutil import _bool, _load_plugin_config
    except ImportError:
        from source_context import in_public_interaction
        from textutil import _bool, _load_plugin_config
    config = _load_plugin_config()
    if in_public_interaction() or _bool(config.get("shadow"), False):
        return {}
    root = Path(config.get("data_dir") or get_hermes_home() / "sanae-mem").expanduser()
    db = root / "sanae.db"
    if not db.is_file():
        return {}
    notes = {}
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        for value in paths:
            path = Path(value)
            if not path.is_absolute() or not path.is_file():
                continue
            try:
                with path.open("rb") as image:
                    digest = hashlib.file_digest(image, "sha256").hexdigest()
            except OSError:
                logger.debug("sanae_mem: inbound meme image unavailable", exc_info=True)
                continue
            row = conn.execute(
                "SELECT r.title, r.meaning, r.id FROM meme_assets a JOIN meme_records r ON r.id=a.record_id "
                "WHERE a.sha256=?", (digest,),
            ).fetchone()
            if row is not None:
                title = " ".join(str(row[0] or "未命名").split())
                meaning = " ".join(str(row[1] or "只收藏了这张素材，未记文字解释").split())
                notes[value] = f"[已知梗图] 我收过的记录#{row[2]}「{title}」：{meaning}"
    return notes


def _meme_now() -> str:
    return datetime.now().astimezone().isoformat()


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class MemesMixin:
    """Mixin for SanaeMemProvider: independent meme/expression memory zone."""


    def _meme_root(self) -> Path:
        root = Path(self._data_dir) / "memes"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _meme_abs(self, rel: str) -> Path:
        root = self._meme_root().resolve()
        target = (root / str(rel or "")).resolve()
        if target != root and root not in target.parents:
            raise ValueError("meme asset path escapes the meme root")
        return target

    def _meme_rel(self, path: Path) -> str:
        root = self._meme_root().resolve()
        target = Path(path).resolve()
        if target != root and root not in target.parents:
            raise ValueError("meme asset path escapes the meme root")
        return target.relative_to(root).as_posix()

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
    def _meme_norm(value: Any) -> str:
        text = str(value or "").casefold()
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)

    @staticmethod
    def _meme_text(title: str, aliases: List[str], meaning: str) -> str:
        return "\n".join(part for part in (title, "、".join(aliases), meaning) if part).strip()

    @staticmethod
    def _meme_text_fingerprint(title: str, aliases: List[str], meaning: str) -> str:
        text = MemesMixin._meme_text(title, aliases, meaning)
        return _fingerprint(text) if text else ""

    @staticmethod
    def _meme_text_field(value: Any) -> str:
        """Only trim edges and unify newlines; keep case, punctuation and inner space."""
        return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @classmethod
    def _meme_text_identity(cls, title: str, aliases: List[str], meaning: str) -> Tuple[Any, ...]:
        """Exact text identity for a save with no images.

        A non-empty title owns the identity; aliases never split it. An empty title
        uses the order-independent alias set. Meaning is always complete text, not
        a lossy normalized form.
        """
        title = cls._meme_text_field(title)
        meaning = cls._meme_text_field(meaning)
        if title:
            return ("title", title, meaning)
        alias_set = tuple(sorted({
            cleaned for alias in aliases
            if (cleaned := cls._meme_text_field(alias))
        }))
        return ("aliases", alias_set, meaning)

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
        """Insert one live record; callers own the surrounding transaction."""
        now = now or _meme_now()
        cursor = self._conn.execute(
            "INSERT INTO meme_records "
            "(title, meaning, aliases, source, source_ref, source_time, "
            " created_at, updated_at, vec, owner_key, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title, meaning, json.dumps(aliases, ensure_ascii=False),
                source, source_ref, source_time, now, now, vector,
                owner_key, str(getattr(self, "_session_id", "") or ""),
            ),
        )
        return int(cursor.lastrowid), now

    def _meme_find_text_duplicate(
        self, title: str, aliases: List[str], meaning: str
    ) -> Optional[int]:
        """Earliest live record with the same exact text identity, across all owners."""
        if self._conn is None:
            return None
        wanted = self._meme_text_identity(title, aliases, meaning)
        rows = self._conn.execute(
            "SELECT id, title, aliases, meaning FROM meme_records ORDER BY id"
        ).fetchall()
        for row in rows:
            if self._meme_text_identity(
                row["title"], self._meme_alias_list(row["aliases"]), row["meaning"]
            ) == wanted:
                return int(row["id"])
        return None

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
    ) -> Dict[str, Any]:
        """Save a text-only meme inside a write transaction.

        The tombstone and live-record checks are repeated after the write lock is
        acquired. A stale lock-external check cannot authorize an insert or a
        restore.
        """
        text = self._meme_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text)[0])
        owner_key = self._write_owner()
        outcome = "forgotten"
        record_id: Optional[int] = None
        restored = False
        with self._lock:
            if self._conn.in_transaction:
                raise RuntimeError("meme save must start its own write transaction")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                text_forgotten = self._meme_is_forgotten(text_fp)
                if not (text_forgotten and not restore_deleted):
                    duplicate_id = self._meme_find_text_duplicate(
                        title, aliases, meaning)
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
                            self._meme_clear_forgotten([text_fp])
                            restored = True
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
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
        row = self._meme_record_row(record_id or 0)
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
            return payload
        payload.update({"stored": True, "reason": "created"})
        if restored:
            payload["restored"] = True
            payload["restored_fingerprints"] = [text_fp]
        return payload

    def _meme_is_forgotten(self, fingerprint: str) -> bool:
        if not fingerprint or self._conn is None:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM meme_forgotten WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return row is not None

    def _meme_mark_forgotten(self, fingerprints: List[str]) -> None:
        now = _meme_now()
        for fingerprint in fingerprints:
            if fingerprint:
                self._conn.execute(
                    "INSERT OR IGNORE INTO meme_forgotten(fingerprint, created_at) VALUES (?, ?)",
                    (fingerprint, now),
                )

    def _meme_clear_forgotten(self, fingerprints: List[str]) -> None:
        for fingerprint in sorted({item for item in fingerprints if item}):
            self._conn.execute(
                "DELETE FROM meme_forgotten WHERE fingerprint=?", (fingerprint,)
            )

    def _meme_archive_root(self) -> Path:
        root = Path(provider_home(self)) / "memory-archive"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _meme_append_archive(self, payload: Dict[str, Any]) -> None:
        path = self._meme_archive_root() / "memes-archived.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _meme_copy_asset(self, record_id: int, asset: Any) -> Dict[str, Any]:
        source = self._meme_abs(str(asset["path"]))
        if not source.is_file():
            raise FileNotFoundError(f"meme asset missing from disk: {source}")
        target_dir = self._meme_archive_root() / "memes" / str(int(record_id))
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".bin"
        target = target_dir / f"{int(asset['id'])}-{str(asset['sha256'])[:12]}{suffix}"
        shutil.copy2(source, target)
        return {
            "id": int(asset["id"]),
            "path": str(asset["path"] or ""),
            "sha256": str(asset["sha256"] or ""),
            "mime": str(asset["mime"] or ""),
            "bytes": int(asset["bytes"] or 0),
            "width": int(asset["width"] or 0),
            "height": int(asset["height"] or 0),
            "archived_path": str(target),
        }

    def _meme_archive(self, row: Any, assets: List[Any]) -> Dict[str, Any]:
        payload = {
            "archived_at": _meme_now(),
            "kind": "record_deleted",
            "record": {
                "id": int(row["id"]),
                "title": str(row["title"] or ""),
                "meaning": str(row["meaning"] or ""),
                "aliases": self._meme_alias_list(row["aliases"]),
                "source": str(row["source"] or ""),
                "source_ref": str(row["source_ref"] or ""),
                "source_time": str(row["source_time"] or ""),
                "created_at": str(row["created_at"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            },
            "assets": [self._meme_copy_asset(int(row["id"]), asset) for asset in assets],
        }
        self._meme_append_archive(payload)
        return payload

    def _meme_archive_asset(self, record_id: int, asset: Any) -> Dict[str, Any]:
        archived = self._meme_copy_asset(int(record_id), asset)
        self._meme_append_archive({
            "archived_at": _meme_now(),
            "kind": "asset_deleted",
            "record_id": int(record_id),
            "asset": archived,
        })
        return archived

    @staticmethod
    def _meme_alias_list(value: Any) -> List[str]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value or "[]")
            except Exception:
                return []
            return [str(item).strip() for item in parsed if str(item).strip()] if isinstance(parsed, list) else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


    def _meme_read_source(self, src: str) -> bytes:
        text = str(src or "").strip()
        if not text:
            raise ValueError("image source is empty")
        if text.startswith("data:"):
            return self._meme_decode_data_url(text)
        if text.startswith(("http://", "https://")):
            try:
                from tools.image_source import ResolveContext, resolve_image_source
            except Exception as exc:
                raise RuntimeError(
                    "URL image import requires the Hermes runtime (tools.image_source)"
                ) from exc
            resolved = _run_coro_sync(resolve_image_source(text, ResolveContext()))
            return bytes(resolved.data)
        if text.lower().startswith("file://"):
            text = text[7:]
        path = Path(os.path.expanduser(text))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _meme_decode_data_url(value: str) -> bytes:
        header, _, payload = str(value).partition(",")
        if ";base64" not in header:
            raise ValueError("image data URL must be base64-encoded")
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError(f"invalid image data URL: {exc}") from exc

    @staticmethod
    def _meme_validate_image(data: bytes) -> Dict[str, Any]:
        if not data:
            raise ValueError("image is empty")
        if Image is None:
            raise RuntimeError("Pillow is required for meme image storage")
        try:
            with Image.open(io.BytesIO(data)) as image:
                fmt = str(image.format or "").upper()
                if fmt not in _FORMAT_TO_MIME:
                    raise ValueError(f"unsupported image format: {fmt or 'unknown'}")
                image.load()
                width, height = image.size
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"cannot decode image: {exc}") from exc
        mime, ext = _FORMAT_TO_MIME[fmt]
        return {"mime": mime, "ext": ext, "width": int(width), "height": int(height)}

    def _meme_store_file(self, data: bytes, sha256: str, ext: str) -> Tuple[str, bool]:
        root = self._meme_root()
        rel = f"{sha256[:2]}/{sha256}{ext}"
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return rel, False
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return rel, True

    def _meme_remove_file(self, rel: str) -> None:
        try:
            path = self._meme_abs(rel)
        except ValueError:
            return
        try:
            path.unlink(missing_ok=True)
            if path.parent != self._meme_root().resolve():
                path.parent.rmdir()
        except OSError:
            logger.debug("sanae_mem: meme asset cleanup skipped for %s", rel)


    def _meme_chat_root(self) -> Path:
        return Path(self._data_dir) / "memes" / "chat"

    @staticmethod
    def _meme_chat_size(width: int, height: int) -> tuple[int, int]:
        longest = max(int(width), int(height))
        if longest <= CHAT_STICKER_MAX_EDGE:
            return int(width), int(height)
        scale = CHAT_STICKER_MAX_EDGE / longest
        return max(1, round(int(width) * scale)), max(1, round(int(height) * scale))

    def _meme_chat_target(self, row: Any) -> Path:
        suffix = Path(str(row["path"] or "")).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            suffix = ".jpg"
        if suffix not in _IMAGE_EXTS:
            raise MemeChatCopyError(f"unsupported chat copy image type: {suffix or 'unknown'}")
        name = (
            f"{row['sha256']}_{CHAT_STICKER_COPY_VERSION}_"
            f"{CHAT_STICKER_MAX_EDGE}{suffix}"
        )
        return self._meme_chat_root() / name

    def _meme_remove_chat_copy(self, row: Any) -> None:
        try:
            target = self._meme_chat_target(row)
        except (MemeChatCopyError, ValueError, OSError):
            return
        try:
            target.unlink(missing_ok=True)
            if target.parent != self._meme_chat_root().resolve():
                target.parent.rmdir()
        except OSError:
            logger.debug("sanae_mem: chat copy cleanup skipped for %s", target)

    @staticmethod
    def _meme_save_static_copy(source: Path, target: Path, fmt: str) -> None:
        if Image is None or ImageOps is None:
            raise MemeChatCopyError("Pillow is unavailable")
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                target_size = MemesMixin._meme_chat_size(width, height)
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                if fmt == "JPEG":
                    image.convert("RGB").save(target, format="JPEG", quality=90, optimize=True)
                elif fmt == "PNG":
                    image.save(target, format="PNG")
                elif fmt == "WEBP":
                    image.save(target, format="WEBP", quality=90)
                elif fmt == "BMP":
                    image.save(target, format="BMP")
                else:
                    raise MemeChatCopyError(f"unsupported chat copy format: {fmt}")
        except MemeChatCopyError:
            raise
        except Exception as exc:
            raise MemeChatCopyError(f"static chat copy failed: {exc}") from exc

    def _meme_save_gif_copy(self, source: Path, target: Path) -> None:
        if Image is None:
            raise MemeChatCopyError("Pillow is unavailable")
        try:
            with Image.open(source) as image:
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                loop = image.info.get("loop")
                target_size = MemesMixin._meme_chat_size(*image.size)
                durations: List[int] = []
                frames: List[Any] = []
                for index in range(frame_count):
                    image.seek(index)
                    durations.append(int(image.info.get("duration", 0) or 0))
                    frame = image.convert("RGBA")
                    frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                    alpha = frame.getchannel("A").point(
                        lambda value: 255 if value >= 128 else 0)
                    frame.putalpha(alpha)
                    frames.append(frame)
                if not frames:
                    raise MemeChatCopyError("GIF has no decodable frames")
                save_kwargs: Dict[str, Any] = {
                    "save_all": True,
                    "append_images": frames[1:],
                    "duration": durations,
                    "disposal": 2,
                    "optimize": False,
                }
                if loop is not None:
                    save_kwargs["loop"] = loop
                frames[0].save(target, format="GIF", **save_kwargs)

            with Image.open(target) as check:
                saved_frame_count = int(getattr(check, "n_frames", 1) or 1)
                if saved_frame_count < 1 or saved_frame_count != len(frames):
                    raise MemeChatCopyError("GIF chat copy lost frames")
                if max(check.size) > CHAT_STICKER_MAX_EDGE:
                    raise MemeChatCopyError("GIF chat copy exceeds sticker size limit")
        except MemeChatCopyError:
            raise
        except Exception as exc:
            raise MemeChatCopyError(f"GIF chat copy failed: {exc}") from exc

    def _meme_chat_copy(self, row: Any) -> tuple[Path, int, int]:
        source = self._meme_abs(str(row["path"]))
        if not source.is_file():
            raise MemeChatCopyError(f"chat copy source missing: {source}")
        try:
            with Image.open(source) as image:
                fmt = str(image.format or "").upper()
                is_gif = fmt == "GIF" or source.suffix.lower() == ".gif"
                if is_gif:
                    width, height = image.size
                elif ImageOps is not None:
                    width, height = ImageOps.exif_transpose(image).size
                else:
                    width, height = image.size
        except Exception as exc:
            raise MemeChatCopyError(f"chat copy source unreadable: {exc}") from exc

        if max(width, height) <= CHAT_STICKER_MAX_EDGE:
            return source, int(width), int(height)

        target = self._meme_chat_target(row)
        expected_width, expected_height = self._meme_chat_size(width, height)
        if target.is_file():
            try:
                with Image.open(target) as image:
                    if image.size == (expected_width, expected_height):
                        return target, expected_width, expected_height
            except Exception:
                pass
            target.unlink(missing_ok=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            if is_gif:
                self._meme_save_gif_copy(source, tmp)
            else:
                self._meme_save_static_copy(source, tmp, fmt or source.suffix.lstrip(".").upper())
            with Image.open(tmp) as check:
                if check.size != (expected_width, expected_height):
                    raise MemeChatCopyError(
                        f"chat copy size mismatch: expected {(expected_width, expected_height)}, "
                        f"got {check.size}"
                    )
            os.replace(tmp, target)
        except MemeChatCopyError:
            raise
        except Exception as exc:
            raise MemeChatCopyError(f"chat copy write failed: {exc}") from exc
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return target, expected_width, expected_height


    def _meme_record_row(self, record_id: int) -> Optional[Any]:
        if self._conn is None:
            return None
        return self._conn.execute(
            "SELECT * FROM meme_records WHERE id=?", (int(record_id),)
        ).fetchone()

    def _meme_asset_rows(self, record_id: int) -> List[Any]:
        if self._conn is None:
            return []
        return self._conn.execute(
            "SELECT * FROM meme_assets WHERE record_id=? ORDER BY id", (int(record_id),)
        ).fetchall()

    def _meme_asset_by_sha(self, sha256: str) -> Optional[Any]:
        if self._conn is None:
            return None
        return self._conn.execute(
            "SELECT * FROM meme_assets WHERE sha256=?", (str(sha256),)
        ).fetchone()

    def _meme_asset_payload(self, row: Any, *, purpose: str = "original") -> Dict[str, Any]:
        original = str(self._meme_abs(row["path"]))
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
            chat_path, send_width, send_height = self._meme_chat_copy(row)
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
        self, row: Any, assets: Optional[List[Any]] = None, *,
        score: Optional[float] = None, match: str = "", purpose: str = "original",
    ) -> Dict[str, Any]:
        aliases = self._meme_alias_list(row["aliases"])
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


    def _tool_meme_save(self, args: Dict[str, Any]) -> Dict[str, Any]:
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
            data = self._meme_read_source(src)
            meta = self._meme_validate_image(data)
            sha256 = hashlib.sha256(data).hexdigest()
            if any(entry["sha256"] == sha256 for entry in prepared):
                continue
            prepared.append(
                {"data": data, "sha256": sha256, **meta, "source_ref": src}
            )

        restore_deleted = args.get("restore_deleted") is True
        forgotten_images = [
            entry for entry in prepared
            if self._meme_is_forgotten(f"image:{entry['sha256']}")
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
            )
        text_forgotten = bool(text_fp and self._meme_is_forgotten(text_fp))
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

        owner_key = self._write_owner()
        now = _meme_now()
        text = self._meme_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text)[0])
        forgotten_fps = {f"image:{entry['sha256']}" for entry in forgotten_images}
        restored_fps = [
            f"image:{entry['sha256']}" for entry in new_prepared
            if f"image:{entry['sha256']}" in forgotten_fps
        ]
        if text_forgotten:
            restored_fps.append(text_fp)
        created_files: List[str] = []
        try:
            with self._lock:
                with self._conn:
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
                        rel, was_created = self._meme_store_file(
                            entry["data"], entry["sha256"], entry["ext"])
                        if was_created:
                            created_files.append(rel)
                        self._conn.execute(
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
                self._meme_remove_file(rel)
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

    def _tool_meme_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        record_id = int(args.get("record_id") or 0)
        row = self._meme_record_row(record_id)
        if row is None:
            raise ValueError(f"meme record not found: {record_id}")

        title = str(row["title"] or "")
        meaning = str(row["meaning"] or "")
        aliases = self._meme_alias_list(row["aliases"])
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
            asset = self._conn.execute(
                "SELECT * FROM meme_assets WHERE id=? AND record_id=?",
                (asset_id, record_id),
            ).fetchone()
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
            data = self._meme_read_source(src)
            meta = self._meme_validate_image(data)
            sha256 = hashlib.sha256(data).hexdigest()
            if any(entry["sha256"] == sha256 for entry in prepared):
                continue
            if self._meme_is_forgotten(f"image:{sha256}"):
                forgotten_images.append(sha256)
                continue
            existing_asset = self._meme_asset_by_sha(sha256)
            if existing_asset is not None:
                if int(existing_asset["record_id"]) == record_id:
                    continue
                if int(existing_asset["id"]) in remove_ids:
                    pass
                else:
                    raise ValueError(
                        f"image already belongs to meme record {int(existing_asset['record_id'])}"
                    )
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
        text = self._meme_text(title, aliases, meaning)
        vector = _pack_vector(_embed_text(text)[0])
        removed_rels: List[str] = []
        removed_assets: List[Any] = []
        archived_assets: List[str] = []
        created_files: List[str] = []
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "UPDATE meme_records SET title=?, meaning=?, aliases=?, source=?, "
                        "source_ref=?, source_time=?, updated_at=?, vec=? WHERE id=?",
                        (
                            title, meaning, json.dumps(aliases, ensure_ascii=False),
                            source, source_ref, source_time, now, vector, record_id,
                        ),
                    )
                    for asset_id in remove_ids:
                        asset = self._conn.execute(
                            "SELECT * FROM meme_assets WHERE id=? AND record_id=?",
                            (asset_id, record_id),
                        ).fetchone()
                        if asset is not None:
                            archived = self._meme_archive_asset(record_id, asset)
                            archived_assets.append(archived["archived_path"])
                            removed_rels.append(str(asset["path"]))
                            removed_assets.append(asset)
                            self._conn.execute(
                                "DELETE FROM meme_assets WHERE id=? AND record_id=?",
                                (asset_id, record_id),
                            )
                    for entry in prepared:
                        rel, was_created = self._meme_store_file(
                            entry["data"], entry["sha256"], entry["ext"])
                        if was_created:
                            created_files.append(rel)
                        self._conn.execute(
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
                self._meme_remove_file(rel)
            raise

        for rel in removed_rels:
            self._meme_remove_file(rel)
        for asset in removed_assets:
            self._meme_remove_chat_copy(asset)
        row = self._meme_record_row(record_id)
        payload = self._meme_payload(row) if row is not None else {}
        payload.update({
            "stored": True,
            "reason": "updated",
            "archived_assets": archived_assets,
        })
        return payload

    def _tool_meme_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = args.get("asset_id")
        record_id = args.get("record_id")
        if asset_id not in (None, ""):
            asset = self._conn.execute(
                "SELECT * FROM meme_assets WHERE id=?", (int(asset_id),)
            ).fetchone()
            if asset is None:
                raise ValueError(f"meme asset not found: {asset_id}")
            if record_id not in (None, "") and int(record_id) != int(asset["record_id"]):
                raise ValueError("asset does not belong to the given record")
            rel = str(asset["path"])
            archived = self._meme_archive_asset(int(asset["record_id"]), asset)
            with self._lock:
                with self._conn:
                    self._conn.execute("DELETE FROM meme_assets WHERE id=?", (int(asset_id),))
            self._meme_remove_chat_copy(asset)
            self._meme_remove_file(rel)
            return {
                "deleted_asset": int(asset_id),
                "record_id": int(asset["record_id"]),
                "archived_path": archived["archived_path"],
                "reason": "deleted_asset",
            }

        if record_id in (None, ""):
            raise ValueError("record_id or asset_id is required")
        record_id = int(record_id)
        row = self._meme_record_row(record_id)
        if row is None:
            raise ValueError(f"meme record not found: {record_id}")
        assets = self._meme_asset_rows(record_id)
        archived = self._meme_archive(row, assets)
        fingerprints = [
            self._meme_text_fingerprint(
                str(row["title"] or ""),
                self._meme_alias_list(row["aliases"]),
                str(row["meaning"] or ""),
            )
        ]
        fingerprints.extend(f"image:{asset['sha256']}" for asset in assets)
        with self._lock:
            with self._conn:
                self._meme_mark_forgotten(fingerprints)
                self._conn.execute("DELETE FROM meme_records WHERE id=?", (record_id,))
        for asset in assets:
            self._meme_remove_chat_copy(asset)
            self._meme_remove_file(str(asset["path"]))
        return {
            "deleted_record": record_id,
            "deleted_assets": len(assets),
            "archived_assets": [asset["archived_path"] for asset in archived["assets"]],
            "reason": "deleted_record",
        }


    def _meme_fts_ids(self, query: str) -> set[int]:
        if self._conn is None or not query:
            return set()
        try:
            safe = query.replace('"', '""')
            return {
                int(row[0])
                for row in self._conn.execute(
                    "SELECT rowid FROM meme_fts WHERE meme_fts MATCH ?",
                    (f'"{safe}"',),
                ).fetchall()
            }
        except sqlite3.Error:
            return set()

    def _meme_candidate_rows(self) -> List[Any]:
        if self._conn is None:
            return []
        return self._conn.execute("SELECT * FROM meme_records").fetchall()

    def _meme_rank(
        self, query: str, top_k: int, *, purpose: str = "original"
    ) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        query = str(query or "").strip()
        limit = max(1, min(10, int(top_k)))
        rows = self._meme_candidate_rows()
        if not query:
            return [
                self._meme_payload(row, score=0.0, match="recent", purpose=purpose)
                for row in sorted(rows, key=lambda item: int(item["id"]), reverse=True)[:limit]
            ]

        query_tokens = set(_tokens(query))
        query_norm = self._meme_norm(query)
        query_vector, embedding_real = _embed_text(query, query=True)
        fts_ids = self._meme_fts_ids(query)
        now = datetime.now().astimezone()
        scored: List[Tuple[float, Any, str, float, float, bool]] = []
        for row in rows:
            aliases = self._meme_alias_list(row["aliases"])
            text = self._meme_text(
                str(row["title"] or ""), aliases, str(row["meaning"] or ""))
            row_tokens = set(_tokens(text))
            lexical = (
                len(query_tokens & row_tokens) / max(1, len(query_tokens))
                if query_tokens else 0.0
            )
            semantic = _cosine(query_vector, _unpack_vector(row["vec"]))
            alias_hit = any(
                alias_norm and (
                    alias_norm == query_norm
                    or (len(alias_norm) >= 2 and alias_norm in query_norm)
                )
                for alias_norm in (self._meme_norm(alias) for alias in aliases)
            )
            fts_hit = int(row["id"]) in fts_ids
            try:
                created = datetime.fromisoformat(str(row["created_at"]))
                age_days = max(0.0, (now - created).total_seconds() / 86400)
            except Exception:
                age_days = _MEME_RECENCY_DAYS
            recency = math.exp(-age_days / _MEME_RECENCY_DAYS)
            if not (
                alias_hit
                or lexical > 0
                or fts_hit
                or (embedding_real and semantic >= 0.25)
            ):
                continue
            score = (
                0.55 * lexical
                + 0.30 * max(0.0, semantic)
                + 0.10 * recency
                + (0.50 if alias_hit else 0.0)
                + (0.05 if fts_hit else 0.0)
            )
            if alias_hit:
                match = "alias"
            elif lexical > 0 and semantic < 0.35:
                match = "text"
            else:
                match = "semantic"
            scored.append((score, row, match, max(0.0, float(semantic)), lexical, fts_hit))
        scored.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
        results: List[Dict[str, Any]] = []
        for score, row, match, semantic, lexical, fts_hit in scored[:limit]:
            payload = self._meme_payload(
                row, score=score, match=match, purpose=purpose)
            payload["_semantic"] = semantic
            payload["_lexical"] = lexical
            payload["_fts_hit"] = fts_hit
            payload["_embedding_real"] = bool(embedding_real)
            results.append(payload)
        return results

    def _tool_meme_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._shadow or self._conn is None:
            return []
        query = str(args.get("query") or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 5))))
        purpose = str(args.get("purpose") or "chat_sticker").strip().lower()
        if purpose not in {"chat_sticker", "original"}:
            raise ValueError("purpose must be 'chat_sticker' or 'original'")
        results = self._meme_rank(query, top_k, purpose=purpose)
        for item in results:
            item.pop("_semantic", None)
            item.pop("_lexical", None)
            item.pop("_fts_hit", None)
            item.pop("_embedding_real", None)
        return results


    def _meme_message_text(self, value: Any) -> str:
        if isinstance(value, str):
            text = _IMAGE_HINT_RE.sub(" ", value).strip()
            if text.startswith("data:"):
                return ""
            try:
                candidate = Path(os.path.expanduser(text))
                if candidate.suffix.lower() in _IMAGE_EXTS and candidate.is_file():
                    return ""
            except (OSError, ValueError):
                pass
            return text
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                parts.append(self._meme_message_text(item))
            return " ".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            kind = str(value.get("type") or "")
            if kind in {"text", "input_text"}:
                return str(value.get("text") or "").strip()
            if kind in {"image", "image_url", "input_image", "image_path"}:
                return ""
            if "text" in value:
                return str(value.get("text") or "").strip()
        return ""

    def _meme_image_bytes(self, value: Any, depth: int = 0) -> List[bytes]:
        if depth > 6 or value is None:
            return []
        if isinstance(value, bytes):
            return [value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("data:image/") or text.startswith("data:application/octet-stream"):
                try:
                    return [self._meme_decode_data_url(text)]
                except Exception:
                    return []
            if text.startswith(("http://", "https://")):
                return []

            def _read_path(candidate: str) -> Optional[bytes]:
                candidate = candidate.strip().strip("'\"")
                if candidate.lower().startswith("file://"):
                    candidate = candidate[7:]
                path = Path(os.path.expanduser(candidate))
                if path.suffix.lower() not in _IMAGE_EXTS:
                    return None
                try:
                    return path.read_bytes() if path.is_file() else None
                except OSError:
                    return None

            hinted = [match.group(1) for match in _IMAGE_HINT_RE.finditer(text)]
            noted = []
            for match in _VISION_IMAGE_URL_RE.finditer(text):
                candidate = match.group(1).strip()
                candidate = re.sub(r"\s*~?\]\s*$", "", candidate).strip().strip("'\"")
                if candidate:
                    noted.append(candidate)
            candidates = [*hinted, *noted]
            if candidates:
                images = [data for data in (_read_path(item) for item in candidates) if data]
                if images:
                    return images
            direct = _read_path(text)
            return [direct] if direct else []
        if isinstance(value, list):
            images: List[bytes] = []
            for item in value:
                images.extend(self._meme_image_bytes(item, depth + 1))
            return images
        if isinstance(value, dict):
            kind = str(value.get("type") or "")
            if kind in {"image", "image_url", "input_image", "image_path"}:
                for key in ("image_url", "url", "path", "image_path"):
                    if key in value:
                        found = self._meme_image_bytes(value.get(key), depth + 1)
                        if found:
                            return found
                source = value.get("source")
                if isinstance(source, dict):
                    if str(source.get("type") or "") == "base64":
                        try:
                            raw = base64.b64decode(str(source.get("data") or ""), validate=True)
                            return [raw] if raw else []
                        except Exception:
                            return []
                data = value.get("data")
                if isinstance(data, str) and data:
                    try:
                        raw = base64.b64decode(data, validate=True)
                        return [raw] if raw else []
                    except Exception:
                        return []
                return []
            images: List[bytes] = []
            for key in ("image_url", "url", "path", "image_path", "source", "content"):
                if key in value:
                    images.extend(self._meme_image_bytes(value.get(key), depth + 1))
            return images
        return []

    def _meme_alias_record(self, text: str) -> Optional[Any]:
        if self._conn is None:
            return None
        query_norm = self._meme_norm(text)
        if not query_norm:
            return None
        for row in self._conn.execute("SELECT * FROM meme_records ORDER BY id DESC").fetchall():
            for alias in self._meme_alias_list(row["aliases"]):
                alias_norm = self._meme_norm(alias)
                if alias_norm and (
                    alias_norm == query_norm
                    or (len(alias_norm) >= 2 and alias_norm in query_norm)
                ):
                    return row
        return None

    def _meme_record_for_image(self, data: bytes) -> Optional[Any]:
        if self._conn is None:
            return None
        sha256 = hashlib.sha256(data).hexdigest()
        asset = self._meme_asset_by_sha(sha256)
        if asset is None:
            return None
        return self._meme_record_row(int(asset["record_id"]))

    @staticmethod
    def _meme_public_terms(title: str, aliases: List[str]) -> List[str]:
        title = str(title or "")
        if "/" in title:
            return []
        terms = [re.split(r"[:：]", title, maxsplit=1)[0], *aliases]
        result: List[str] = []
        for term in terms:
            term = str(term or "").strip()
            if len(term) < 2 or any(char in term for char in "\n\r\x00"):
                continue
            if term not in result:
                result.append(term)
        return result

    def _meme_public_text_hits(
        self, parts: List[str], rows: List[Any]
    ) -> List[Tuple[Any, str]]:
        segments = [str(part) for part in parts if isinstance(part, str) and part]
        if not segments:
            return []
        hits: List[Tuple[Any, str]] = []
        for row in rows:
            for term in self._meme_public_terms(
                row["title"], self._meme_alias_list(row["aliases"])
            ):
                if any(term in segment for segment in segments):
                    hits.append((row, term))
        return hits

    def known_meme_terms(self, parts: List[str]) -> List[str]:
        """Word faces from collected memes that appear verbatim in public parts.

        Read-only and public-safe: only ``title``/``aliases`` are loaded, and only
        the matched word faces are returned. Explanations, source references,
        record ids and annotations after a colon are never exposed.
        """
        if self._shadow or self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT title, aliases FROM meme_records ORDER BY id"
        ).fetchall()
        terms: List[str] = []
        for _row, term in self._meme_public_text_hits(parts, rows):
            if term not in terms:
                terms.append(term)
        return terms

    def meme_context_for_public_material(self, parts: List[str]) -> str:
        """Matched explanations for a private main turn, or "" in a public turn."""
        if self._shadow or self._conn is None:
            return ""
        try:
            from .source_context import in_public_interaction
        except ImportError:
            from source_context import in_public_interaction
        if in_public_interaction():
            return ""
        rows = self._conn.execute(
            "SELECT id, title, meaning, aliases FROM meme_records ORDER BY id"
        ).fetchall()
        seen: set[int] = set()
        lines: List[str] = []
        for row, term in self._meme_public_text_hits(parts, rows):
            record_id = int(row["id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            meaning = str(row["meaning"] or "").strip()
            if meaning:
                lines.append(f"- {term}：{meaning}")
        return "\n".join(lines)

    def _meme_available_sticker_summary(self) -> str:
        """One-line roster of image memes she may send; "" in public turns or shadow."""
        if self._shadow or self._conn is None:
            return ""
        try:
            from .source_context import in_public_interaction
        except ImportError:
            from source_context import in_public_interaction
        if in_public_interaction():
            return ""
        rows = self._conn.execute(
            "SELECT DISTINCT r.id, r.title FROM meme_records r "
            "JOIN meme_assets a ON a.record_id = r.id ORDER BY r.id"
        ).fetchall()
        titles: List[str] = []
        for row in rows:
            title = str(row["title"] or "").strip()
            if not title:
                continue
            head = re.split(r"[：:]", title, maxsplit=1)[0].strip()
            name = head or title
            if name not in titles:
                titles.append(name)
        if not titles:
            return ""
        return (
            "当前库里双方共享、平时闲聊应景或斗图回敬时可搜取发送的有图表情："
            + "、".join(titles)
            + "（直接甩图或配半句短话回敬即可，不必解释图意，也不必每轮都发）。"
        )

    def meme_context_for_turn(self, user_message: Any, *, known_context: Any = None) -> str:
        """Return a short private-memory block for this turn, or "".

        Text aliases are matched exactly; images are matched by
        their original-byte SHA-256, so a re-sent collected image brings its
        explanation without a vision call. Unknown images stay untouched.
        """
        if self._shadow or self._conn is None:
            return ""
        try:
            from .source_context import in_public_interaction
        except ImportError:
            from source_context import in_public_interaction
        if in_public_interaction():
            return ""

        records: List[Tuple[Any, str]] = []
        seen: set[int] = {
            int(record_id) for record_id in re.findall(
                r"(?m)^\[已知梗图\] 我收过的记录#(\d+)「", self._meme_message_text(known_context))
        }
        for data in self._meme_image_bytes(user_message):
            row = self._meme_record_for_image(data)
            if row is not None and int(row["id"]) not in seen:
                seen.add(int(row["id"]))
                records.append((row, "image"))
        text = self._meme_message_text(user_message)
        if text:
            alias_row = self._meme_alias_record(text)
            if alias_row is not None:
                if int(alias_row["id"]) not in seen:
                    seen.add(int(alias_row["id"]))
                    records.append((alias_row, "alias"))
        if not records:
            return ""
        lines = [
            "[梗与表情记忆] 以下是双方共享图库里收过的梗/表情（仅作理解语气与心意的内部参考）；"
            "平时打趣或斗图回敬时可直接调 sanae_meme_search 取图回敬，绝不要在回复里复述画面文字、拆解或解释这张梗图的意思，也不要声称是当前用户刚告诉她的："
        ]
        for row, via in records[:2]:
            title = str(row["title"] or "").strip() or "未命名"
            meaning = str(row["meaning"] or "").strip()
            aliases = self._meme_alias_list(row["aliases"])
            alias_text = f"；别名：{'、'.join(aliases)}" if aliases else ""
            body = meaning or "（无文字解释，仅素材）"
            lines.append(f"- 记录#{int(row['id'])}「{title}」：{body}{alias_text}")
            if via == "image":
                lines.append("  （当前消息里有与图库完全相同的图；可直接按这条理解，不必重新识别。）")
        return "\n".join(lines)
