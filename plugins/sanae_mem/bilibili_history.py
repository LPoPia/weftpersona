"""Built-in decoding of persisted B站 provenance and receipt records."""

from __future__ import annotations

import re
import sqlite3

from .infra.source_descriptors import MemeEvidenceRoute, PrivatePeerRoute, ReceiptTable, SourceDescriptor

_PLATFORM = "bilibili"
BILIBILI_PLUGIN = "bilibili_public"
_DM_PREFIX = "bilibili:dm:"
_VIDEO_PREFIX = "https://www.bilibili.com/video/"
_BVID = re.compile(r"BV[0-9A-Za-z]{10}")
_RECEIPTS = ReceiptTable("bilibili_reply_receipts", "B站私信", "B站回复（平台已受理）")


def _matches_private_ref(ref: str, *, owner_key: str, peer: str) -> bool:
    if not ref.startswith(_DM_PREFIX):
        return False
    account, _, tail = ref[len(_DM_PREFIX):].partition(":")
    sender, _, message_id = tail.partition(":")
    return bool(account and sender == peer and message_id)


def _video_url(object_id: str) -> str:
    return f"{_VIDEO_PREFIX}{object_id}/"


def _matches_original(object_id: str, locator: str) -> bool:
    return bool(object_id and locator.rstrip("/") == f"{_VIDEO_PREFIX}{object_id}")


def _legacy_video_id(stored_id: str, ref: str) -> str:
    if stored_id:
        return stored_id if _BVID.fullmatch(stored_id) else ""
    if ref.startswith(_VIDEO_PREFIX):
        object_id = ref.removeprefix(_VIDEO_PREFIX).removesuffix("/")
        return object_id if _BVID.fullmatch(object_id) else ""
    return ""


def _legacy_video_mode(raw: str) -> str:
    if "在B站收到该视频的抽样画面分析，音轨未核验：" in raw:
        return "video_analysis"
    if "在B站阅读了该视频的字幕，未观看画面：" in raw:
        return "captions_only"
    return ""


def _video_id_in_text(text: str) -> str:
    match = _BVID.search(text)
    return match.group() if match else ""


BILIBILI_SOURCE = SourceDescriptor(
    platform=_PLATFORM,
    label="B站",
    private_peer=PrivatePeerRoute(_PLATFORM, _matches_private_ref),
    receipts=_RECEIPTS,
    meme_evidence=MemeEvidenceRoute(
        BILIBILI_PLUGIN, frozenset({"comment", "danmaku", "caption", "title", "description"}),
        _matches_original,
    ),
    legacy_video_id=_legacy_video_id,
    legacy_video_mode=_legacy_video_mode,
    legacy_video_in_text=_video_id_in_text,
    legacy_video_url=_video_url,
    legacy_video_kind=lambda kind: kind in {"", "self", "unknown", "maintainer"},
    legacy_unscoped_video=True,
    legacy_video_storage_key=lambda object_id: object_id,
    legacy_video_ref_prefix=_VIDEO_PREFIX,
    legacy_storage_id_glob="BV" + "[0-9A-Za-z]" * 10,
    legacy_storage_id_length=12,
)


def record_bilibili_reply(
    conn: sqlite3.Connection, *, owner: str, receipt_id: str, at: float,
    text: str, visibility: str,
) -> bool:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {_RECEIPTS.table} (
        owner_key TEXT NOT NULL, receipt_id TEXT NOT NULL, at REAL NOT NULL,
        text TEXT NOT NULL, visibility TEXT NOT NULL, PRIMARY KEY (owner_key, receipt_id)
    )""")
    cursor = conn.execute(
        f"INSERT OR IGNORE INTO {_RECEIPTS.table} VALUES (?, ?, ?, ?, ?)",
        (owner, receipt_id, at, text, visibility),
    )
    return cursor.rowcount == 1
