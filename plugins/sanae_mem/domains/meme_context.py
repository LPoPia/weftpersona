from __future__ import annotations

import re
import logging
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, List, Sequence, Tuple
from urllib.parse import urlparse

from . import meme_assets, meme_search
from ..storage.sqlite_facade import connect_readonly

logger = logging.getLogger(__name__)


def known_inbound_images(
    paths: List[str],
    db_path: str | Path,
    *,
    owner_key: str = "",
    owner_keys: Sequence[str] = (),
    allow_legacy_blank_owner: bool = False,
) -> dict[str, str]:
    """Return explanations for exact stored images without opening a writable store."""
    db = Path(db_path)
    if not db.is_file():
        return {}
    owners = sorted({str(value) for value in (owner_keys or (owner_key,)) if str(value)})
    if owners:
        private_clause = "r.owner_key IN (%s)" % ",".join("?" for _ in owners)
        if allow_legacy_blank_owner:
            private_clause = f"({private_clause} OR r.owner_key='')"
    else:
        private_clause = "0"
    notes: dict[str, str] = {}
    with closing(connect_readonly(db)) as conn:
        for value in paths:
            path = Path(value)
            if not path.is_absolute() or not path.is_file():
                continue
            try:
                digest = meme_assets.file_sha256(path)
            except OSError:
                logger.debug("sanae_mem: inbound meme image unavailable", exc_info=True)
                continue
            row = conn.execute(
                "SELECT r.title, r.meaning, r.id FROM meme_assets a JOIN meme_records r ON r.id=a.record_id "
                "WHERE a.sha256=? AND ((r.visibility_scope='public' AND r.tier='eye_familiar' "
                "AND r.evidence_status='confirmed') "
                f"OR (r.visibility_scope='private' AND {private_clause}))",
                (digest, *owners),
            ).fetchone()
            if row is not None:
                title = " ".join(str(row[0] or "未命名").split())
                meaning = " ".join(str(row[1] or "只收藏了这张素材，未记文字解释").split())
                notes[value] = f"[已知梗图] 我收过的记录#{row[2]}「{title}」：{meaning}"
    return notes


def message_text(value: Any) -> str:
    if isinstance(value, str):
        text = meme_assets.IMAGE_HINT_RE.sub(" ", value).strip()
        if text.startswith("data:"):
            return ""
        try:
            candidate = Path(text).expanduser()
            if candidate.suffix.lower() in meme_assets.IMAGE_EXTS and candidate.is_file():
                return ""
        except (OSError, ValueError):
            pass
        return text
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.append(message_text(item))
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


def known_meme_terms(conn: Any, parts: List[str], *, shadow: bool) -> List[str]:
    """Return only verbatim public-safe word faces, never explanations or ids."""
    if shadow or conn is None:
        return []
    rows = conn.execute(
        "SELECT title, aliases, tier, visibility_scope FROM meme_records "
        "WHERE tier='eye_familiar' AND visibility_scope='public' "
        "AND evidence_status='confirmed' ORDER BY id"
    ).fetchall()
    terms: List[str] = []
    for _row, term in meme_search.public_text_hits(parts, rows):
        if term not in terms:
            terms.append(term)
    return terms


def public_meme_senses(conn: Any, parts: List[str], *, shadow: bool) -> List[Dict[str, Any]]:
    if shadow or conn is None:
        return []
    rows = conn.execute(
        "SELECT r.id, r.title, r.meaning, r.aliases, r.tier, r.visibility_scope, "
        "r.source_ref, sm.extra_metadata FROM meme_records r "
        "LEFT JOIN source_materials sm ON sm.id=r.source_material_id "
        "WHERE r.tier='eye_familiar' AND r.visibility_scope='public' "
        "AND r.evidence_status='confirmed' ORDER BY r.id"
    ).fetchall()
    return [{"id": row["id"], "term": term, "meaning": row["meaning"].strip(),
             "source": _verified_source_url(row)}
            for row, term in meme_search.public_text_hits(parts, rows) if str(row["meaning"] or "").strip()]


def _owner_read_clause(
    owner_key: str, owner_keys: Sequence[str] = (), *, allow_legacy_blank_owner: bool = False,
) -> Tuple[str, Tuple[Any, ...]]:
    owners = sorted({str(value) for value in (owner_keys or (owner_key,)) if str(value)})
    if not owners:
        return "owner_key=''", ()
    clause = "owner_key IN (%s)" % ",".join("?" for _ in owners)
    if allow_legacy_blank_owner:
        clause = f"({clause} OR owner_key='')"
    return clause, tuple(owners)


def context_for_public_material(
    conn: Any, parts: List[str], *, shadow: bool, public: bool, public_only: bool = False,
    owner_key: str = "", allow_legacy_blank_owner: bool = False,
    owner_keys: Sequence[str] = (),
) -> str:
    if shadow or conn is None or (public and not public_only):
        return ""
    if public_only:
        lines = {}
        for sense in public_meme_senses(conn, parts, shadow=shadow):
            source = sense["source"]
            suffix = f"（出处：{source}）" if source else ""
            lines.setdefault(sense["id"], f"- {sense['term']}：{sense['meaning']}{suffix}")
        return "\n".join(lines.values())
    if owner_key:
        private_owner_clause, params = _owner_read_clause(
            owner_key, owner_keys, allow_legacy_blank_owner=allow_legacy_blank_owner)
        private_owner_clause = private_owner_clause.replace("owner_key", "r.owner_key")
        rows = conn.execute(
            "SELECT r.id, r.title, r.meaning, r.aliases, r.tier, r.visibility_scope, "
            "r.source_ref, sm.extra_metadata FROM meme_records r "
            "LEFT JOIN source_materials sm ON sm.id=r.source_material_id "
            "WHERE (r.visibility_scope='public' AND r.tier='eye_familiar' "
            "AND r.evidence_status='confirmed') OR "
            "(r.visibility_scope='private' AND r.tier='common' AND "
            f"{private_owner_clause}) ORDER BY r.id", params
        ).fetchall()
    else:
        return ""
    seen: set[int] = set()
    lines: List[str] = []
    for row, term in meme_search.public_text_hits(parts, rows):
        record_id = int(row["id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        meaning = str(row["meaning"] or "").strip()
        if meaning:
            source_url = _verified_source_url(row)
            source_suffix = f"（出处：{source_url}）" if source_url else ""
            lines.append(f"- {term}：{meaning}{source_suffix}")
    return "\n".join(lines)


def _verified_source_url(row: Any) -> str:
    """Return only the verified page URL, never the fetched page text or quote."""
    metadata = row["extra_metadata"] if "extra_metadata" in row.keys() else ""
    try:
        meta = json.loads(str(metadata or "{}"))
    except (TypeError, ValueError):
        meta = {}
    candidate = str(meta.get("verification_url") or "").strip() if isinstance(meta, dict) else ""
    if not candidate:
        candidate = str(row["source_ref"] or "").strip() if "source_ref" in row.keys() else ""
    parsed = urlparse(candidate)
    if "visibility_scope" in row.keys() and row["visibility_scope"] != "public":
        return ""
    return candidate if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def available_sticker_summary(
    conn: Any, *, shadow: bool, public: bool, owner_key: str = "",
    allow_legacy_blank_owner: bool = False, owner_keys: Sequence[str] = (),
) -> str:
    if shadow or conn is None or public or not owner_key:
        return ""
    private_owner_clause, params = _owner_read_clause(
        owner_key, owner_keys, allow_legacy_blank_owner=allow_legacy_blank_owner)
    private_owner_clause = private_owner_clause.replace("owner_key", "r.owner_key")
    rows = conn.execute(
        "SELECT DISTINCT r.id, r.title FROM meme_records r "
        "JOIN meme_assets a ON a.record_id = r.id WHERE "
        "(r.visibility_scope='public' AND r.tier='eye_familiar' "
        "AND r.evidence_status='confirmed') OR "
        "(r.visibility_scope='private' AND r.tier='common' AND "
        f"{private_owner_clause}) ORDER BY r.id", params
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


def context_for_turn(
    conn: Any,
    user_message: Any,
    *,
    known_context: Any = None,
    shadow: bool,
    public: bool,
    owner_key: str = "",
    allow_legacy_blank_owner: bool = False,
    owner_keys: Sequence[str] = (),
) -> str:
    """Build an owner-scoped private overlay or a public eye-familiar overlay."""
    if shadow or conn is None:
        return ""

    public_rows = conn.execute(
        "SELECT * FROM meme_records WHERE tier='eye_familiar' "
        "AND visibility_scope='public' AND evidence_status='confirmed' ORDER BY id DESC"
    ).fetchall()
    private_rows: List[Any] = []
    private_params: Tuple[Any, ...] = ()
    if not public and owner_key:
        owner_clause, private_params = _owner_read_clause(
            owner_key, owner_keys, allow_legacy_blank_owner=allow_legacy_blank_owner)
        private_rows = conn.execute(
            "SELECT * FROM meme_records WHERE tier='common' AND visibility_scope='private' "
            f"AND {owner_clause} ORDER BY id DESC", private_params,
        ).fetchall()
    permitted_ids = {int(row["id"]) for row in [*public_rows, *private_rows]}

    records: List[tuple[Any, str]] = []
    seen: set[int] = {
        int(record_id) for record_id in re.findall(
            r"(?m)^\[已知梗图\] 我收过的记录#(\d+)「", message_text(known_context))
    }
    for data in meme_assets.image_bytes(user_message):
        row = meme_search.record_for_image(conn, data)
        if row is not None and int(row["id"]) not in permitted_ids:
            row = None
        if row is not None and int(row["id"]) not in seen:
            seen.add(int(row["id"]))
            records.append((row, "image"))
    text = message_text(user_message)
    if text:
        for row, _term in meme_search.public_text_hits([text], public_rows):
            record_id = int(row["id"])
            if record_id not in seen:
                seen.add(record_id)
                records.append((row, "alias"))
        for row in meme_search.alias_records(text, private_rows):
            record_id = int(row["id"])
            if record_id not in seen:
                seen.add(record_id)
                records.append((row, "alias"))

    sections: List[str] = []
    # This Tier-2 reminder is private, short, and limited to the verified owner's
    # reviewed common-tier text entries. Public and unknown scopes never receive it.
    if not public and owner_key:
        reminder_clause, reminder_params = _owner_read_clause(
            owner_key, owner_keys, allow_legacy_blank_owner=allow_legacy_blank_owner)
        reminder_rows = conn.execute(
            "SELECT title, meaning FROM meme_records WHERE tier='common' "
            "AND visibility_scope='private' AND " + reminder_clause +
            " ORDER BY updated_at DESC, id DESC LIMIT 3",
            reminder_params,
        ).fetchall()
        reminders = []
        for row in reminder_rows:
            title = " ".join(str(row["title"] or "").split())[:24]
            meaning = " ".join(str(row["meaning"] or "").split())[:42]
            if title and meaning:
                reminders.append(f"{title}：{meaning}")
        if reminders:
            sections.append("[常玩梗短提醒] " + "；".join(reminders))

    if not records and not sections:
        return ""
    if records:
        meme_lines: List[str] = [
            "[梗与表情记忆] 以下是双方共享图库里收过的梗/表情（仅作理解语气与心意的内部参考）；"
            "平时打趣或斗图回敬时可直接调 sanae_meme_search 取图回敬，绝不要在回复里复述画面文字、拆解或解释这张梗图的意思，也不要声称是当前用户刚告诉她的："
        ]
        for row, via in records[:2]:
            title = str(row["title"] or "").strip() or "未命名"
            meaning = str(row["meaning"] or "").strip()
            aliases = meme_search.parse_aliases(row["aliases"])
            alias_text = f"；别名：{'、'.join(aliases)}" if aliases else ""
            body = meaning or "（无文字解释，仅素材）"
            meme_lines.append(f"- 记录#{int(row['id'])}「{title}」：{body}{alias_text}")
            if via == "image":
                meme_lines.append("  （当前消息里有与图库完全相同的图；可直接按这条理解，不必重新识别。）")
        sections.append("\n".join(meme_lines))
    return "\n\n".join(sections)
