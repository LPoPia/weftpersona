from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

_RECENCY_DAYS = 180.0


def parse_aliases(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "[]")
        except Exception:
            return []
        return [str(item).strip() for item in parsed if str(item).strip()] if isinstance(parsed, list) else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _norm(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold(), flags=re.UNICODE)


def record_text(title: str, aliases: List[str], meaning: str) -> str:
    return "\n".join(part for part in (title, "、".join(aliases), meaning) if part).strip()


def fts_ids(conn: Any, query: str) -> set[int]:
    if conn is None or not query:
        return set()
    try:
        safe = query.replace('"', '""')
        return {
            int(row[0])
            for row in conn.execute(
                "SELECT rowid FROM meme_fts WHERE meme_fts MATCH ?",
                (f'"{safe}"',),
            ).fetchall()
        }
    except sqlite3.Error:
        return set()


def candidate_rows(conn: Any) -> List[Any]:
    if conn is None:
        return []
    return conn.execute("SELECT * FROM meme_records").fetchall()


def rank_records(
    conn: Any,
    query: str,
    top_k: int,
    *,
    embed_text: Callable[..., Any],
    tokens: Callable[[str], Any],
    cosine: Callable[[Any, Any], float],
    unpack_vector: Callable[[Any], Any],
    rows: Optional[List[Any]] = None,
    matched_fts_ids: Optional[set[int]] = None,
) -> List[dict[str, Any]]:
    if conn is None and rows is None:
        return []
    query = str(query or "").strip()
    limit = max(1, min(10, int(top_k)))
    rows = candidate_rows(conn) if rows is None else rows
    if not query:
        return [
            {"row": row, "score": 0.0, "match": "recent"}
            for row in sorted(rows, key=lambda item: int(item["id"]), reverse=True)[:limit]
        ]

    query_tokens = set(tokens(query))
    query_norm = _norm(query)
    query_vector, embedding_real = embed_text(query, query=True)
    if matched_fts_ids is None:
        matched_fts_ids = fts_ids(conn, query)
    now = datetime.now().astimezone()
    scored: List[Tuple[float, Any, str, float, float, bool]] = []
    for row in rows:
        aliases = parse_aliases(row["aliases"])
        text = record_text(str(row["title"] or ""), aliases, str(row["meaning"] or ""))
        row_tokens = set(tokens(text))
        lexical = (
            len(query_tokens & row_tokens) / max(1, len(query_tokens))
            if query_tokens else 0.0
        )
        semantic = cosine(query_vector, unpack_vector(row["vec"]))
        alias_hit = any(
            alias_norm and (
                alias_norm == query_norm
                or (len(alias_norm) >= 2 and alias_norm in query_norm)
            )
            for alias_norm in (_norm(alias) for alias in aliases)
        )
        fts_hit = int(row["id"]) in matched_fts_ids
        try:
            created = datetime.fromisoformat(str(row["created_at"]))
            age_days = max(0.0, (now - created).total_seconds() / 86400)
        except Exception:
            age_days = _RECENCY_DAYS
        recency = math.exp(-age_days / _RECENCY_DAYS)
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
    return [
        {
            "row": row,
            "score": score,
            "match": match,
            "semantic": semantic,
            "lexical": lexical,
            "fts_hit": fts_hit,
            "embedding_real": bool(embedding_real),
        }
        for score, row, match, semantic, lexical, fts_hit in scored[:limit]
    ]


def alias_records(text: str, rows: List[Any]) -> List[Any]:
    query_norm = _norm(text)
    if not query_norm:
        return []
    matched: List[Any] = []
    for row in rows:
        for alias in [str(row["title"] or ""), *parse_aliases(row["aliases"])]:
            alias_norm = _norm(alias)
            if alias_norm and (
                alias_norm == query_norm
                or (len(alias_norm) >= 2 and alias_norm in query_norm)
            ):
                matched.append(row)
                break
    return matched


def alias_record(conn: Any, text: str) -> Optional[Any]:
    if conn is None:
        return None
    rows = conn.execute("SELECT * FROM meme_records ORDER BY id DESC").fetchall()
    matches = alias_records(text, rows)
    return matches[0] if matches else None


def record_for_image(conn: Any, data: bytes) -> Optional[Any]:
    if conn is None:
        return None
    sha256 = hashlib.sha256(data).hexdigest()
    asset = conn.execute(
        "SELECT * FROM meme_assets WHERE sha256=?", (sha256,)
    ).fetchone()
    if asset is None:
        return None
    return conn.execute(
        "SELECT * FROM meme_records WHERE id=?", (int(asset["record_id"]),)
    ).fetchone()


def public_terms(title: str, aliases: List[str]) -> List[str]:
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


def public_text_hits(parts: List[str], rows: List[Any]) -> List[Tuple[Any, str]]:
    segments = [str(part) for part in parts if isinstance(part, str) and part]
    if not segments:
        return []
    hits: List[Tuple[Any, str]] = []
    for row in rows:
        for term in public_terms(row["title"], parse_aliases(row["aliases"])):
            norm_term = _norm(term)
            if norm_term and any(norm_term in _norm(segment) for segment in segments):
                hits.append((row, term))
    return hits
