"""Read-only SQLite access and rendering for partner impressions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .sqlite_facade import connect_readonly

ImpressionTarget = tuple[Path, str, str]


@dataclass(frozen=True)
class Impression:
    revision: int
    text: str
    speaker: str
    evidence: str
    turn_id: str
    updated_at: float
    baseline: str = ""
    recent: str = ""
    # Canonical provenance JSON of the turn this revision was written from ("" = old row).
    source_context: str = ""


def _parse_sections(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    baseline, recent = "", ""
    if "【长期底色】" in text or "【阶段近况】" in text:
        b_idx = text.find("【长期底色】")
        r_idx = text.find("【阶段近况】")
        if b_idx != -1 and r_idx != -1:
            if b_idx < r_idx:
                baseline = text[b_idx + len("【长期底色】"):r_idx].strip()
                recent = text[r_idx + len("【阶段近况】"):].strip()
            else:
                recent = text[r_idx + len("【阶段近况】"):b_idx].strip()
                baseline = text[b_idx + len("【长期底色】"):].strip()
        elif b_idx != -1:
            baseline = text[b_idx + len("【长期底色】"):].strip()
        elif r_idx != -1:
            recent = text[r_idx + len("【阶段近况】"):].strip()
        return baseline, recent
    return text.strip(), ""


def _compose_sections(baseline: str, recent: str, fallback: str = "") -> str:
    b, r = (baseline or "").strip(), (recent or "").strip()
    if b and r:
        return f"【长期底色】{b}\n【阶段近况】{r}"
    if b:
        return f"【长期底色】{b}"
    if r:
        return f"【阶段近况】{r}"
    return (fallback or "").strip()


def _has_impression_table(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='impression_revisions'"
        ).fetchone()
        is not None
    )


def _read_impression_with_conn(
    conn: sqlite3.Connection,
    target: ImpressionTarget,
    *,
    as_of: float | None = None,
    owners: Sequence[str] = (),
) -> Impression | None:
    _, profile_key, owner_key = target
    keys = sorted({str(owner_key or ""), *(str(value) for value in owners if str(value))})
    if len(keys) > 1:
        placeholders = ",".join("?" for _ in keys)
        owner_clause = f"owner_key IN ({placeholders})"
        params: list = [profile_key, *keys]
        order = " ORDER BY updated_at DESC, revision DESC LIMIT 1"
    else:
        owner_clause = "owner_key=?"
        params = [profile_key, str(owner_key or "")]
        order = " ORDER BY revision DESC LIMIT 1"
    if as_of is not None:
        owner_clause += " AND recorded_at<=?"
        params.append(as_of)
    columns = {
        str(col[1])
        for col in conn.execute("PRAGMA table_info(impression_revisions)").fetchall()
    }
    source_expr = "source_context" if "source_context" in columns else "'' AS source_context"
    row = conn.execute(
        "SELECT revision, text, speaker, evidence, turn_id, updated_at, baseline, recent, "
        f"{source_expr} "
        "FROM impression_revisions WHERE profile_key=? AND " + owner_clause + order,
        params,
    ).fetchone()
    if row is None:
        return None
    values = dict(row)
    b = str(values.get("baseline") or "").strip()
    r = str(values.get("recent") or "").strip()
    if not b and not r and values["text"]:
        b, r = _parse_sections(values["text"])
    return Impression(
        revision=int(values["revision"]),
        text=str(values["text"] or ""),
        speaker=str(values["speaker"] or ""),
        evidence=str(values["evidence"] or ""),
        turn_id=str(values["turn_id"] or ""),
        updated_at=float(values["updated_at"]),
        baseline=b,
        recent=r,
        source_context=str(values.get("source_context") or ""),
    )


def read_impression(target: ImpressionTarget, *, as_of: float | None = None,
                    owners: Sequence[str] = ()) -> Impression | None:
    """Read one person's impression; bound accounts' old rows stay readable where they are."""
    directory, _, _ = target
    db_path = directory / "sanae.db"
    if not db_path.exists():
        return None
    with closing(connect_readonly(db_path)) as conn:
        if not _has_impression_table(conn):
            return None
        return _read_impression_with_conn(conn, target, as_of=as_of, owners=owners)
