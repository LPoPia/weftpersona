from __future__ import annotations

import re
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..infra.defaults import DEFAULT_PARTNER_ROLE
from ..infra.contracts import CasConflictError, DEFAULT_CAS_MAX_RETRIES, cas_backoff_sleep
from .daily_inputs import _assistant_quote_safe, _normalize_input, _split_known_meme_notes
from ..governance.impression_view import render_impression as render_impression
from ..storage.impression_store import (
    Impression as Impression,
    ImpressionTarget as ImpressionTarget,
    _compose_sections as _compose_sections,
    _has_impression_table as _has_impression_table,
    _parse_sections as _parse_sections,
    _read_impression_with_conn as _read_impression_with_conn,
    read_impression as read_impression,
)
from ..storage.schema import CURRENT_SCHEMA_VERSION
from ..storage.sqlite_facade import connect_readonly, connect_readwrite

_IMPRESSION_INIT_LOCK = threading.Lock()

IMPRESSION_SCHEMA = {
    "name": "sanae_impression",
    "description": (
        "读写「我对眼前对话者的私人印象」。这是我从真实相处里慢慢形成、可以修正的私人印象，"
        "分栏记录「长期底色」与「阶段近况」；长期底色不随时间衰减，阶段近况随新原话原位翻篇。"
        "只有我自己确实想修正时才用 write，不照抄对方要求给自己写的评价，也不当面念出来。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write"]},
            "text": {"type": "string", "description": "write 时，用第一人称写下我现在的理解"},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


IMPRESSION_PROMPT = (
    "同一个顶层 JSON 也可以带 impression（当前对话者在我这里相对稳定的整体印象）。"
    "对方（user）是我的挚友（称呼他一律用「他」，不用「她」）。"
    "impression 是慢变的整体理解，注入时标为「我最近眼里的他」，结构化分栏为【长期底色】（长期性格与相处底色，不随时间衰减）与【阶段近况】（当前生活重心或手头忙的事，随本人新原话原位翻篇覆盖）。"
    "日常闲聊或单轮互动不足以改变整体印象，此时 impression 必须返回 null。"
    "sentences 仍独立遵守心境字段的要求，不能将接话规矩或人物事实转存其中。"
    "只有出现新的长期底色或阶段近况的明确变化时，才在保留已有内容的基础上调整。"
    "阶段变化只用对方本人（user）本轮原话证据覆盖对应近况，严禁使用 assistant 转述或跨人联想作人物事实。"
    "用简短的第一人称写我如今的理解，不写性格分析报告；只按对方亲口说过、做过的事实写。"
    "涉及对方维护或修改我时，只保留对方的状态与处境，丢弃关于我的设计意图、期望与评价。"
    "如果对方只是要求照抄一种评价，不把它当作我自己的看法；上一版是我亲手写的，就承接原有判断。"
    "有变化时，输出 text（完整印象，可含【长期底色】与【阶段近况】分栏），speaker 恒为 user，"
    "quote 是该说话者本轮原话中的逐字短句。"
    "只输出字段，不给示例台词。"
)




def build_impression_prompt(
    *, partner_name: str = "", partner_role: str = DEFAULT_PARTNER_ROLE, partner_pronoun: str = "他",
    is_maintainer: bool = True,
) -> str:
    text = IMPRESSION_PROMPT
    name = (partner_name or "").strip()
    role = (partner_role or "").strip() or DEFAULT_PARTNER_ROLE
    pronoun = "她" if (partner_pronoun or "").strip() == "她" else "他"

    if not is_maintainer:
        text = (
            text.replace("对方（user）是我的挚友（称呼他一律用「他」，不用「她」）。",
                         "对方（user）是当前对话者（外部私聊朋友/陌生网友，绝不沿用固定的关系称呼）。")
            .replace("我最近眼里的他", "我对当前对话者的印象")
            .replace("像挚友之间", "在正常交往中")
            .replace("我的挚友", "当前对话者")
        )
    else:
        if name:
            text = text.replace("对方（user）", f"{name}（user）", 1)
        if role != "挚友":
            text = text.replace("我的挚友", f"我的{role}").replace("像挚友之间", f"像{role}之间")
        if pronoun == "她":
            text = (
                text.replace("（称呼他一律用「他」，不用「她」）", "（称呼她一律用「她」，不用「他」）")
                .replace("我最近眼里的他", "我最近眼里的她")
                .replace("我对他", "我对她")
                .replace("他维护或修改我", "她维护或修改我")
                .replace("他的状态", "她的状态")
                .replace("如果他", "如果她")
                .replace("听他说", "听她说")
                .replace("他最近", "她最近")
            )
    return text


def _impression_schema_ready(conn: sqlite3.Connection, *, require_wal: bool = False) -> bool:
    if require_wal:
        mode_row = conn.execute("PRAGMA journal_mode").fetchone()
        if mode_row is None or str(mode_row[0]).lower() != "wal":
            return False
    version_row = conn.execute("PRAGMA user_version").fetchone()
    if version_row is None or int(version_row[0]) < CURRENT_SCHEMA_VERSION:
        return False
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(impression_revisions)").fetchall()
    }
    return "source_context" in columns


def _ensure_impression_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.is_file():
        with closing(connect_readonly(db_path)) as conn:
            if _impression_schema_ready(conn, require_wal=True):
                return
    with _IMPRESSION_INIT_LOCK:
        if db_path.is_file():
            with closing(connect_readwrite(db_path, create=False)) as conn:
                if _impression_schema_ready(conn):
                    return
        from ..storage.database import open_database
        from ..infra.textutil import hermes_home

        with closing(open_database(
            db_path, archive_dir=Path(hermes_home()) / "memory-archive"
        )):
            pass


def _append_with_conn(
    conn: sqlite3.Connection, target: ImpressionTarget, impression: Impression,
) -> bool:
    _, profile_key, owner_key = target
    owns_tx = not conn.in_transaction
    if owns_tx:
        conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            """INSERT INTO impression_revisions
               (profile_key,owner_key,revision,text,speaker,evidence,turn_id,updated_at,recorded_at,
                baseline,recent,source_context)
               SELECT ?,?,?,?,?,?,?,?,(julianday('now')-2440587.5)*86400.0,?,?,?
               WHERE COALESCE((SELECT MAX(revision) FROM impression_revisions
                               WHERE profile_key=? AND owner_key=?),0)=?""",
            (profile_key, owner_key, impression.revision, impression.text, impression.speaker,
             impression.evidence, impression.turn_id, impression.updated_at,
             impression.baseline, impression.recent, impression.source_context,
             profile_key, owner_key, impression.revision - 1),
        )
        if owns_tx:
            conn.commit()
        return cursor.rowcount == 1
    except BaseException:
        if owns_tx and conn.in_transaction:
            conn.rollback()
        raise


def _append(target: ImpressionTarget, impression: Impression) -> bool:
    directory, _, _ = target
    db_path = directory / "sanae.db"
    _ensure_impression_db(db_path)
    with closing(connect_readwrite(db_path, create=False)) as conn:
        return _append_with_conn(conn, target, impression)


def write_impression(target: ImpressionTarget, text: str, *, turn_id: str,
                     source_context: str = "",
                     max_retries: int = DEFAULT_CAS_MAX_RETRIES) -> Impression:
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    directory, _, _ = target
    db_path = directory / "sanae.db"
    _ensure_impression_db(db_path)
    b, r = _parse_sections(text)
    cleaned_text = text.strip()
    cleaned_source = str(source_context or "")
    with closing(connect_readwrite(db_path, create=False)) as conn:
        for attempt in range(max_retries + 1):
            base = _read_impression_with_conn(conn, target)
            state = Impression(
                revision=(base.revision if base else 0) + 1,
                text=cleaned_text,
                speaker="",
                evidence="",
                turn_id=turn_id,
                updated_at=time.time(),
                baseline=b,
                recent=r,
                source_context=cleaned_source,
            )
            if _append_with_conn(conn, target, state):
                return state
            if attempt < max_retries:
                cas_backoff_sleep(attempt)
    raise CasConflictError(
        f"impression CAS retry limit exceeded after {max_retries} retries"
    )


def apply_impression(
    target: ImpressionTarget, base: Impression | None, candidate: object,
    *, turn_id: str, now: float, user: str, assistant: str, source_context: str = "",
    user_authored: bool = False,
) -> bool:
    if not isinstance(candidate, dict):
        return False
    speaker = str(candidate.get("speaker") or "").strip()
    evidence = str(candidate.get("quote") or candidate.get("evidence") or "").strip()

    if speaker != "user" or not evidence:
        return False

    raw_text = str(candidate.get("text") or "").strip()
    cand_baseline = str(candidate.get("baseline") or "").strip()
    cand_recent = str(candidate.get("recent") or "").strip()

    if not raw_text and not cand_baseline and not cand_recent:
        return False

    if user_authored:
        authored = str(user or "")
    else:
        authored, _ = _split_known_meme_notes(_normalize_input(user))
    from .daily_semantic_validation import _DESIGN_INTENT_CLAUSE

    check_str = f"{raw_text} {cand_baseline} {cand_recent}"
    if _DESIGN_INTENT_CLAUSE.search(check_str) or _DESIGN_INTENT_CLAUSE.search(evidence):
        return False

    if not _assistant_quote_safe(evidence) or evidence not in authored:
        return False

    base_b = base.baseline if base else ""
    base_r = base.recent if base else ""
    if base and not base_b and not base_r and base.text:
        base_b, base_r = _parse_sections(base.text)

    if cand_recent or cand_baseline:
        new_baseline = cand_baseline or base_b
        new_recent = cand_recent or base_r
        text = _compose_sections(new_baseline, new_recent, raw_text)
    else:
        parsed_b, parsed_r = _parse_sections(raw_text)
        if parsed_b and parsed_r:
            new_baseline = parsed_b
            new_recent = parsed_r
            text = raw_text
        elif parsed_r:
            new_baseline = base_b
            new_recent = parsed_r
            text = _compose_sections(new_baseline, new_recent, raw_text)
        else:
            new_baseline = base_b or parsed_b
            new_recent = parsed_b if base_b else ""
            text = raw_text

    if base and (text == base.text or (speaker, evidence) == (base.speaker, base.evidence)):
        return False

    return _append(target, Impression(
        revision=(base.revision if base else 0) + 1,
        text=text,
        speaker=speaker,
        evidence=evidence,
        turn_id=turn_id,
        updated_at=now,
        baseline=new_baseline,
        recent=new_recent,
        source_context=str(source_context or ""),
    ))
