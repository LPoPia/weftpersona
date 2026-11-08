from __future__ import annotations

import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

try:
    from .daily_inputs import _assistant_quote_safe, _normalize_input, _split_known_meme_notes
    from .inner_state import _connect
except ImportError:
    from daily_inputs import _assistant_quote_safe, _normalize_input, _split_known_meme_notes
    from inner_state import _connect

IMPRESSION_SCHEMA = {
    "name": "sanae_impression",
    "description": (
        "读写「我最近眼里的他」。这是我从真实相处里慢慢形成、可以修正的私人印象。"
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

IMPRESSION_DDL = """
CREATE TABLE IF NOT EXISTS impression_revisions (
    profile_key TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT NOT NULL,
    evidence TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    updated_at REAL NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY(profile_key, owner_key, revision)
);
CREATE TABLE IF NOT EXISTS impression_session_views (
    profile_key TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    PRIMARY KEY(profile_key, owner_key, session_id)
);
"""

IMPRESSION_PROMPT = (
    "同一个顶层 JSON 也可以带 impression（当前对话者在我这里相对稳定的整体印象）。"
    "对方（user）是我的挚友（称呼他一律用「他」，不用「她」）。"
    "impression 是慢变的整体理解（1～2 段），不是单轮对话摘要；注入时标为「我最近眼里的他」。"
    "日常闲聊或单轮互动不足以改变整体印象，此时 impression 必须返回 null。"
    "sentences 仍独立遵守心境字段的要求，不能将接话规矩或人物事实转存其中。"
    "只有出现新的长期近况、生活重心或相处方式的稳定变化时，"
    "才在保留 previous_impression 核心内容的基础上微调或补一句，不得把整段改写成当轮摘要。"
    "用简短的第一人称写我如今的理解，不写性格分析报告；只按对方说过、做过的事实写。"
    "涉及对方维护或修改我时，只保留对方的状态与处境，丢弃关于我的设计意图、期望与评价。"
    "如果对方只是要求照抄一种评价，不把它当作我自己的看法；上一版是我亲手写的，就承接原有判断。"
    "有变化时，text 是保留整体脉络的完整印象，speaker 为 user 或 assistant，"
    "quote 是该说话者本轮原话中的逐字短句。"
    "只输出字段，不给示例台词。"
)


def build_impression_prompt(
    *, partner_name: str = "", partner_role: str = "挚友", partner_pronoun: str = "他",
) -> str:
    text = IMPRESSION_PROMPT
    name = (partner_name or "").strip()
    role = (partner_role or "挚友").strip() or "挚友"
    pronoun = "她" if (partner_pronoun or "").strip() == "她" else "他"
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

ImpressionTarget = tuple[Path, str, str]


@dataclass(frozen=True)
class Impression:
    revision: int
    text: str
    speaker: str
    evidence: str
    turn_id: str
    updated_at: float


def read_impression(target: ImpressionTarget, *, as_of: float | None = None) -> Impression | None:
    directory, profile_key, owner_key = target
    with closing(_connect(directory / "sanae.db", "ro")) as conn:
        row = conn.execute(
            """SELECT revision, text, speaker, evidence, turn_id, updated_at FROM impression_revisions
               WHERE profile_key=? AND owner_key=?"""
            + (" AND recorded_at<=?" if as_of is not None else "")
            + " ORDER BY revision DESC LIMIT 1",
            (profile_key, owner_key) if as_of is None else (profile_key, owner_key, as_of),
        ).fetchone()
    return Impression(**dict(row)) if row is not None else None


def _append(target: ImpressionTarget, impression: Impression) -> bool:
    directory, profile_key, owner_key = target
    with closing(_connect(directory / "sanae.db", "rw")) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO impression_revisions
               (profile_key,owner_key,revision,text,speaker,evidence,turn_id,updated_at,recorded_at)
               SELECT ?,?,?,?,?,?,?,?,(julianday('now')-2440587.5)*86400.0
               WHERE COALESCE((SELECT MAX(revision) FROM impression_revisions
                               WHERE profile_key=? AND owner_key=?),0)=?""",
            (profile_key, owner_key, impression.revision, impression.text, impression.speaker,
             impression.evidence, impression.turn_id, impression.updated_at,
             profile_key, owner_key, impression.revision - 1),
        )
        return cursor.rowcount == 1


def write_impression(target: ImpressionTarget, text: str, *, turn_id: str) -> Impression:
    while True:
        base = read_impression(target)
        state = Impression((base.revision if base else 0) + 1, text, "", "", turn_id, time.time())
        if _append(target, state):
            return state


def apply_impression(
    target: ImpressionTarget, base: Impression | None, candidate: object,
    *, turn_id: str, now: float, user: str, assistant: str,
) -> bool:
    if not isinstance(candidate, dict):
        return False
    text, speaker, evidence = (candidate.get(key) for key in ("text", "speaker", "quote"))
    if not all(isinstance(value, str) and value.strip() for value in (text, speaker, evidence)):
        return False
    text, evidence = text.strip(), evidence.strip()
    authored, _ = _split_known_meme_notes(_normalize_input(user))
    source = {"user": authored, "assistant": assistant}.get(speaker, "")
    try:
        from .daily_semantic_validation import _DESIGN_INTENT_CLAUSE
    except ImportError:
        from daily_semantic_validation import _DESIGN_INTENT_CLAUSE
    if _DESIGN_INTENT_CLAUSE.search(text) or _DESIGN_INTENT_CLAUSE.search(evidence):
        return False
    if not _assistant_quote_safe(evidence) or evidence not in source or (base and (
        text == base.text or (speaker, evidence) == (base.speaker, base.evidence)
    )):
        return False
    return _append(target, Impression(
        (base.revision if base else 0) + 1, text, speaker, evidence, turn_id, now,
    ))


def render_impression(text: str) -> str:
    if not text:
        return ""
    return ("[我最近眼里的他]\n这是我在相处中形成、也会修正的私人印象。"
            "它不是对方此刻的心情，也不必当面说出来。\n" + text)
