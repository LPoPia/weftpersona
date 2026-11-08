from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from .daily_inputs import (
        _is_question_clause, _is_second_person_clause,
        _looks_secret, _scrub_face_tags, _split_clauses,
    )
    from .daily_rules import (
        BLOCK_MARKER, KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
        MAX_ACTIVE_PER_KIND, MAX_BASIS_CHARS, MAX_BLOCK_CHARS, MAX_BLOCK_TEXT_CHARS,
        MAX_QUOTE_CHARS, MAX_TEXT_CHARS, MAX_YESTERDAY_CHARS, MAX_YESTERDAY_ITEMS,
        SOURCE_INFERENCE, SOURCE_STATEMENT, STALE_TTL_DAYS, STALE_TTL_MAX_DAYS, THREAD_TTL_DAYS,
        _BLOCK_LABELS, _BLOCK_NOTE, _CHAT_META, _Decision,
        _INFER_LABEL, _ROUTINE, _SUSTAINED_HORIZON,
        _TRANSIENT_STATE, _YESTERDAY_LABEL, _YESTERDAY_NOTE,
        _norm_text, _now_iso,
    )
    from .textutil import _tokens
except ImportError:
    from daily_inputs import (
        _is_question_clause, _is_second_person_clause,
        _looks_secret, _scrub_face_tags, _split_clauses,
    )
    from daily_rules import (
        BLOCK_MARKER, KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
        MAX_ACTIVE_PER_KIND, MAX_BASIS_CHARS, MAX_BLOCK_CHARS, MAX_BLOCK_TEXT_CHARS,
        MAX_QUOTE_CHARS, MAX_TEXT_CHARS, MAX_YESTERDAY_CHARS, MAX_YESTERDAY_ITEMS,
        SOURCE_INFERENCE, SOURCE_STATEMENT, STALE_TTL_DAYS, STALE_TTL_MAX_DAYS, THREAD_TTL_DAYS,
        _BLOCK_LABELS, _BLOCK_NOTE, _CHAT_META, _Decision,
        _INFER_LABEL, _ROUTINE, _SUSTAINED_HORIZON,
        _TRANSIENT_STATE, _YESTERDAY_LABEL, _YESTERDAY_NOTE,
        _norm_text, _now_iso,
    )
    from textutil import _tokens


def _yesterday(today: str) -> str:
    try:
        return (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    except ValueError:
        return ""


def _row_source(row: Any) -> str:
    try:
        value = str(row["source_kind"] or "").strip()
    except Exception:
        return SOURCE_STATEMENT
    return value or SOURCE_STATEMENT


def _sustained_horizon(text: str) -> bool:
    return _SUSTAINED_HORIZON.search(str(text or "")) is not None


def _default_ttl(today: str, sustained: bool = False) -> Optional[str]:
    days = STALE_TTL_MAX_DAYS if sustained else STALE_TTL_DAYS
    try:
        return (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    except ValueError:
        return None


def _shares_content(left: str, right: str, min_len: int = 3) -> bool:
    a, b = _norm_text(left), _norm_text(right)
    if len(a) < min_len or len(b) < min_len:
        return False
    return any(a[index:index + min_len] in b for index in range(len(a) - min_len + 1))


def _is_transient_state(text: str) -> bool:
    return _TRANSIENT_STATE.search(str(text or "")) is not None


_THREAD_COMMON_WORDS = (
    "好奇", "随口", "提到", "提起", "那个", "什么", "怎么", "到底", "为什么", "一下",
    "了解", "琢磨", "查查", "看看", "自己", "早苗", "今天", "最近", "觉得", "打算",
    "回头", "细聊", "说的",
)


def _thread_core_text(text: str) -> str:
    value = str(text or "")
    for word in _THREAD_COMMON_WORDS:
        value = value.replace(word, " ")
    return value


def _thread_core_tokens(text: str) -> set[str]:
    """Core tokens of a curiosity thread: ASCII words plus Chinese 2/3-grams."""
    value = _thread_core_text(text)
    tokens = set(_tokens(value))
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.update(run[index:index + 2] for index in range(len(run) - 1))
        tokens.update(run[index:index + 3] for index in range(len(run) - 2))
    return {token for token in tokens if len(token) >= 2}


def _thread_same_topic(thread_text: str, target_text: str) -> bool:
    """Same curiosity item, judged by core proper nouns / bigrams rather than wording."""
    left_text, right_text = _thread_core_text(thread_text), _thread_core_text(target_text)
    shared = _thread_core_tokens(thread_text) & _thread_core_tokens(target_text)
    if not shared:
        return False
    if any(token.isascii() for token in shared):
        return True
    if any(len(token) >= 3 and token in right_text for token in shared):
        return True
    return sum(1 for token in shared if len(token) == 2) >= 2


_THREAD_ACTIVITY_HEAD_RE = re.compile(r"定时(?:自主)?活动完成：[^。！\n]+[。！]?")
_THREAD_SEGMENT_RE = re.compile(r"[。！？!?\n]")
_THREAD_UNEXPLORED_RE = re.compile(
    r"没查|还没|未查|没看|还没看|没去查|没搜|没空|没来得及|没弄清|没搞懂|没弄明白|"
    r"不知道|不清楚|还不懂|没细看|先不查|先不看|以后再|改天再|下次再|晚点再|有空再|"
    r"回头再|等有空|改天看|以后有空|打算去查|想找时间查|还在好奇|还是好奇"
)
_THREAD_EXPLORED_RE = re.compile(
    r"B站观看|收到该视频的抽样画面分析|阅读了该视频的字幕|查了|查到|查清|搜了|搜到|"
    r"翻了|翻到|看了|看到|读了|读到|找到|了解了|去了解|研究了一下|琢磨明白|"
    r"弄明白|搞明白|搞懂|弄清|看懂|原来|原理是|其实是|靠的是|核心是|机制是|结构是|是因为"
)


_THREAD_RECORD_SPLIT_RE = re.compile(r"\n(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*[|｜]|\s*-\s)")
_VIDEO_WATCH_RECORD_RE = re.compile(r"B站观看|收到该视频的抽样画面分析|阅读了该视频的字幕")
_BOOK_TITLE_RE = re.compile(r"《[^》]{2,80}》")


def _thread_exploration_completed(thread_text: str, activity_text: str) -> bool:
    """True only when an activity record actually explored the thread (not deferred/unknown)."""
    raw = str(activity_text or "").strip()
    if not raw:
        return False
    thread_titles = set(_BOOK_TITLE_RE.findall(str(thread_text or "")))
    records = _THREAD_RECORD_SPLIT_RE.split(raw)
    for record in records:
        body = _THREAD_ACTIVITY_HEAD_RE.sub("", record)
        if _THREAD_UNEXPLORED_RE.search(body):
            continue
        if _VIDEO_WATCH_RECORD_RE.search(record):
            rec_titles = set(_BOOK_TITLE_RE.findall(record))
            if thread_titles and rec_titles and (thread_titles & rec_titles):
                # Watching the origin video that spawned the curiosity is not exploring the answer.
                continue
        matched = [
            piece for segment in _THREAD_SEGMENT_RE.split(body)
            if (piece := segment.strip()) and _thread_same_topic(thread_text, piece)
        ]
        if matched and any(_THREAD_EXPLORED_RE.search(piece) for piece in matched):
            return True
    return False


def _resolve_daily_expiry(decision: "_Decision", row: Any, today: str) -> Optional[str]:
    existing = ""
    stored = ""
    if row is not None:
        try:
            existing = str(row["expires_on"] or "")
            stored = str(row["text"] or "")
        except Exception:
            existing, stored = "", ""
    if str(decision.kind) == KIND_THREAD:
        if existing:
            return existing
        if decision.expires_on:
            return decision.expires_on
        try:
            return (date.fromisoformat(today) + timedelta(days=THREAD_TTL_DAYS)).isoformat()
        except ValueError:
            return None
    if decision.expires_on:
        return decision.expires_on
    if str(decision.kind) != KIND_CONSTRAINT:
        return existing or None
    quote = str(decision.quote or "")
    text = str(decision.text or "")
    if _is_transient_state(text or quote):
        return today
    if existing and existing > today and not _shares_content(quote, stored):
        return existing
    return _default_ttl(today, _sustained_horizon(quote or text or stored))


_QUOTED_ROW = re.compile(r'^[“"「『].*[”"」』]$', re.S)


def _view_noise(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if _QUOTED_ROW.match(value):
        return True
    clauses = _split_clauses(value) or [value]
    return (
        any(_CHAT_META.fullmatch(clause) for clause in clauses)
        or any(_is_question_clause(clause) for clause in clauses)
        or any(_is_second_person_clause(clause) for clause in clauses)
    )


def _yesterday_items(rows: Sequence[Any]) -> List[str]:
    primary: List[str] = []
    routine: List[str] = []
    seen: set = set()
    for row in rows:
        text = str(row["text"] or "").strip()
        if not text or _view_noise(text):
            continue
        key = _norm_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        if _ROUTINE.search(text):
            routine.append(text)
        else:
            primary.append(text)
    ordered = primary if primary else routine
    picked: List[str] = []
    total = 0
    for text in ordered:
        if len(picked) >= MAX_YESTERDAY_ITEMS:
            break
        if picked and total + len(text) > MAX_YESTERDAY_CHARS:
            break
        picked.append(text)
        total += len(text)
    return picked


def _fact_rank(row: Any) -> Tuple[int, int]:
    text = str(row["text"] or "")
    return (1 if _ROUTINE.search(text) else 0, -int(row["entry_id"]))


class DailyStateMixin:

    def _daily_apply(
        self, scope: Tuple[str, str], decisions: List[_Decision], today: str, session_id: str,
    ) -> bool:
        owner, profile = scope
        now = _now_iso()
        try:
            self.expire_stale_daily_facts(
                owner_key=owner, profile_key=profile, today=today, now=now,
            )
            for decision in decisions:
                self._daily_apply_one(owner, profile, decision, today, now, session_id)
        except Exception as exc:
            logger.warning("sanae_mem: daily_context write failed (%s)", type(exc).__name__)
            return False
        return True

    def _daily_apply_one(
        self, owner: str, profile: str, decision: _Decision, today: str, now: str, session_id: str,
    ) -> None:
        kind, slot, action = decision.kind, decision.slot, decision.action
        if kind == KIND_THREAD and (action == "clear" or not decision.text):
            if action != "clear":
                return
        if action == "clear":
            row = self.find_active_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot,
            )
            if row is None:
                return
            for _attempt in (0, 1):
                if self.cas_supersede_daily_entry(
                    owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
                    version=int(row["version"]), status="closed", now=now,
                ):
                    return
                row = self.find_active_daily_entry(
                    owner_key=owner, profile_key=profile, kind=kind, slot=slot,
                )
                if row is None:
                    return
            raise RuntimeError("daily_context clear conflict")

        text = _scrub_face_tags(str(decision.text or "")).strip()
        quote = _scrub_face_tags(str(decision.quote or ""))[:MAX_QUOTE_CHARS]
        if kind == KIND_THREAD and _ROUTINE.search(text):
            return
        if _looks_secret(text) or _looks_secret(quote):
            logger.info("sanae_mem: daily_context rejected (secret)")
            return
        row = self.find_active_daily_entry(
            owner_key=owner, profile_key=profile, kind=kind, slot=slot,
        )
        if kind == KIND_THREAD and row is None:
            for candidate in self.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=today):
                if str(candidate["kind"]) != KIND_THREAD:
                    continue
                candidate_text = str(candidate["text"] or "")
                candidate_quote = str(candidate["source_quote"] or "")
                if (_shares_content(text, candidate_text)
                        or _shares_content(quote, candidate_quote)
                        or _thread_same_topic(
                            f"{text} {quote}", f"{candidate_text} {candidate_quote}")):
                    row = candidate
                    slot = str(candidate["slot"])
                    break
        expires = _resolve_daily_expiry(decision, row, today)
        opened_on = str(row["opened_on"] or today) if (kind == KIND_THREAD and row is not None) else today
        if row is not None:
            if not text or len(text) > MAX_TEXT_CHARS:
                raise RuntimeError("daily_context update not storable")
            if (str(row["text"] or "") == text
                    and str(row["expires_on"] or "") == str(expires or "")
                    and _row_source(row) == decision.source_kind):
                return
            if not self.cas_update_daily_entry(
                owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
                version=int(row["version"]), text=text, opened_on=opened_on,
                expires_on=expires, source_quote=quote,
                source_turn=session_id, now=now, source_kind=decision.source_kind,
            ):
                raise RuntimeError("daily_context update conflict")
            return

        if not text or len(text) > MAX_TEXT_CHARS:
            return
        if kind != KIND_FACT:
            stats = self.daily_active_stats(owner_key=owner, profile_key=profile)
            if stats["counts"].get(kind, 0) >= MAX_ACTIVE_PER_KIND.get(kind, 0):
                logger.info("sanae_mem: daily_context rejected (%s quota)", kind)
                return
        try:
            self.open_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot, text=text,
                opened_on=today, now=now, expires_on=expires,
                source_quote=quote, source_turn=session_id,
                source_kind=decision.source_kind,
                author_kind=self._daily_author_kind(),
            )
            return
        except Exception:
            row = self.find_active_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot,
            )
            if row is None:
                raise
            expires = _resolve_daily_expiry(decision, row, today)
            if kind == KIND_THREAD:
                opened_on = str(row["opened_on"] or today)
        if not self.cas_update_daily_entry(
            owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
            version=int(row["version"]), text=text, opened_on=opened_on,
            expires_on=expires, source_quote=quote,
            source_turn=session_id, now=now, source_kind=decision.source_kind,
        ):
            raise RuntimeError("daily_context update conflict")

    def _daily_render(
        self, scope: Tuple[str, str], today: str, suppressed: Sequence[Tuple[str, str]] = (),
    ) -> str:
        owner, profile = scope
        try:
            rows = self.list_active_daily_entries(
                owner_key=owner, profile_key=profile, today=today,
            )
            yesterday_rows = self.list_daily_entries_for_day(
                owner_key=owner, profile_key=profile, day=_yesterday(today),
                kind=KIND_FACT, include_statuses=("active", "expired"),
                source_kind=SOURCE_STATEMENT,
            )
        except Exception as exc:
            logger.warning("sanae_mem: daily_context read failed (%s)", type(exc).__name__)
            return ""
        skip = set(suppressed)
        labels = dict(_BLOCK_LABELS)
        lines: List[str] = []
        for row in rows:
            if str(row["kind"]) != KIND_CONSTRAINT:
                continue
            if (KIND_CONSTRAINT, str(row["slot"])) in skip:
                continue
            text = str(row["text"] or "").strip()
            if text:
                lines.append("· %s：%s" % (labels[KIND_CONSTRAINT], text))
        fact_rows = [
            row for row in rows
            if str(row["kind"]) == KIND_FACT and _row_source(row) != SOURCE_INFERENCE
        ]
        fact_rows.sort(key=_fact_rank)
        used = 0
        for row in fact_rows:
            if (KIND_FACT, str(row["slot"])) in skip:
                continue
            text = str(row["text"] or "").strip()
            if not text or _view_noise(text):
                continue
            if used + len(text) > MAX_BLOCK_TEXT_CHARS:
                continue
            used += len(text)
            lines.append("· %s：%s" % (labels[KIND_FACT], text))
        for row in rows:
            if str(row["kind"]) != KIND_FACT or _row_source(row) != SOURCE_INFERENCE:
                continue
            if (KIND_FACT, str(row["slot"])) in skip:
                continue
            text = str(row["text"] or "").strip()
            if not text:
                continue
            basis = str(row["source_quote"] or "").strip()[:MAX_BASIS_CHARS]
            if basis:
                lines.append("· %s：%s（未确认；依据「%s」）" % (_INFER_LABEL, text, basis))
            else:
                lines.append("· %s：%s（未确认）" % (_INFER_LABEL, text))
        stars = _yesterday_items(yesterday_rows)
        if stars:
            lines.append("· %s：%s%s" % (_YESTERDAY_LABEL, "；".join(stars), _YESTERDAY_NOTE))
        if not lines:
            return ""
        header = BLOCK_MARKER + "\n" + _BLOCK_NOTE
        block = header
        for line in lines:
            candidate = block + "\n" + line
            if len(candidate) > MAX_BLOCK_CHARS:
                break
            block = candidate
        if block == header:
            return ""
        return block
