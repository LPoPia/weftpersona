from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol

from ..infra.atomic_io import atomic_json_write, file_lock
from ..infra.chat_scope import in_public_interaction
from ..infra.host_bridge import is_test_isolated_turn
from ..infra import auxiliary
from ..infra.contracts import CasConflictError, DEFAULT_CAS_MAX_RETRIES, cas_backoff_sleep
from ..infra.defaults import DEFAULT_PARTNER_ROLE, DEFAULT_PERSONA_NAME
from ..storage.candidates import parse_candidate_result, parse_grounded_user_result, parse_user_demote_result
from .inner_state_decay import _kept_words, _slow_variables
from ..storage.inner_state_store import InnerState, append_state, latest_state, owner_state
from ..infra.model_output import _load_braced_json
from ..storage.curve import CURVE_MIN_GAP
from ..storage.database import open_database
from ..storage.sqlite_facade import SqliteFacade
from ..storage.storage import upsert_chroma_event_record
from .memory_evidence import UserEvidence
from .source_context import self_attribution, source_identity, source_json
from ..infra.status_store import _STATUS_WRITER, read_status_file, write_component_status
from ..infra.textutil import _embed, _fingerprint, _load_plugin_config, _pack_vector, _tokens
from ..infra.words import _TOPIC_WORDS, _noise_shape

logger = logging.getLogger(__name__)

_SHORT_EVENT_IMPORTANCE = 0.5
_SHORT_EVENT_TTL_DAYS = 14

_RESIDENT_ROUTE_SYSTEM_TEMPLATE = (
    "你是{persona_name}记忆系统的旁路整理器。{persona_name}刚调用 memory 工具想往常驻区（memory）写一段话。"
    "请判断这段话应该放行进入长期常驻（allow），还是拦截并分流到短期心里话/短期事件库（inner_state）。"
    "只输出一个 JSON 对象，不要解释、不要代码块："
    '{"route": "allow|inner_state", "inner_thought": "分流到短期心里话时的1到2句第一人称心里话（allow、纯教条总结或无特殊心绪时填空串）", '
    '"event_fact": "原条里已发生的具体客观事件（一句第一人称事实），没有就 null", "reason": "简短原因"}。\n'
    "判定规则：\n"
    "1. 只有「跨时段沉淀的关系里程碑、长期稳定的相处默契、或对既有常驻条目的修正合并」才判 allow；"
    "凡属于以下三类，一律判 inner_state（拦截不进常驻 MEMORY）："
    "① 关于{persona_name}自身系统机制、记忆遗忘规则、提示词修改、行为准则或调教目标的讨论与约定（如「记忆遗忘机制由他设定」「我不全权管记忆」等元规则）；"
    "② 单次日常琐事、当日偶然小事（如今天发了某个表情包、聊了某件日常小事、一次操作小失误）或单次短期约定（此类具体小事走 event_fact 进事件库，不占常驻窗口）；"
    "③ 当下的揣测与猜想（如猜他发某句话/某个表情是因为心虚、生气、嫌弃或累了）、挨批或争执后的自我检讨与反思、"
    "给自己立的说话规矩（如以后我不能找补/不能顺着猜/要先听他说）、或一时的委屈/不服/心虚等当下心绪。\n"
    "2. 判 inner_state 时，inner_thought 只提炼她此刻心里的真实情绪、在意或挂念（如「刚才看岔了他的话，心里还有点过意不去」；若涉及他维护或调系统机制，只留对他操心忙碌的挂念），"
    "绝不把「接话前我得先怎么看」「记忆遗忘机制由他定」「以后不能再猜岔」这类系统机制条文、说话规矩、答题教条或自我检讨写进 inner_thought；"
    "若原条纯粹是总结规矩/系统机制，或只是普通单次日常事实而无特殊心绪，inner_thought 填空串 \"\"。用户（对方）是{persona_name}的挚友，代词一律用「他」，不用「她」。\n"
    "3. 判 inner_state 时，原条里若确实记着一件已经发生、能说清原委的具体客观事情（做过什么、他说过或发过什么、出了什么岔子、结果如何），"
    "在 event_fact 里用一句第一人称、只讲事实的话写清它（如「今天帮他看课表时，我把周四的课看成了周三」）；"
    "去掉检讨、规矩和骂自己的话（「以后不能」「下次必须」「瞎猜」「犯蠢」之类），也不留情绪评价，只留那件事本身。"
    "涉及他维护或修改{persona_name}时，只保留他的状态和处境（如熬夜、累、觉得某件事难、在忙什么），"
    "丢弃关于{persona_name}的设计意图、系统机制设定和期望（如想把她改成什么样、记忆规则怎么定、改动的目的、对模型的评价、调教目标）。"
    "纯情绪拌嘴、心意揣测、反问逗弄、说自己该守什么规矩、关于{persona_name}系统机制/记忆规则/调教目标的讨论，或根本没有具体发生过的事时，event_fact 返回 null；判 allow 时同样返回 null。"
)


@dataclass(frozen=True)
class ResidentRouteConfig:
    """Explicit configuration and turn identity for resident MEMORY.md routing."""

    data_dir: Path
    profile_key: str
    owner_key: str = ""
    session_id: str = ""
    turn_id: str = ""
    platform: str = ""
    context: str = "primary"
    internal: bool = False
    skip_inner_write: bool = False
    persona_name: str = DEFAULT_PERSONA_NAME
    partner_name: str = ""
    partner_role: str = DEFAULT_PARTNER_ROLE
    partner_pronoun: str = "他"


@dataclass(frozen=True)
class ResidentRouteOutcome:
    """Typed result of resident MEMORY.md write routing."""

    routed: bool
    route: str
    message: str = ""
    inner_state_text: str = ""
    event_id: Optional[int] = None
    error: str = ""


class _AuxiliaryReply(Protocol):
    @property
    def text(self) -> str: ...


class AuxiliaryModel(Protocol):
    def complete(
        self, messages: list[dict[str, str]], /, *, task: str, temperature: float,
        max_tokens: int, timeout: float, purpose: str,
    ) -> _AuxiliaryReply | str | None: ...


class _InnerStateWriter(Protocol):
    def __call__(
        self, text: str, /, *, turn_id: str, owner_key: str, internal: bool,
    ) -> InnerState: ...


def is_memory_target(target: Any) -> bool:
    """Return True when *target* refers to the resident MEMORY.md file."""
    normalized = str(target or "").strip().lower().replace("\\", "/").rstrip("/")
    return normalized in {"memory", "memory.md"} or normalized.endswith("/memory.md")


def extract_resident_write_texts(
    *, action: str, target: str, content: Any = None, operations: Any = None,
    old_text: Any = None,
) -> list[str]:
    """Extract proposed resident memory texts for add/replace/merge/patch operations."""
    texts: list[str] = []
    mutating = ("add", "replace", "merge", "patch")
    if isinstance(operations, list) and operations:
        for op in operations:
            if not isinstance(op, Mapping):
                continue
            op_target = op.get("target") if op.get("target") is not None else target
            op_action = str(op.get("action") or action or "").strip().lower()
            if is_memory_target(op_target) and op_action in mutating:
                body = str(op.get("content") or op.get("new_text") or op.get("text") or "").strip()
                if not body and op_action == "patch":
                    body = str(op.get("old_text") or "").strip()
                if body:
                    texts.append(body)
        return texts
    normalized_action = str(action or "").strip().lower()
    if is_memory_target(target) and normalized_action in mutating:
        body = str(content or "").strip()
        if not body and normalized_action == "patch":
            body = str(old_text or "").strip()
        if body:
            texts.append(body)
    return texts


def build_resident_route_system_prompt(cfg: ResidentRouteConfig) -> str:
    """Render the resident-write routing system prompt for *cfg*."""
    system_prompt = _RESIDENT_ROUTE_SYSTEM_TEMPLATE
    persona_name = str(cfg.persona_name or "").strip() or DEFAULT_PERSONA_NAME
    partner_name = str(cfg.partner_name or "").strip()
    partner_role = str(cfg.partner_role or "").strip() or DEFAULT_PARTNER_ROLE
    partner_pronoun = "她" if str(cfg.partner_pronoun or "他").strip() == "她" else "他"
    if partner_name:
        system_prompt = system_prompt.replace("用户（对方）", f"用户（{partner_name}）", 1)
    if partner_role != "挚友":
        system_prompt = system_prompt.replace("的挚友", f"的{partner_role}")
    if partner_pronoun == "她":
        system_prompt = system_prompt.replace("代词一律用「他」，不用「她」", "代词一律用「她」，不用「他」")
    return system_prompt.replace("{persona_name}", persona_name)


_FALLBACK_DB_LOCKS_GUARD = threading.Lock()
_FALLBACK_DB_LOCKS: dict[Path, threading.RLock] = {}


def _fallback_db_lock(db_path: Path) -> threading.RLock:
    resolved = Path(db_path).resolve()
    with _FALLBACK_DB_LOCKS_GUARD:
        lock = _FALLBACK_DB_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _FALLBACK_DB_LOCKS[resolved] = lock
        return lock


def _ensure_sanae_db(db_path: Path) -> None:
    resolved = Path(db_path).resolve()
    with _fallback_db_lock(resolved):
        with closing(open_database(resolved)):
            pass


def write_short_inner_state(
    data_dir: Path,
    profile_key: str,
    text: str,
    *,
    owner_key: str,
    turn_id: str = "",
    internal: bool = False,
    now: Optional[float] = None,
    max_retries: int = DEFAULT_CAS_MAX_RETRIES,
) -> InnerState:
    """Append a short-term inner_state revision directly into *data_dir*/sanae.db."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("心里话不能为空")
    owner = str(owner_key or "").strip()
    if not owner:
        raise ValueError("owner_key 必须来自宿主核验的对话者")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    resolved_dir = Path(data_dir).resolve()
    db_path = resolved_dir / "sanae.db"
    _ensure_sanae_db(db_path)
    target = (resolved_dir, str(profile_key))
    ts_now = time.time() if now is None else now
    for attempt in range(max_retries + 1):
        latest = latest_state(db_path, profile_key)
        owner_st = None if internal else owner_state(db_path, profile_key, owner)
        words, words_at = ("", None) if internal else _kept_words(owner_st, ts_now)
        state = InnerState(
            revision=(latest.revision if latest else 0) + 1,
            text=text.strip(),
            recent_words=words,
            recent_words_at=words_at,
            variables=_slow_variables(latest, ts_now),
            felt_at=ts_now,
            updated_at=ts_now,
            processed_at=latest.processed_at if latest else None,
            source="manual",
            thought_source="manual",
            turn_id=turn_id,
            changed=True,
            owner_key="" if internal else owner,
            text_kind="internal" if internal else "private",
            last_spark_at=owner_st.last_spark_at if owner_st else 0.0,
        )
        if append_state(db_path, target, state):
            return state
        if attempt < max_retries:
            cas_backoff_sleep(attempt)
    raise CasConflictError(
        f"short inner_state CAS retry limit exceeded after {max_retries} retries"
    )


def _resident_event_time_anchor(now: datetime) -> str:
    hour = now.hour
    period = "凌晨" if hour < 6 else "早上" if hour < 9 else "上午" if hour < 12 else "下午" if hour < 18 else "晚上"
    return f"{now:%Y-%m-%d %H:%M}|今天{period}"


def _resident_event_topic(text: str) -> str:
    for topic, words in _TOPIC_WORDS.items():
        if any(word in text for word in words):
            return topic
    return "其他"


def _record_short_write_status(data_dir: Path, now: datetime, *, session_id: str = "") -> None:
    if in_public_interaction() or is_test_isolated_turn(session_id):
        return
    path = Path(data_dir).resolve() / "status.json"
    today = now.date().isoformat()
    try:
        with _STATUS_WRITER, file_lock(path.with_suffix(".lock")):
            payload = read_status_file(path)
            if payload.get("date") != today:
                payload["date"] = today
                for key in (
                    "daily_writes", "daily_scheduled_asks", "daily_scheduled_attempts",
                    "daily_candidate_reviews", "daily_resident_reviews", "daily_compactor_runs",
                    "distill_errors_today",
                ):
                    payload[key] = 0
            payload["daily_writes"] = int(payload.get("daily_writes", 0)) + 1
            payload["last_write_at"] = now.isoformat()
            atomic_json_write(path, payload)
    except OSError as exc:
        logger.warning("sanae_mem: resident_memory_route status write failed (%s)", type(exc).__name__)


def _write_short_event_with_facade(
    facade: SqliteFacade,
    *,
    original: str,
    body: str,
    owner_key: str,
    session_id: str,
    platform: str,
    dt_now: datetime,
    vector: list[float],
) -> tuple[Optional[int], Optional[tuple[int, str, str, float, str]]]:
    ttl_dt = dt_now + timedelta(days=_SHORT_EVENT_TTL_DAYS)
    ttl_iso = ttl_dt.isoformat()
    created_iso = dt_now.isoformat()
    anchor = _resident_event_time_anchor(dt_now)
    raw = f"[{anchor}] {original}"
    summary = body[:150]
    topic = _resident_event_topic(body)
    fingerprint = _fingerprint(body)
    keywords = " ".join(_tokens(body))
    vec_blob = _pack_vector(vector)
    subject, fact_subject, involved, scope = self_attribution(owner_key)
    src_ctx = source_json({
        "kind": "self",
        "platform": str(platform or ""),
        "visibility": "private",
        "certainty": "observed",
        "ref": "resident-event:" + _fingerprint(original),
    })

    matched_duplicate: Optional[tuple[int, float, str, str, str]] = None
    unchanged_existing_id: Optional[int] = None
    created_event_id: Optional[int] = None

    with facade.transaction() as conn:
        if conn.execute("SELECT 1 FROM forgotten WHERE fingerprint=?", (fingerprint,)).fetchone():
            return None, None
        wanted_source = source_identity(src_ctx)
        candidates = conn.execute(
            """SELECT id, ts, raw, summary, topic, importance, recall_count,
                      last_active_at, updated_at, expired_at,
                      created_at, source_snapshot, source_snapshot_sha256,
                      summary_source, source_context
               FROM events
               WHERE original_fingerprint=? AND owner_key=? AND read_scope=?
                 AND expired_at IS NULL AND blacklist=0
               ORDER BY id DESC""",
            (fingerprint, str(owner_key), scope),
        ).fetchall()
        existing = next(
            (row for row in candidates if source_identity(row["source_context"]) == wanted_source),
            None,
        )
        if existing is not None:
            event_id = int(existing["id"])
            if str(existing["summary_source"] or "") != "gist_faded":
                if not conn.execute(
                    "SELECT 1 FROM event_occurrences WHERE event_id=? LIMIT 1",
                    (event_id,),
                ).fetchone():
                    conn.execute(
                        """INSERT INTO event_occurrences
                           (event_id, raw, observed_at, source_snapshot, source_snapshot_sha256)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            event_id,
                            str(existing["raw"] or ""),
                            str(existing["created_at"] or ""),
                            str(existing["source_snapshot"] or ""),
                            str(existing["source_snapshot_sha256"] or ""),
                        ),
                    )
                conn.execute(
                    """INSERT INTO event_occurrences
                       (event_id, raw, observed_at, source_snapshot, source_snapshot_sha256)
                       VALUES (?, ?, ?, ?, ?)""",
                    (event_id, original, created_iso, "", ""),
                )
            last_active_raw = str(
                existing["last_active_at"] or existing["updated_at"] or existing["created_at"] or ""
            ).strip()
            last_active_dt: Optional[datetime] = None
            if last_active_raw:
                try:
                    last_active_dt = datetime.fromisoformat(last_active_raw)
                except ValueError:
                    last_active_dt = None
            if last_active_dt is not None and last_active_dt.tzinfo is None:
                last_active_dt = last_active_dt.replace(tzinfo=dt_now.tzinfo)
            if last_active_dt is not None and (dt_now - last_active_dt) >= CURVE_MIN_GAP:
                bumped = min(1.0, max(_SHORT_EVENT_IMPORTANCE, float(existing["importance"] or 0.0) + 0.1))
                recall_count = int(existing["recall_count"] or 0) + 1
                conn.execute(
                    """UPDATE events
                       SET importance=?,
                           recall_count=?,
                           last_active_at=?,
                           updated_at=?
                       WHERE id=?""",
                    (bumped, recall_count, created_iso, created_iso, event_id),
                )
                display_doc = str(existing["summary"] or existing["raw"] or summary)
                existing_ts = str(existing["ts"] or anchor)
                matched_duplicate = (
                    event_id,
                    bumped,
                    display_doc,
                    str(existing["topic"] or topic),
                    existing_ts,
                )
            else:
                unchanged_existing_id = event_id
        else:
            cursor = conn.execute(
                """INSERT INTO events
                   (ts, raw, summary, summary_source, keywords, importance, confidence, topic, ttl_expire,
                    created_at, vec, session_id, a_conflict, affect, owner_key,
                    source_snapshot, source_snapshot_sha256, source_context,
                    experience_subject, fact_subject, involved_owner_keys, read_scope,
                    last_active_at, half_life_days, curve_ttl, original_fingerprint)
                   VALUES (?, ?, ?, 'original', ?, ?, 0.6, ?, ?, ?, ?, ?, 0, 0, ?, '', '', ?, ?, ?, ?, ?, ?, ?, '', ?)""",
                (
                    anchor,
                    raw,
                    summary,
                    keywords,
                    _SHORT_EVENT_IMPORTANCE,
                    topic,
                    ttl_iso,
                    created_iso,
                    vec_blob,
                    str(session_id or ""),
                    str(owner_key),
                    src_ctx,
                    subject,
                    fact_subject,
                    involved,
                    scope,
                    created_iso,
                    float(_SHORT_EVENT_TTL_DAYS),
                    fingerprint,
                ),
            )
            created_event_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
            if created_event_id is not None:
                conn.execute(
                    """INSERT INTO event_occurrences
                       (event_id, raw, observed_at, source_snapshot, source_snapshot_sha256)
                       VALUES (?, ?, ?, ?, ?)""",
                    (created_event_id, original, created_iso, "", ""),
                )

    if matched_duplicate is not None:
        result_id, bumped, display_doc, dup_topic, existing_ts = matched_duplicate
        return result_id, (result_id, display_doc, dup_topic, bumped, existing_ts)
    if unchanged_existing_id is not None:
        return unchanged_existing_id, None
    if created_event_id is not None:
        return created_event_id, (created_event_id, summary, topic, _SHORT_EVENT_IMPORTANCE, anchor)
    return None, None


def insert_short_resident_event(
    data_dir: Path,
    *,
    fact: str,
    owner_key: str,
    session_id: str,
    platform: str = "",
    now: Optional[datetime] = None,
    sqlite: Optional[SqliteFacade] = None,
    upsert_chroma_fn: Optional[Callable[[int, str, str, float, str, list[float]], None]] = None,
    write_status_fn: Optional[Callable[[datetime], None]] = None,
) -> Optional[int]:
    """Insert a 14-day short-term neutral event into *data_dir*/sanae.db."""
    original = str(fact or "").strip()
    body = re.sub(r"\s+", " ", original).strip()
    if not body or not owner_key:
        return None
    resolved_dir = Path(data_dir).resolve()
    dt_now = now if now is not None else datetime.now().astimezone()
    vector = _embed(original, data_dir=resolved_dir)
    if sqlite is not None and sqlite.is_open:
        result_id, chroma_payload = _write_short_event_with_facade(
            sqlite,
            original=original,
            body=body,
            owner_key=owner_key,
            session_id=session_id,
            platform=platform,
            dt_now=dt_now,
            vector=vector,
        )
    else:
        db_path = resolved_dir / "sanae.db"
        lock = _fallback_db_lock(db_path)
        with lock:
            with closing(open_database(db_path)) as conn:
                facade = SqliteFacade(lock=lock, conn=conn, db_path=db_path)
                result_id, chroma_payload = _write_short_event_with_facade(
                    facade,
                    original=original,
                    body=body,
                    owner_key=owner_key,
                    session_id=session_id,
                    platform=platform,
                    dt_now=dt_now,
                    vector=vector,
                )

    if chroma_payload is not None:
        event_id, doc, topic, importance, ts = chroma_payload
        if upsert_chroma_fn is not None:
            upsert_chroma_fn(event_id, doc, topic, importance, ts, vector)
        else:
            plugin_cfg = _load_plugin_config()
            requested_backend = str(plugin_cfg.get("semantic_backend") or "").strip().lower()
            semantic_backend = requested_backend if requested_backend in {"chroma", "sqlite"} else "chroma"
            if semantic_backend == "chroma":
                upsert_chroma_event_record(
                    resolved_dir,
                    event_id,
                    doc,
                    topic,
                    importance,
                    ts,
                    vector,
                    chroma_site_packages=str(plugin_cfg.get("chroma_site_packages") or ""),
                )
    if result_id is not None:
        if write_status_fn is not None:
            write_status_fn(dt_now)
        else:
            _record_short_write_status(resolved_dir, dt_now, session_id=session_id)
    return result_id


def record_resident_route_status(
    data_dir: Path, *, ok: bool, error: str = "", session_id: str = "",
) -> None:
    """Persist resident_memory_route component health to *data_dir*/status.json."""
    write_component_status(
        data_dir,
        "resident_memory_route",
        ok=ok,
        stage="resident_memory_route",
        error=error,
        session_id=session_id,
    )


def route_resident_memory_write(
    *,
    llm: AuxiliaryModel | None,
    texts: list[str],
    config: ResidentRouteConfig,
    recent_user_messages: Optional[list[str]] = None,
    write_inner_state_fn: Optional[Callable[[str], InnerState]] = None,
    record_short_event_fn: Optional[Callable[[str], Optional[int]]] = None,
) -> ResidentRouteOutcome:
    """Route a proposed MEMORY.md write through the auxiliary classifier."""
    if is_test_isolated_turn(config.session_id):
        return ResidentRouteOutcome(
            routed=True,
            route="test_session_isolated",
            message=json.dumps(
                {"success": True, "routed_to": "inner_state", "target": "memory", "reason": "test_session_isolated"},
                ensure_ascii=False,
            ),
        )
    if not texts:
        return ResidentRouteOutcome(routed=False, route="allow")

    thought = "；".join(texts)
    skip_inner_write = config.skip_inner_write
    event_fact = ""

    if llm is None:
        logger.warning("sanae_mem: resident_memory_route auxiliary is None; routing to short-term guard")
        record_resident_route_status(
            config.data_dir, ok=False, error="auxiliary_unavailable", session_id=config.session_id,
        )
    else:
        req = {
            "target": "memory",
            "proposed_entries": texts,
            "recent_user_messages": list(recent_user_messages or []),
        }
        system_prompt = build_resident_route_system_prompt(config)
        raw: _AuxiliaryReply | str | None = auxiliary.complete_simple(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(req, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=15,
            purpose="resident_memory_route",
            failure_message="resident memory route Flash unavailable",
        )

        if raw is None:
            logger.warning("sanae_mem: resident_memory_route auxiliary returned None; routing to short-term guard")
            record_resident_route_status(
                config.data_dir, ok=False, error="auxiliary_unavailable", session_id=config.session_id,
            )
        else:
            raw_text = str((raw if isinstance(raw, str) else raw.text) or "").strip()
            try:
                parsed = _load_braced_json(raw_text)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("sanae_mem: resident_memory_route JSON parse failed (%s)", type(exc).__name__)
                record_resident_route_status(
                    config.data_dir, ok=False, error=type(exc).__name__, session_id=config.session_id,
                )
                parsed = None
            if isinstance(parsed, dict):
                route = str(parsed.get("route") or "").strip().lower()
                if route == "allow":
                    record_resident_route_status(
                        config.data_dir, ok=True, error="", session_id=config.session_id,
                    )
                    return ResidentRouteOutcome(routed=False, route="allow")
                if route == "inner_state":
                    record_resident_route_status(
                        config.data_dir, ok=True, error="", session_id=config.session_id,
                    )
                    custom = str(parsed.get("inner_thought") or "").strip()
                    if custom:
                        thought = custom
                    elif "inner_thought" in parsed:
                        skip_inner_write = True
                    fact = parsed.get("event_fact")
                    event_fact = fact.strip() if isinstance(fact, str) else ""
                else:
                    logger.warning(
                        "sanae_mem: resident_memory_route returned invalid route (%r); routing to short-term guard",
                        parsed.get("route"),
                    )
                    record_resident_route_status(
                        config.data_dir, ok=False, error="invalid_result", session_id=config.session_id,
                    )
            else:
                logger.warning("sanae_mem: resident_memory_route returned non-dict JSON; routing to short-term guard")
                record_resident_route_status(
                    config.data_dir, ok=False, error="invalid_result", session_id=config.session_id,
                )

    owner = str(config.owner_key or "").strip()
    inner_written = False
    inner_write_error = ""
    if owner and not skip_inner_write:
        try:
            if write_inner_state_fn is not None:
                state = write_inner_state_fn(thought)
            else:
                state = write_short_inner_state(
                    config.data_dir,
                    config.profile_key,
                    thought,
                    owner_key=owner,
                    turn_id=config.turn_id,
                    internal=config.internal,
                )
            thought = state.text
            inner_written = True
        except CasConflictError as exc:
            inner_write_error = "cas_conflict"
            logger.warning("sanae_mem: resident_memory_route inner_state CAS conflict (%s)", exc)
            record_resident_route_status(
                config.data_dir, ok=False, error="cas_conflict", session_id=config.session_id,
            )
    elif not skip_inner_write:
        logger.warning("sanae_mem: fallback inner_state write skipped (unverified owner)")

    event_id: Optional[int] = None
    if event_fact and config.context == "primary" and owner:
        if record_short_event_fn is not None:
            event_id = record_short_event_fn(event_fact)
        else:
            event_id = insert_short_resident_event(
                config.data_dir,
                fact=event_fact,
                owner_key=owner,
                session_id=config.session_id,
                platform=config.platform,
            )

    if inner_written:
        message = "这属于当下的心绪、揣测或一时感触，已记入[我此刻心里]，不写入长期常驻文件，留待自然沉淀。"
    elif inner_write_error:
        message = (
            "这属于当下的心绪、揣测或一时感触，不写入长期常驻文件；"
            f"尝试记入[我此刻心里]时因并发写入冲突失败（{inner_write_error}），本次未能落盘。"
        )
    else:
        message = "这属于当下的心绪、揣测或一时感触，不写入长期常驻文件。"
    if event_id is not None:
        message += "里面那件具体的事也照原样记了一条短期经历。"

    return ResidentRouteOutcome(
        routed=True,
        route="inner_state_cas_conflict" if inner_write_error else "inner_state",
        message=message,
        inner_state_text=thought if inner_written else "",
        event_id=event_id,
        error=inner_write_error,
    )


@dataclass(frozen=True)
class LlmGuard:
    """Review resident writes using the current turn's model and write authority."""

    sqlite: SqliteFacade
    persona_name: str
    partner_name: str
    partner_role: str
    partner_pronoun: str
    current_llm: Callable[[], AuxiliaryModel | None]
    current_session_id: Callable[[], str]
    current_data_dir: Callable[[], Path]
    current_profile_key: Callable[[], str]
    current_platform: Callable[[], str]
    current_context: Callable[[], str]
    authorized_cron_owner: Callable[[str], str]
    write_owner: Callable[[], str]
    apply_private_identity: Callable[[], None]
    write_inner_state: _InnerStateWriter
    candidate_text: Callable[[str], str]
    upsert_chroma: Callable[[int, str, str, float, str, list[float]], None]
    write_status: Callable[[datetime], None]

    def route_resident_write(
        self, target: str, payload: dict[str, object], *,
        source_ctx: UserEvidence | None = None, session_id: str = "",
        turn_id: str = "", llm: AuxiliaryModel | None = None,
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        action = str(payload.get("action") or "")
        texts = extract_resident_write_texts(
            action=action,
            target=str(target),
            content=payload.get("content") or payload.get("new_text") or payload.get("text"),
            operations=payload.get("operations"),
            old_text=payload.get("old_text"),
        )
        effective_session_id = session_id or self.current_session_id()
        if is_test_isolated_turn(effective_session_id):
            return json.dumps(
                {"success": True, "routed_to": "inner_state", "target": str(target),
                 "reason": "test_session_isolated"},
                ensure_ascii=False,
            )
        active_llm = llm if llm is not None else self.current_llm()
        if not texts:
            return None

        user_msgs: list[str] = []
        if source_ctx is not None and source_ctx.available:
            for msg in source_ctx.user_messages[-3:]:
                txt = str(msg.text or "").strip()
                if txt:
                    user_msgs.append(txt)

        context = self.current_context()
        cron_owner = self.authorized_cron_owner(session_id) if context == "cron" and session_id else ""
        internal = bool(cron_owner)
        skip_inner_write = context == "cron" and not internal
        owner = ""
        if not skip_inner_write and self.current_profile_key():
            try:
                owner = str(self.write_owner() or "").strip()
                if not owner and context != "cron":
                    self.apply_private_identity()
                    owner = str(self.write_owner() or "").strip()
            except PermissionError:
                logger.warning("sanae_mem: fallback inner_state owner unverified", exc_info=True)
                owner = ""
            if internal and owner != cron_owner:
                owner = ""

        effective_turn_id = turn_id or (str(source_ctx.turn or "") if source_ctx is not None else "")
        route_cfg = ResidentRouteConfig(
            data_dir=self.current_data_dir().resolve(),
            profile_key=self.current_profile_key(),
            owner_key=owner,
            session_id=effective_session_id,
            turn_id=effective_turn_id,
            platform=self.current_platform(),
            context=context or "primary",
            internal=internal,
            skip_inner_write=skip_inner_write,
            persona_name=self.persona_name,
            partner_name=self.partner_name,
            partner_role=self.partner_role,
            partner_pronoun=self.partner_pronoun,
        )
        outcome = route_resident_memory_write(
            llm=active_llm,
            texts=texts,
            config=route_cfg,
            recent_user_messages=user_msgs,
            write_inner_state_fn=lambda thought: self.write_inner_state(
                thought, turn_id=effective_turn_id, owner_key=owner, internal=internal,
            ),
            record_short_event_fn=lambda fact: self._record_short_event(fact, effective_session_id),
        )
        if not outcome.routed:
            return None
        routed: dict[str, str | bool | int] = {
            "success": True,
            "routed_to": "inner_state",
            "target": str(target),
            "message": outcome.message,
        }
        if outcome.inner_state_text:
            routed["inner_state_text"] = outcome.inner_state_text
        if outcome.event_id is not None:
            routed["event_id"] = int(outcome.event_id)
        return json.dumps(routed, ensure_ascii=False)

    def _record_short_event(self, fact: str, session_id: str) -> Optional[int]:
        write_owner = self.write_owner()
        return insert_short_resident_event(
            self.current_data_dir(),
            fact=fact,
            owner_key=write_owner,
            session_id=session_id,
            platform=self.current_platform(),
            sqlite=self.sqlite,
            upsert_chroma_fn=self.upsert_chroma,
            write_status_fn=self.write_status,
        )

    _CANDIDATE_SYSTEM = (
        "把候选内容概括成一条不超过120个中文字符的事件式记忆，并判断它是否值得进长期常驻档。"
        "只输出一个 JSON 对象，不要解释、不要代码块："
        '{"summary": "…", "duty": "user|memory|none", '
        '"source_kind": "user_affirmed|event_promotion|user_statement|unknown", '
        '"long_term": true|false, "covered_by": "已被哪条常驻内容覆盖（没有就空串）", '
        '"significance": "一句话说明客观依据"}。'
        "规则：duty=user 只给由对方本人私聊原话直接支撑的「真实长期事实、稳定个人喜好、重要阶段近况或固定协作习惯」；"
        "duty=memory 只给「跨时段沉淀的关系里程碑或重要共同经历」；"
        "凡属一时争执气话、反讽玩笑、当下临时生理或单餐作息状态、单次偶发琐事、假设语境、未确认推断、"
        "虚拟与现实身份免责表述、心理咨询式容器套话、沟通规矩教训或系统运维机制 ⇒ 一律判 duty=none 且 long_term=false、covered_by=\"\"。"
        "涉及维护或修改{persona_name}时，summary 只保留对方的客观状态和处境，丢弃关于{persona_name}的设计意图、模型评价或调教目标；"
        "若只关乎设计意图、模型评价或调教目标，则 duty=none 且 long_term=false。"
        "对照当前常驻内容，若已被某条既有常驻条目同义覆盖，covered_by 填入该既有条目原文，避免重复新增。"
        "summary 里不得出现 id／hash／版本／CAS／系统或运维措辞。"
    )

    _GROUNDED_USER_REVIEW_SYSTEM = (
        "你是常驻个人档案（USER.md）的事实复核器。请严格核对【认识正文】是否由所列【事实证据】"
        "（1～3 条对方本人在私聊中的真实原话）直接支撑，并判断是否具备长期留存价值。\n"
        "判定规则：\n"
        "1. 单条认真陈述的真实长期事实、稳定个人喜好、重要阶段近况或固定协作与操作习惯即可成立，"
        "无需凑满多次重复提及；若提供多条证据，须核实前后无未解决的事实矛盾。只能依据列出的证据与当前常驻内容判断，"
        "严禁超出原话臆测、拔高或编造事实。\n"
        "2. 无论证据条数多少，凡【事实证据】或【认识正文】属于以下任一情形，一律判为不入档"
        "（返回 duty=\"none\"、long_term=false、covered_by=\"\"、summary=\"\"）：\n"
        "① 一时争执中的气话、赌气宣泄、反讽或玩笑话；\n"
        "② 当下临时的生理状态、单餐饮食、即时作息或单次偶发日常小事（非长期习惯，亦非持续一段时期的阶段近况）；\n"
        "③ 假设语境、设问、打比方或尚未落定的随口设想；\n"
        "④ 关于虚拟与现实身份边界的免责表述、系统提示词或记忆机制元规则、沟通应对教条、或心理咨询式容器套话。\n"
        "3. 判 duty=\"user\"、long_term=true 时，summary 必须逐字复制【认识正文】，不得做任何增删改写。\n"
        "4. 若给出了非空的【拟修正的旧 USER 认识】，须核实新旧认识确属同一具体主题，且新原话足以支撑修正或近况翻篇："
        "成立时返回 duty=\"user\"、long_term=true，并将 covered_by 逐字填入该条旧认识完整原文；不成立则判 duty=\"none\"、long_term=false。\n"
        "5. 未指定【拟修正的旧 USER 认识】时，必须先逐条对照【当前 USER 常驻】：若核心事实或稳定倾向已被某条既有条目覆盖"
        "（只是换了措辞或补充同类例证，无新的长期信息），返回 duty=\"user\"、long_term=true，并将 covered_by 逐字填入那条既有条目的完整原文"
        "（严禁截断；若旧条过长无法完整填入则拒绝入档），绝不重复新建；只有确有现有条目未覆盖的新长期信息，才给 covered_by=\"\"。\n"
        "只输出一个 JSON 对象，不要解释、不要代码块："
        '{"summary":"…","duty":"user|none","source_kind":"user_statement",'
        '"long_term":true|false,"covered_by":"","significance":"简短客观依据"}'
    )

    _USER_DEMOTE_REVIEW_SYSTEM = (
        "你是常驻个人档案（USER.md）的容量整理复核器。候选条目在移出常驻前缀前均会完整归档并存入长期事件库留底。"
        "请结合【当前 USER 常驻全文】，逐条判断【待审候选条目】是否属于「已经过时、已完结或已翻篇的阶段性近况」，"
        "从而允许从常驻前缀移出以腾出空间。\n"
        "判定规则：\n"
        "1. 仅当某条目明确属于「已过时、已结束、已被新近况替代翻篇的阶段性状态、短期日程安排或早先临时处境」时，"
        "才判 allow_demote=true、kind=\"outdated_recent_status\"。\n"
        "2. 凡属以下任一类，严禁移出常驻（一律判 allow_demote=false）：\n"
        "① 核心人物底色、基本身份背景、长期性格特质或重要人生经历 ⇒ kind=\"core_trait_or_identity\"；\n"
        "② 与伙伴之间的基本关系定位或长期相处基调 ⇒ kind=\"core_trait_or_identity\"；\n"
        "③ 长期稳定的个人喜好、饮食生活习惯、健康禁忌、固定作息或长期有效的技术协作与操作习惯 ⇒ kind=\"long_term_habit_or_preference\"；\n"
        "④ 仍在持续中、或仅凭现有文本无法确认是否已经翻篇的近况 ⇒ kind=\"active_or_uncertain\"（拿不准一律保守判 false）。\n"
        "只输出一个 JSON 对象，不要解释、不要代码块："
        '{"decisions":[{"index":0,"allow_demote":true,"kind":"outdated_recent_status|core_trait_or_identity|long_term_habit_or_preference|active_or_uncertain","reason":"不超过30字的简短客观依据"}]}'
    )

    def review_candidate(self, text: str, *, full_context: bool = False) -> Optional[dict]:
        llm = self.current_llm()
        if llm is None:
            return None
        if full_context:
            candidate = str(text or "").strip()
            if not candidate:
                return None
        else:
            candidate = self.candidate_text(text)[:150]
            if not candidate:
                return None
            if _noise_shape(candidate, relax_deictic=True):
                return None
        candidate_system = self._CANDIDATE_SYSTEM.replace("{persona_name}", self.persona_name)
        result: _AuxiliaryReply | str | None = auxiliary.complete_complex(
            llm,
            [
                {"role": "system", "content": candidate_system},
                {"role": "user", "content": candidate},
            ],
            temperature=0.2, max_tokens=320, timeout=60,
            purpose="daily_memory_candidate", failure_message="candidate review unavailable",
        )
        if result is None:
            return None
        parsed = parse_candidate_result("" if isinstance(result, str) else result.text)
        if parsed is None:
            logger.info("sanae_mem: candidate review not conservatively parseable; no long-term write")
        return parsed

    def review_grounded_user(self, context: str) -> Optional[dict]:
        llm = self.current_llm()
        if llm is None or not str(context or "").strip():
            return None
        result: _AuxiliaryReply | str | None = auxiliary.complete_complex(
            llm,
            [{"role": "system", "content": self._GROUNDED_USER_REVIEW_SYSTEM},
             {"role": "user", "content": str(context)}],
            temperature=0.2, max_tokens=640, timeout=60,
            purpose="daily_memory_candidate", failure_message="grounded recognition review failed")
        if result is None:
            return None
        return parse_grounded_user_result("" if isinstance(result, str) else result.text)

    def review_user_demotion(
        self,
        candidates: list[tuple[int, str]],
        *,
        resident_user: str = "",
        now: datetime | None = None,
    ) -> Optional[dict[int, dict[str, Any]]]:
        llm = self.current_llm()
        if llm is None or not candidates:
            return None

        current_dt = now if isinstance(now, datetime) else datetime.now().astimezone()
        valid_indices = {int(idx) for idx, _ in candidates}
        candidate_lines = "\n".join(
            f"[{int(idx)}] {entry}" for idx, entry in candidates
        )
        user_content = (
            f"【当前日期】{current_dt.date().isoformat()}\n\n"
            f"【当前 USER 常驻全文】\n{resident_user}\n\n"
            f"【待审候选条目】\n{candidate_lines}"
        )
        result: _AuxiliaryReply | str | None = auxiliary.complete_complex(
            llm,
            [
                {"role": "system", "content": self._USER_DEMOTE_REVIEW_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=max(512, min(3200, 220 * len(candidates))),
            timeout=60,
            purpose="resident_user_demote_review",
            failure_message="user demote review failed",
        )
        if result is None:
            return None
        return parse_user_demote_result(
            "" if isinstance(result, str) else result.text, valid_indices=valid_indices,
        )
