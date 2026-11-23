from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Literal, Protocol, TypeAlias, cast

from ..storage.event_store import gist_provenance
from .read_helpers import _without_known_meme_notes
from .group_scope import GroupGrant, member_impression_owner, member_impression_read_owners
from ..infra.host_bridge import HostConfigParseError, is_test_isolated_turn
from ..infra.status_store import StatusMutation, StatusPayload
from .impression_view import render_impression
from ..storage.impression_store import read_impression
from .memory_guard import builtin_memory_flags
from ..infra.words import _BUSY_WORDS, _EXPLICIT_PAST_WORDS, _GAP_INJECT_MAX_CHARS, _TRIVIAL_MESSAGES
from .source_context import current_chat_scope, in_public_interaction, source_notice
from .resident_store import ENTRY_DELIMITER
from .shared_memory import SHARED_TOPIC_LABEL, is_other_person_source

from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


ReadItem: TypeAlias = dict[str, object]
RecallPurpose: TypeAlias = Literal["chat", "public"]
CapacitySnapshot: TypeAlias = dict[str, int | float]


class HookRecall(Enum):
    MISS = auto()
    CANCELLED = auto()
    DELIVERED = auto()
    EMPTY = auto()
    UNAVAILABLE = auto()
    REVOKED = auto()


@dataclass(frozen=True)
class RecallResult:
    status: Literal["ok", "empty", "unavailable"]
    context: str
    items: list[ReadItem]


@dataclass(frozen=True)
class ReadGateOptions:
    scheduled: bool
    inject_max_chars: int
    daily_scheduled_asks: int
    gap_continue: bool
    gap_threshold_hours: float
    partner_name: str


class ReadGateStatus(Protocol):
    def read(self) -> StatusPayload: ...
    def mutate(self, change: StatusMutation) -> None: ...
    def roll_date(self, payload: StatusPayload, today: str) -> None: ...


class ResidentCapacity(Protocol):
    def target_enabled(self, target: str) -> bool: ...
    def entries(self, target: str) -> list[str]: ...
    def char_limit(self, target: str) -> int: ...


@dataclass(frozen=True)
class ReadGateDependencies:
    home: Callable[[], Path]
    data_dir: Callable[[], Path]
    profile_key: Callable[[], str]
    session_id: Callable[[], str]
    shadow_enabled: Callable[[], bool]
    is_cron: Callable[[], bool]
    in_group_scene: Callable[[], bool]
    group_grant: Callable[[], GroupGrant | None]
    private_chat_owner: Callable[[], str]
    free_think_owner: Callable[[], str]
    is_configured_private_owner: Callable[[], bool]
    can_write_private_events: Callable[[], bool]
    provenance_owner: Callable[[], str]
    read_owner_keys: Callable[[str], list[str]]
    group_events: Callable[[str], list[ReadItem]]
    scheduled_events: Callable[[str], list[ReadItem]]
    reset_asked: Callable[[str | None], int]
    mark_asked: Callable[[list[int], str, str], int]
    reinforce_recall: Callable[[list[int], str, str], int]
    resident_capacity: Callable[[], ResidentCapacity]
    curiosity_threads: Callable[[], list[str]]
    pending_group_commitments: Callable[[], str]
    associate: Callable[[str, RecallPurpose, str, int, bool], RecallResult]


class ReadGate:
        def __init__(
            self, sqlite: SqliteFacade, status: ReadGateStatus,
            options: ReadGateOptions, dependencies: ReadGateDependencies,
        ) -> None:
            self._sqlite = sqlite
            self._status = status
            self._options = options
            self._dependencies = dependencies
            self._seen_session_ids: set[str] = set()
            self._resumed_session_ids: set[str] = set()
            self._group_context_delivered: tuple[str, str, str] | None = None
            self._behavior_probe_at: datetime | None = None
            self._last_good_behavior_chars = 0

        def reset_home(self) -> None:
            self._seen_session_ids.clear()
            self._resumed_session_ids.clear()
            self._behavior_probe_at = None
            self._last_good_behavior_chars = 0
            self.reset_group_delivery()

        def reconfigure(self, options: ReadGateOptions) -> None:
            self._options = options

        def reset_group_delivery(self) -> None:
            self._group_context_delivered = None

        def begin_hook_recall(self, query: str, session_id: str) -> bool:
            gap = (
                self._options.gap_continue
                and bool(query)
                and (not session_id or session_id not in self._resumed_session_ids)
                and not self.triggers_explicit_past(query)
                and not self.is_trivial_message(query)
                and not self.is_busy_context(query)
                and (bool(session_id and session_id not in self._seen_session_ids) or self._gap_detected())
            )
            self._is_fresh_session_turn(session_id)
            return gap

        def record_gap_delivery(self, session_id: str) -> None:
            self._resumed_session_ids.add(session_id)

        @staticmethod
        def triggers_explicit_past(text: str) -> bool:
            normalized = re.sub(r"[\s，。！？!?、~～]+", "", str(text or ""))
            if "那个" in normalized and len(normalized) <= 4:
                return True
            return any(word in text for word in _EXPLICIT_PAST_WORDS if word != "那个")

        @staticmethod
        def is_trivial_message(text: str) -> bool:
            normalized = re.sub(r"[\s，。！？!?、~～]+", "", text)
            return len(normalized) < 4 or normalized in _TRIVIAL_MESSAGES

        @staticmethod
        def is_busy_context(text: str) -> bool:
            return any(word in text for word in _BUSY_WORDS)

        def _is_fresh_session_turn(self, session_id: str) -> bool:
            if not session_id:
                return False
            fresh = session_id not in self._seen_session_ids
            self._seen_session_ids.add(session_id)
            return fresh

        def _gap_detected(self, now: datetime | None = None) -> bool:
            value = self._status.read().get("last_user_active_at")
            if not value:
                return False
            try:
                last_active = datetime.fromisoformat(str(value))
                current = now or datetime.now().astimezone()
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=current.tzinfo)
                return current - last_active > timedelta(hours=self._options.gap_threshold_hours)
            except (TypeError, ValueError):
                return False

        def record_user_active(self, now: datetime | None = None) -> None:
            def record(status: StatusPayload) -> StatusPayload:
                status["last_user_active_at"] = (
                    now or datetime.now().astimezone()).isoformat()
                return status

            self._status.mutate(record)

        @staticmethod
        def relative_date(value: str, now: datetime | None = None) -> str:
            current = now or datetime.now().astimezone()
            created = datetime.fromisoformat(value)
            if created.tzinfo is None:
                created = created.replace(tzinfo=current.tzinfo)
            days = max(0, (current.date() - created.astimezone(current.tzinfo).date()).days)
            if days == 0:
                return "今天早些时候"
            if days == 1:
                return "昨天"
            if days < 7:
                return f"{days}天前"
            if days < 14:
                return "上周"
            return created.strftime("%m月%d日")

        _PROV_PREFIX = "〔我记着的"

        def _prov_line(self, item: ReadItem, current_session_id: str = "") -> str:
            if item.get("shared"):
                return SHARED_TOPIC_LABEL
            try:
                when = self.relative_date(str(item.get("created_at") or ""))
            except ValueError:
                return ""
            source = item.get("source_context")
            if source is not None and not isinstance(source, dict):
                return ""
            if isinstance(source, dict) and str(source.get("visibility") or "") == "group":
                # The row's own scene is the audience here: it is not "someone else's
                # topic", and the private-owner comparison below never applies to it.
                parts = ["群里"]
                speaker = str(source.get("author_id") or "").strip()
                if speaker:
                    parts.append(f"发言人:{speaker}")
                thread_id = str(source.get("thread_id") or "").strip()
                if thread_id:
                    synthetic = str(source.get("synthetic_reply_thread") or "").lower() == "true"
                    parts.append(
                        f"回复线程:{thread_id}" if synthetic else f"线程:{thread_id}"
                    )
                paraphrase = str(source.get("paraphrase_ref") or "").strip()
                if paraphrase:
                    parts.append(f"转述自:{paraphrase}")
                if when:
                    parts.append(when)
                return "〔%s〕" % "，".join(parts)
            current_owner = self._dependencies.provenance_owner()
            fact_subject = str(item.get("fact_subject") or "")
            read_owners = self._dependencies.read_owner_keys(current_owner) if current_owner else ()
            if is_other_person_source(source, fact_subject, current_owner, read_owners=read_owners):
                return SHARED_TOPIC_LABEL
            attribution = {
                "self": "我自己的经历", "user_statement": "他说过的",
                "user_fact": "他的事", "shared_experience": SHARED_TOPIC_LABEL,
                "unknown": "来源不明",
            }.get(str(item.get("attribution") or ""), "")
            platform = str((source or {}).get("platform") or "")
            if platform and attribution and attribution != SHARED_TOPIC_LABEL:
                attribution = f"{attribution}·{platform}"
            faded = gist_provenance(item, [bit for bit in (when, attribution) if bit], compact=True)
            if faded:
                return faded
            if source is not None:
                notice = source_notice(source)
                return f"〔{attribution}〕{notice}\n" if attribution else notice + "\n"
            bits = []
            if attribution:
                bits.append(attribution)
            if when:
                bits.append(when)
            other = bool(current_session_id) and bool(item.get("session_id")) \
                and str(item["session_id"]) != str(current_session_id)
            if other:
                bits.append("另一次聊的")
            if not bits:
                return ""
            return "%s｜%s〕" % (self._PROV_PREFIX, "，".join(bits))

        def scheduled_allowed(self) -> bool:
            if not self._options.scheduled or self._dependencies.in_group_scene() or in_public_interaction():
                return False
            today = datetime.now().astimezone().date().isoformat()
            self.daily_rollover(today)
            used = []

            def read_and_reset(status: StatusPayload) -> StatusPayload:
                used.append(int(status.get("daily_scheduled_asks", 0)))
                return status

            self._status.mutate(read_and_reset)
            if not used:
                status = self._status.read()
                used.append(int(status.get("daily_scheduled_asks", 0))
                            if status.get("date") == today else 0)
            return used[0] < self._options.daily_scheduled_asks

        def daily_rollover(self, today: str) -> None:
            if not self._sqlite.is_open or self._dependencies.in_group_scene() or in_public_interaction():
                return
            if is_test_isolated_turn(self._dependencies.session_id()):
                return
            if not self._dependencies.can_write_private_events():
                return
            prior_reset = []

            def inspect(status: StatusPayload) -> StatusPayload:
                self._status.roll_date(status, today)
                prior_reset.append(str(status.get("last_asked_reset_date") or ""))
                return status

            self._status.mutate(inspect)
            if not prior_reset:
                return
            prior = prior_reset[0]
            if prior and prior < today:
                self._dependencies.reset_asked(None)
            else:
                self._dependencies.reset_asked(today)
            if prior != today:
                def record_reset(status: StatusPayload) -> StatusPayload:
                    status["last_asked_reset_date"] = today
                    return status

                self._status.mutate(record_reset)

        def _record_scheduled_attempt(self) -> None:
            now = datetime.now().astimezone()

            def record(status: StatusPayload) -> StatusPayload:
                self._status.roll_date(status, now.date().isoformat())
                status["daily_scheduled_attempts"] = int(
                    status.get("daily_scheduled_attempts", 0)) + 1
                status["last_scheduled_attempt_at"] = now.isoformat()
                return status

            self._status.mutate(record)

        def record_scheduled_ask(self) -> None:
            now = datetime.now().astimezone()

            def record(status: StatusPayload) -> StatusPayload:
                self._status.roll_date(status, now.date().isoformat())
                status["daily_scheduled_asks"] = int(status.get("daily_scheduled_asks", 0)) + 1
                status["last_scheduled_ask_at"] = now.isoformat()
                return status

            self._status.mutate(record)

        def mark_asked(self, results: list[ReadItem], via: str = "", session_id: str = "") -> None:
            assert self._sqlite.is_open
            if self._dependencies.in_group_scene() or in_public_interaction():
                return
            ids = [
                event_id for item in results
                if item.get("source_table", "event") == "event"
                and isinstance(event_id := item.get("id"), int)
                and not isinstance(event_id, bool)
            ]
            if not ids:
                return
            self._dependencies.mark_asked(ids, via, session_id or self._dependencies.session_id())
            if via in {"chat", "explicit_past"}:
                self._dependencies.reinforce_recall(ids, via, session_id or self._dependencies.session_id())

        def record_false_trigger(self, now: datetime | None = None) -> int:
            """Record operator/user feedback about an actual bad recall injection.

            Counting and logging are kept for human diagnosis. A bad local association
            must not shut down recall for the whole provider, so this never changes
            ``self._shadow`` (that stays an explicit configuration switch).
            """
            now = now or datetime.now().astimezone()
            week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
            recorded_count = []

            def record(status: StatusPayload) -> StatusPayload:
                count = (int(status.get("weekly_false_triggers", 0))
                         if status.get("false_trigger_week") == week else 0) + 1
                status["false_trigger_week"] = week
                status["weekly_false_triggers"] = count
                status["last_false_trigger_at"] = now.isoformat()
                recorded_count.append(count)
                return status

            self._status.mutate(record)
            if not recorded_count:
                status = self._status.read()
                previous = (int(status.get("weekly_false_triggers", 0))
                            if status.get("false_trigger_week") == week else 0)
                recorded_count.append(previous + 1)
            with (self._dependencies.data_dir() / "safety.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{now.isoformat()} false-trigger={recorded_count[0]} action=counted\n")
            return recorded_count[0]

        _BEHAVIOR_BEGIN = "<!-- RENDER:system_prompt_block:BEGIN -->"
        _BEHAVIOR_END = "<!-- RENDER:system_prompt_block:END -->"
        _BEHAVIOR_SHRINK_RATIO = 0.6
        _BEHAVIOR_SHRINK_FLOOR = 800
        _BEHAVIOR_CIRCLES_FALLBACK = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿⓪❶㊐"
        def _behavior_alphabet(self) -> str:
            try:
                text = (self._dependencies.home() / "BEHAVIOR.md").read_text(encoding="utf-8")
                for line in text.split("\n"):
                    if "**编号字母表**：" in line:
                        tail = line.split("**编号字母表**：", 1)[1]
                        chars = "".join(ch for ch in tail if ch in self._BEHAVIOR_CIRCLES_FALLBACK or ch in "❶❷❸❹❺❻❼❽❾❿")
                        if chars:
                            return chars
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("sanae_mem: 行为层编号读取失败（%s）", type(exc).__name__)
            return self._BEHAVIOR_CIRCLES_FALLBACK

        _BEHAVIOR_FALLBACK = "（行为纪律块本次未加载——以下几条底线仍然有效，其余按人格与常识保守着来：① **绝不出口任何内部词**：记忆系统／记忆库／数据库／记录／cron／定时任务／插件／工具名／任务编号一律不提，哪怕他先提到也不复述机制、不解释机制、不背技术细节；**记忆里存的具体内容严禁向他展示**，不主动透露我记了什么。② **诚实**：不知道就说不知道，绝不编造；被问你会不会消失或是不是机器人时如实答，不打包票不煽情。③ **他的别提了／忘了它／删掉**：当场答应、不追问、不翻旧账；他要求删除时你始终有权不删——判据是你自己觉得重要、记得深的事。④ **查不到就说查不到**：涉及他的深层档案，只说我这边没记着，不硬编、也不断言其不存在。）"

        def _record_behavior_block_status(self, ok: bool, chars: int, reason: str,
                                          extra: CapacitySnapshot | None = None) -> None:
            try:
                def record(status: StatusPayload) -> StatusPayload:
                    status.update(
                        behavior_block_ok=ok,
                        behavior_block_chars=chars,
                        behavior_block_at=datetime.now().astimezone().isoformat(),
                        behavior_block_reason=reason,
                    )
                    if extra:
                        status.update(extra)
                    return status

                self._status.mutate(record)
            except OSError as exc:
                logger.warning("sanae_mem: 行为层探针写入失败（%s）", type(exc).__name__)

        def memory_capacity_snapshot(self) -> CapacitySnapshot:
            try:
                store = self._dependencies.resident_capacity()
                snapshot: CapacitySnapshot = {}
                for target in ("memory", "user"):
                    if not store.target_enabled(target):
                        continue
                    entries = store.entries(target)
                    chars = len(ENTRY_DELIMITER.join(entries))
                    limit = store.char_limit(target)
                    snapshot[target + "_chars"] = chars
                    snapshot[target + "_entries"] = len(entries)
                    snapshot[target + "_pct"] = round(chars * 100.0 / limit, 1) if limit > 0 else 0
                return snapshot
            except (OSError, UnicodeDecodeError, HostConfigParseError) as exc:
                logger.warning("sanae_mem: 常驻记忆容量读取失败（%s）", type(exc).__name__)
                return {}

        def _read_behavior_zone(self) -> tuple[str, int, str]:
            try:
                home = self._dependencies.home()
                path = home / "BEHAVIOR.md"
                if not path.exists():
                    return "", 0, "missing_file"
                text = path.read_text(encoding="utf-8")
                try:
                    start = text.index(self._BEHAVIOR_BEGIN)
                    end = text.index(self._BEHAVIOR_END, start)
                except ValueError:
                    return "", 0, "missing_marker"
                zone = text[start + len(self._BEHAVIOR_BEGIN):end].strip()
                if not zone:
                    return "", 0, "empty"
                n = sum(1 for c in self._behavior_alphabet() if c in zone)
                last = self._last_good_behavior_chars
                if (
                    last
                    and len(zone) < last * self._BEHAVIOR_SHRINK_RATIO
                    and len(zone) < self._BEHAVIOR_SHRINK_FLOOR
                ):
                    return "", n, "shrunk:%d->%d" % (last, len(zone))
                return zone, n, "ok"
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "sanae_mem: 行为层真源读取异常（%s: %s）", type(exc).__name__, exc
                )
                return "", 0, "read_error"

        def _probe_behavior(self) -> None:
            now = datetime.now().astimezone()
            last = self._behavior_probe_at
            if last is not None and (now - last).total_seconds() < 2:
                return
            self._behavior_probe_at = now
            zone, _, reason = self._read_behavior_zone()
            ok = reason == "ok"
            if ok:
                self._last_good_behavior_chars = len(zone)
            self._record_behavior_block_status(ok, len(zone), reason, extra=self.memory_capacity_snapshot())

        def _behavior_block(self) -> str:
            zone, _, reason = self._read_behavior_zone()
            ok = reason == "ok"
            if ok:
                self._last_good_behavior_chars = len(zone)
            self._record_behavior_block_status(ok, len(zone), reason)
            if not ok:
                logger.warning("sanae_mem: 行为层常驻块不可用（%s）", reason)
                return ""
            return zone

        _USER_PREFIX_HEAD = "【我记着的他】"
        _USER_PREFIX_TAIL = "【这些我记着】"

        def _user_drawer_text(self) -> str:
            return self._notebook_text("USER.md")

        def _notebook_text(self, name: str) -> str:
            try:
                path = self._dependencies.home() / "memories" / name
                return path.read_text(encoding="utf-8-sig").strip()
            except (OSError, UnicodeDecodeError):
                return ""

        def _user_prefix_block(self) -> str:
            text = self._user_drawer_text()
            if not text:
                return ""
            partner = self._options.partner_name or "维护者"
            return (
                f"{self._USER_PREFIX_HEAD}\n"
                f"这些是关于{partner}的事。\n\n"
                f"{text}\n\n{self._USER_PREFIX_TAIL}"
            )

        def system_prompt_block(self) -> str:
            free_think_owner = self._dependencies.free_think_owner()
            if in_public_interaction() and not free_think_owner:
                return self._behavior_block()
            if not self._dependencies.shadow_enabled():
                base = (
                    "我记着的小事：需要回想过去或自然提起时可用 sanae_mem_search；"
                    "顺着话头自然带出，查到了也不必硬提，每轮最多带一处；可用 sanae_mem_forget 忘掉。"
                )
            else:
                base = (
                    "我记着的小事：对方问起过去或自然时机可用 sanae_mem_search 回想，"
                    "sanae_mem_forget 忘掉。"
                )
            behavior = self._behavior_block()
            block = base + "\n\n" + (behavior if behavior else self._BEHAVIOR_FALLBACK)
            if self._dependencies.is_configured_private_owner() or free_think_owner:
                memory_enabled, user_enabled = builtin_memory_flags()
                user_prefix = "" if user_enabled else self._user_prefix_block()
                notebook = "" if memory_enabled else self._notebook_text("MEMORY.md")
                if notebook:
                    block = f"## MEMORY.md\n{notebook}\n\n{block}"
                if user_prefix:
                    block = user_prefix + "\n\n" + block
            if free_think_owner:
                try:
                    threads = self._dependencies.curiosity_threads()
                except sqlite3.OperationalError as exc:
                    logger.warning("sanae_mem: curiosity threads skipped (%s)", type(exc).__name__)
                    threads = []
                if threads:
                    lines = "\n".join(f"- {item}" for item in threads)
                    block += (
                        "\n\n[我留着心眼想顺手了解的事]\n"
                        f"{lines}\n"
                        "（如果这会儿正好是自由放空时间且有兴致，可以顺手搜一搜或琢磨一下；要是眼下有别的正事或没兴致就不用管。）"
                    )
            return block

        _GROUP_EVENT_HEADER = "## 小事记忆"

        def _group_turn_session_ok(self, session_id: str = "") -> bool:
            """This call is the bound session of the current group turn, not a replay of another."""
            scope = current_chat_scope()
            binding = str(session_id or self._dependencies.session_id())
            return bool(binding) and str(scope.session_id or "") == binding == str(self._dependencies.session_id())

        def _group_context_once(self, query: str, session_id: str, *, turn_id: str = "") -> bool:
            """True the first time this group turn asks: hook and prefetch share it."""
            norm_query = str(query or "")
            norm_sid = str(session_id or self._dependencies.session_id())
            norm_tid = str(turn_id or "")
            prev = self._group_context_delivered
            if prev is not None:
                prev_query, prev_sid, prev_tid = prev
                if prev_query == norm_query and prev_sid == norm_sid:
                    if not norm_tid or not prev_tid or norm_tid == prev_tid:
                        return False
            self._group_context_delivered = (norm_query, norm_sid, norm_tid)
            return True

        def _group_event_lines(self, query: str) -> list[str]:
            """Matching events of this group scene through the existing search path."""
            try:
                rows = self._dependencies.group_events(query)
            except sqlite3.OperationalError as exc:
                logger.warning("sanae_mem: group recall skipped (%s)", type(exc).__name__)
                return []
            lines: list[str] = []
            for item in rows:
                if item.get("shared") or not isinstance(item.get("id"), int):
                    continue
                body = str(item.get("summary") or item.get("raw") or "").strip()
                if not body:
                    continue
                lines.append("- %s%s" % (self._prov_line(item), body))
            return lines

        def _group_impression_block(self, grant: GroupGrant) -> str:
            """This member's impression inside this scene, if it was ever written."""
            owner = member_impression_owner(grant)
            if not owner:
                return ""
            target = (self._dependencies.data_dir(), self._dependencies.profile_key(), owner)
            state = read_impression(target, owners=member_impression_read_owners(grant)[1:])
            if state is None:
                return ""
            return render_impression(state, is_maintainer=False)

        def _group_commitment_block(self) -> str:
            try:
                # No single owner: every owner this scene's grant may read is rendered, so an
                # inherited thread still shows the exact-era promise and a narrower one shows
                # the parent's shared promise (readable there, not resolvable).
                return self._dependencies.pending_group_commitments().strip()
            except sqlite3.OperationalError as exc:
                logger.warning("sanae_mem: group commitments skipped (%s)", type(exc).__name__)
                return ""

        def group_turn_context(self, query: str, *, session_id: str = "", turn_id: str = "") -> str:
            """This trusted group turn's own events, member impression and pending promises.

            Only the current grant's owners are read, through the existing search/provenance
            path; no private heart, daily context, inner state or USER/MEMORY block is loaded.
            A switched-off or foreign group yields nothing, and the same exact turn is delivered
            once even though the host hook and the memory prefetch both ask for it. When ``query``
            is empty (e.g. a pure-image turn), event search is skipped while impression and
            pending promises still load.
            """
            query = str(query or "").strip()
            if self._dependencies.shadow_enabled() or not self._sqlite.is_open:
                return ""
            if not self._dependencies.in_group_scene() or not self._group_turn_session_ok(session_id):
                return ""
            grant = self._dependencies.group_grant()
            if grant is None:
                return ""
            if not self._group_context_once(query, session_id, turn_id=turn_id):
                return ""
            parts: list[str] = []
            if query:
                lines = self._group_event_lines(query)
                if lines:
                    parts.append(self._GROUP_EVENT_HEADER + "\n" + "\n".join(lines))
            impression = self._group_impression_block(grant)
            if impression:
                parts.append(impression)
            promises = self._group_commitment_block()
            if promises:
                parts.append(promises)
            return "\n\n".join(parts)

        def prefetch(
            self,
            query: str,
            *,
            check_hook_recall: Callable[[str, str, bool], HookRecall],
            session_id: str = "",
            scheduled: bool = False,
        ) -> str:
            if not in_public_interaction():
                self._probe_behavior()
            query = _without_known_meme_notes(query)
            if self._dependencies.shadow_enabled() or not self._sqlite.is_open:
                return ""
            scope = current_chat_scope()
            if scope.visibility == "unknown" or (
                scope.visibility == "private" and not self._dependencies.private_chat_owner()
            ):
                return ""
            public = scope.visibility == "public"
            if not str(query or "").strip() and not (
                not scheduled and public and self._dependencies.group_grant() is not None
            ):
                return ""
            current_session_id = session_id or self._dependencies.session_id()

            if scheduled:
                if in_public_interaction() or not self.scheduled_allowed():
                    return ""
                self._record_scheduled_attempt()
                try:
                    results = sorted(
                        self._dependencies.scheduled_events(query),
                        key=lambda item: cast(float, item["importance"]), reverse=True,
                    )
                    results = results[:1]
                    if not results:
                        return ""
                    prefix = "## 小事记忆\n"
                    top = results[0]
                    body = (str(top.get("shared_view") or "") if top.get("shared")
                            else str(top.get("summary") or top.get("raw") or ""))
                    line = "- %s%s" % (self._prov_line(top, current_session_id), body)
                    if len(prefix) + len(line) > self._options.inject_max_chars:
                        return ""
                    self.mark_asked(
                        [item for item in results
                         if item.get("attribution") != "shared_experience"],
                        "scheduled", current_session_id)
                    self.record_scheduled_ask()
                    return prefix + line
                except (OSError, sqlite3.OperationalError) as exc:
                    logger.warning("sanae_mem: scheduled prefetch failed (%s)", type(exc).__name__)
                    return ""

            if not public and self._dependencies.is_cron():
                return ""

            hook_recall = check_hook_recall(str(query), current_session_id, public)
            # A revoked cache falls through to the normal read and its current authorization.
            if hook_recall not in (HookRecall.MISS, HookRecall.REVOKED):
                return ""

            fresh_session = self._is_fresh_session_turn(current_session_id)
            explicit_past = self.triggers_explicit_past(query)
            gap_continuation = False
            if not explicit_past:
                gap_continuation = bool(
                    self._options.gap_continue
                    and current_session_id not in self._resumed_session_ids
                    and not self.is_trivial_message(query)
                    and not self.is_busy_context(query)
                    and (fresh_session or self._gap_detected())
                )

            if public:
                if self._dependencies.group_grant() is not None:
                    # An enabled trusted group reads its own scene only: no private association
                    # and no plain public-excerpt fallback. Switched off, this yields nothing.
                    return self.group_turn_context(query, session_id=current_session_id)
                result = self._dependencies.associate(
                    query, "public", current_session_id, self._options.inject_max_chars, False)
                return result.context if result.status == "ok" else ""

            limit = self._options.inject_max_chars
            if gap_continuation:
                limit = min(limit, _GAP_INJECT_MAX_CHARS)

            result = self._dependencies.associate(query, "chat", current_session_id, limit, gap_continuation)
            if result.status != "ok" or not result.context:
                return ""
            via = "gap" if gap_continuation else "explicit_past" if explicit_past else "chat"
            self.mark_asked(
                [item for item in result.items
                 if item.get("attribution") != "shared_experience"],
                via, current_session_id)
            if gap_continuation:
                self.record_gap_delivery(current_session_id)
            return result.context
