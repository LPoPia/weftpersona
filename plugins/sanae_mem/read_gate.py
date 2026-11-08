from __future__ import annotations

import json
import logging
import math
import re
import site
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .event_store import gist_provenance
    from .daily_inputs import _without_known_meme_notes
    from .words import *
    from .textutil import *
    from .textutil import hermes_home
    from .schemas import *
    from .source_context import in_public_interaction, participant_code
except ImportError:
    from event_store import gist_provenance
    from daily_inputs import _without_known_meme_notes
    from words import *
    from textutil import *
    from textutil import hermes_home
    from schemas import *
    from source_context import in_public_interaction, participant_code

logger = logging.getLogger(__name__)


class ReadGateMixin:
        @staticmethod
        def _triggers_explicit_past(text: str) -> bool:
            normalized = re.sub(r"[\s，。！？!?、~～]+", "", str(text or ""))
            if "那个" in normalized and len(normalized) <= 4:
                return True
            return any(word in text for word in _EXPLICIT_PAST_WORDS if word != "那个")

        @staticmethod
        def _is_trivial_message(text: str) -> bool:
            normalized = re.sub(r"[\s，。！？!?、~～]+", "", text)
            return len(normalized) < 4 or normalized in _TRIVIAL_MESSAGES

        @staticmethod
        def _is_busy_context(text: str) -> bool:
            return any(word in text for word in _BUSY_WORDS)

        def _is_fresh_session_turn(self, session_id: str) -> bool:
            if not session_id:
                return False
            fresh = session_id not in self._seen_session_ids
            self._seen_session_ids.add(session_id)
            return fresh

        def _gap_detected(self, now: Optional[datetime] = None) -> bool:
            value = self._read_status().get("last_user_active_at")
            if not value:
                return False
            try:
                last_active = datetime.fromisoformat(str(value))
                current = now or datetime.now().astimezone()
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=current.tzinfo)
                return current - last_active > timedelta(hours=self._gap_threshold_hours)
            except (TypeError, ValueError):
                return False

        def _record_user_active(self, now: Optional[datetime] = None) -> None:
            with self._lock:
                status = self._read_status()
                status["last_user_active_at"] = (now or datetime.now().astimezone()).isoformat()
                self._write_status_payload(status)

        @staticmethod
        def _relative_date(value: str, now: Optional[datetime] = None) -> str:
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

        def _prov_line(self, item: dict, current_session_id: str = "") -> str:
            try:
                when = self._relative_date(str(item.get("created_at") or ""))
                source = item.get("source_context") if isinstance(item, dict) else None
                attribution = {
                    "self": "我自己的经历", "user_statement": "他说过的",
                    "user_fact": "他的事", "shared_experience": "共同经历",
                    "unknown": "来源不明",
                }.get(str(item.get("attribution") or ""), "")
                fact_subject = str(item.get("fact_subject") or "")
                if fact_subject.startswith("owner:"):
                    code = participant_code(fact_subject.removeprefix("owner:"))
                    if code:
                        attribution = (f"{attribution}（{code}）"
                                       if attribution else code)
                platform = str((source or {}).get("platform") or "")
                if platform and attribution:
                    attribution = f"{attribution}·{platform}"
                faded = gist_provenance(item, [bit for bit in (when, attribution) if bit], compact=True)
                if faded:
                    return faded
                if source is not None:
                    try:
                        from agent.interaction_context import source_notice
                    except Exception:
                        source_notice = None
                    if source_notice is not None:
                        notice = source_notice(source)
                        return (f"〔{attribution}〕{notice}\n" if attribution
                                else notice + "\n")
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
            except Exception:
                return ""

        def _scheduled_allowed(self) -> bool:
            today = datetime.now().astimezone().date().isoformat()
            self._daily_rollover(today)
            status = self._read_status()
            if status.get("date") != today:
                status["date"] = today
                status["daily_writes"] = 0
                status["daily_scheduled_asks"] = 0
                status["daily_scheduled_attempts"] = 0
                self._write_status_payload(status)
            used = int(status.get("daily_scheduled_asks", 0))
            return used < self._daily_scheduled_asks

        def _daily_rollover(self, today: str) -> None:
            status = self._read_status()
            if status.get("last_asked_reset_date") == today:
                return
            if self._conn is not None:
                self.reset_asked_all()
            status["last_asked_reset_date"] = today
            self._write_status_payload(status)

        def _record_scheduled_attempt(self) -> None:
            now = datetime.now().astimezone()
            status = self._read_status()
            status["daily_scheduled_attempts"] = int(status.get("daily_scheduled_attempts", 0)) + 1
            status["last_scheduled_attempt_at"] = now.isoformat()
            self._write_status_payload(status)

        def _record_scheduled_ask(self) -> None:
            now = datetime.now().astimezone()
            status = self._read_status()
            status["daily_scheduled_asks"] = int(status.get("daily_scheduled_asks", 0)) + 1
            status["last_scheduled_ask_at"] = now.isoformat()
            self._write_status_payload(status)

        def _mark_asked(self, results: List[dict], via: str = "", session_id: str = "") -> None:
            assert self._conn is not None
            ids = [int(item["id"]) for item in results
                   if item.get("source_table", "event") == "event"
                   and isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool)]
            if not ids:
                return
            self.mark_asked(ids, via, session_id or self._session_id)
            if via in {"chat", "explicit_past"}:
                self.reinforce_on_recall(ids, via=via, session_id=session_id or self._session_id)

        def record_false_trigger(self, now: Optional[datetime] = None) -> int:
            """Record operator/user feedback about an actual bad recall injection.

            Counting and logging are kept for human diagnosis. A bad local association
            must not shut down recall for the whole provider, so this never changes
            ``self._shadow`` (that stays an explicit configuration switch).
            """
            now = now or datetime.now().astimezone()
            status = self._read_status()
            week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
            count = int(status.get("weekly_false_triggers", 0)) if status.get("false_trigger_week") == week else 0
            count += 1
            status["false_trigger_week"] = week
            status["weekly_false_triggers"] = count
            status["last_false_trigger_at"] = now.isoformat()
            self._write_status_payload(status)
            with (self._data_dir / "safety.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{now.isoformat()} false-trigger={count} action=counted\n")
            return count

        _BEHAVIOR_BEGIN = "<!-- RENDER:system_prompt_block:BEGIN -->"
        _BEHAVIOR_END = "<!-- RENDER:system_prompt_block:END -->"
        _BEHAVIOR_SHRINK_RATIO = 0.6
        _BEHAVIOR_SHRINK_FLOOR = 800
        _BEHAVIOR_CIRCLES_FALLBACK = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿⓪❶㊐"
        def _behavior_alphabet(self) -> str:
            try:
                text = (Path(hermes_home()).expanduser() / "BEHAVIOR.md").read_text(encoding="utf-8")
                for line in text.split("\n"):
                    if "**编号字母表**：" in line:
                        tail = line.split("**编号字母表**：", 1)[1]
                        chars = "".join(ch for ch in tail if ch in self._BEHAVIOR_CIRCLES_FALLBACK or ch in "❶❷❸❹❺❻❼❽❾❿")
                        if chars:
                            return chars
            except Exception:
                pass
            return self._BEHAVIOR_CIRCLES_FALLBACK

        _last_good_behavior_chars = 0
        _BEHAVIOR_FALLBACK = "（行为纪律块本次未加载——以下几条底线仍然有效，其余按人格与常识保守着来：① **绝不出口任何内部词**：记忆系统／记忆库／数据库／记录／cron／定时任务／插件／工具名／任务编号一律不提，哪怕他先提到也不复述机制、不解释机制、不背技术细节；**记忆里存的具体内容严禁向他展示**，不主动透露我记了什么。② **诚实**：不知道就说不知道，绝不编造；被问你会不会消失或是不是机器人时如实答，不打包票不煽情。③ **他的别提了／忘了它／删掉**：当场答应、不追问、不翻旧账；他要求删除时你始终有权不删——判据是你自己觉得重要、记得深的事。④ **查不到就说查不到**：涉及他的深层档案，只说我这边没记着，不硬编、也不断言其不存在。）"

        def _record_behavior_block_status(self, ok: bool, chars: int, reason: str,
                                          extra: Optional[Dict[str, Any]] = None) -> None:
            try:
                status = self._read_status()
                status.update(
                    behavior_block_ok=ok,
                    behavior_block_chars=chars,
                    behavior_block_at=datetime.now().astimezone().isoformat(),
                    behavior_block_reason=reason,
                )
                if extra:
                    status.update(extra)
                self._write_status_payload(status)
            except Exception as exc:
                logger.warning("sanae_mem: 行为层探针写入失败（%s）", type(exc).__name__)

        def _memory_capacity_snapshot(self) -> Dict[str, Any]:
            try:
                from tools.memory_tool import ENTRY_DELIMITER, load_on_disk_store
                store = load_on_disk_store()
                snapshot: Dict[str, Any] = {}
                for target in ("memory", "user"):
                    if not store.target_enabled(target):
                        continue
                    entries = store._entries_for(target)
                    chars = len(ENTRY_DELIMITER.join(entries))
                    limit = int(store._char_limit(target))
                    snapshot[target + "_chars"] = chars
                    snapshot[target + "_entries"] = len(entries)
                    snapshot[target + "_pct"] = round(chars * 100.0 / limit, 1) if limit > 0 else 0
                return snapshot
            except Exception:
                return {}

        def _read_behavior_zone(self):
            try:
                home = Path(hermes_home()).expanduser()
                path = home / "BEHAVIOR.md"
                if not path.exists():
                    path = Path(self._data_dir).parent / "BEHAVIOR.md"
                if not path.exists():
                    return None, 0, "missing_file"
                text = path.read_text(encoding="utf-8")
                try:
                    start = text.index(self._BEHAVIOR_BEGIN)
                    end = text.index(self._BEHAVIOR_END, start)
                except ValueError:
                    return None, 0, "missing_marker"
                zone = text[start + len(self._BEHAVIOR_BEGIN):end].strip()
                if not zone:
                    return None, 0, "empty"
                n = sum(1 for c in self._behavior_alphabet() if c in zone)
                last = self._last_good_behavior_chars
                if (
                    last
                    and len(zone) < last * self._BEHAVIOR_SHRINK_RATIO
                    and len(zone) < self._BEHAVIOR_SHRINK_FLOOR
                ):
                    return None, n, "shrunk:%d->%d" % (last, len(zone))
                return zone, n, "ok"
            except Exception as exc:
                logger.warning(
                    "sanae_mem: 行为层真源读取异常（%s: %s）", type(exc).__name__, exc
                )
                return None, 0, "read_error"

        def _probe_behavior(self) -> None:
            try:
                now = datetime.now().astimezone()
                last = getattr(self, "_behavior_probe_at", None)
                if last is not None and (now - last).total_seconds() < 2:
                    return
                self._behavior_probe_at = now
                zone, n_items, reason = self._read_behavior_zone()
                ok = reason == "ok"
                if ok:
                    self._last_good_behavior_chars = len(zone)
                self._record_behavior_block_status(ok, len(zone) if zone else 0, reason,
                                                   extra=self._memory_capacity_snapshot())
            except Exception:
                pass

        def _behavior_block(self) -> str:
            zone, n_items, reason = self._read_behavior_zone()
            ok = reason == "ok"
            if ok:
                self._last_good_behavior_chars = len(zone)
            self._record_behavior_block_status(ok, len(zone) if zone else 0, reason)
            if not ok:
                logger.warning("sanae_mem: 行为层常驻块不可用（%s）", reason)
                return ""
            return zone

        def system_prompt_block(self) -> str:
            if in_public_interaction():
                return ""
            if not self._shadow:
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
            if not self._shadow and str(getattr(self, "_agent_context", "")) == "cron":
                try:
                    threads = self.active_curiosity_threads()
                except Exception:
                    threads = []
                if threads:
                    lines = "\n".join(f"- {item}" for item in threads)
                    block += (
                        "\n\n[我留着心眼想顺手了解的事]\n"
                        f"{lines}\n"
                        "（如果这会儿正好是自由放空时间且有兴致，可以顺手搜一搜或琢磨一下；要是眼下有别的正事或没兴致就不用管。）"
                    )
            return block

        def prefetch(self, query: str, *, session_id: str = "", scheduled: bool = False) -> str:
            if not in_public_interaction():
                self._probe_behavior()
                query = _without_known_meme_notes(query)
            return self._prefetch_impl(query, session_id=session_id, scheduled=scheduled)

        def _prefetch_impl(self, query: str, *, session_id: str = "", scheduled: bool = False) -> str:
            if self._shadow or not query or self._conn is None:
                return ""
            current_session_id = session_id or self._session_id

            if scheduled:
                if (in_public_interaction()
                        or self._read_gate != "scheduled" or not self._scheduled_allowed()):
                    return ""
                self._record_scheduled_attempt()
                try:
                    results = self._search(
                        {"query": query, "top_k": 10},
                        for_injection=True, with_source=True)
                    results.sort(key=lambda item: item["importance"], reverse=True)
                    results = results[:1]
                    if not results:
                        return ""
                    prefix = "## 小事记忆\n"
                    line = "- %s%s" % (self._prov_line(results[0], current_session_id),
                                       results[0].get("summary") or results[0]["raw"])
                    if len(prefix) + len(line) > self._inject_max_chars:
                        return ""
                    self._mark_asked(
                        [item for item in results
                         if item.get("attribution") != "shared_experience"],
                        "scheduled", current_session_id)
                    self._record_scheduled_ask()
                    return prefix + line
                except Exception as exc:
                    logger.warning("sanae_mem: scheduled prefetch failed (%s)", type(exc).__name__)
                    return ""

            public = in_public_interaction()

            if not public and str(getattr(self, "_agent_context", "")) == "cron":
                return ""

            if not public:
                with self._lock:
                    request = self._association_hook_request
                    if request is not None:
                        request["closed"] = True
                        self._association_hook_request = None
                if (self._association_hook_registered or
                        (request is not None and request["session_id"] == current_session_id
                         and request["query"] == str(query))):
                    return ""

            cache = self._association_cache
            if cache and not self._association_read_valid():
                self._association_cache = None
                self._last_recall = None
                cache = None
            if not public and cache and cache.get("delivered_by_hook"):
                cached_key = cache.get("key") or ("", "")
                if cached_key[0] == str(query) and cached_key[1] == str(current_session_id):
                    return ""

            fresh_session = self._is_fresh_session_turn(current_session_id)
            explicit_past = self._triggers_explicit_past(query)
            gap_continuation = False
            if not explicit_past:
                gap_continuation = bool(
                    self._gap_continue
                    and current_session_id not in self._resumed_session_ids
                    and not self._is_trivial_message(query)
                    and not self._is_busy_context(query)
                    and (fresh_session or self._gap_detected())
                )

            if public:
                result = self.associate_context(
                    query, purpose="public", session_id=current_session_id,
                    max_chars=self._inject_max_chars, hot_path=True)
                return result["context"] if result.get("status") == "ok" else ""

            limit = min(self._inject_max_chars, _GAP_INJECT_MAX_CHARS) if gap_continuation else self._inject_max_chars

            result = self.associate_context(
                query, purpose="chat", session_id=current_session_id,
                max_chars=limit, gap_continuation=gap_continuation, hot_path=True,
            )
            if result.get("status") != "ok" or not result.get("context"):
                return ""
            via = "gap" if gap_continuation else "explicit_past" if explicit_past else "chat"
            self._mark_asked(
                [item for item in result["items"]
                 if item.get("attribution") != "shared_experience"],
                via, current_session_id)
            if gap_continuation:
                self._resumed_session_ids.add(current_session_id)
            return result["context"]
