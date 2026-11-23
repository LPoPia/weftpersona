from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Protocol

from .group_scope import exclude_group_rows
from ..infra.host_bridge import is_test_isolated_turn
from .resident_store import ENTRY_DELIMITER as _RESIDENT_DELIMITER, ResidentStore, joint_baseline, parse_entries
from .source_context import in_public_interaction, source_notice

if TYPE_CHECKING:
    from ..infra.status_store import StatusMutation, StatusPayload
    from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


class _PutEventReview(Protocol):
    def __call__(
        self, event_id: int, /, *, target: str, duty: str, content_sha256: str,
        event_sha256: str, event_version: str, baseline_sha256: str, source_kind: str,
        significance: str, reviewed_text: str, now: str,
    ) -> str: ...


@dataclass(frozen=True)
class PromotionReviewDependencies:
    sqlite: SqliteFacade
    current_session_id: Callable[[], str]
    memory_dir: Callable[[], Path]
    write_owner: Callable[[], str]
    read_scope_clause: Callable[[str], tuple[str, list[str]]]
    event_source_context: Callable[[int], dict | None]
    get_event_review: Callable[[int, str], sqlite3.Row | None]
    put_event_review: _PutEventReview
    review_candidate: Callable[[str], dict | None]
    model_available: Callable[[], bool]
    distill_daily_cap: Callable[[], int]
    resident_review_daily_cap: Callable[[], int]
    read_status: Callable[[], StatusPayload]
    mutate_status: Callable[[StatusMutation], None]
    roll_status_date: Callable[[StatusPayload, str], None]
    write_candidate_status: Callable[[bool, datetime, str], None]


class PromotionReview:
        def __init__(self, dependencies: PromotionReviewDependencies) -> None:
            self._dependencies = dependencies
            self._sqlite = dependencies.sqlite

        def read_resident_texts(self, memory_dir: Path | None = None) -> tuple[str, str] | None:
            try:
                store = ResidentStore(Path(memory_dir) if memory_dir is not None
                                      else self._dependencies.memory_dir())
                return (_RESIDENT_DELIMITER.join(store.entries("user")),
                        _RESIDENT_DELIMITER.join(store.entries("memory")))
            except (OSError, ValueError):
                return None

        def resident_baseline(self, memory_dir: Path | None = None) -> str | None:
            resident = self.read_resident_texts(memory_dir)
            return joint_baseline(*resident) if resident is not None else None

        def can_review_candidate(self, now: datetime) -> bool:
            status = self._dependencies.read_status()
            today = now.date().isoformat()
            used = int(status.get("daily_candidate_reviews", 0)) if status.get("date") == today else 0
            return used < max(0, int(self._dependencies.distill_daily_cap()))

        def note_candidate_call(self, now: datetime) -> None:
            today = now.date().isoformat()

            def record(status: dict) -> dict:
                self._dependencies.roll_status_date(status, today)
                prior = int(status.get("daily_candidate_reviews", 0))
                status["daily_candidate_reviews"] = prior + 1
                return status

            try:
                self._dependencies.mutate_status(record)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "sanae_mem: daily candidate review quota status not written (%s)",
                    type(exc).__name__,
                )

        def can_review_resident(self, now: datetime) -> bool:
            status = self._dependencies.read_status()
            today = now.date().isoformat()
            used = int(status.get("daily_resident_reviews", 0)) if status.get("date") == today else 0
            return used < max(0, int(self._dependencies.resident_review_daily_cap()))

        def reserve_resident_call(
            self, now: datetime,
        ) -> Literal["", "quota_exhausted", "quota_unavailable"]:
            """Claim one shared resident review slot under the status file's writer lock."""
            today = now.date().isoformat()
            cap = max(0, int(self._dependencies.resident_review_daily_cap()))
            reserved = False

            def record(status: dict) -> dict:
                nonlocal reserved
                self._dependencies.roll_status_date(status, today)
                count = int(status.get("daily_resident_reviews", 0))
                if count < cap:
                    status["daily_resident_reviews"] = count + 1
                    reserved = True
                return status

            try:
                self._dependencies.mutate_status(record)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "sanae_mem: daily resident review quota status not written (%s)",
                    type(exc).__name__,
                )
                return "quota_unavailable"
            return "" if reserved else "quota_exhausted"

        def _review_owner(self, owner_key: Optional[str] = None) -> str:
            if owner_key is not None:
                return str(owner_key or "")
            return str(self._dependencies.write_owner() or "")

        def review_promotion_candidate(self, event_id: int, now: Optional[datetime] = None, *,
                                       context: Optional[Dict[str, Any]] = None,
                                       allow_llm: bool = True,
                                       owner_key: Optional[str] = None,
                                       maintenance: bool = False,
                                       enforce_daily_cap: bool = True) -> dict:
            """在既有 sanae_extract 额度内，对事件生成摘要、职责、长期意义和覆盖判断。

            真实调用尝试即计额度；失败或解析不保守返回 none，事件原文保留。只返回候选提案，
            不直接写长期记忆。后端保全正文、目标、事件与常驻 baseline，供重放核对。

            按 id 读取只取**当前可信 owner**（无绑定时是 `owner_key=''` 的那些行）的行：
            默认从可信写绑定解析 owner，SQL 层就带 `owner_key=?`，别人的行返回
            `event_not_found`，不会先把 raw/summary 取出来再遮罩。`maintenance=True` 是唯一的
            不设 owner 读取，只由内部维护流（`distill`）显式声明；任何漏传参数的调用都按当前
            owner 处理，不会自动升级成全库。"""
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            if is_test_isolated_turn(self._dependencies.current_session_id()):
                return {"action": "none", "ok": False, "reason": "test_session_isolated"}
            if in_public_interaction():
                return {"action": "none", "reason": "review_unavailable_in_public_interaction"}
            now = now or datetime.now().astimezone()
            owner = "" if maintenance else self._review_owner(owner_key)
            sql = ("SELECT id, raw, ts, summary, updated_at, created_at, "
                   "experience_subject, fact_subject FROM events WHERE id=? AND blacklist=0"
                   " AND " + exclude_group_rows())
            params: List[Any] = [int(event_id)]
            if not maintenance:
                clause, owner_params = self._dependencies.read_scope_clause(owner)
                sql += f" AND {clause}"
                params.extend(owner_params)
            row = self._sqlite.fetchone(sql, tuple(params))
            if row is None:
                return {"action": "none", "reason": "event_not_found", "event_id": int(event_id)}
            from ..storage.candidates import build_model_context, review as _review
            resident = self.read_resident_texts()
            resident_ok = resident is not None
            resident_user, resident_memory = resident if resident is not None else ("", "")
            source = self._dependencies.event_source_context(int(event_id))
            source_kind = str((source or {}).get("kind") or (context or {}).get("source_kind") or "unknown")
            version = str(row["updated_at"] or row["created_at"] or "")
            baseline_sha = joint_baseline(resident_user, resident_memory)
            content_sha = hashlib.sha256(str(row["raw"] or "").encode("utf-8")).hexdigest()
            if not resident_ok:
                proposal = {"action": "none", "reason": "resident_unavailable"}
                return dict(proposal, proposal=proposal, review_sha256="",
                            event_version=version, baseline_sha256=baseline_sha,
                            event_sha256=content_sha, parsed=False,
                            context_truncated=False, resident_available=False)
            subject_source = dict(source) if isinstance(source, dict) else {}
            if subject_source:
                for key in ("experience_subject", "fact_subject"):
                    value = str(row[key] or "")
                    if value:
                        subject_source[key] = value
                expected_notice = source_notice(subject_source)
            else:
                expected_notice = ""
            if allow_llm:
                existing = self._dependencies.get_event_review(int(event_id), "")
                if (existing is not None
                        and str(existing["duty"] or "") in ("user", "memory", "none")
                        and str(existing["event_sha256"] or "") == content_sha
                        and str(existing["event_version"] or "") == version
                        and str(existing["baseline_sha256"] or "") == baseline_sha
                        and (not expected_notice
                             or str(existing["target"] or "") != "memory"
                             or not str(existing["reviewed_text"] or "").strip()
                             or expected_notice in str(existing["reviewed_text"] or ""))):
                    existing_target = str(existing["target"] or "")
                    covered_row = self._sqlite.fetchone(
                        "SELECT covered_sha256 FROM recognition_review_targets WHERE review_sha256=?",
                        (str(existing["review_sha256"]),),
                    )
                    covered_entry = ""
                    reusable = True
                    if covered_row is not None and str(covered_row["covered_sha256"] or ""):
                        target_pool = parse_entries(
                            resident_user if existing_target == "user" else resident_memory
                        )
                        matched = [
                            item for item in target_pool
                            if hashlib.sha256(item.encode("utf-8")).hexdigest() == str(covered_row["covered_sha256"])
                        ]
                        if matched:
                            covered_entry = matched[0]
                        else:
                            reusable = False
                    if reusable:
                        action = (
                            "merge_evidence" if (existing_target and covered_entry)
                            else "propose_new" if existing_target
                            else "none"
                        )
                        proposal = {"action": action,
                                    "target": existing_target,
                                    "text": str(existing["reviewed_text"] or ""),
                                    "significance": str(existing["significance"] or ""),
                                    "background": {"event_id": int(event_id), "event_sha256": content_sha,
                                                   "ts": str(row["ts"] or ""),
                                                   "event_chars": len(str(row["raw"] or "")),
                                                   "source_kind": str(existing["source_kind"] or "unknown"),
                                                   "review_sha256": str(existing["review_sha256"]),
                                                   "baseline_sha256": baseline_sha,
                                                   "event_version": version}}
                        if covered_entry:
                            proposal["covered_by"] = covered_entry
                        if proposal["action"] == "none":
                            proposal["reason"] = "stored_review_no_long_term"
                        return dict(proposal, proposal=proposal,
                                    review_sha256=str(existing["review_sha256"]),
                                    event_version=version, baseline_sha256=baseline_sha,
                                    event_sha256=content_sha,
                                    parsed=bool(proposal["text"] or str(existing["duty"] or "") in ("user", "memory", "none")),
                                    context_truncated=bool((context or {}).get("truncated")),
                                    resident_available=resident_ok, reused=True)
            if context is None:
                context = build_model_context(event_raw=str(row["raw"] or ""), source_kind=source_kind,
                                              resident_user=resident_user, resident_memory=resident_memory)
            if subject_source:
                context = dict(context, text=expected_notice
                               + "\nExternal claims remain attributed and unverified, not maintainer confirmation.\n"
                               + str(context.get("text") or ""))
            candidate = None
            review_skip_reason = ""
            quota_ok = (self.can_review_candidate(now) if enforce_daily_cap
                        else self.can_review_resident(now))
            if allow_llm and quota_ok and not context.get("truncated"):
                if self._dependencies.model_available():
                    if enforce_daily_cap:
                        self.note_candidate_call(now)
                    else:
                        review_skip_reason = self.reserve_resident_call(now)
                if not review_skip_reason:
                    try:
                        candidate = self._dependencies.review_candidate(str(context.get("text") or ""))
                    except OSError as exc:
                        logger.warning("sanae_mem: candidate review failed (%s)", type(exc).__name__)
                        review_skip_reason = "review_error"
            else:
                review_skip_reason = ("quota_exhausted" if not quota_ok else
                                      "context_truncated" if context.get("truncated") else
                                      "review_disabled")
                logger.info("sanae_mem: candidate review skipped (quota=%s truncated=%s resident=%s)",
                            quota_ok, context.get("truncated"), resident_ok)
            proposal = _review({"id": int(row["id"]), "raw": str(row["raw"]), "ts": str(row["ts"]),
                                "source_kind": source_kind}, candidate)
            if allow_llm and candidate is None:
                self._dependencies.write_candidate_status(
                    False, now, review_skip_reason or "candidate_parse_failed")
            elif candidate is not None and self._dependencies.read_status().get(
                    "components", {}).get("resident_review", {}).get("ok") is False:
                self._dependencies.write_candidate_status(True, now, "")
            if source and proposal.get("target") == "user" and source.get("kind") != "maintainer":
                proposal = {"action": "none", "reason": "event_source_not_maintainer",
                            "background": proposal.get("background", {})}
            elif source and proposal.get("target") == "memory" and proposal.get("text"):
                proposal["text"] += "\n" + expected_notice
            review_sha = ""
            if candidate is not None:
                try:
                    with self._sqlite.transaction() as conn:
                        next_review_sha = self._dependencies.put_event_review(
                            int(row["id"]), target=str(proposal.get("target") or ""),
                            duty=str(candidate.get("duty") or ""),
                            content_sha256=hashlib.sha256(str(proposal.get("text") or "").encode("utf-8")).hexdigest(),
                            event_sha256=content_sha,
                            event_version=version, baseline_sha256=baseline_sha, source_kind=source_kind,
                            significance=str(proposal.get("significance") or ""),
                            reviewed_text=str(proposal.get("text") or ""), now=now.isoformat())
                        if next_review_sha and str(proposal.get("action") or "") == "merge_evidence":
                            covered_text = str(proposal.get("covered_by") or "")
                            conn.execute(
                                "INSERT OR REPLACE INTO recognition_review_targets"
                                "(review_sha256,covered_sha256) VALUES(?,?)",
                                (next_review_sha, hashlib.sha256(covered_text.encode("utf-8")).hexdigest()),
                            )
                        review_sha = next_review_sha
                except sqlite3.Error as exc:
                    logger.warning("sanae_mem: candidate review could not be stored (%s)", type(exc).__name__)
                    review_sha = ""
                    proposal = {
                        "action": "none",
                        "reason": "review_storage_failed",
                        "background": proposal.get("background", {}),
                    }
            review_managed = bool(review_sha) and str(proposal.get("action")) == "propose_new"
            if review_managed:
                proposal["background"] = dict(proposal.get("background") or {},
                                              review_sha256=review_sha, baseline_sha256=baseline_sha,
                                              event_version=version)
            try:
                def record(status: dict) -> dict:
                    status["last_candidate_review_at"] = now.isoformat()
                    status["last_candidate_review"] = {
                        "event_id": int(row["id"]), "action": proposal.get("action", "none"),
                        "duty": proposal.get("target", ""), "parsed": bool(candidate),
                        "stored": bool(review_sha),
                    }
                    return status

                self._dependencies.mutate_status(record)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "sanae_mem: last candidate review status not written (%s)",
                    type(exc).__name__,
                )
            return dict(proposal, proposal=proposal, review_sha256=review_sha, event_version=version,
                        baseline_sha256=baseline_sha,
                        event_sha256=proposal.get("background", {}).get("event_sha256", ""),
                        parsed=bool(candidate), context_truncated=bool(context.get("truncated")),
                        resident_available=resident_ok, review_skip_reason=review_skip_reason)

        @classmethod
        def _migration_binding_id(cls, target: str, content_sha256: str) -> int:
            seed = f"migration|{target}|{content_sha256}"
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
            return -(int(digest, 16) + 2)

        def review_migration_text(self, text: str, target: str,
                                  now: Optional[datetime] = None) -> dict:
            """对迁入 memory/user 的既有正文做语义审核并保全记录，不直接写长期记忆。

            沿用 sanae_extract 额度，调用方自报目标或来源不能授资格。审核通过返回 ok=True 和记录摘要，
            否则返回 ok=False 及 reason，保留原来源。"""
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            if is_test_isolated_turn(self._dependencies.current_session_id()):
                return {"action": "none", "ok": False, "reason": "test_session_isolated"}
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if str(target) not in ("memory", "user"):
                return {"ok": False, "reason": "not_a_long_term_target"}
            from ..storage.candidates import build_model_context
            resident = self.read_resident_texts()
            resident_ok = resident is not None
            resident_user, resident_memory = resident if resident is not None else ("", "")
            context = build_model_context(event_raw=body, source_kind="migration",
                                          resident_user=resident_user, resident_memory=resident_memory)
            candidate = None
            quota_ok = self.can_review_candidate(now)
            if quota_ok and not (context.get("truncated") or not resident_ok):
                if self._dependencies.model_available():
                    self.note_candidate_call(now)
                try:
                    candidate = self._dependencies.review_candidate(str(context.get("text") or ""))
                except OSError as exc:
                    logger.warning("sanae_mem: migration review failed (%s)", type(exc).__name__)
                    return {"ok": False, "reason": "review_failed"}
            if candidate is None:
                reason = ("context_truncated" if context.get("truncated")
                          else "resident_unavailable" if not resident_ok
                          else "quota_exhausted" if not quota_ok
                          else "candidate_parse_failed")
                return {"ok": False, "reason": reason, "target": str(target)}
            if str(candidate.get("duty") or "") != str(target) or not candidate.get("long_term"):
                return {"ok": False, "reason": "not_a_long_term_candidate_for_target",
                        "target": str(target), "duty": candidate.get("duty", "")}
            content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            baseline_sha = joint_baseline(resident_user, resident_memory)
            binding_id = self._migration_binding_id(str(target), content_sha)
            review_sha = self._dependencies.put_event_review(
                binding_id, target=str(target), duty=str(candidate.get("duty") or target),
                content_sha256=content_sha, event_sha256="", event_version="",
                baseline_sha256=baseline_sha, source_kind="migration",
                significance=str(candidate.get("significance") or ""),
                reviewed_text=body, now=now.isoformat())
            return {"ok": True, "review_sha256": review_sha, "baseline_sha256": baseline_sha,
                    "content_sha256": content_sha, "target": str(target),
                    "event_sha256": "", "event_version": "", "significance": candidate.get("significance", "")}

        def migration_binding(self, declared: Dict[str, Any]) -> Optional[dict]:
            """按（由目标＋正文 hash 决定的绑定 id，载荷声明的记录 id）回读迁移审核记录。

            取不到 ⇒ None（宿主按"核验不了"拒写、留源）。
            """
            declared = declared or {}
            content_sha = str(declared.get("content_sha256") or "")
            target = str(declared.get("duty") or declared.get("target") or "")
            review_sha = str(declared.get("review_sha256") or "")
            if not content_sha or not target:
                return None
            row = self._dependencies.get_event_review(self._migration_binding_id(target, content_sha), review_sha)
            if row is None:
                return None
            return {key: row[key] for key in row.keys()}

        def review_legacy_migration_text(self, text: str, target: str, *, source_sha256: str,
                                         source_ref: str, source_rel: str, batch_id: str,
                                         now: Optional[datetime] = None) -> dict:
            """**本批已授权保全迁移**的审核记录保全（内部方法；不是工具面、不是授权来源）。

            与 `review_migration_text` 的区别：那条的语义结论来自辅助槽**现场判定**（可能因额度／
            截断／解析失败待定）。这条的结论**已由本批逐项真实语义审读作出并冻结**；授权与正文
            核验在**宿主内部函数**（`tools.memory_organize.stage_legacy_authorized_migration`：
            清单必须逐字节等于冻结摘要、源整档 hash、切片位置逐字、切片 hash 三道核对）——
            本方法**不是**授权来源，只是把那条已核过的结论按确切正文 hash＋目标＋常驻 baseline
            ＋来源片段指纹**保全**成一条后端 `event_reviews` 记录，供共用重放回读逐项核对。

            ⛔ 公开工具面没有通往这里的模式（`sanae_mem_review` 不认 `legacy_migration`）：
            caller 自报 text／target／hash／batch **不**建立任何权限。
            """
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if str(target) not in ("memory", "user"):
                return {"ok": False, "reason": "not_a_long_term_target"}
            if not str(source_sha256 or "") or not str(source_ref or "") or not str(batch_id or ""):
                return {"ok": False, "reason": "source_fragment_fingerprint_required"}
            resident = self.read_resident_texts()
            if resident is None:
                return {"ok": False, "reason": "resident_unavailable"}
            resident_user, resident_memory = resident
            content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            baseline_sha = joint_baseline(resident_user, resident_memory)
            version = f"legacy-batch|{batch_id}|{source_rel}|{source_ref}"
            binding_id = self._migration_binding_id(str(target), content_sha)
            review_sha = self._dependencies.put_event_review(
                binding_id, target=str(target), duty=str(target),
                content_sha256=content_sha, event_sha256=str(source_sha256),
                event_version=version, baseline_sha256=baseline_sha, source_kind="legacy_snapshot",
                significance="", reviewed_text=body, now=now.isoformat())
            return {"ok": True, "review_sha256": review_sha, "baseline_sha256": baseline_sha,
                    "content_sha256": content_sha, "target": str(target),
                    "source_sha256": str(source_sha256), "source_ref": str(source_ref),
                    "batch_id": str(batch_id), "duty": str(target),
                    "long_term": True, "reason": ""}
