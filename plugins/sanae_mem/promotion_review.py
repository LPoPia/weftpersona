from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RESIDENT_DELIMITER = "\n§\n"


def _normalize_resident_text(text: str) -> str:
    entries = [part.strip() for part in str(text or "").split(_RESIDENT_DELIMITER)]
    return _RESIDENT_DELIMITER.join(dict.fromkeys(part for part in entries if part))


class PromotionReviewMixin:
        @staticmethod
        def _resident_entries(text: str) -> List[str]:
            return [part for part in str(text or "").split(_RESIDENT_DELIMITER) if part]

        def _resident_texts(self, memory_dir: Optional[Path] = None) -> tuple:
            """Read USER/MEMORY with one rule for the provider root and a host memory dir.

            Missing files count as empty (never created); UTF-8-sig is accepted; any other
            read/decode error marks the read failed. Entries are normalized and deduplicated
            on the existing delimiter.
            """
            try:
                from .textutil import provider_home
            except ImportError:
                from textutil import provider_home
            try:
                memories = (Path(memory_dir) if memory_dir is not None
                            else Path(provider_home(self)) / "memories")
                out = []
                read_ok = True
                for name in ("USER.md", "MEMORY.md"):
                    try:
                        text = (memories / name).read_text(encoding="utf-8-sig")
                    except FileNotFoundError:
                        text = ""
                    except (OSError, ValueError):
                        read_ok = False
                        text = ""
                    out.append(_normalize_resident_text(text))
            except Exception:
                self._resident_read_ok = False
                return ("", "")
            self._resident_read_ok = read_ok
            return (out[0], out[1])

        def resident_baseline(self, memory_dir: Optional[Path] = None) -> Optional[str]:
            resident_user, resident_memory = PromotionReviewMixin._resident_texts(self, memory_dir)
            if not getattr(self, "_resident_read_ok", False):
                return None
            return PromotionReviewMixin._resident_baseline_sha256(resident_user, resident_memory)

        @staticmethod
        def _resident_baseline_sha256(resident_user: str, resident_memory: str) -> str:
            payload = json.dumps([str(resident_user), str(resident_memory)],
                                 ensure_ascii=False, separators=(",", ":"))
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        def _llm_calls_today(self, now: datetime) -> int:
            status = self._read_status()
            today = now.date().isoformat()
            return int(status.get("daily_candidate_reviews", 0)) if status.get("date") == today else 0

        def _note_llm_call(self, now: datetime) -> None:
            status = self._read_status()
            today = now.date().isoformat()
            prior = int(status.get("daily_candidate_reviews", 0)) if status.get("date") == today else 0
            status["date"] = today
            status["daily_candidate_reviews"] = prior + 1
            try:
                self._write_status_payload(status)
            except Exception:
                pass

        def review_promotion_candidate(self, event_id: int, now: Optional[datetime] = None, *,
                                       context: Optional[Dict[str, Any]] = None,
                                       allow_llm: bool = True) -> dict:
            """在既有 sanae_extract 额度内，对事件生成摘要、职责、长期意义和覆盖判断。

            真实调用尝试即计额度；失败或解析不保守返回 none，事件原文保留。只返回候选提案，
            不直接写长期记忆。后端保全正文、目标、事件与常驻 baseline，供重放核对。"""
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            try:
                from .source_context import in_public_interaction
            except ImportError:
                from source_context import in_public_interaction
            if in_public_interaction():
                return {"action": "none", "reason": "review_unavailable_in_public_interaction"}
            now = now or datetime.now().astimezone()
            row = self._conn.execute(
                "SELECT id, raw, ts, summary, updated_at, created_at, "
                "experience_subject, fact_subject FROM events WHERE id=? AND blacklist=0",
                (int(event_id),),
            ).fetchone()
            if row is None:
                return {"action": "none", "reason": "event_not_found", "event_id": int(event_id)}
            try:
                from .candidates import build_model_context, review as _review
            except ImportError:
                from candidates import build_model_context, review as _review
            resident_user, resident_memory = self._resident_texts()
            resident_ok = bool(getattr(self, "_resident_read_ok", False))
            source_kind = str((context or {}).get("source_kind") or "unknown")
            version = str(row["updated_at"] or row["created_at"] or "")
            baseline_sha = self._resident_baseline_sha256(resident_user, resident_memory)
            content_sha = hashlib.sha256(str(row["raw"] or "").encode("utf-8")).hexdigest()
            if not resident_ok:
                proposal = {"action": "none", "reason": "resident_unavailable"}
                return dict(proposal, proposal=proposal, review_sha256="",
                            event_version=version, baseline_sha256=baseline_sha,
                            event_sha256=content_sha, parsed=False,
                            context_truncated=False, resident_available=False)
            source = self.event_source_context(int(event_id))
            subject_source = dict(source) if isinstance(source, dict) else {}
            if subject_source:
                for key in ("experience_subject", "fact_subject"):
                    value = str(row[key] or "")
                    if value:
                        subject_source[key] = value
                from agent.interaction_context import source_notice
                expected_notice = source_notice(subject_source)
            else:
                expected_notice = ""
            if allow_llm:
                existing = self.get_event_review(int(event_id))
                if (existing is not None
                        and str(existing["duty"] or "") in ("user", "memory")
                        and str(existing["event_sha256"] or "") == content_sha
                        and str(existing["event_version"] or "") == version
                        and str(existing["baseline_sha256"] or "") == baseline_sha
                        and (not expected_notice
                             or str(existing["target"] or "") != "memory"
                             or not str(existing["reviewed_text"] or "").strip()
                             or expected_notice in str(existing["reviewed_text"] or ""))):
                    proposal = {"action": "propose_new" if str(existing["target"] or "") else "none",
                                "target": str(existing["target"] or ""),
                                "text": str(existing["reviewed_text"] or ""),
                                "significance": str(existing["significance"] or ""),
                                "background": {"event_id": int(event_id), "event_sha256": content_sha,
                                               "ts": str(row["ts"] or ""),
                                               "event_chars": len(str(row["raw"] or "")),
                                               "source_kind": str(existing["source_kind"] or "unknown"),
                                               "review_sha256": str(existing["review_sha256"]),
                                               "baseline_sha256": baseline_sha,
                                               "event_version": version}}
                    if proposal["action"] == "none":
                        proposal["reason"] = "stored_review_no_long_term"
                    return dict(proposal, proposal=proposal,
                                review_sha256=str(existing["review_sha256"]),
                                event_version=version, baseline_sha256=baseline_sha,
                                event_sha256=content_sha, parsed=bool(proposal["text"]),
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
            quota_ok = self._llm_calls_today(now) < max(0, int(self._distill_daily_cap))
            if allow_llm and quota_ok and not context.get("truncated"):
                if self._llm is not None:
                    self._note_llm_call(now)
                try:
                    candidate = self._maybe_llm_candidate(str(context.get("text") or ""),
                                                          full_context=True)
                except Exception as exc:
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
            if source and proposal.get("target") == "user" and source.get("kind") != "maintainer":
                proposal = {"action": "none", "reason": "event_source_not_maintainer",
                            "background": proposal.get("background", {})}
            elif source and proposal.get("target") == "memory" and proposal.get("text"):
                proposal["text"] += "\n" + expected_notice
            review_sha = ""
            if candidate is not None:
                try:
                    review_sha = self.put_event_review(
                        int(row["id"]), target=str(proposal.get("target") or ""),
                        duty=str(candidate.get("duty") or ""),
                        content_sha256=hashlib.sha256(str(proposal.get("text") or "").encode("utf-8")).hexdigest(),
                        event_sha256=content_sha,
                        event_version=version, baseline_sha256=baseline_sha, source_kind=source_kind,
                        significance=str(proposal.get("significance") or ""),
                        reviewed_text=str(proposal.get("text") or ""), now=now.isoformat())
                except Exception as exc:
                    logger.warning("sanae_mem: candidate review could not be stored (%s)", type(exc).__name__)
            review_managed = bool(review_sha) and str(proposal.get("action")) == "propose_new"
            if review_managed:
                proposal["background"] = dict(proposal.get("background") or {},
                                              review_sha256=review_sha, baseline_sha256=baseline_sha,
                                              event_version=version)
            try:
                status = self._read_status()
                status["last_candidate_review_at"] = now.isoformat()
                status["last_candidate_review"] = {
                    "event_id": int(row["id"]), "action": proposal.get("action", "none"),
                    "duty": proposal.get("target", ""), "parsed": bool(candidate),
                    "stored": bool(review_sha),
                }
                self._write_status_payload(status)
            except Exception:
                pass
            return dict(proposal, proposal=proposal, review_sha256=review_sha, event_version=version,
                        baseline_sha256=baseline_sha,
                        event_sha256=proposal.get("background", {}).get("event_sha256", ""),
                        parsed=bool(candidate), context_truncated=bool(context.get("truncated")),
                        resident_available=resident_ok, review_skip_reason=review_skip_reason)

        def review_outcome(self, event_id: int, now: Optional[datetime] = None) -> dict:
            """给宿主固定入口用：返回**可暂存**的审核结论（含后端记录摘要与确切的候选正文）。

            与 `distill` 共用同一实现（同一次 `sanae_extract` 调用）与同一日额度。
            """
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            review = self.review_promotion_candidate(int(event_id), now)
            review["ok"] = bool(review.get("resident_available", True))
            return review

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
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if str(target) not in ("memory", "user"):
                return {"ok": False, "reason": "not_a_long_term_target"}
            try:
                from .candidates import build_model_context, parse_candidate_result, review as _review
            except ImportError:
                from candidates import build_model_context, parse_candidate_result, review as _review
            resident_user, resident_memory = self._resident_texts()
            resident_ok = bool(getattr(self, "_resident_read_ok", False))
            context = build_model_context(event_raw=body, source_kind="migration",
                                          resident_user=resident_user, resident_memory=resident_memory)
            candidate = None
            quota_ok = self._llm_calls_today(now) < max(0, int(self._distill_daily_cap))
            if quota_ok and not (context.get("truncated") or not resident_ok):
                if self._llm is not None:
                    self._note_llm_call(now)
                try:
                    candidate = self._maybe_llm_candidate(str(context.get("text") or ""),
                                                          full_context=True)
                except Exception as exc:
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
            baseline_sha = self._resident_baseline_sha256(resident_user, resident_memory)
            binding_id = self._migration_binding_id(str(target), content_sha)
            review_sha = self.put_event_review(
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
            row = self.get_event_review(self._migration_binding_id(target, content_sha), review_sha)
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
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if str(target) not in ("memory", "user"):
                return {"ok": False, "reason": "not_a_long_term_target"}
            if not str(source_sha256 or "") or not str(source_ref or "") or not str(batch_id or ""):
                return {"ok": False, "reason": "source_fragment_fingerprint_required"}
            resident_user, resident_memory = self._resident_texts()
            resident_ok = bool(getattr(self, "_resident_read_ok", False))
            if not resident_ok:
                return {"ok": False, "reason": "resident_unavailable"}
            content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            baseline_sha = self._resident_baseline_sha256(resident_user, resident_memory)
            version = f"legacy-batch|{batch_id}|{source_rel}|{source_ref}"
            binding_id = self._migration_binding_id(str(target), content_sha)
            review_sha = self.put_event_review(
                binding_id, target=str(target), duty=str(target),
                content_sha256=content_sha, event_sha256=str(source_sha256),
                event_version=version, baseline_sha256=baseline_sha, source_kind="legacy_snapshot",
                significance="", reviewed_text=body, now=now.isoformat())
            return {"ok": True, "review_sha256": review_sha, "baseline_sha256": baseline_sha,
                    "content_sha256": content_sha, "target": str(target),
                    "source_sha256": str(source_sha256), "source_ref": str(source_ref),
                    "batch_id": str(batch_id), "duty": str(target),
                    "long_term": True, "reason": ""}
