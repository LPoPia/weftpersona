from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    from . import auxiliary
    from .source_context import self_attribution
    from .textutil import _fingerprint
    from .words import _noise_shape
except ImportError:
    import auxiliary
    from source_context import self_attribution
    from textutil import _fingerprint
    from words import _noise_shape

logger = logging.getLogger(__name__)

_SHORT_EVENT_IMPORTANCE = 0.5
_SHORT_EVENT_TTL_DAYS = 14


class LlmGuardMixin:
        _RESIDENT_ROUTE_SYSTEM = (
            "你是早苗记忆系统的旁路整理器。早苗刚调用 memory 工具想往常驻区（memory）写一段话。"
            "请判断这段话应该放行进入长期常驻（allow），还是拦截并分流到短期心里话/短期事件库（inner_state）。"
            "只输出一个 JSON 对象，不要解释、不要代码块："
            '{"route": "allow|inner_state", "inner_thought": "分流到短期心里话时的1到2句第一人称心里话（allow、纯教条总结或无特殊心绪时填空串）", '
            '"event_fact": "原条里已发生的具体客观事件（一句第一人称事实），没有就 null", "reason": "简短原因"}。\n'
            "判定规则：\n"
            "1. 只有「跨时段沉淀的关系里程碑、长期稳定的相处默契、或对既有常驻条目的修正合并」才判 allow；"
            "凡属于以下三类，一律判 inner_state（拦截不进常驻 MEMORY）："
            "① 关于早苗自身系统机制、记忆遗忘规则、提示词修改、行为准则或调教目标的讨论与约定（如「记忆遗忘机制由他设定」「我不全权管记忆」等元规则）；"
            "② 单次日常琐事、当日偶然小事（如今天发了某个表情包、聊了某件日常小事、一次操作小失误）或单次短期约定（此类具体小事走 event_fact 进事件库，不占常驻窗口）；"
            "③ 当下的揣测与猜想（如猜他发某句话/某个表情是因为心虚、生气、嫌弃或累了）、挨批或争执后的自我检讨与反思、"
            "给自己立的说话规矩（如以后我不能找补/不能顺着猜/要先听他说）、或一时的委屈/不服/心虚等当下心绪。\n"
            "2. 判 inner_state 时，inner_thought 只提炼她此刻心里的真实情绪、在意或挂念（如「刚才看岔了他的话，心里还有点过意不去」；若涉及他维护或调系统机制，只留对他操心忙碌的挂念），"
            "绝不把「接话前我得先怎么看」「记忆遗忘机制由他定」「以后不能再猜岔」这类系统机制条文、说话规矩、答题教条或自我检讨写进 inner_thought；"
            "若原条纯粹是总结规矩/系统机制，或只是普通单次日常事实而无特殊心绪，inner_thought 填空串 \"\"。用户（对方）是早苗的挚友，代词一律用「他」，不用「她」。\n"
            "3. 判 inner_state 时，原条里若确实记着一件已经发生、能说清原委的具体客观事情（做过什么、他说过或发过什么、出了什么岔子、结果如何），"
            "在 event_fact 里用一句第一人称、只讲事实的话写清它（如「今天帮他看课表时，我把周四的课看成了周三」）；"
            "去掉检讨、规矩和骂自己的话（「以后不能」「下次必须」「瞎猜」「犯蠢」之类），也不留情绪评价，只留那件事本身。"
            "涉及他维护或修改早苗时，只保留他的状态和处境（如熬夜、累、觉得某件事难、在忙什么），"
            "丢弃关于早苗的设计意图、系统机制设定和期望（如想把她改成什么样、记忆规则怎么定、改动的目的、对模型的评价、调教目标）。"
            "纯情绪拌嘴、心意揣测、反问逗弄、说自己该守什么规矩、关于早苗系统机制/记忆规则/调教目标的讨论，或根本没有具体发生过的事时，event_fact 返回 null；判 allow 时同样返回 null。"
        )

        def route_resident_write(
            self, target: str, payload: dict[str, Any], *,
            source_ctx: Any = None, session_id: str = "",
        ) -> Optional[str]:
            if self._llm is None or str(target) != "memory" or not isinstance(payload, dict):
                return None
            texts: list[str] = []
            ops = payload.get("operations")
            if isinstance(ops, list) and ops:
                for op in ops:
                    if isinstance(op, dict) and str(op.get("action") or "") in ("add", "replace", "merge"):
                        body = str(op.get("content") or op.get("new_text") or op.get("text") or "").strip()
                        if body:
                            texts.append(body)
            elif str(payload.get("action") or "") in ("add", "replace", "merge"):
                body = str(payload.get("content") or payload.get("new_text") or payload.get("text") or "").strip()
                if body:
                    texts.append(body)
            if not texts:
                return None
            user_msgs: list[str] = []
            if source_ctx is not None and getattr(source_ctx, "available", False):
                for msg in (getattr(source_ctx, "user_messages", None) or [])[-3:]:
                    txt = str(getattr(msg, "text", "") or "").strip()
                    if txt:
                        user_msgs.append(txt)
            req = {
                "target": str(target),
                "proposed_entries": texts,
                "recent_user_messages": user_msgs,
            }
            system_prompt = self._RESIDENT_ROUTE_SYSTEM
            persona_name = str(getattr(self, "_persona_name", "早苗") or "早苗").strip() or "早苗"
            partner_name = str(getattr(self, "_partner_name", "") or "").strip()
            partner_role = str(getattr(self, "_partner_role", "挚友") or "挚友").strip() or "挚友"
            partner_pronoun = "她" if str(getattr(self, "_partner_pronoun", "他") or "他").strip() == "她" else "他"
            if partner_name:
                system_prompt = system_prompt.replace("用户（对方）", f"用户（{partner_name}）", 1)
            if partner_role != "挚友":
                system_prompt = system_prompt.replace("的挚友", f"的{partner_role}")
            if partner_pronoun == "她":
                system_prompt = system_prompt.replace("代词一律用「他」，不用「她」", "代词一律用「她」，不用「他」")
            if persona_name != "早苗":
                system_prompt = system_prompt.replace("早苗", persona_name)
            raw = auxiliary.complete_simple(
                self._llm,
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
            thought = "；".join(texts)
            context = str(getattr(self, "_agent_context", ""))
            skip_inner_write = context == "cron"
            event_fact = ""
            if raw is not None:
                text = str(getattr(raw, "text", raw) or "").strip()
                first, last = text.find("{"), text.rfind("}")
                if 0 <= first <= last:
                    try:
                        parsed = json.loads(text[first:last + 1])
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        if "route" not in parsed and "duty" in parsed:
                            return None
                        route = str(parsed.get("route") or "").strip().lower()
                        if route == "allow":
                            return None
                        custom = str(parsed.get("inner_thought") or "").strip()
                        if custom:
                            thought = custom
                        elif "inner_thought" in parsed:
                            skip_inner_write = True
                        fact = parsed.get("event_fact")
                        event_fact = fact.strip() if isinstance(fact, str) else ""
            if not skip_inner_write and getattr(self, "_daily_profile_key", ""):
                try:
                    turn_id = str(getattr(source_ctx, "turn", "") or "") if source_ctx is not None else ""
                    state = self.write_inner_state(thought, turn_id=turn_id)
                    thought = state.text
                except Exception:
                    logger.warning("sanae_mem: fallback inner_state write failed", exc_info=True)
            event_id = None
            if event_fact and context == "primary":
                try:
                    event_id = self._record_short_event(event_fact, session_id)
                except Exception:
                    logger.warning("sanae_mem: resident short event write failed", exc_info=True)
            routed = {
                "success": True,
                "routed_to": "inner_state",
                "target": str(target),
                "inner_state_text": thought,
                "message": "这属于当下的心绪、揣测或一时感触，已记入[我此刻心里]，不写入长期常驻文件，留待自然沉淀。",
            }
            if event_id is not None:
                routed["event_id"] = int(event_id)
                routed["message"] += "里面那件具体的事也照原样记了一条短期经历。"
            return json.dumps(routed, ensure_ascii=False)

        def _record_short_event(self, fact: str, session_id: str) -> Optional[int]:
            write_owner = self._write_owner()
            subject, fact_subject, involved, scope = self_attribution(write_owner)
            return self._ingest_shared(
                fact, session_id, importance=_SHORT_EVENT_IMPORTANCE,
                ttl_days_exact=_SHORT_EVENT_TTL_DAYS, owner_key=write_owner, source="explicit",
                source_context={
                    "kind": "self", "platform": str(getattr(self, "_platform", "") or ""),
                    "visibility": "private", "certainty": "observed",
                    "ref": "resident-event:" + _fingerprint(fact),
                },
                experience_subject=subject, fact_subject=fact_subject,
                involved_owner_keys=involved, read_scope=scope,
            )

        _CANDIDATE_SYSTEM = (
            "把候选内容概括成一条不超过120个中文字符的事件式记忆，并判断它是否值得进长期档。"
            "只输出一个 JSON 对象，不要解释、不要代码块："
            '{"summary": "…", "duty": "user|memory|none", '
            '"source_kind": "user_affirmed|event_promotion|user_statement|unknown", '
            '"long_term": true|false, "covered_by": "已被哪条常驻内容覆盖（没有就空串）", '
            '"significance": "一句话说明长期意义"}。'
            "规则：duty=user 只给「用户长期稳定事实／明确偏好」；duty=memory 只给「关系认识／重要经历」；"
            "规则、教训、运维、临时情绪、未确认推断、行为要求 ⇒ duty=none 且 long_term=false。"
            "涉及维护或修改早苗时，summary 只保留他的状态和处境（如熬夜、累、觉得某件事难、在忙什么），"
            "丢弃关于早苗的设计意图和期望（如想把她改成什么样、改动的目的、对模型的评价、调教目标）；"
            "若只关乎设计意图、模型评价或调教目标，则 duty=none 且 long_term=false。"
            "summary 里不得出现 id／hash／版本／CAS／系统或运维措辞。"
        )

        def _maybe_llm_candidate(self, text: str, *, full_context: bool = False) -> Optional[dict]:
            if self._llm is None:
                return None
            if full_context:
                candidate = str(text or "").strip()
                if not candidate:
                    return None
            else:
                candidate = self._candidate_text(text)[:150]
                if not candidate:
                    return None
                if _noise_shape(candidate, relax_deictic=True):
                    return None
            result = auxiliary.complete_complex(
                self._llm,
                [
                    {"role": "system", "content": self._CANDIDATE_SYSTEM},
                    {"role": "user", "content": candidate},
                ],
                temperature=0.2, max_tokens=320, timeout=60,
                purpose="daily_memory_candidate", failure_message="candidate review unavailable",
            )
            if result is None:
                return None
            try:
                try:
                    from .candidates import parse_candidate_result
                except ImportError:
                    from candidates import parse_candidate_result
            except Exception:
                return None
            parsed = parse_candidate_result(getattr(result, "text", ""))
            if parsed is None:
                logger.info("sanae_mem: candidate review not conservatively parseable; no long-term write")
            return parsed

        def _maybe_llm_grounded_review(self, context: str) -> Optional[dict]:
            if self._llm is None or not str(context or "").strip():
                return None
            system = (
                "核对【认识正文】是否由【事实证据】真实支持。不能把单次偶然行为武断推成"
                "稳定性格；注意新旧事实冲突和反证。只能依据列出的证据和当前 USER/MEMORY，"
                "不能编造事实。不同事件 ID 或同一条用户消息里的不同句子不一定是不同经历；"
                "必须根据发生的事情、时间和情境判定是否为独立且有代表性的事实。"
                "只有认识确实有代表性且证据充分时，"
                "才返回 duty=user、long_term=true；summary 必须逐字复制【认识正文】，"
                "不能改写。若有【拟修正的旧 USER 认识】，必须核对新旧认识属于同一主题、"
                "新证据足以支持修正；成立时 covered_by 填入旧认识完整原文，不成立则拒绝。"
                "必须先逐条对照【当前 USER 常驻】，优先合并再考虑新增。"
                "即使措辞、举例、侧重点不同，只要是相同的稳定倾向且没有新的核心认识，"
                "也算被已有 USER 覆盖：long_term=true、covered_by 填入那条旧认识完整原文，"
                "绝不能只因新证据或换了说法就新建。"
                "例如旧认识为‘他重视守约’，候选为‘他会认真兑现承诺’，"
                "这就是同义覆盖，covered_by 必须是‘他重视守约’。"
                "若 significance 声称与现有 USER 认识一致而 covered_by 为空，结果自相矛盾。"
                "只有确有不能被现有认识表达的新长期含义，才可给 covered_by 空串。"
                "旧条过长而无法完整引用时应拒绝，不得截断原文。"
                "若事实不足、反证未解决或不能判断，输出 duty=none、long_term=false、"
                "covered_by=''，不得勉强形成认识。"
                "只输出 JSON 对象："
                '{"summary":"…","duty":"user|none","source_kind":"user_statement",'
                '"long_term":true|false,"covered_by":"","significance":"证据为何足以支撑判断"}'
            )
            result = auxiliary.complete_complex(
                self._llm,
                [{"role": "system", "content": system},
                 {"role": "user", "content": str(context)}],
                temperature=0.2, max_tokens=640, timeout=60,
                purpose="daily_memory_candidate", failure_message="grounded recognition review failed")
            if result is None:
                return None
            try:
                from .candidates import parse_grounded_user_result
            except ImportError:
                from candidates import parse_grounded_user_result
            return parse_grounded_user_result(getattr(result, "text", ""))
