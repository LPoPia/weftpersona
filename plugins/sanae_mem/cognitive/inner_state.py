from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Protocol

from ..infra import auxiliary
from ..infra.contracts import CasConflictError, DEFAULT_CAS_MAX_RETRIES, cas_backoff_sleep
from ..infra.defaults import DEFAULT_PARTNER_ROLE, DEFAULT_PERSONA_NAME
from .daily_rules import DAILY_SEMANTIC_TIMEOUT
from .inner_state_life import (
    INNER_STATE_MARKER, _HALF_LIVES, _afterglow, _kept_words,
    _live_words, _slow_variables, _slept_over, _thought_at,
    render_inner_state as _render_inner_state,
)
from .inner_state_model import (
    _CRON_HEARTBEAT_PROMPT, _INNER_EXPRESSION, _INNER_PREDICATE_RE,
    _RECAP_CLAUSE_RE, _SHARED_QUOTED_RE, _SHARED_SENTENCE_MAX,
    _SHARED_SENTENCE_SPLIT, _SPARK_PROMPT, _UPDATE_PROMPT, _decode_update,
    _reject_reply_parrot, _shared_forbidden_names, _shared_quotes_private_words,
    _shared_sentences, _shared_text_safe, _update_object,
)
from .inner_state_spark import (
    _CN_RUN_RE, _SPARK_COOLDOWN, _SPARK_NOTE_RE, _UNSPOKEN_NOTE_RE,
    _body_tokens, _clean_spark, _clean_unspoken_link, _compact_distant_memory_text,
    _distant_memory_attribution, _format_spark_note, _format_unspoken_note,
    _sample_distant_memories, _strip_unspoken_note,
)
from .inner_state_store import (
    _clock_state, _connect, _latest_state, _row_visibility, _shared_state,
    _shared_table_exists, _state_from_row, _states, _visible_kind,
    append_state, inspect_inner_state, latest_state,
    owner_state, read_shared_state, read_state, write_shared_state,
)
from .inner_state_types import InnerState, InnerStateTarget, InnerStateView

logger = logging.getLogger(__name__)


class _Model(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, task: str, temperature: float,
        max_tokens: int, timeout: float, purpose: str,
    ) -> object: ...


class _MutateStatus(Protocol):
    def __call__(
        self, change: Callable[[dict[str, object]], dict[str, object]], *,
        session_id: str, data_dir: Path,
    ) -> None: ...


class _RecordFlash(Protocol):
    def __call__(
        self, *, owner_key: str, user_text: str, assistant_text: str, turn_id: str,
        previous_state: InnerState | None, current_state: InnerState, now: float,
    ) -> object: ...


@dataclass(frozen=True)
class InnerStatePersona:
    name: str
    partner_name: str
    partner_role: str
    partner_pronoun: str


@dataclass(frozen=True)
class InnerStateDependencies:
    current_target: Callable[[], InnerStateTarget]
    current_session_id: Callable[[], str]
    current_read_owners: Callable[[], tuple[str, ...]]
    current_maintainer: Callable[[], str]
    current_persona: Callable[[], InnerStatePersona]
    current_model: Callable[[], _Model | None]
    mutate_status: _MutateStatus
    record_flash: _RecordFlash


class InnerStateService:
    def __init__(self, dependencies: InnerStateDependencies) -> None:
        self._deps = dependencies

    def inner_state_target(self) -> InnerStateTarget:
        return self._deps.current_target()

    def latest_inner_state(self, *, target: InnerStateTarget | None = None) -> InnerState | None:
        """The global head metadata of the single profile chain."""
        data_dir, profile_key = target if target is not None else self.inner_state_target()
        return latest_state(data_dir / "sanae.db", profile_key)

    def _owner_state(self, target: InnerStateTarget, owner_key: str) -> InnerState | None:
        data_dir, profile_key = target
        return owner_state(data_dir / "sanae.db", profile_key, owner_key)

    def read_inner_state(
        self, *, target: InnerStateTarget | None = None, owner_key: str = "",
        shared_only: bool = False, internal: bool = False,
    ) -> InnerStateView | None:
        data_dir, profile_key = target if target is not None else self.inner_state_target()
        return read_state(
            data_dir / "sanae.db", profile_key, owner_key=owner_key,
            shared_only=shared_only, internal=internal)

    def render_inner_state(
        self, state: InnerState | None = None, *, now: float | None = None,
        target: InnerStateTarget | None = None,
        session_id: str = "", owner_key: str = "", internal: bool = False,
    ) -> str:
        resolved_target = target if target is not None else self.inner_state_target()
        data_dir, _profile_key = resolved_target
        return _render_inner_state(
            data_dir / "sanae.db", resolved_target, state,
            now=time.time() if now is None else now,
            session_id=session_id or self._deps.current_session_id(),
            owner_key=owner_key, internal=internal,
        )

    def _append_inner_state(self, target: InnerStateTarget, state: InnerState) -> bool:
        data_dir, _profile_key = target
        return append_state(data_dir / "sanae.db", target, state)

    def write_inner_state(
        self, text: str, *, owner_key: str, turn_id: str = "", now: float | None = None,
        internal: bool = False, max_retries: int = DEFAULT_CAS_MAX_RETRIES,
    ) -> InnerState:
        """Keep handwriting verbatim in the verified owner's private row.

        ``internal=True`` is the trusted autonomous handwriting: the words stay in
        the internal segment instead of the execution owner's private row, and no
        spoken words are carried over. Wiring passes it only from a verified cron
        receipt; no tool argument can select it.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("心里话不能为空")
        owner = str(owner_key or "").strip()
        if not owner:
            raise ValueError("owner_key 必须来自宿主核验的对话者")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        now = time.time() if now is None else now
        target = self.inner_state_target()
        for attempt in range(max_retries + 1):
            latest = self.latest_inner_state(target=target)
            owner_state = None if internal else self._owner_state(target, owner)
            words, words_at = ("", None) if internal else _kept_words(owner_state, now)
            state = InnerState(
                revision=(latest.revision if latest else 0) + 1, text=text.strip(),
                recent_words=words, recent_words_at=words_at,
                variables=_slow_variables(latest, now),
                felt_at=now, updated_at=now,
                processed_at=latest.processed_at if latest else None,
                source="manual", thought_source="manual", turn_id=turn_id, changed=True,
                owner_key="" if internal else owner,
                text_kind="internal" if internal else "private",
                last_spark_at=owner_state.last_spark_at if owner_state else 0.0,
            )
            if self._append_inner_state(target, state):
                return state
            if attempt < max_retries:
                cas_backoff_sleep(attempt)
        raise CasConflictError(
            f"inner_state CAS retry limit exceeded after {max_retries} retries"
        )

    def record_inner_state_failure(self, reason: str, *, session_id: str = "",
                                   now: float | None = None, target: InnerStateTarget | None = None) -> None:
        stamp = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc)
        data_dir, _ = target if target is not None else self.inner_state_target()

        def record(payload: dict[str, object]) -> dict[str, object]:
            old = payload.get("inner_state")
            inner = dict(old) if isinstance(old, dict) else {}
            count = inner.get("failure_count")
            inner.update({
                "last_failure": reason, "last_failure_at": stamp.isoformat(),
                "failure_count": (count if type(count) is int else 0) + 1,
            })
            payload["inner_state"] = inner
            return payload

        self._deps.mutate_status(record, session_id=session_id, data_dir=data_dir)

    def _record_memory_axis_flash(
        self, *, appended: bool, turn_id: str, user_text: str, assistant_text: str,
        previous_state: InnerState | None, current_state: InnerState, now: float, owner_key: str,
    ) -> None:
        if not appended:
            return
        if not (str(user_text or "").strip() or str(assistant_text or "").strip()):
            return
        try:
            self._deps.record_flash(
                owner_key=owner_key, user_text=str(user_text or ""),
                assistant_text=str(assistant_text or ""), turn_id=turn_id,
                previous_state=previous_state, current_state=current_state, now=now,
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: inner-state flash promotion skipped (%s)",
                           type(exc).__name__)

    def _read_shared_state(self, target: InnerStateTarget) -> InnerState | None:
        data_dir, profile_key = target
        return read_shared_state(data_dir / "sanae.db", profile_key)

    def _write_shared_state(
        self, target: InnerStateTarget, *, text: str, variables: Mapping[str, float],
        now: float, source_revision: int,
    ) -> bool:
        data_dir, profile_key = target
        return write_shared_state(
            data_dir / "sanae.db", profile_key, text=text, variables=variables,
            now=now, source_revision=source_revision)

    def _write_shared_sentence(
        self, target: InnerStateTarget, raw: object, *, forbidden_names: tuple[str, ...],
        private_sources: tuple[str, ...], variables: Mapping[str, float], now: float,
        revision: int,
    ) -> None:
        parts = [
            safe for safe in (
                _shared_text_safe(text, forbidden_names=forbidden_names,
                                  private_sources=private_sources)
                for text in _shared_sentences(raw))
            if safe
        ]
        if not parts:
            return
        try:
            self._write_shared_state(
                target, text="".join(parts), variables=variables, now=now,
                source_revision=revision)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: shared inner view skipped (%s)", type(exc).__name__)

    def ensure_shared_inner_state(
        self, *, target: InnerStateTarget | None = None, owner_key: str,
        now: float | None = None, since: float | None = None, session_id: str = "",
    ) -> InnerState | None:
        """The current safe common view, read locally with no model call.

        Ordinary reads never call this: it exists for the nightly journal, which
        needs a real common mood on a day when nobody but the maintainer has
        talked to her. With no safe view updated at/after ``since`` there is no
        grounded common mood and ``None`` is returned as is; the caller neither
        retries nor generates a safe view here. ``owner_key``, ``now`` and
        ``session_id`` stay in the signature for those existing readers.
        """
        resolved = target if target is not None else self.inner_state_target()
        _data_dir, profile_key = resolved
        if not profile_key:
            return None
        existing = self._read_shared_state(resolved)
        if existing is not None and (since is None or existing.updated_at >= since):
            return existing
        return None

    def update_inner_state(
        self, *, turn_id: str, user_text: str, assistant_text: str, reasoning: str = "",
        experiences: str = "", image_context: str = "", session_id: str = "", now: float | None = None,
        target: InnerStateTarget | None = None,
        impression_target: tuple[Path, str, str] | None = None,
        preserve_processed_at: bool = False, owner_key: str, internal: bool = False,
    ) -> bool:
        """Synthesize from this owner's fragment, then commit only if the chain is current.

        ``internal=True`` is the trusted autonomous heartbeat: it reads and writes
        the internal segment instead of the execution owner's private row, so a
        real activity that digested someone else's private matter stays hers.
        The qualification is the caller's verified cron receipt; no tool argument
        reaches this flag.
        """
        now = time.time() if now is None else now
        target = target if target is not None else self.inner_state_target()
        owner = str(owner_key or "").strip()
        heartbeat = bool(preserve_processed_at and not user_text.strip())
        if heartbeat and not internal:
            self.record_inner_state_failure(
                "autonomous_scope_required", session_id=session_id, now=now, target=target)
            return False
        if internal:
            heartbeat = True
        if heartbeat:
            impression_target = None
        if not owner:
            self.record_inner_state_failure(
                "owner_unverified", session_id=session_id, now=now, target=target)
            return False
        try:
            from .impression import apply_impression, build_impression_prompt, read_impression
            latest = self.latest_inner_state(target=target)
            view = self.read_inner_state(target=target, owner_key=owner, internal=internal)
            private = view.private if view is not None else None
            shared = view.shared if view is not None else None
            thought_locked = False
            if latest is not None and latest.updated_at > now:
                return False
            processed_at = max(
                now,
                (latest.processed_at + 1e-6)
                if latest is not None and latest.processed_at is not None else now,
            )
            if private is not None:
                same_turn = bool(turn_id and private.turn_id == turn_id)
                if same_turn and private.source != "manual":
                    return False
                thought_locked = same_turn
            impression = (read_impression(impression_target, owners=self._deps.current_read_owners())
                          if impression_target is not None else None)
            if impression and (impression.updated_at > now or impression.turn_id == turn_id):
                impression_target = None
            private_clean = _strip_unspoken_note(private.text) if private else ""
            private_live = _strip_unspoken_note(_thought_at(private, now)) if private else ""
            shared_live = _strip_unspoken_note(_thought_at(shared, now)) if shared else ""
            live_words, live_words_at = (
                ("", None) if internal else
                (private.recent_words, private.recent_words_at) if heartbeat and private
                else _kept_words(private, now)
            )
            if private_live and shared_live:
                primary_seg = (private if private.felt_at >= shared.felt_at else shared)
            elif private_live:
                primary_seg = private
            elif shared_live:
                primary_seg = shared
            else:
                primary_seg = private or shared
            primary_live = private_live if primary_seg is private else shared_live
            primary_clean = _strip_unspoken_note(primary_seg.text) if primary_seg else ""
            slept = bool(primary_seg) and _slept_over(primary_seg.felt_at, now)
            prev_vars = _slow_variables(latest, now)

            def _segment_note(segment, live_text):
                return {
                    "text": live_text,
                    "seconds_since_feeling": max(0.0, now - segment.felt_at) if live_text else None,
                }

            previous = {
                "thought": primary_live,
                "earlier_text": primary_clean if primary_live and not slept else "",
                "recent_words": live_words,
                "source": primary_seg.thought_source if primary_seg is not None else "",
                "seconds_since_feeling": (
                    max(0.0, now - primary_seg.felt_at)
                    if primary_seg is not None and primary_live else None),
                "variables": prev_vars,
                "shared": _segment_note(shared, shared_live) if shared is not None else None,
            } if (private is not None or shared is not None or latest is not None) else None
            if previous is not None:
                if internal:
                    previous["internal"] = ({
                        **_segment_note(private, private_live),
                        "source": private.thought_source,
                    } if private is not None else None)
                else:
                    previous["private"] = ({
                        **_segment_note(private, private_live),
                        "recent_words": live_words, "source": private.thought_source,
                    } if private is not None else None)
            payload = {
                "previous": previous, "user": user_text, "delivered_reply": assistant_text,
                "reasoning": (reasoning or "").strip() or None, "lived_experiences": experiences,
            }
            persona = self._deps.current_persona()
            persona_name = persona.name.strip() or DEFAULT_PERSONA_NAME
            partner_name = persona.partner_name.strip()
            partner_role = persona.partner_role.strip() or DEFAULT_PARTNER_ROLE
            partner_pronoun = persona.partner_pronoun.strip() or "他"
            maintainer_owner = self._deps.current_maintainer()
            is_maintainer = bool(maintainer_owner) and owner == maintainer_owner
            prompt = _CRON_HEARTBEAT_PROMPT if heartbeat else _UPDATE_PROMPT
            if not heartbeat:
                if is_maintainer:
                    if partner_name:
                        prompt = prompt.replace(
                            "user 是对方本轮说的话", f"user 是{partner_name}本轮说的话", 1)
                elif partner_name:
                    prompt = prompt.replace(
                        "user 是对方本轮说的话",
                        f"user 是当前正在跟你说话的另一个人本轮说的话，不是{partner_name}说的", 1)
                else:
                    prompt = prompt.replace(
                        "user 是对方本轮说的话",
                        "user 是当前正在跟你说话的另一个人本轮说的话", 1)
                if not is_maintainer:
                    prompt += (
                        "\ntension 与 closeness 仍表示我和挚友的关系，"
                        "不因为眼前这位外人的互动另算一段关系；他真实的态度可以自然影响我此刻的心情。")
            prompt = prompt.replace("挚友", partner_role if is_maintainer else DEFAULT_PARTNER_ROLE)
            prompt = prompt.replace("{persona_name}", persona_name)
            if image_context:
                payload["user_image_context"] = image_context
                prompt += ("\n图片线索可帮助理解本轮：它是发图、已知梗含义或视觉观察，"
                           "不属于用户原话；心里话里可以留着我对他发图心意的揣测，但别把揣测当成他亲口说过的事实。")
            if impression_target is not None:
                imp_prompt = build_impression_prompt(
                    partner_name=partner_name if is_maintainer else "",
                    partner_role=partner_role if is_maintainer else "外部对话者",
                    partner_pronoun=partner_pronoun if is_maintainer else "对方",
                    is_maintainer=is_maintainer,
                )
                prompt += "\n\n" + imp_prompt
                payload["previous_impression"] = {
                    "text": impression.text, "speaker": impression.speaker,
                    "evidence": impression.evidence, "handwritten": not impression.speaker,
                } if impression else None
            if not heartbeat and not thought_locked and prev_vars["tension"] <= 0.2:
                distant = _sample_distant_memories(
                    target[0] / "sanae.db", owner_key=owner, profile_key=target[1],
                    user_text=user_text, assistant_text=assistant_text, turn_id=turn_id, now=now,
                )
                if distant:
                    payload["distant_memories"] = distant
                    spark_prompt = _SPARK_PROMPT.replace("{persona_name}", persona_name)
                    prompt += spark_prompt
            if thought_locked:
                prompt += ("\n本轮心里话已由她亲手记下，sentences 和 variables 均返回 null。"
                           "照常整理 recent_words 和可用的 impression。")
            else:
                prompt += (
                    "\n最终输出为一个 JSON 对象，各字段分别整理。sentences 用完整内心陈述保留自身感受与所牵挂的人事，"
                    "不重复事实整理、人物印象、解释或建议；没有有据的新变化就返回 null。"
                )
            raw = auxiliary.complete_simple(
                self._deps.current_model(),
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.0, max_tokens=768, timeout=DAILY_SEMANTIC_TIMEOUT, purpose="inner_state_update",
                failure_message="inner state Flash unavailable",
            )
            if raw is None:
                self.record_inner_state_failure("flash_unavailable", session_id=session_id, now=now, target=target)
                return False
            try:
                raw = _update_object(raw)
                thought, variables, recent, unspoken, spark, candidate = _decode_update(
                    raw, thought_locked=thought_locked, delivered_reply=assistant_text,
                    lived_experiences=experiences,
                    heartbeat_variables=prev_vars if heartbeat else None)
            except ValueError:
                self.record_inner_state_failure("invalid_flash_output", session_id=session_id, now=now, target=target)
                return False
            changed_impression = False
            if heartbeat:
                recent, unspoken, spark = "", "", ""
            if impression_target is not None:
                changed_impression = apply_impression(
                    impression_target, impression, candidate, turn_id=turn_id, now=now,
                    user=user_text, assistant=assistant_text,
                )
            if recent and recent in assistant_text:
                words, words_at = recent, now
            else:
                words, words_at = live_words, live_words_at
            refreshed = words_at != live_words_at
            if thought_locked:
                state = replace(private,
                                revision=(latest.revision if latest else private.revision) + 1,
                                recent_words=words, recent_words_at=words_at,
                                processed_at=private.processed_at if preserve_processed_at else processed_at,
                                source="flash", changed=words != live_words or refreshed)
                locked_appended = self._append_inner_state(target, state)
                if not internal:
                    self._record_memory_axis_flash(
                        appended=locked_appended, turn_id=turn_id, user_text=user_text,
                        assistant_text=assistant_text, previous_state=latest, current_state=state,
                        now=now, owner_key=owner,
                    )
                return locked_appended or changed_impression
            unspoken_note = _format_unspoken_note(
                unspoken, assistant_text, user_text=user_text, reasoning=reasoning
            )
            fresh = thought is not None and (private is None or thought != private_clean)
            cleared = bool(private and private_clean and not private_live)
            core_text = thought if fresh else (
                private_clean if (private_live and private is not None) else "")
            effective_vars = variables if variables is not None else prev_vars
            vars_changed = variables is not None and any(
                abs(effective_vars[key] - prev_vars[key]) > 1e-6 for key in _HALF_LIVES)
            spark_note = (
                _format_spark_note(spark, assistant_text)
                if (effective_vars["tension"] <= 0.2 and prev_vars["tension"] <= 0.2)
                else ""
            )
            notes = "".join(n for n in (unspoken_note, spark_note) if n)
            text = f"{core_text}{notes}" if core_text and notes else (core_text or notes)
            text_changed = (private is None and bool(text)) or (
                private is not None and text != private.text)
            state = InnerState(
                revision=(latest.revision if latest else 0) + 1, text=text, recent_words=words,
                recent_words_at=words_at, variables=effective_vars,
                felt_at=now if (fresh or private is None) else private.felt_at,
                updated_at=now,
                processed_at=(latest.processed_at if latest else None) if preserve_processed_at else processed_at,
                source="flash", turn_id=turn_id,
                thought_source="flash" if fresh else (private.thought_source if private else "flash"),
                changed=(fresh or cleared or text_changed or words != live_words
                         or refreshed or vars_changed),
                owner_key="" if internal else owner,
                text_kind="internal" if internal else "private",
                last_spark_at=max(
                    float(private.last_spark_at if private is not None else 0.0),
                    float(now) if spark_note and not internal else 0.0,
                ),
            )
            appended = self._append_inner_state(target, state)
            if not internal:
                self._record_memory_axis_flash(
                    appended=appended, turn_id=turn_id, user_text=user_text,
                    assistant_text=assistant_text, previous_state=latest, current_state=state,
                    now=now, owner_key=owner,
                )
            if appended:
                forbidden = _shared_forbidden_names(partner_name)
                if internal:
                    candidate, sources = thought or "", (experiences,)
                else:
                    candidate = raw.get("shared_sentences")
                    sources = (user_text, assistant_text, thought or "", private_clean)
                self._write_shared_sentence(
                    target, candidate, forbidden_names=forbidden, private_sources=sources,
                    variables=effective_vars, now=now, revision=state.revision)
            return appended or changed_impression
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: inner state update failed (%s)", type(exc).__name__)
            self.record_inner_state_failure("update_error", session_id=session_id, now=now, target=target)
            return False


def _main() -> None:
    from .daily_context import daily_profile_key
    from ..infra.textutil import hermes_home
    parser = argparse.ArgumentParser(description="Read Sanae's current inner state and recent revisions.")
    parser.add_argument("--home", type=Path, required=True, help="Hermes profile home")
    parser.add_argument("--db", type=Path, help="sanae.db (defaults to <home>/sanae-mem/sanae.db)")
    parser.add_argument("--recent", type=int, default=5, help="number of previous revisions to display")
    parser.add_argument("--owner", default="",
                        help="verified interlocutor owner_key; omit for the shared view only")
    args = parser.parse_args()
    home = Path(hermes_home(args.home))
    db = args.db.expanduser() if args.db else home / "sanae-mem" / "sanae.db"
    print(json.dumps(
        inspect_inner_state(db, daily_profile_key(home), recent=args.recent, owner_key=args.owner),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
