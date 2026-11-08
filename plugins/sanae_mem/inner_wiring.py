from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from agent.memory_provider import ctx_bound, spawn_context_thread

try:
    from .daily_inputs import delivered_user_input, flatten_user_text
    from .impression import read_impression, render_impression, write_impression
    from .inner_state import INNER_STATE_MARKER, _SPARK_NOTE_RE
    from .self_utterances import _previous_reply_time
except ImportError:
    from daily_inputs import delivered_user_input, flatten_user_text
    from impression import read_impression, render_impression, write_impression
    from inner_state import INNER_STATE_MARKER, _SPARK_NOTE_RE
    from self_utterances import _previous_reply_time

logger = logging.getLogger(__name__)
_ACTIVITY_LEAD = (
    "（以下是我自己的空闲经历；顺口提起时只先丢半句引子或一句简单印象，留白看他接不接茬；"
    "哪怕正聊在共同兴头上，也只用一两句日常大白话带过核心意思，绝不照抄下面的书面术语或分步说明）\n我这阵子刚做完的事："
)
_WAIT_SECONDS = 3.0
_Target = tuple[Path, str]

INNER_STATE_SCHEMA = {
    "name": "sanae_inner_state",
    "description": (
        "读写我此刻的心里话。read 是看看自己现在心里怎么想；"
        "write 是我想亲手记下自己的当下感受，用第一人称写两到四句话。"
        "心里话是给我自己的，不必告诉对方。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write"]},
            "text": {"type": "string", "description": "write 时的心里话"},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


@dataclass
class _Updates:
    queue: deque[tuple[Callable[[], None], threading.Event]]
    latest: threading.Event


_updates_lock = threading.Lock()
_updates: dict[_Target, _Updates] = {}


def _drain(target: _Target, updates: _Updates) -> None:
    while True:
        with _updates_lock:
            if not updates.queue:
                del _updates[target]
                return
            job, finished = updates.queue.popleft()
        try:
            job()
        except Exception as exc:
            logger.warning("sanae_mem: inner state worker failed (%s)", type(exc).__name__)
        finally:
            finished.set()


def _submit(target: _Target, job: Callable[[], None]) -> None:
    job = ctx_bound(job)
    finished = threading.Event()
    with _updates_lock:
        updates = _updates.get(target)
        if updates is not None:
            updates.latest = finished
            updates.queue.append((job, finished))
            return
        updates = _updates[target] = _Updates(deque([(job, finished)]), finished)
        try:
            spawn_context_thread(
                _drain, name="sanae-inner-state", args=(target, updates),
            ).start()
        except Exception:
            del _updates[target]
            raise


def _wait_for_pending(target: _Target) -> None:
    with _updates_lock:
        updates = _updates.get(target)
        latest = updates.latest if updates else None
    if latest is not None:
        latest.wait(_WAIT_SECONDS)


class InnerWiringMixin:
    def _cron_inner_scope(self, *, session_id="", synthetic=False, **_):
        if self._shadow or synthetic or str(self._agent_context) != "cron":
            return None
        from gateway.session_cron import current_execution
        try:
            from .life_context import _trusted_cron_home
        except ImportError:
            from life_context import _trusted_cron_home
        binding = current_execution()
        scope = self._daily_scope()
        session = str(session_id or (binding[1] if binding else "") or self._session_id or "")
        if scope is None or _trusted_cron_home(self._hermes_home, scope[0], session) is None:
            return None
        return scope

    def _inner_turn_scope(
        self, *, session_id="", turn_id="", platform="", chat_type="",
        sender_id="", sender_id_alt="", synthetic=False, require_current=True, **_,
    ):
        if (self._shadow or synthetic or chat_type not in {"dm", "private", "direct"}
                or not str(turn_id or "").strip()):
            return None
        for sender in (sender_id, sender_id_alt):
            scope = self._daily_turn_scope(
                session_id, platform, sender, require_current=require_current,
            )
            if scope is not None and self._daily_source_allows(sender):
                return scope
        return None

    def _turn_user_text(self, turn: dict) -> str:
        message = turn.get("user_message")
        if message is None:
            message = turn.get("user_text")
        text = flatten_user_text(message) if message is not None else ""
        if str(text or "").strip():
            return str(text)
        session = str(turn.get("session_id") or "")
        current = getattr(self, "_current_turn_input", None)
        if (isinstance(current, tuple) and len(current) == 3
                and str(current[0]) == session and str(current[2] or "").strip()):
            return str(current[2] or "")
        request = getattr(self, "_association_hook_request", None)
        if isinstance(request, dict) and str(request.get("session_id") or "") == session:
            return str(request.get("query") or "")
        return ""

    def current_turn_context(self, **turn) -> str:
        scope = self._inner_turn_scope(**turn) or self._cron_inner_scope(**turn)
        if scope is None:
            return ""
        target = self.inner_state_target()
        _wait_for_pending(target)
        scope = self._inner_turn_scope(**turn) or self._cron_inner_scope(**turn)
        if scope is None:
            return ""
        session = str(turn.get("session_id") or "")
        cron = str(self._agent_context) == "cron"
        state = self.read_inner_state(target=target)
        block = self.render_inner_state(
            state, target=target, session_id=session, consume_spark=not cron,
        ) if state is not None else ""
        facts = self._recent_activities(scope[0], state, allow_trusted_cron=cron, **turn)
        world = []
        if facts:
            world.append("\n".join((_ACTIVITY_LEAD, facts)))
        if not cron and not turn.get("durable_native_thread", False):
            imp = read_impression((target[0], scope[1], scope[0]))
            rendered = render_impression(imp.text if imp else "")
            if rendered:
                world.append(rendered)
        if not world and not block:
            return ""
        outer = "[我眼中的世界]\n" + "\n\n".join(world) if world else ""
        inner = "[我内心的感受]\n" + (block or INNER_STATE_MARKER)
        return "\n\n".join(part for part in (outer, inner) if part)

    def _recent_activities(self, owner, state, *, session_id="", platform="",
                           allow_trusted_cron=False, **_) -> str:
        """What she finished on her own since the last delivered reply, as plain facts.

        The floor takes the later of the flash watermark and the previous delivered reply, so a
        failed flash update or a handwritten state cannot replay an activity after it was read.
        """
        try:
            from .life_context import collect_recent_activities
        except ImportError:
            from life_context import collect_recent_activities

        home = Path(self._hermes_home)
        try:
            replied_at = _previous_reply_time(home, session_id, platform)
        except Exception as exc:
            logger.warning("sanae_mem: previous reply time unavailable (%s)", type(exc).__name__)
            replied_at = None
        floors = [stamp for stamp in (state.processed_at if state else None, replied_at) if stamp is not None]
        try:
            facts = collect_recent_activities(
                home=home, data_dir=self._data_dir, owner_key=owner,
                session_id=session_id, since=max(floors) if floors else None, through=time.time(),
                allow_trusted_cron=allow_trusted_cron,
            )
        except Exception as exc:
            logger.warning("sanae_mem: recent activity read failed (%s)", type(exc).__name__)
            return ""
        if facts:
            try:
                self.resolve_explored_curiosity_threads(facts)
            except Exception as exc:
                logger.warning("sanae_mem: curiosity thread resolve failed (%s)", type(exc).__name__)
        return facts

    def session_system_context(self, *, session_started_at, durable_native_thread=False, **turn) -> str:
        scope = self._inner_turn_scope(**turn)
        if scope is None:
            return ""
        if durable_native_thread:
            # The native thread persists and replays its own developer instructions.
            if not getattr(self, "_impression_native_warned", False):
                logger.warning("sanae_mem: private impression omitted on codex_app_server durable thread")
                self._impression_native_warned = True
            return ""
        return ""

    def _tool_impression(self, args, **turn) -> dict:
        scope = self._inner_turn_scope(**turn)
        if scope is None:
            raise PermissionError("对他的印象只在本人的私聊中使用。")
        owner, profile = scope
        target = (self.inner_state_target()[0], profile, owner)
        operation = args.get("operation")
        if operation == "read":
            state = read_impression(target)
            return {"text": state.text if state is not None else ""}
        if operation == "write":
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请写下我对他的印象。")
            state = write_impression(target, text.strip(), turn_id=turn["turn_id"])
            return {"text": state.text, "note": "我已记下新版，下一轮就能带上。"}
        raise ValueError("请选择 read 或 write。")

    def _tool_inner_state(self, args, **turn) -> dict:
        scope = self._inner_turn_scope(**turn) or self._cron_inner_scope(**turn)
        if scope is None:
            raise PermissionError("心里话只在本人的私聊中使用。")
        operation = args.get("operation")
        if operation == "read":
            session = str(turn.get("session_id") or "")
            return {"text": self.render_inner_state(
                target=self.inner_state_target(), session_id=session,
                consume_spark=True,
            )}
        if operation == "write":
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请写下我此刻的心里话。")
            turn_id = str(turn.get("turn_id") or "")
            if str(self._agent_context) == "cron":
                from gateway.session_cron import current_execution
                _, _, job_id, execution_id = current_execution()
                turn_id = f"cron:{job_id}:{execution_id}"
            state = self.write_inner_state(text, turn_id=turn_id)
            return {"text": state.text}
        raise ValueError("请选择 read 或 write。")

    def _on_cron_heartbeat(self, *, session_id="", assistant_content="", synthetic=False) -> None:
        scope = self._cron_inner_scope(session_id=session_id, synthetic=synthetic)
        if scope is None or self._llm is None:
            return
        from gateway.session_cron import current_execution
        from gateway.response_filters import is_autonomous_silence_response
        try:
            from .life_context import collect_life_experiences
        except ImportError:
            from life_context import collect_life_experiences
        _, bound_session, job_id, execution_id = current_execution()
        session = str(session_id or bound_session)
        target, now = self.inner_state_target(), time.time()
        turn_id = f"cron:{job_id}:{execution_id}"

        def update():
            if self._cron_inner_scope(session_id=session) != scope:
                return
            base = self.read_inner_state(target=target)
            if base and (base.updated_at > now or (base.turn_id == turn_id and base.source != "manual")):
                return
            experiences = collect_life_experiences(
                home=self._hermes_home, data_dir=target[0], owner_key=scope[0], session_id=session,
                since=base.updated_at if base else None, through=now, allow_trusted_cron=True,
            )
            output = str(assistant_content or "").strip()
            if output and output != "HEARTBEAT_OK" and not is_autonomous_silence_response(output):
                experiences = "\n".join(part for part in (experiences, output) if part)
            if experiences:
                self.update_inner_state(
                    turn_id=turn_id, user_text="", assistant_text="", experiences=experiences,
                    session_id=session, now=now, target=target, preserve_processed_at=True,
                )

        _submit(target, update)

    def on_delivered_turn(
        self, *, session_id, turn_id, platform, sender_id, sender_id_alt, chat_type,
        session_key, user_message, assistant_response, reasoning, delivered_at, has_native_image=False,
    ) -> None:
        scope = self._inner_turn_scope(
            session_id=session_id, turn_id=turn_id, platform=platform, chat_type=chat_type,
            sender_id=sender_id, sender_id_alt=sender_id_alt,
            require_current=False,
        )
        if (scope is None
                or not isinstance(assistant_response, str) or not assistant_response.strip()):
            return
        target = self.inner_state_target()
        home = self._hermes_home

        def fail(reason: str) -> None:
            self.record_inner_state_failure(reason, session_id=session_id, target=target)

        if not isinstance(delivered_at, (int, float)) or not math.isfinite(delivered_at):
            fail("delivery_time_invalid")
            return
        user_text, image_context = delivered_user_input(user_message, has_native_image=has_native_image)

        def update() -> None:
            try:
                try:
                    from .life_context import collect_life_experiences
                except ImportError:
                    from life_context import collect_life_experiences

                previous = self.read_inner_state(target=target)
                experiences = collect_life_experiences(
                    home=home, data_dir=target[0], owner_key=scope[0], session_id=session_id,
                    since=previous.processed_at if previous is not None else None,
                    through=delivered_at,
                )
            except Exception:
                fail("life_context_failed")
                return
            if experiences:
                try:
                    self.resolve_explored_curiosity_threads(experiences)
                except Exception as exc:
                    logger.warning(
                        "sanae_mem: curiosity thread resolve failed (%s)", type(exc).__name__)
            try:
                self.update_inner_state(
                    turn_id=turn_id, user_text=user_text, assistant_text=assistant_response,
                    reasoning=reasoning or "", experiences=experiences, image_context=image_context,
                    session_id=session_id, now=delivered_at, target=target,
                    impression_target=(target[0], scope[1], scope[0]),
                )
            except Exception:
                fail("update_failed")

        try:
            _submit(target, update)
        except Exception:
            fail("worker_start_failed")
