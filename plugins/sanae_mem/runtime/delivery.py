from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from sanae_mem.governance.runtime_registry import owns_delivery, register_delivery_owner

from . import group_ingest
from ..cognitive.daily_context import daily_profile_key
from .delivery_receipts import clean_receipt_text, runtime_footer_pattern
from .group_context import group_feature_enabled
from ..infra.atomic_io import atomic_json_write, file_lock
from ..infra.clock import shanghai_datetime
from ..infra.host_bridge import (
    delivery_receipt_fields, get_hermes_home, hermes_home_scope, is_chat_delivery_receipt,
    is_test_isolated_turn, receipt_plugin_user_context, receipt_transcript, turn_envelope_scope,
)
from ..infra.status_store import _STATUS_WRITER, _component_error_code, read_status_file
from ..cognitive.inner_wiring import _submit
from .private_delivery_settler import (
    VerifiedPrivateDelivery, prune_turn_user_context, read_turn_user_context, settle_private_delivery,
)
from .provider import SanaeMemProvider
from ..governance.source_context import ChatScope, PUBLIC_CHAT_TYPES, current_chat_scope
from ..infra.textutil import _bool

logger = logging.getLogger(__name__)


def _record_group_delivery_status(data_dir: Path, *, ok: bool, reason: str = "") -> None:
    """Persist group delivery outcome to *data_dir*/status.json without chat text."""
    path = Path(data_dir).expanduser().resolve() / "status.json"
    recorded_at = shanghai_datetime(None).isoformat(timespec="seconds")
    error_code = _component_error_code(reason) if not ok else ""
    try:
        with _STATUS_WRITER, file_lock(path.with_suffix(".lock")):
            latest = read_status_file(path)
            raw_components = latest.get("components")
            components: dict[str, object] = (
                dict(raw_components) if isinstance(raw_components, dict) else {}
            )
            previous = components.get("group_delivery")
            if ok:
                if not (isinstance(previous, dict) and previous.get("ok") is False):
                    return
                components["group_delivery"] = {
                    "ok": True,
                    "stage": "apply_group_delivery",
                    "at": recorded_at,
                    "failure_count": 0,
                }
            else:
                prev_count = previous.get("failure_count") if isinstance(previous, dict) else 0
                components["group_delivery"] = {
                    "ok": False,
                    "stage": "apply_group_delivery",
                    "error": error_code,
                    "at": recorded_at,
                    "failure_count": (prev_count if type(prev_count) is int else 0) + 1,
                }
            latest["components"] = components
            atomic_json_write(path, latest)
    except OSError as status_exc:
        logger.warning(
            "sanae_mem: group delivery status write failed (%s)",
            type(status_exc).__name__,
        )


def _open_private_hook_provider(prototype: SanaeMemProvider, scope: ChatScope, home: Path) -> SanaeMemProvider:
    return prototype.open_runtime_worker(home, prototype.runtime_config(home),
                                         session_id=scope.session_id, platform=scope.platform)


class DeliveryHistory:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recent_turn_ids: dict[tuple[str, str, str], deque[tuple[str, float]]] = {}
        self.recent_inbound_turn_ids: dict[tuple[str, str, str, str], deque[tuple[str, float]]] = {}
        self.applied_deliveries: dict[tuple[str, ...], float] = {}

    def _prune_maps(self, now_ts: float) -> None:
        cutoff = now_ts - 300.0
        for key, entries in list(self.recent_turn_ids.items()):
            kept = deque(item for item in entries if item[1] >= cutoff)
            if kept:
                self.recent_turn_ids[key] = kept
            else:
                self.recent_turn_ids.pop(key, None)
        for key, entries in list(self.recent_inbound_turn_ids.items()):
            kept = deque(item for item in entries if item[1] >= cutoff)
            if kept:
                self.recent_inbound_turn_ids[key] = kept
            else:
                self.recent_inbound_turn_ids.pop(key, None)
        for key, ts_val in list(self.applied_deliveries.items()):
            if ts_val < cutoff:
                self.applied_deliveries.pop(key, None)
        prune_turn_user_context(now_ts)

    @staticmethod
    def _push_turn_id(
        entries: deque[tuple[str, float]] | None,
        target_tid: str,
        observed_at: float,
    ) -> deque[tuple[str, float]]:
        queue = (
            deque(item for item in entries if item[0] != target_tid)
            if entries is not None
            else deque()
        )
        queue.append((target_tid, observed_at))
        return queue

    def _discard_turn_id(self, reply_key: tuple[str, str, str], target_tid: str) -> None:
        if not target_tid:
            return
        entries = self.recent_turn_ids.get(reply_key)
        if entries is not None:
            kept = deque(item for item in entries if item[0] != target_tid)
            if kept:
                self.recent_turn_ids[reply_key] = kept
            else:
                self.recent_turn_ids.pop(reply_key, None)
        home_key, sid, reply_text = reply_key
        for in_key, in_entries in list(self.recent_inbound_turn_ids.items()):
            if in_key[0] == home_key and in_key[1] == sid and in_key[2] == reply_text:
                kept_in = deque(item for item in in_entries if item[0] != target_tid)
                if kept_in:
                    self.recent_inbound_turn_ids[in_key] = kept_in
                else:
                    self.recent_inbound_turn_ids.pop(in_key, None)

    def observe(self, home: Path, session_id: str, turn_id: str, message_id: str, assistant_response: str) -> None:
        final_text = assistant_response.strip()
        observed_at = time.time()
        home_key = str(Path(home).resolve())
        sid = str(session_id)
        inbound_id = str(message_id).strip()
        reply_key = (home_key, sid, final_text)
        inbound_key = (home_key, sid, final_text, inbound_id)
        tid = str(turn_id)
        with self.lock:
            self._prune_maps(observed_at)
            self.recent_turn_ids[reply_key] = self._push_turn_id(
                self.recent_turn_ids.get(reply_key), tid, observed_at,
            )
            if inbound_id:
                self.recent_inbound_turn_ids[inbound_key] = self._push_turn_id(
                    self.recent_inbound_turn_ids.get(inbound_key), tid, observed_at,
                )


class DeliveryRouter:
    def __init__(self, home: Path, provider: SanaeMemProvider, *, discovery: bool) -> None:
        self.home = home
        self.provider = provider
        self.discovery = discovery
        self.history = DeliveryHistory()
        self._attached = False

    def attach(self) -> Callable[[], None]:
        self.history, release = register_delivery_owner(self.home, self, self.discovery, self.history)
        self._attached = True
        return release

    def owns_delivery(self) -> bool:
        return self._attached and owns_delivery(self.home, self)

    def observe(self, session_id: str, turn_id: str, message_id: str, assistant_response: str) -> None:
        if self.owns_delivery():
            self.history.observe(self.home, session_id, turn_id, message_id, assistant_response)

    def on_transport_receipt(self, receipt_obj: object) -> None:
        # The host pins the dispatching turn's runtime HOME around this synchronous emit, so the
        # receipt is read in the HOME it was actually sent from. Only the provider registered in
        # that same HOME owns it: a provider loaded for another profile would otherwise apply the
        # same delivery to its own data_dir. Checked before any dedup claim, config or text read.
        home = get_hermes_home().resolve()
        if home != self.home:
            return
        if not self.owns_delivery():
            return
        if not is_chat_delivery_receipt(receipt_obj):
            return
        prototype = self.provider
        live_cfg = prototype.runtime_config(home)
        if _bool(live_cfg.get("shadow"), False):
            return
        receipt = delivery_receipt_fields(receipt_obj)
        session_id = str(receipt.session_id or "").strip()
        platform = str(receipt.platform or "").strip().lower()
        chat_type_raw = str(receipt.chat_type or "").strip().lower()
        chat_id = str(receipt.chat_id or "").strip()
        sender_id = str(receipt.sender_id or "").strip()
        sender_alt = str(receipt.sender_id_alt or "").strip()
        scope_id = str(receipt.scope_id or "").strip()
        receipt_session_key = str(receipt.session_key or "").strip()
        thread_id = str(receipt.thread_id or "").strip()
        # The sending turn's runtime profile, named by the host. An older host sends none: the
        # worker envelope then stays unnamed rather than guessing one from the key or inbound context.
        profile = str(receipt.profile or "").strip()
        delivered_text = str(receipt.delivered_text or "").strip()
        receipt_clean_text = str(receipt.clean_text or "").strip()
        user_msg = str(receipt.user_message or "").strip()
        delivered_at = float(receipt.delivered_at or time.time())
        private_chat = chat_type_raw in {"dm", "private", "direct", "c2c"}
        group_chat = chat_type_raw in PUBLIC_CHAT_TYPES
        if (not session_id or not platform or not chat_id or not (sender_id or sender_alt)
                or not (private_chat or group_chat)
                or not (receipt_clean_text or delivered_text) or not user_msg):
            return
        # Slack needs the workspace, and for a private chat the dispatch-time key the sender
        # actually used: the same sender ID exists in every workspace. A group receipt instead
        # needs only its workspace; its scene facts are the ones below.
        if platform == "slack" and (not scope_id or (private_chat and not receipt_session_key)):
            return

        footer_pat = None
        if not receipt_clean_text:
            footer_pat = runtime_footer_pattern(
                home, platform, str(receipt.model or "")
            )
        clean_reply = clean_receipt_text(delivered_text, receipt_clean_text, footer_pat)

        incoming_id_early = str(receipt.incoming_message_id or "").strip()
        if group_chat and not incoming_id_early:
            return

        explicit_turn_id = str(receipt.turn_id or "").strip()
        outbound_msg_id = str(receipt.message_id or "").strip()
        home_key = str(Path(home).resolve())
        reply_key = (home_key, session_id, clean_reply)
        turn_key_explicit = (
            (home_key, session_id, "turn", explicit_turn_id)
            if explicit_turn_id
            else None
        )
        msg_dedup_key = (
            (home_key, session_id, "msg", outbound_msg_id)
            if outbound_msg_id
            else None
        )
        in_reply_dedup_key = (
            (home_key, session_id, "in_reply", incoming_id_early, clean_reply)
            if incoming_id_early
            else None
        )
        lookup_ts = time.time()
        with self.history.lock:
            self.history._prune_maps(lookup_ts)
            if turn_key_explicit is not None and turn_key_explicit in self.history.applied_deliveries:
                self.history._discard_turn_id(reply_key, explicit_turn_id)
                return
            if msg_dedup_key is not None and msg_dedup_key in self.history.applied_deliveries:
                return
            if (
                not explicit_turn_id
                and not outbound_msg_id
                and in_reply_dedup_key is not None
                and in_reply_dedup_key in self.history.applied_deliveries
            ):
                return
            matched_turn_id = ""
            if explicit_turn_id:
                self.history._discard_turn_id(reply_key, explicit_turn_id)
            elif incoming_id_early:
                inbound_key = (home_key, session_id, clean_reply, incoming_id_early)
                in_entries = self.history.recent_inbound_turn_ids.get(inbound_key)
                if in_entries:
                    matched_turn_id = in_entries.popleft()[0]
                    if not in_entries:
                        self.history.recent_inbound_turn_ids.pop(inbound_key, None)
                    self.history._discard_turn_id(reply_key, matched_turn_id)
            else:
                entries = self.history.recent_turn_ids.get(reply_key)
                if entries:
                    matched_turn_id = entries.popleft()[0]
                    if not entries:
                        self.history.recent_turn_ids.pop(reply_key, None)
                    self.history._discard_turn_id(reply_key, matched_turn_id)
            resolved_turn_id = (
                explicit_turn_id
                or matched_turn_id
                or f"receipt:{outbound_msg_id or incoming_id_early or int(delivered_at * 1000)}"
            )
            turn_dedup_key = (home_key, session_id, "turn", resolved_turn_id)
            if turn_dedup_key in self.history.applied_deliveries:
                return
            self.history.applied_deliveries[turn_dedup_key] = lookup_ts
            if msg_dedup_key is not None:
                self.history.applied_deliveries[msg_dedup_key] = lookup_ts
            if in_reply_dedup_key is not None:
                self.history.applied_deliveries[in_reply_dedup_key] = lookup_ts

        data_dir = Path(live_cfg.get("data_dir") or Path(home) / "sanae-mem").expanduser().resolve()
        profile_key = daily_profile_key(home)
        if group_chat:
            # A group row must carry the inbound message it answers; this send's own id is not a
            # substitute for it. Every other scene fact comes straight from this receipt.
            incoming_id = str(receipt.incoming_message_id or "").strip()
            parent_chat_id = str(receipt.parent_chat_id or "").strip()
            thread_visibility = str(
                receipt.thread_visibility or "").strip().lower()
            occurred_at = str(receipt.occurred_at or "").strip()
            reply_to = str(receipt.reply_to_message_id or "").strip()
            is_bot = receipt.is_bot
            inbound_coverage = str(receipt.inbound_coverage or "").strip()
            synthetic_reply_thread = receipt.synthetic_reply_thread is True
            delivery_ref = str(receipt.message_id or "").strip()
            transcript = receipt_transcript(receipt_obj)

            def run_group_update() -> None:
                # Rebuilt from this receipt alone: the inbound turn's profile/scene has already
                # been cleared, and the group worker must never inherit either.
                with hermes_home_scope(home), turn_envelope_scope(
                    inherit_context=False,
                    platform=platform,
                    chat_type=chat_type_raw,
                    chat_id=chat_id,
                    sender_id=sender_id or sender_alt,
                    sender_id_alt=sender_alt,
                    scope_id=scope_id,
                    session_id=session_id,
                    session_key=receipt_session_key,
                    message_id=incoming_id,
                    thread_id=thread_id,
                    parent_chat_id=parent_chat_id,
                    thread_visibility=thread_visibility,
                    occurred_at=occurred_at,
                    reply_to_message_id=reply_to,
                    is_bot=is_bot,
                    inbound_coverage=inbound_coverage,
                    synthetic_reply_thread=synthetic_reply_thread,
                    profile=profile,
                ):
                    scope = current_chat_scope()
                    if scope.visibility != "public":
                        return
                    config = prototype.runtime_config(home)
                    if _bool(config.get("shadow"), False) or not group_feature_enabled(
                            config, scope, "long_term_memory"):
                        return
                    worker = _open_private_hook_provider(prototype, scope, home)
                    try:
                        if not worker.permits_group_delivery():
                            return
                        group_result = group_ingest.apply_group_delivery(
                            worker,
                            user_message=user_msg,
                            assistant_response=clean_reply,
                            turn_id=resolved_turn_id,
                            delivered_at=delivered_at,
                            delivery_ref=delivery_ref,
                            transcript=transcript,
                            user_message_authored=(
                                receipt.user_message_authored is True
                            ),
                        )
                    finally:
                        worker.shutdown()
                    if not isinstance(group_result, dict) or not group_result.get("ok"):
                        reason = (
                            str(group_result.get("reason") or "group_delivery_failed")
                            if isinstance(group_result, dict)
                            else "group_delivery_failed"
                        )
                        logger.warning(
                            "sanae_mem: group delivery failed (turn=%s, reason=%s)",
                            resolved_turn_id,
                            reason,
                        )
                        if not is_test_isolated_turn(session_id):
                            _record_group_delivery_status(data_dir, ok=False, reason=reason)
                    elif not is_test_isolated_turn(session_id):
                        _record_group_delivery_status(data_dir, ok=True)

            _submit(prototype.background_tasks, (data_dir, profile_key, "group"), run_group_update)
            return

        norm_chat_type = "dm" if chat_type_raw == "c2c" else chat_type_raw
        # The receipt's dispatch-time key wins. Older hosts send none; other platforms keep the
        # legacy synthesized key there, Slack was refused above.
        session_key = receipt_session_key or f"agent:main:{platform}:{norm_chat_type}:{chat_id}"
        outbound_msg_id = str(receipt.message_id or "").strip()
        incoming_id = str(receipt.incoming_message_id or "").strip()
        envelope_msg_id = incoming_id or outbound_msg_id or "transport-ack"
        occurred_at = str(receipt.occurred_at or "").strip()
        reply_to = str(receipt.reply_to_message_id or "").strip()
        is_bot = receipt.is_bot
        has_native_image = receipt.has_native_image is True
        transcript = receipt_transcript(receipt_obj)
        user_message_authored = receipt.user_message_authored is True
        plugin_user_context = receipt_plugin_user_context(receipt_obj)
        if not plugin_user_context:
            plugin_user_context = read_turn_user_context(home, session_id, resolved_turn_id)
        delivery = VerifiedPrivateDelivery(
            session_id=session_id,
            turn_id=resolved_turn_id,
            platform=platform,
            chat_id=chat_id,
            chat_type=norm_chat_type,
            sender_id=sender_id or sender_alt,
            sender_id_alt=sender_alt,
            user_message=user_msg,
            assistant_response=clean_reply,
            delivered_at=delivered_at,
            message_id=outbound_msg_id,
            incoming_message_id=incoming_id,
            has_native_image=has_native_image,
            transcript=transcript,
            user_message_authored=user_message_authored,
            profile=profile,
            scope_id=scope_id,
            session_key=session_key,
            thread_id=thread_id,
            occurred_at=occurred_at,
            reply_to_message_id=reply_to,
            is_bot=is_bot,
            plugin_user_context=plugin_user_context,
        )

        def run_verified_update() -> None:
            # The worker runs off-turn: rebuild the whole envelope from this receipt only, so the
            # profile, scope and scene can never be inherited from whatever inbound context is left.
            with hermes_home_scope(home), turn_envelope_scope(
                inherit_context=False,
                platform=platform,
                chat_type=norm_chat_type,
                chat_id=chat_id,
                sender_id=sender_id or sender_alt,
                sender_id_alt=sender_alt,
                scope_id=scope_id,
                session_id=session_id,
                session_key=session_key,
                message_id=envelope_msg_id,
                thread_id=thread_id,
                profile=profile,
                occurred_at=occurred_at,
                reply_to_message_id=reply_to,
                is_bot=is_bot,
                write_origin="foreground",
                is_messaging_surface=True,
            ):
                scope = current_chat_scope()
                if not scope.owner:
                    return
                if _bool(prototype.runtime_config(home).get("shadow"), False):
                    return
                worker = _open_private_hook_provider(prototype, scope, home)
                try:
                    settle_private_delivery(worker, delivery)
                finally:
                    worker.shutdown()

        _submit(prototype.background_tasks, (data_dir, profile_key, "private"), run_verified_update)
