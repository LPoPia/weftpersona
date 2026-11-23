from __future__ import annotations

from contextlib import closing
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any

from ..infra.host_bridge import is_delegated_child_turn, query_verified_cron_sends
from ..governance.source_context import current_chat_scope
from ..governance.source_receipts import read_source_receipts


logger = logging.getLogger(__name__)
HEADER = "[我刚才在别处说过]"
_SEPARATOR = "｜"
_CONTINUATION = "> "


def _readable_owners(owner_key: Any, *, data_dir: Any = None, home: Any = None) -> tuple[str, ...]:
    if isinstance(owner_key, (list, tuple, set, frozenset)):
        return tuple(sorted({str(k).strip() for k in owner_key if str(k).strip()}))
    owner = str(owner_key or "").strip()
    if not owner:
        return ()
    from ..governance.account_binding import current_profile, read_owners
    from ..infra.host_bridge import get_hermes_home
    from ..infra.textutil import _load_plugin_config

    cfg = _load_plugin_config()
    resolved_home = Path(home) if home is not None else get_hermes_home()
    directory = Path(data_dir if data_dir is not None else (cfg.get("data_dir") or resolved_home / "sanae-mem"))
    path = directory.expanduser().resolve() / "sanae.db"
    profile = current_profile(resolved_home)
    if path.is_file():
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
                return read_owners(owner, config=cfg, connection=conn, profile=profile)
        except sqlite3.Error:
            return (owner,)
    cfg_scoped = {**cfg, "data_dir": str(directory)} if data_dir is not None else cfg
    return read_owners(owner, config=cfg_scoped, profile=profile)


def _owner_sql(column: str, owners: tuple[str, ...]) -> tuple[str, list[str]]:
    if len(owners) == 1:
        return f"{column}=?", [owners[0]]
    return f"{column} IN (%s)" % ",".join("?" for _ in owners), list(owners)


def _previous_reply_time(owner_key, session_id: str, platform: str = "", *,
                         home: Path | None = None, data_dir: Path | None = None) -> float | None:
    from ..infra.host_bridge import get_hermes_home
    from ..infra.textutil import _load_plugin_config

    home = Path(home) if home is not None else get_hermes_home()
    directory = Path(data_dir or _load_plugin_config().get("data_dir") or home / "sanae-mem")
    path = directory.expanduser().resolve() / "sanae.db"
    owners = _readable_owners(owner_key, data_dir=data_dir, home=home)
    if not owners or not session_id or not path.is_file():
        return None
    owner_clause, owner_params = _owner_sql("owner_key", owners)
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute(
                f"SELECT delivered_at FROM session_reply_watermarks "
                f"WHERE {owner_clause} AND session_id=? ORDER BY delivered_at DESC LIMIT 1",
                (*owner_params, session_id),
            ).fetchone()
            if row is None and platform:
                row = conn.execute(
                    f"SELECT delivered_at FROM session_reply_watermarks "
                    f"WHERE {owner_clause} AND platform=? AND session_id!=? ORDER BY delivered_at DESC LIMIT 1",
                    (*owner_params, platform, session_id),
                ).fetchone()
            return float(row[0]) if row is not None else None
    except sqlite3.OperationalError:
        return None


def _private_receipt_route(owner_key: str, session_id: str, home=None):
    from ..infra.host_bridge import get_hermes_home
    from ..governance.source_context import resolve_chat_identity
    from ..infra.textutil import _load_plugin_config

    scope = current_chat_scope()
    if (not owner_key or not session_id or scope.session_id != session_id
            or scope.visibility != "private" or is_delegated_child_turn()):
        return None
    active_home = get_hermes_home().resolve()
    if home is not None and Path(home).resolve() != active_home:
        return None
    config = _load_plugin_config()
    identity = resolve_chat_identity(scope, config)
    if identity.private_owner_key != owner_key:
        if not (
            scope.owner == owner_key
            and not config.get("memory_identity")
            and not config.get("maintainer_account_bindings")
        ):
            return None
    return active_home, scope.platform, scope.actor, scope.chat_id, scope.thread_id


def recent_self_utterances(
    owner_key: str, session_id: str, *, home=None, data_dir=None, since=None,
    unconsumed_only: bool = False, read_owners=(),
) -> str:
    """Read receipts only in the current bound, private main turn; never append a chat row."""
    route = _private_receipt_route(owner_key, session_id, home)
    if route is None:
        return ""
    owners = _readable_owners(read_owners or owner_key, data_dir=data_dir, home=route[0])
    clock = datetime.now().astimezone()
    window_floor = clock.timestamp() - 12 * 60 * 60
    if since is None and unconsumed_only:
        try:
            since = _previous_reply_time(owners, session_id, route[1], home=route[0], data_dir=data_dir)
        except Exception as exc:
            logger.warning("Could not read previous reply time (%s)", type(exc).__name__)
    lower = window_floor if since is None else max(window_floor, since)
    rows = _self_utterance_rows(
        owners, route, data_dir=data_dir, since=lower, through=clock.timestamp(),
    )
    if not rows:
        return ""
    return "\n".join([HEADER, *_format_experience_rows(rows, clock.tzinfo)])


def _confirmed_cron_sent_rows(owner_key, route, *, since, through):
    """Read only transport-confirmed sends for the exact private route."""
    _, platform, _, chat_id, thread_id = route
    owners = owner_key if isinstance(owner_key, (list, tuple, set, frozenset)) else (owner_key,)
    out = []
    for owner in dict.fromkeys(str(k) for k in owners if k):
        out.extend(query_verified_cron_sends(
            bound_owner=owner, platform=platform, chat_id=chat_id, thread_id=thread_id,
            since_ts=since, until_ts=through,
        ))
    return out


def _self_utterance_rows(owner_key, route, *, data_dir, since, through, strict=False, read_owners=()):
    owners = _readable_owners(read_owners or owner_key, data_dir=data_dir, home=route[0] if route else None)
    rows = []
    try:
        rows.extend((stamp, "定时主动发言", text) for stamp, text in _confirmed_cron_sent_rows(
            owners, route, since=since, through=through))
    except Exception as exc:
        if strict:
            raise
        logger.warning("Could not read cron send receipts (%s)", type(exc).__name__)
    try:
        receipts = read_source_receipts(data_dir, owners, since, through)
        rows.extend((item.delivered_at, item.label, item.text) for item in receipts)
    except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
        if strict:
            raise
        logger.warning("Could not read external source send receipts (%s)", type(exc).__name__)
    return [(stamp, label, text) for stamp, label, text in rows if since < stamp <= through]


def _format_experience_rows(rows, tzinfo):
    for stamp, label, text in sorted(rows):
        timestamp = datetime.fromtimestamp(stamp, tz=tzinfo).strftime("%Y-%m-%d %H:%M")
        first, *tail = text.split("\n")
        yield _SEPARATOR.join((timestamp, label, first))
        yield from (_CONTINUATION + line for line in tail)
