"""Clean bridge between ViveMind (``sanae_mem``) and ``hermes_host_contracts``.

Uses only public host extension contracts when available and degrades cleanly on
unmodified upstream Hermes checkouts without importing custom/private underscore
internals.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, fields as dataclass_fields
import logging
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, TypeVar, runtime_checkable

from .contracts import ViveMindError

logger = logging.getLogger(__name__)

try:
    import hermes_host_contracts as _hc
except ImportError:
    _hc = None  # type: ignore[assignment]

try:
    import hermes_test_isolation as _hti
except ImportError:
    _hti = None  # type: ignore[assignment]

try:
    from agent.memory_provider import MemoryProvider, RecallStatus, ctx_bound, spawn_context_thread
except ImportError:
    class MemoryProvider:
        """Standalone base for the provider's own lifecycle implementation."""

    @dataclass(frozen=True)
    class RecallStatus:
        provider_label: str
        count: int
        glyph: str = "🧠"

    def ctx_bound(fn):
        context = copy_context()
        return lambda *args, **kwargs: context.run(fn, *args, **kwargs)

    def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
        return threading.Thread(target=ctx_bound(target), name=name, daemon=daemon,
                                args=args, kwargs=kwargs)


class _StandaloneTurnEnvelope(SimpleNamespace):
    is_messaging_surface: bool = False


_turn_envelope: ContextVar[_StandaloneTurnEnvelope | None] = ContextVar(
    "vivemind_turn_envelope", default=None,
)
_hermes_home_override: ContextVar[Path | None] = ContextVar("vivemind_hermes_home_override", default=None)


class HostConfigParseError(ViveMindError, ValueError):
    """Raised when ``config.yaml`` cannot be parsed by any available YAML/JSON loader."""


def _host_contract_fn(name: str) -> Callable[..., Any] | None:
    """Return a public callable from ``hermes_host_contracts``, or ``None`` if absent."""
    if _hc is None:
        return None
    fn = getattr(_hc, name, None)
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(f"hermes_host_contracts.{name} is not callable ({type(fn).__name__})")
    return fn


def _test_isolation_fn(name: str) -> Callable[..., Any] | None:
    """Return a public callable from ``hermes_test_isolation``, or ``None`` if absent."""
    if _hti is None:
        return None
    fn = getattr(_hti, name, None)
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(f"hermes_test_isolation.{name} is not callable ({type(fn).__name__})")
    return fn


def _register_host_contract(name: str, *args: Any) -> bool:
    """Register via an optional host contract, failing fast if the contract exists and raises."""
    fn = _host_contract_fn(name)
    if fn is None:
        return False
    try:
        fn(*args)
    except Exception:
        logger.exception("host_bridge: %s failed", name)
        raise
    return True


def _unregister_host_contract(name: str, *args: Any) -> None:
    """Unregister via an optional host contract, logging a warning if teardown raises."""
    fn = _host_contract_fn(name)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:
        logger.warning("host_bridge: %s failed", name, exc_info=True)


def get_hermes_home() -> Path:
    pinned = _hermes_home_override.get()
    if pinned is not None:
        return pinned
    return get_host_runtime_home()


def get_host_runtime_home() -> Path:
    fn = _host_contract_fn("get_hermes_home")
    if fn is not None:
        return Path(fn())
    return Path(os.environ.get("HERMES_HOME_OVERRIDE") or os.environ.get("HERMES_HOME")
                or (Path.home() / ".hermes")).expanduser()


@contextmanager
def hermes_home_scope(home: Path | str | None):
    """Temporarily bind ``get_hermes_home()`` inside ViveMind to *home*."""
    if home is None:
        yield
        return
    token = _hermes_home_override.set(Path(home).expanduser())
    try:
        yield
    finally:
        _hermes_home_override.reset(token)


def get_process_hermes_home() -> Path:
    fn = _host_contract_fn("get_process_hermes_home")
    return Path(fn()) if fn is not None else get_hermes_home()


def get_subprocess_home() -> Path:
    return Path(os.environ.get("HERMES_HOME_OVERRIDE") or os.environ.get("HERMES_HOME")
                or str(get_hermes_home())).expanduser()


@dataclass(frozen=True)
class _ParseLocation:
    line: int | None = None
    column: int | None = None
    position: int | None = None


def _extract_parse_location(exc: BaseException | None) -> _ParseLocation:
    """Return 1-based line/column or byte/char position for known parse exceptions."""
    if exc is None:
        return _ParseLocation()
    import json

    if isinstance(exc, json.JSONDecodeError):
        return _ParseLocation(line=exc.lineno, column=exc.colno)
    try:
        import yaml
        from yaml.reader import ReaderError as PyYamlReaderError
    except ImportError:
        yaml = None  # type: ignore[assignment]
        PyYamlReaderError = None  # type: ignore[assignment,misc]
    if yaml is not None and isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark or exc.context_mark
        if mark is not None:
            return _ParseLocation(line=mark.line + 1, column=mark.column + 1)
    if PyYamlReaderError is not None and isinstance(exc, PyYamlReaderError):
        if isinstance(exc.position, int):
            return _ParseLocation(position=exc.position)
    try:
        from ruamel.yaml.error import MarkedYAMLError as RuamelMarkedYAMLError
        from ruamel.yaml.reader import ReaderError as RuamelReaderError
    except ImportError:
        RuamelMarkedYAMLError = None  # type: ignore[assignment,misc]
        RuamelReaderError = None  # type: ignore[assignment,misc]
    if RuamelMarkedYAMLError is not None and isinstance(exc, RuamelMarkedYAMLError):
        mark = exc.problem_mark or exc.context_mark
        if mark is not None:
            return _ParseLocation(line=mark.line + 1, column=mark.column + 1)
    if RuamelReaderError is not None and isinstance(exc, RuamelReaderError):
        if isinstance(exc.position, int):
            return _ParseLocation(position=exc.position)
    return _ParseLocation()


def _format_parse_error_detail(
    cause_name: str,
    location: _ParseLocation,
) -> str:
    """Format a sanitized parse-error summary containing only class name and position."""
    if location.line is not None and location.column is not None:
        loc_str = f"line {location.line}, column {location.column}"
    elif location.line is not None:
        loc_str = f"line {location.line}"
    elif location.position is not None:
        loc_str = f"position {location.position}"
    else:
        loc_str = ""
    if cause_name and loc_str:
        return f"{cause_name} at {loc_str}"
    return cause_name or loc_str


def _load_yaml_text(raw: str) -> Any:
    try:
        import yaml
    except ImportError:
        yaml = None  # type: ignore[assignment]
    if yaml is not None:
        safe_load = yaml.safe_load
        try:
            return safe_load(raw)
        except (
            yaml.YAMLError,
            ValueError,
            LookupError,
            TypeError,
            AttributeError,
            ArithmeticError,
            RecursionError,
        ) as exc:
            raise HostConfigParseError(
                _format_parse_error_detail(type(exc).__name__, _extract_parse_location(exc))
            ) from exc
    try:
        from ruamel.yaml import YAML, YAMLError
    except ImportError:
        import json

        try:
            return json.loads(raw)
        except (ValueError, TypeError, RecursionError) as exc:
            raise HostConfigParseError(
                _format_parse_error_detail(type(exc).__name__, _extract_parse_location(exc))
            ) from exc
    loader = YAML(typ="safe").load
    try:
        return loader(raw)
    except (
        YAMLError,
        ValueError,
        LookupError,
        TypeError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as exc:
        raise HostConfigParseError(
            _format_parse_error_detail(type(exc).__name__, _extract_parse_location(exc))
        ) from exc


def load_host_config(home: Path | None = None) -> dict[str, Any]:
    path = (Path(home) if home is not None else get_hermes_home()) / "config.yaml"
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError) as exc:
        if isinstance(exc, UnicodeDecodeError):
            detail = f"{type(exc).__name__} at byte {exc.start}"
        elif isinstance(exc, OSError) and exc.errno is not None:
            detail = f"{type(exc).__name__} (errno {exc.errno})"
        else:
            detail = type(exc).__name__
        logger.warning(
            "host_bridge: failed to read host config %s (%s)",
            path,
            detail,
        )
        return {}
    if not raw.strip():
        return {}
    try:
        value = _load_yaml_text(raw)
    except HostConfigParseError as exc:
        cause = exc.__cause__
        cause_name = type(cause).__name__ if cause is not None else ""
        detail = _format_parse_error_detail(cause_name, _extract_parse_location(cause))
        if detail:
            logger.warning(
                "host_bridge: failed to parse host config %s (%s: %s)",
                path,
                type(exc).__name__,
                detail,
            )
        else:
            logger.warning(
                "host_bridge: failed to parse host config %s (%s)",
                path,
                type(exc).__name__,
            )
        return {}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        logger.warning(
            "host_bridge: host config %s root must be a mapping, got %s",
            path,
            type(value).__name__,
        )
        return {}
    return dict(value)


def resolve_plugin_context(ctx: Any, plugin_id: str = "sanae_mem") -> Any:
    fn = _host_contract_fn("resolve_plugin_context")
    if hasattr(ctx, "llm") or fn is None:
        return ctx
    resolved = fn(ctx, plugin_id=plugin_id)
    return resolved if resolved is not None else ctx


@contextmanager
def turn_envelope_scope(inherit_context: bool = True, **fields: Any):
    """Bind a turn envelope for a callback, built from *fields*.

    ``inherit_context=False`` (host contracts present) builds a COMPLETE new envelope instead of
    letting the host fill omitted fields from ambient session ContextVars / the active agent.
    Off-turn work (a delivery receipt applied by a queue worker) uses this so a cleared or
    unrelated inbound context — its profile, scope or scene — is never inherited. A host that
    predates an envelope field keeps its own default for it. The flag never reaches the host: it
    selects which envelope is handed over.
    """
    scope_fn = _host_contract_fn("turn_envelope_scope")
    if scope_fn is not None:
        if not inherit_context:
            from hermes_host_contracts import TurnEnvelope

            supported = {spec.name for spec in dataclass_fields(TurnEnvelope)}
            with scope_fn(
                    TurnEnvelope(**{key: value for key, value in fields.items() if key in supported})
            ) as scoped:
                yield scoped
            return
        with scope_fn(**fields) as envelope:
            yield envelope
        return
    envelope = _StandaloneTurnEnvelope(**fields)
    token = _turn_envelope.set(envelope)
    try:
        yield envelope
    finally:
        _turn_envelope.reset(token)


def has_native_turn_context() -> bool:
    import importlib
    try:
        mod = importlib.import_module("agent.memory_manager")
    except ImportError:
        return False
    manager_cls = getattr(mod, "MemoryManager", None)
    return manager_cls is not None and callable(getattr(manager_cls, "current_turn_context", None))


def hermes_source_root() -> Path | None:
    fn = _host_contract_fn("hermes_source_root")
    return Path(fn()) if fn is not None else None


def list_plugin_cron_jobs(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    fn = _host_contract_fn("list_plugin_cron_jobs")
    return fn(include_disabled=include_disabled) if fn is not None else []


def pause_plugin_cron_job(job_id: str, *, reason: str = "") -> dict[str, Any] | None:
    fn = _host_contract_fn("pause_plugin_cron_job")
    return fn(job_id, reason=reason) if fn is not None else None


def sync_plugin_cron_job(*, name: str, enabled: bool,
                         match_fn: Callable[[Mapping[str, Any]], bool], refresh: bool = False,
                         paused_reason: str = "Disabled by plugin configuration",
                         **settings: Any) -> dict[str, Any] | None:
    fn = _host_contract_fn("sync_plugin_cron_job")
    if fn is None:
        return None
    return fn(name=name, enabled=enabled, match_fn=match_fn,
              refresh=refresh, paused_reason=paused_reason, **settings)


def register_cron_completion_handler(handler: Callable[[Any], None]) -> bool:
    return _register_host_contract("register_cron_completion_handler", handler)


def unregister_cron_completion_handler(handler: Callable[[Any], None]) -> None:
    _unregister_host_contract("unregister_cron_completion_handler", handler)


def get_cognitive_prompt_policy() -> Any | None:
    fn = _host_contract_fn("get_cognitive_prompt_policy")
    return fn() if fn is not None else None


@runtime_checkable
class PromptPolicyHomeCoverage(Protocol):
    def covers_builtin_memory_for_current_home(self) -> bool: ...


def has_effective_cognitive_prompt_policy() -> bool:
    """Keep the legacy host contract unless the policy reports its own HOME coverage."""
    policy = get_cognitive_prompt_policy()
    if isinstance(policy, PromptPolicyHomeCoverage):
        return policy.covers_builtin_memory_for_current_home()
    return policy is not None


def has_host_contracts() -> bool:
    return _hc is not None


def is_test_session_id(session_id: str | None) -> bool:
    """True when *session_id* itself names or is armed as a test-isolated session."""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    if sid.startswith("T:") or sid.startswith("test_") or ":test:" in sid:
        return True
    is_test_session = _test_isolation_fn("is_test_session")
    if is_test_session is not None:
        try:
            return bool(is_test_session(sid))
        except (OSError, ValueError, sqlite3.Error):
            logger.warning(
                "host_bridge: hermes_test_isolation.is_test_session failed for session_id=%r; failing closed",
                sid,
                exc_info=True,
            )
            return True
    return False


def is_test_isolated_turn(session_id: str | None = None) -> bool:
    sid = str(session_id or "").strip()
    if sid.startswith("T:"):
        return True
    is_test_session = _test_isolation_fn("is_test_session")
    if is_test_session is not None:
        try:
            if is_test_session(sid or None):
                return True
        except Exception:
            logger.warning(
                "host_bridge: hermes_test_isolation.is_test_session failed for session_id=%r",
                sid or None,
                exc_info=True,
            )
    get_envelope = _host_contract_fn("current_turn_envelope")
    if get_envelope is not None:
        try:
            env = get_envelope()
        except Exception:
            logger.warning(
                "host_bridge: current_turn_envelope failed in is_test_isolated_turn",
                exc_info=True,
            )
        else:
            if env is not None and bool(getattr(env, "is_test_isolated", False)):
                return True
    return False


def current_turn_envelope() -> Any | None:
    fn = _host_contract_fn("current_turn_envelope")
    if fn is None:
        return _turn_envelope.get()
    try:
        return fn()
    except Exception:
        logger.warning("host_bridge: current_turn_envelope failed", exc_info=True)
        return None


def current_turn_is_messaging_surface() -> bool:
    envelope = _hc.current_turn_envelope() if _hc is not None else _turn_envelope.get()
    return envelope is not None and envelope.is_messaging_surface


@dataclass(frozen=True)
class HostChatFields:
    platform: str
    chat_type: str
    chat_id: str
    sender_id: str
    sender_id_alt: str
    session_key: str
    message_id: str
    scope_id: str
    parent_chat_id: str
    reply_to_message_id: str
    occurred_at: str
    inbound_coverage: str
    is_bot: bool | None
    thread_visibility: str
    profile: str
    thread_id: str
    session_id: str
    source: str
    synthetic_reply_thread: bool
    restricted: bool
    is_delegated_child: bool = False


def host_chat_fields(envelope: object) -> HostChatFields:
    def text(field: str, *, strip: bool = True) -> str:
        value = str(getattr(envelope, field, "") or "")
        return value.strip() if strip else value

    bot = getattr(envelope, "is_bot", None)
    delegated = bool(getattr(envelope, "is_delegated_child", False))
    return HostChatFields(
        platform=text("platform").lower(),
        chat_type=text("chat_type").lower(),
        chat_id=text("chat_id"),
        sender_id=text("sender_id"),
        sender_id_alt=text("sender_id_alt"),
        session_key=text("session_key"),
        message_id=text("message_id", strip=False),
        scope_id=text("scope_id"),
        parent_chat_id=text("parent_chat_id"),
        reply_to_message_id=text("reply_to_message_id"),
        occurred_at=text("occurred_at"),
        inbound_coverage=text("inbound_coverage") or "unknown",
        is_bot=bot if isinstance(bot, bool) else None,
        thread_visibility=text("thread_visibility"),
        profile=text("profile"),
        thread_id=text("thread_id", strip=False),
        session_id=text("session_id", strip=False),
        source=text("source").lower(),
        synthetic_reply_thread=getattr(envelope, "synthetic_reply_thread", False) is True,
        restricted=bool(delegated or getattr(envelope, "restricted_audience", "")),
        is_delegated_child=delegated,
    )


def is_delegated_child_turn() -> bool:
    envelope = current_turn_envelope()
    return bool(envelope is not None and getattr(envelope, "is_delegated_child", False))


def current_cron_binding() -> tuple[str, str, str] | None:
    """Return verified ``(session_id, job_id, execution_id)`` for the live cron run, or ``None``."""
    env = verified_cron_envelope()
    if env is None:
        return None
    try:
        if env.job_id and env.execution_id and not env.cancelled:
            return (str(env.session_id or ""), str(env.job_id), str(env.execution_id))
    except AttributeError:
        logger.warning(
            "host_bridge: malformed cron envelope returned by host (%s)",
            type(env).__name__,
            exc_info=True,
        )
    return None


def verified_cron_envelope() -> Any | None:
    """Return the public contract's current verified cron identity, when available."""
    fn = _host_contract_fn("current_cron_envelope")
    if fn is None:
        return None
    try:
        return fn(reload_live_state=True)
    except Exception:
        logger.warning("host_bridge: current_cron_envelope failed", exc_info=True)
        return None


def resolve_task_cwd(task_id: str = "default") -> list[Path]:
    """Resolve candidate workspace CWDs for relative file-tool path checks."""
    candidates: list[Path] = []
    fn = _host_contract_fn("resolve_task_cwd")
    if fn is not None:
        try:
            resolved = fn(task_id or None)
        except Exception:
            logger.warning(
                "host_bridge: resolve_task_cwd failed for task_id=%r",
                task_id,
                exc_info=True,
            )
            resolved = None
        if resolved:
            try:
                p = Path(resolved).expanduser()
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "host_bridge: invalid task cwd %r (%s: %s)",
                    resolved,
                    type(exc).__name__,
                    exc,
                )
            else:
                if p.is_absolute() and p not in candidates:
                    candidates.append(p)
    if not candidates:
        candidates.append(Path.cwd())
    return candidates


def query_verified_cron_sends(
    *,
    bound_owner: str,
    platform: str = "",
    chat_id: str = "",
    thread_id: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
) -> list[tuple[float, str]]:
    """Read transport-confirmed cron sends from ``hermes_host_contracts``."""
    if not bound_owner:
        return []
    fn = _host_contract_fn("query_verified_cron_sends")
    if fn is None:
        return []
    try:
        rows = fn(
            since_ts=since_ts,
            until_ts=until_ts,
            bound_owner=bound_owner,
            platform=platform or None,
            chat_id=chat_id or None,
            thread_id=thread_id,
            limit=200,
        )
    except Exception:
        logger.warning(
            "host_bridge: query_verified_cron_sends failed (bound_owner=%r, platform=%r)",
            bound_owner,
            platform,
            exc_info=True,
        )
        return []
    if rows is None:
        return []
    out: list[tuple[float, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            logger.warning(
                "host_bridge: ignoring malformed verified cron send row (%s)",
                type(row).__name__,
            )
            continue
        row_plat = str(row.get("platform") or "").strip().lower()
        row_chat_id = str(row.get("chat_id") or row.get("target") or "")
        text = str(row.get("delivered_text") or "").strip()
        try:
            stamp = float(row.get("delivered_at") or 0.0)
        except (TypeError, ValueError):
            logger.warning(
                "host_bridge: ignoring verified cron send row with invalid delivered_at=%r",
                row.get("delivered_at"),
            )
            continue
        if not text or stamp <= 0:
            continue
        if str(row.get("bound_owner") or "") != bound_owner:
            continue
        if platform and row_plat != platform.lower():
            continue
        if chat_id and row_chat_id != chat_id:
            continue
        if thread_id is not None and str(row.get("thread_id") or "") != str(thread_id):
            continue
        if since_ts is not None and stamp <= since_ts or until_ts is not None and stamp > until_ts:
            continue
        out.append((stamp, text))
    out.sort(key=lambda item: item[0])
    return out


def register_delivery_receipt_handler(handler: Callable[[Any], None]) -> bool:
    return _register_host_contract("register_delivery_receipt_handler", handler)


def unregister_delivery_receipt_handler(handler: Callable[[Any], None]) -> None:
    _unregister_host_contract("unregister_delivery_receipt_handler", handler)


def register_cognitive_prompt_policy(policy: Any) -> bool:
    return _register_host_contract("register_cognitive_prompt_policy", policy)


def unregister_cognitive_prompt_policy(policy: Any = None) -> None:
    _unregister_host_contract("unregister_cognitive_prompt_policy", policy)


def register_inbound_merge_policy(policy: Any) -> bool:
    return _register_host_contract("register_inbound_merge_policy", policy)


def get_inbound_merge_policy() -> Any | None:
    fn = _host_contract_fn("get_inbound_merge_policy")
    return fn() if fn is not None else None


def unregister_inbound_merge_policy(policy: Any = None) -> None:
    _unregister_host_contract("unregister_inbound_merge_policy", policy)


def register_memory_mutation_gate(gate: Any) -> bool:
    return _register_host_contract("register_memory_mutation_gate", gate)


def get_memory_mutation_gate() -> Any | None:
    fn = _host_contract_fn("get_memory_mutation_gate")
    return fn() if fn is not None else None


def unregister_memory_mutation_gate(gate: Any = None) -> None:
    _unregister_host_contract("unregister_memory_mutation_gate", gate)


def register_cron_hooks(
    *,
    prompt_preparer: Callable[[Mapping[str, Any], str, Any], str | None] | None = None,
    pre_delivery_gate: Callable[[Mapping[str, Any], str, Any], dict[str, Any] | bool | None] | None = None,
) -> bool:
    if _hc is None:
        return False
    ok = False
    registered_preparer = False
    if prompt_preparer is not None and _register_host_contract(
        "register_cron_prompt_preparer", prompt_preparer
    ):
        registered_preparer = True
        ok = True
    if pre_delivery_gate is not None:
        try:
            if _register_host_contract("register_cron_pre_delivery_gate", pre_delivery_gate):
                ok = True
        except Exception:
            if registered_preparer:
                _unregister_host_contract("unregister_cron_prompt_preparer", prompt_preparer)
            raise
    return ok


def unregister_cron_hooks(
    *,
    prompt_preparer: Callable[[Mapping[str, Any], str, Any], str | None] | None = None,
    pre_delivery_gate: Callable[[Mapping[str, Any], str, Any], dict[str, Any] | bool | None] | None = None,
) -> None:
    if _hc is None:
        return
    if prompt_preparer is not None:
        _unregister_host_contract("unregister_cron_prompt_preparer", prompt_preparer)
    if pre_delivery_gate is not None:
        _unregister_host_contract("unregister_cron_pre_delivery_gate", pre_delivery_gate)


def register_group_context_policy(policy: Any) -> bool:
    return _register_host_contract("register_group_context_policy", policy)


def get_group_context_policy() -> Any | None:
    fn = _host_contract_fn("get_group_context_policy")
    return fn() if fn is not None else None


def unregister_group_context_policy(policy: Any = None) -> None:
    _unregister_host_contract("unregister_group_context_policy", policy)


def register_group_reply_policy(policy: Any) -> bool:
    return _register_host_contract("register_group_reply_policy", policy)


def get_group_reply_policy() -> Any | None:
    fn = _host_contract_fn("get_group_reply_policy")
    return fn() if fn is not None else None


def unregister_group_reply_policy(policy: Any = None) -> None:
    _unregister_host_contract("unregister_group_reply_policy", policy)


def group_reply_decision(**fields: Any) -> Any:
    """The host's public ``GroupReplyDecision``, or None on a host that has no such seam."""
    factory = _host_contract_fn("GroupReplyDecision")
    return factory(**fields) if factory is not None else None


_ProviderT = TypeVar("_ProviderT", contravariant=True)


class PluginRegistration(Protocol[_ProviderT]):
    def register_memory_provider(self, provider: _ProviderT, /) -> None: ...
    def register_hook(self, hook: str, callback: Callable[..., object], /) -> object: ...


def register_plugin_tool(
    ctx: object, *, name: str, toolset: str, schema: dict[str, object],
    handler: Callable[..., str], check_fn: Callable[[], bool],
) -> None:
    register = getattr(ctx, "register_tool", None)
    if not callable(register):
        raise TypeError("The host context does not support plugin tool registration")
    register(name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn)


def plugin_unload_registrar(ctx: object) -> Callable[[Callable[[], None]], object] | None:
    on_unload = getattr(ctx, "on_unload", None)
    return on_unload if callable(on_unload) else None


def cron_profile_home(envelope: object) -> Path:
    home = getattr(envelope, "profile_home", "")
    return Path(home).expanduser().resolve() if home else get_hermes_home().resolve()


class CronCompletion(Protocol):
    @property
    def status(self) -> str: ...
    @property
    def job_id(self) -> str: ...
    @property
    def execution_id(self) -> str: ...
    @property
    def session_id(self) -> str: ...
    @property
    def finished_at(self) -> str: ...
    @property
    def final_response(self) -> str: ...


def cron_completion_job_spec(event: CronCompletion) -> dict[str, Any]:
    return dict(getattr(event, "job_spec", None) or {})


@dataclass(frozen=True)
class RuntimeDeliveryReceipt:
    session_id: str
    platform: str
    chat_type: str
    chat_id: str
    sender_id: str
    sender_id_alt: str
    scope_id: str
    session_key: str
    thread_id: str
    profile: str
    delivered_text: str
    clean_text: str
    user_message: str
    model: str
    incoming_message_id: str
    turn_id: str
    message_id: str
    parent_chat_id: str
    thread_visibility: str
    occurred_at: str
    reply_to_message_id: str
    inbound_coverage: str
    synthetic_reply_thread: bool
    has_native_image: bool
    user_message_authored: bool
    delivered_at: float
    is_bot: bool | None


def is_chat_delivery_receipt(receipt: object) -> bool:
    return bool(getattr(receipt, "transport_confirmed", False)) and not bool(getattr(receipt, "is_cron", False))


def receipt_transcript(receipt: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(getattr(receipt, "transcript", ()) or ())


def receipt_plugin_user_context(receipt: object) -> str:
    context = getattr(receipt, "plugin_user_context", "")
    return context.strip() if isinstance(context, str) else ""


def delivery_receipt_fields(receipt: object) -> RuntimeDeliveryReceipt:
    def text(field: str) -> str:
        return str(getattr(receipt, field, "") or "").strip()

    return RuntimeDeliveryReceipt(
        session_id=text("session_id"),
        platform=text("platform"),
        chat_type=text("chat_type"),
        chat_id=text("chat_id"),
        sender_id=text("sender_id"),
        sender_id_alt=text("sender_id_alt"),
        scope_id=text("scope_id"),
        session_key=text("session_key"),
        thread_id=text("thread_id"),
        profile=text("profile"),
        delivered_text=text("delivered_text"),
        clean_text=text("clean_text"),
        user_message=text("user_message"),
        model=text("model"),
        incoming_message_id=text("incoming_message_id"),
        turn_id=text("turn_id"),
        message_id=text("message_id"),
        parent_chat_id=text("parent_chat_id"),
        thread_visibility=text("thread_visibility"),
        occurred_at=text("occurred_at"),
        reply_to_message_id=text("reply_to_message_id"),
        inbound_coverage=text("inbound_coverage"),
        synthetic_reply_thread=getattr(receipt, "synthetic_reply_thread", False) is True,
        has_native_image=getattr(receipt, "has_native_image", False) is True,
        user_message_authored=getattr(receipt, "user_message_authored", False) is True,
        delivered_at=float(getattr(receipt, "delivered_at", 0.0) or 0.0),
        is_bot=getattr(receipt, "is_bot", None),
    )
