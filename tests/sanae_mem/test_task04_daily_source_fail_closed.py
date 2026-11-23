"""Task 04: fail-closed secondary source verification in daily_context.

All runtime tests drive only public entry points (``plugin_entry.register``,
``provider.initialize``, ``provider.handle_pre_llm_call``,
``private_delivery_settler.settle_private_delivery``, and
``hermes_host_contracts.emit_delivery_receipt``) and assemble the outer LLM stub
exclusively via the public ``ctx.llm`` contract with zero private attribute access.
"""

from __future__ import annotations

import ast
from contextlib import closing, contextmanager
import importlib
import json
import logging
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
import yaml

from sanae_mem.cognitive import daily_context, inner_wiring
from sanae_mem import plugin_entry
from sanae_mem.runtime import private_delivery_settler
from sanae_mem.governance import source_context
from sanae_mem.governance.source_context import ChatScope


def _load_host_contracts() -> Any:
    try:
        return importlib.import_module("hermes_host_contracts")
    except ModuleNotFoundError:
        pytest.skip("hermes_host_contracts is unavailable in this environment")


class _StubAuxiliaryLLM:
    """Outermost auxiliary LLM transport stub exposed via public ``ctx.llm``."""

    def __init__(self) -> None:
        self.daily_calls: list[str] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: str = "",
        **_kwargs: Any,
    ) -> SimpleNamespace:
        if purpose == "daily_context_extract":
            self.daily_calls.append(purpose)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "items": [
                            {
                                "kind": "DAILY_FACT",
                                "slot": "clinic_visit",
                                "action": "set",
                                "text": "维护者今天下午要去医院复查",
                                "quote": "我今天下午要去医院复查",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        if purpose == "commitments_update":
            return SimpleNamespace(text=json.dumps({"new_commitments": [], "resolutions": []}))
        return SimpleNamespace(
            text=json.dumps(
                {
                    "sentences": ["知道他今天下午要去复查。"],
                    "variables": {"tension": 0.1, "closeness": 0.6, "energy": 0.2},
                    "shared_sentences": None,
                },
                ensure_ascii=False,
            )
        )


class _PublicPluginContext:
    """Minimal public plugin registration context carrying public ``llm``."""

    def __init__(self, llm: _StubAuxiliaryLLM) -> None:
        self.llm = llm
        self.provider: plugin_entry.SanaeMemProvider | None = None
        self.hooks: dict[str, list[Callable[..., Any]]] = {}
        self.tools: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.aux_tasks: list[tuple[str, dict[str, Any]]] = []
        self.unload_callbacks: list[Callable[[], Any]] = []

    def register_memory_provider(self, provider: plugin_entry.SanaeMemProvider) -> None:
        self.provider = provider

    def register_hook(self, name: str, handler: Callable[..., Any]) -> None:
        self.hooks.setdefault(name, []).append(handler)

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        self.tools.append((args, kwargs))

    def register_auxiliary_task(self, task: str, **kwargs: Any) -> None:
        self.aux_tasks.append((task, kwargs))

    def on_unload(self, callback: Callable[[], Any]) -> None:
        self.unload_callbacks.append(callback)

    def unload(self) -> None:
        for callback in reversed(self.unload_callbacks):
            callback()
        self.unload_callbacks.clear()


@contextmanager
def _registered_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str = "telegram",
    user_id: str = "1001",
    session_id: str = "task04-dm-1",
    scope_id: str = "",
    bindings: list[dict[str, str]] | None = None,
    initialize: bool = True,
) -> Iterator[tuple[plugin_entry.SanaeMemProvider, Path, _StubAuxiliaryLLM, Path]]:
    home = tmp_path / f"home-{session_id}"
    home.mkdir(parents=True, exist_ok=True)
    data_dir = home / "sanae-mem"
    plugin_cfg: dict[str, Any] = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": platform, "user_id": user_id},
    }
    if scope_id:
        plugin_cfg["memory_identity"]["scope_id"] = scope_id
    if bindings:
        plugin_cfg["maintainer_account_bindings"] = bindings
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "test-model",
                "plugins": {
                    "enabled": ["sanae_mem"],
                    "sanae_mem": plugin_cfg,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    llm = _StubAuxiliaryLLM()
    ctx = _PublicPluginContext(llm)
    plugin_entry.register(ctx)
    assert ctx.provider is not None
    provider = ctx.provider
    if initialize:
        provider.initialize(session_id, platform=platform, hermes_home=str(home))
    try:
        yield provider, data_dir / "sanae.db", llm, home
    finally:
        try:
            provider.shutdown()
        finally:
            ctx.unload()


def _turn_envelope(
    contracts: Any,
    *,
    platform: str = "telegram",
    sender_id: str = "1001",
    chat_id: str = "1001",
    session_id: str = "task04-dm-1",
    turn_id: str = "turn-1",
    message_id: str = "in-1",
    chat_type: str = "dm",
    scope_id: str = "",
) -> Any:
    session_key = (
        f"agent:main:{platform}:{scope_id}:{chat_type}:{chat_id}"
        if scope_id
        else f"agent:main:{platform}:{chat_type}:{chat_id}"
    )
    return contracts.TurnEnvelope(
        platform=platform,
        chat_type=chat_type,
        chat_id=chat_id,
        sender_id=sender_id,
        session_id=session_id,
        session_key=session_key,
        message_id=message_id,
        turn_id=turn_id,
        scope_id=scope_id,
        is_messaging_surface=True,
        write_origin="foreground",
    )


def _private_delivery(
    *,
    session_id: str = "task04-dm-1",
    turn_id: str = "turn-1",
    platform: str = "telegram",
    chat_id: str = "1001",
    chat_type: str = "dm",
    sender_id: str = "1001",
    scope_id: str = "",
    message_id: str = "in-1",
    user_message: str = "我今天下午要去医院复查，记住提醒我！",
    assistant_response: str = "好的，复查路上注意安全。",
) -> private_delivery_settler.VerifiedPrivateDelivery:
    session_key = (
        f"agent:main:{platform}:{scope_id}:{chat_type}:{chat_id}"
        if scope_id
        else f"agent:main:{platform}:{chat_type}:{chat_id}"
    )
    return private_delivery_settler.VerifiedPrivateDelivery(
        session_id=session_id,
        turn_id=turn_id,
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id=sender_id,
        user_message=user_message,
        assistant_response=assistant_response,
        delivered_at=time.time(),
        message_id=f"out-{turn_id}",
        incoming_message_id=message_id,
        scope_id=scope_id,
        session_key=session_key,
        transcript=(
            {"role": "user", "content": user_message, "platform_message_id": message_id},
            {"role": "assistant", "content": assistant_response},
        ),
    )


def _table_count(db_path: Path, table: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _daily_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == daily_context.logger.name and rec.levelno >= logging.WARNING
    ]


def test_no_dynamic_source_context_import_or_fail_open_except() -> None:
    """AST checks: no dynamic source_context imports, no fail-open except, logged fail-closed."""
    source_path = Path(daily_context.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    methods: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_resolve_daily_identity",
            "_daily_turn_scope",
            "_daily_source_allows",
        }:
            methods[node.name] = node
    assert set(methods) == {
        "_resolve_daily_identity",
        "_daily_turn_scope",
        "_daily_source_allows",
    }

    for name in ("_resolve_daily_identity", "_daily_source_allows"):
        imports = [
            n for n in ast.walk(methods[name]) if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        assert not imports, f"{name} must reuse top-level imports, found {imports!r}"

    identity_handlers = [
        n for n in ast.walk(methods["_resolve_daily_identity"]) if isinstance(n, ast.ExceptHandler)
    ]
    assert not identity_handlers, "_resolve_daily_identity must not silently swallow exceptions"

    scope_handlers = [
        n for n in ast.walk(methods["_daily_turn_scope"]) if isinstance(n, ast.ExceptHandler)
    ]
    assert len(scope_handlers) == 1
    scope_handler = scope_handlers[0]
    assert any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value is None
        for n in ast.walk(scope_handler)
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "warning"
        for n in ast.walk(scope_handler)
    )

    allows_handlers = [
        n for n in ast.walk(methods["_daily_source_allows"]) if isinstance(n, ast.ExceptHandler)
    ]
    assert len(allows_handlers) == 1
    allows_handler = allows_handlers[0]
    for node in ast.walk(allows_handler):
        if isinstance(node, ast.Return):
            assert (
                isinstance(node.value, ast.Constant) and node.value.value is False
            ), "exception handler in _daily_source_allows must return False (fail-closed)"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warning"
        for node in ast.walk(allows_handler)
    ), "exception handler in _daily_source_allows must log logger.warning"


def test_normal_maintainer_dm_telegram_and_bound_slack_settle_daily_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contracts = _load_host_contracts()
    bindings = [
        {
            "profile": "default",
            "platform": "slack",
            "scope_id": "T_MAIN",
            "user_id": "U_MAINT",
        }
    ]
    tg_env = _turn_envelope(
        contracts, platform="telegram", sender_id="1001", chat_id="1001", session_id="tg-ok"
    )
    with contracts.turn_envelope_scope(tg_env):
        with _registered_provider(
            tmp_path,
            monkeypatch,
            platform="telegram",
            user_id="1001",
            session_id="tg-ok",
            bindings=bindings,
        ) as (provider, db_path, llm, _home):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                res = private_delivery_settler.settle_private_delivery(
                    provider,
                    _private_delivery(
                        session_id="tg-ok",
                        turn_id="tg-turn-1",
                        platform="telegram",
                        chat_id="1001",
                        sender_id="1001",
                        message_id="tg-in-1",
                    ),
                )
            assert res.settled is True
            assert llm.daily_calls == ["daily_context_extract"]
            assert not _daily_warnings(caplog)
            assert _table_count(db_path, "daily_context") == 1

    slack_env = _turn_envelope(
        contracts,
        platform="slack",
        sender_id="U_MAINT",
        chat_id="D_MAIN",
        scope_id="T_MAIN",
        session_id="slack-ok",
    )
    with contracts.turn_envelope_scope(slack_env):
        with _registered_provider(
            tmp_path,
            monkeypatch,
            platform="telegram",
            user_id="1001",
            session_id="slack-ok",
            bindings=bindings,
            initialize=False,
        ) as (provider, db_path, llm, home):
            provider.initialize("slack-ok", platform="slack", hermes_home=str(home))
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                res = private_delivery_settler.settle_private_delivery(
                    provider,
                    _private_delivery(
                        session_id="slack-ok",
                        turn_id="slack-turn-1",
                        platform="slack",
                        chat_id="D_MAIN",
                        sender_id="U_MAINT",
                        scope_id="T_MAIN",
                        message_id="slack-in-1",
                    ),
                )
            assert res.settled is True
            assert llm.daily_calls == ["daily_context_extract"]
            assert not _daily_warnings(caplog)
            assert _table_count(db_path, "daily_context") == 1


def test_resolve_chat_identity_exception_in_settle_private_delivery_fail_closed_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When resolve_chat_identity raises during settlement, daily_context writes 0 rows,
    emits a warning without leaking body text, and steps 3/4/5 of settlement complete."""
    contracts = _load_host_contracts()
    env = _turn_envelope(
        contracts, platform="telegram", sender_id="1001", chat_id="1001", session_id="id-err"
    )
    with contracts.turn_envelope_scope(env):
        with _registered_provider(
            tmp_path, monkeypatch, platform="telegram", user_id="1001", session_id="id-err"
        ) as (provider, db_path, llm, _home):
            # 1. resolve_chat_identity raises on _daily_turn_scope call during settle_private_delivery
            def _raise_identity(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("PRIVATE_BODY_LEAK_GUARD_1")

            with monkeypatch.context() as mp:
                mp.setattr(daily_context, "resolve_chat_identity", _raise_identity)
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    res = private_delivery_settler.settle_private_delivery(
                        provider,
                        _private_delivery(
                            session_id="id-err",
                            turn_id="turn-id-err-1",
                            message_id="in-id-err-1",
                        ),
                    )
                assert res.settled is True
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0
                assert _table_count(db_path, "events") >= 1
                assert _table_count(db_path, "inner_state_revisions") >= 1
                assert any(
                    "daily identity resolution failed (RuntimeError)" in rec.getMessage()
                    for rec in caplog.records
                )
                assert all(
                    "PRIVATE_BODY_LEAK_GUARD_1" not in rec.getMessage()
                    for rec in caplog.records
                )

                # Also verify handle_pre_llm_call fails closed for daily_context without crashing
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    hook_out = provider.handle_pre_llm_call(
                        user_message="今天下午我去医院复查",
                        session_id="id-err",
                        platform="telegram",
                        sender_id="1001",
                        turn_id="turn-hook-err",
                    )
                assert isinstance(hook_out, dict)
                warnings = _daily_warnings(caplog)
                assert any(
                    "daily identity resolution failed (RuntimeError)" in msg for msg in warnings
                )
                assert all("PRIVATE_BODY_LEAK_GUARD_1" not in msg for msg in warnings)

            # 2. First call (_daily_turn_scope) succeeds, second call (_daily_source_allows) raises
            calls = {"count": 0}
            real_resolve = source_context.resolve_chat_identity

            def _raise_on_second_call(*args: Any, **kwargs: Any) -> Any:
                calls["count"] += 1
                if calls["count"] >= 2:
                    raise RuntimeError("PRIVATE_BODY_LEAK_GUARD_2")
                return real_resolve(*args, **kwargs)

            with monkeypatch.context() as mp:
                mp.setattr(daily_context, "resolve_chat_identity", _raise_on_second_call)
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    res = private_delivery_settler.settle_private_delivery(
                        provider,
                        _private_delivery(
                            session_id="id-err",
                            turn_id="turn-id-err-2",
                            message_id="in-id-err-2",
                        ),
                    )
                assert res.settled is True
                assert calls["count"] == 2
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0
                assert any(
                    "daily source check failed (RuntimeError)" in rec.getMessage()
                    for rec in caplog.records
                )
                assert all(
                    "PRIVATE_BODY_LEAK_GUARD_2" not in rec.getMessage()
                    for rec in caplog.records
                )

            # 3. resolve_chat_identity returns None or malformed object without maintainer/private_owner_key
            for idx, (label, bad_identity) in enumerate(
                (
                    ("none-identity", None),
                    ("malformed-identity", SimpleNamespace(person_key="p-only")),
                ),
                1,
            ):
                before_inner = _table_count(db_path, "inner_state_revisions")
                with monkeypatch.context() as mp:
                    mp.setattr(
                        daily_context,
                        "resolve_chat_identity",
                        lambda *_args, _bad=bad_identity, **_kwargs: _bad,
                    )
                    caplog.clear()
                    with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                        res = private_delivery_settler.settle_private_delivery(
                            provider,
                            _private_delivery(
                                session_id="id-err",
                                turn_id=f"turn-bad-id-{idx}",
                                message_id=f"in-bad-id-{idx}",
                            ),
                        )
                    assert res.settled is True, label
                    assert llm.daily_calls == [], label
                    assert _table_count(db_path, "daily_context") == 0, label
                    assert _table_count(db_path, "inner_state_revisions") == before_inner + 1, label
                    assert any(
                        "daily identity resolution failed (TypeError)" in msg
                        for msg in _daily_warnings(caplog)
                    ), f"{label}: got {_daily_warnings(caplog)!r}"

            # 4. First call (_daily_turn_scope) succeeds, second call (_daily_source_allows) returns None
            none_calls = {"count": 0}

            def _none_on_second_call(*args: Any, **kwargs: Any) -> Any:
                none_calls["count"] += 1
                if none_calls["count"] >= 2:
                    return None
                return real_resolve(*args, **kwargs)

            with monkeypatch.context() as mp:
                mp.setattr(daily_context, "resolve_chat_identity", _none_on_second_call)
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    res = private_delivery_settler.settle_private_delivery(
                        provider,
                        _private_delivery(
                            session_id="id-err",
                            turn_id="turn-bad-id-second",
                            message_id="in-bad-id-second",
                        ),
                    )
                assert res.settled is True
                assert none_calls["count"] == 2
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0
                assert any(
                    "non_maintainer_identity" in msg for msg in _daily_warnings(caplog)
                )


def test_current_source_record_and_scope_exceptions_fail_closed_in_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contracts = _load_host_contracts()
    env = _turn_envelope(
        contracts, platform="telegram", sender_id="1001", chat_id="1001", session_id="src-err"
    )
    with contracts.turn_envelope_scope(env):
        with _registered_provider(
            tmp_path, monkeypatch, platform="telegram", user_id="1001", session_id="src-err"
        ) as (provider, db_path, llm, _home):
            # 1. current_source_record raises inside _daily_source_allows
            def _raise_source(*_args: Any, **_kwargs: Any) -> Any:
                raise ValueError("PRIVATE_BODY_LEAK_GUARD_3")

            with monkeypatch.context() as mp:
                mp.setattr(daily_context, "current_source_record", _raise_source)
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    res = private_delivery_settler.settle_private_delivery(
                        provider,
                        _private_delivery(
                            session_id="src-err",
                            turn_id="turn-src-err-1",
                            message_id="in-src-err-1",
                        ),
                    )
                assert res.settled is True
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0
                assert any(
                    "daily source check failed (ValueError)" in rec.getMessage()
                    for rec in caplog.records
                )
                assert all(
                    "PRIVATE_BODY_LEAK_GUARD_3" not in rec.getMessage()
                    for rec in caplog.records
                )

            # 2. current_chat_scope raises on second call (inside _daily_source_allows)
            calls = {"count": 0}
            real_scope = source_context.current_chat_scope

            def _raise_scope_on_second() -> ChatScope:
                calls["count"] += 1
                if calls["count"] >= 2:
                    raise KeyError("PRIVATE_BODY_LEAK_GUARD_4")
                return real_scope()

            with monkeypatch.context() as mp:
                mp.setattr(daily_context, "current_chat_scope", _raise_scope_on_second)
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                    res = private_delivery_settler.settle_private_delivery(
                        provider,
                        _private_delivery(
                            session_id="src-err",
                            turn_id="turn-src-err-2",
                            message_id="in-src-err-2",
                        ),
                    )
                assert res.settled is True
                assert calls["count"] == 2
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0
                assert any(
                    "daily source check failed (KeyError)" in rec.getMessage()
                    for rec in caplog.records
                )
                assert all(
                    "PRIVATE_BODY_LEAK_GUARD_4" not in rec.getMessage()
                    for rec in caplog.records
                )


def test_non_maintainer_source_and_author_platform_mismatches_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contracts = _load_host_contracts()
    env = _turn_envelope(
        contracts, platform="telegram", sender_id="1001", chat_id="1001", session_id="src-chk"
    )
    with contracts.turn_envelope_scope(env):
        with _registered_provider(
            tmp_path, monkeypatch, platform="telegram", user_id="1001", session_id="src-chk"
        ) as (provider, db_path, llm, _home):
            cases: list[tuple[str, dict[str, str] | None, str, str | None]] = [
                (
                    "non-maintainer-kind",
                    {"kind": "external_user", "author_id": "1001", "platform": "telegram"},
                    "source_kind=external_user",
                    None,
                ),
                (
                    "missing-source",
                    None,
                    "source_kind=missing",
                    None,
                ),
                (
                    "author-mismatch",
                    {"kind": "maintainer", "author_id": "imposter-777", "platform": "telegram"},
                    "author_mismatch",
                    "imposter-777",
                ),
                (
                    "empty-author",
                    {"kind": "maintainer", "author_id": "", "platform": "telegram"},
                    "author_mismatch",
                    None,
                ),
                (
                    "platform-mismatch",
                    {"kind": "maintainer", "author_id": "1001", "platform": "discord"},
                    "platform_mismatch",
                    None,
                ),
            ]
            for idx, (label, fake_source, expected_tag, forbidden_secret) in enumerate(cases, 1):
                with monkeypatch.context() as mp:
                    mp.setattr(
                        daily_context,
                        "current_source_record",
                        lambda *_args, _src=fake_source, **_kwargs: _src,
                    )
                    caplog.clear()
                    with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                        res = private_delivery_settler.settle_private_delivery(
                            provider,
                            _private_delivery(
                                session_id="src-chk",
                                turn_id=f"turn-{label}-{idx}",
                                message_id=f"in-{label}-{idx}",
                            ),
                        )
                    assert res.settled is True
                    assert llm.daily_calls == [], f"{label} should block daily_context_extract"
                    assert _table_count(db_path, "daily_context") == 0
                    assert any(expected_tag in rec.getMessage() for rec in caplog.records), (
                        f"{label}: expected warning containing {expected_tag!r}, got "
                        f"{[r.getMessage() for r in caplog.records]!r}"
                    )
                    if forbidden_secret is not None:
                        assert all(
                            forbidden_secret not in rec.getMessage() for rec in caplog.records
                        )


def test_non_private_external_wrong_workspace_and_secondary_scope_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contracts = _load_host_contracts()
    bindings = [
        {
            "profile": "default",
            "platform": "slack",
            "scope_id": "T_MAIN",
            "user_id": "U_MAINT",
        }
    ]
    # 1. External user DM, group chat, Slack without scope_id, Slack wrong workspace T_OTHER
    ext_env = _turn_envelope(
        contracts, platform="telegram", sender_id="9999", chat_id="9999", session_id="ext-dm"
    )
    with contracts.turn_envelope_scope(ext_env):
        with _registered_provider(
            tmp_path,
            monkeypatch,
            platform="telegram",
            user_id="1001",
            session_id="ext-dm",
            bindings=bindings,
        ) as (provider, db_path, llm, _home):
            private_delivery_settler.settle_private_delivery(
                provider,
                _private_delivery(
                    session_id="ext-dm",
                    turn_id="ext-t1",
                    platform="telegram",
                    chat_id="9999",
                    sender_id="9999",
                    message_id="ext-in-1",
                ),
            )
            assert llm.daily_calls == []
            assert _table_count(db_path, "daily_context") == 0

            # Group chat -> rejected as public_interaction
            grp_env = _turn_envelope(
                contracts,
                platform="telegram",
                sender_id="1001",
                chat_id="grp-1",
                chat_type="group",
                session_id="grp-s1",
            )
            with contracts.turn_envelope_scope(grp_env):
                grp_res = private_delivery_settler.settle_private_delivery(
                    provider,
                    _private_delivery(
                        session_id="grp-s1",
                        turn_id="grp-t1",
                        platform="telegram",
                        chat_id="grp-1",
                        chat_type="group",
                        sender_id="1001",
                        message_id="grp-in-1",
                    ),
                )
                assert grp_res.settled is False
                assert grp_res.reason == "public_interaction"
                assert _table_count(db_path, "daily_context") == 0

            # Slack DM missing scope_id -> rejected as public_interaction (unknown scope)
            unscoped_slack = _turn_envelope(
                contracts,
                platform="slack",
                sender_id="U_MAINT",
                chat_id="D_UNSCOPED",
                scope_id="",
                session_id="slack-unscoped",
            )
            with contracts.turn_envelope_scope(unscoped_slack):
                slack_res = private_delivery_settler.settle_private_delivery(
                    provider,
                    _private_delivery(
                        session_id="slack-unscoped",
                        turn_id="slack-unscoped-t1",
                        platform="slack",
                        chat_id="D_UNSCOPED",
                        sender_id="U_MAINT",
                        scope_id="",
                        message_id="slack-unscoped-in-1",
                    ),
                )
                assert slack_res.settled is False
                assert slack_res.reason == "public_interaction"
                assert _table_count(db_path, "daily_context") == 0

            # Slack DM in wrong workspace T_OTHER -> non-maintainer, 0 daily_context rows
            other_ws_slack = _turn_envelope(
                contracts,
                platform="slack",
                sender_id="U_MAINT",
                chat_id="D_OTHER",
                scope_id="T_OTHER",
                session_id="slack-other-ws",
            )
            with contracts.turn_envelope_scope(other_ws_slack):
                private_delivery_settler.settle_private_delivery(
                    provider,
                    _private_delivery(
                        session_id="slack-other-ws",
                        turn_id="slack-other-t1",
                        platform="slack",
                        chat_id="D_OTHER",
                        sender_id="U_MAINT",
                        scope_id="T_OTHER",
                        message_id="slack-other-in-1",
                    ),
                )
                assert llm.daily_calls == []
                assert _table_count(db_path, "daily_context") == 0

    # 2. Secondary check in _daily_source_allows independently rejects unknown/public/non-maintainer
    maint_env = _turn_envelope(
        contracts, platform="telegram", sender_id="1001", chat_id="1001", session_id="sec-scope"
    )
    with contracts.turn_envelope_scope(maint_env):
        with _registered_provider(
            tmp_path, monkeypatch, platform="telegram", user_id="1001", session_id="sec-scope"
        ) as (provider, db_path, llm, _home):
            real_scope = source_context.current_chat_scope
            secondary_cases: list[tuple[str, ChatScope, str]] = [
                ("unknown-scope", ChatScope("unknown"), "scope_visibility=unknown"),
                (
                    "public-scope",
                    ChatScope("public", "telegram", "1001", chat_type="group", chat_id="g1"),
                    "scope_visibility=public",
                ),
                (
                    "non-maintainer-scope",
                    ChatScope(
                        "private",
                        "telegram",
                        "9999",
                        "",
                        "in-x",
                        "dm",
                        "9999",
                        "",
                        "sec-scope",
                        "agent:main:telegram:dm:9999",
                    ),
                    "non_maintainer_identity",
                ),
            ]
            for idx, (label, second_scope, expected_tag) in enumerate(secondary_cases, 1):
                calls = {"count": 0}

                def _swap_second_scope(_second: ChatScope = second_scope, _calls: dict[str, int] = calls) -> ChatScope:
                    _calls["count"] += 1
                    if _calls["count"] >= 2:
                        return _second
                    return real_scope()

                with monkeypatch.context() as mp:
                    mp.setattr(daily_context, "current_chat_scope", _swap_second_scope)
                    caplog.clear()
                    with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                        res = private_delivery_settler.settle_private_delivery(
                            provider,
                            _private_delivery(
                                session_id="sec-scope",
                                turn_id=f"sec-t-{idx}",
                                message_id=f"sec-in-{idx}",
                            ),
                        )
                    assert res.settled is True
                    assert calls["count"] == 2
                    assert llm.daily_calls == [], f"{label} should block daily_context_extract"
                    assert _table_count(db_path, "daily_context") == 0
                    assert any(expected_tag in rec.getMessage() for rec in caplog.records), (
                        f"{label}: expected warning containing {expected_tag!r}, got "
                        f"{[r.getMessage() for r in caplog.records]!r}"
                    )


def test_real_emit_delivery_receipt_end_to_end_fail_closed_and_normal_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real plugin_entry.register(ctx) + emit_delivery_receipt end-to-end verification."""
    contracts = _load_host_contracts()
    with _registered_provider(
        tmp_path,
        monkeypatch,
        platform="telegram",
        user_id="1001",
        session_id="receipt-e2e",
        initialize=False,
    ) as (_provider, db_path, llm, _home):
        def _build_receipt(turn_id: str, msg_id: str, out_id: str) -> Any:
            return contracts.DeliveryReceipt(
                session_id="receipt-e2e",
                platform="telegram",
                chat_id="1001",
                chat_type="dm",
                sender_id="1001",
                turn_id=turn_id,
                message_id=out_id,
                incoming_message_id=msg_id,
                session_key="agent:main:telegram:dm:1001",
                user_message="我今天下午要去医院复查，记住提醒我！",
                delivered_text="好的，复查路上注意安全。",
                clean_text="好的，复查路上注意安全。",
                delivered_at=time.time(),
                is_bot=False,
                transcript=(
                    {
                        "role": "user",
                        "content": "我今天下午要去医院复查，记住提醒我！",
                        "platform_message_id": msg_id,
                    },
                    {"role": "assistant", "content": "好的，复查路上注意安全。"},
                ),
            )

        # Receipt 1: fault injected into daily_context.current_source_record -> fail-closed
        def _raise_source(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("RECEIPT_SECRET_BODY")

        with monkeypatch.context() as mp:
            mp.setattr(daily_context, "current_source_record", _raise_source)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
                dispatched = contracts.emit_delivery_receipt(
                    _build_receipt("rcpt-turn-1", "rcpt-in-1", "rcpt-out-1")
                )
                assert dispatched >= 1
                assert inner_wiring.wait_for_pending_updates(timeout=10.0)
            assert llm.daily_calls == []
            assert _table_count(db_path, "daily_context") == 0
            assert _table_count(db_path, "events") >= 1
            assert _table_count(db_path, "inner_state_revisions") >= 1
            assert any(
                "daily source check failed (RuntimeError)" in rec.getMessage()
                for rec in caplog.records
            )
            assert all("RECEIPT_SECRET_BODY" not in rec.getMessage() for rec in caplog.records)

        # Receipt 2: normal maintainer DM -> settles daily_context cleanly
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=daily_context.logger.name):
            dispatched = contracts.emit_delivery_receipt(
                _build_receipt("rcpt-turn-2", "rcpt-in-2", "rcpt-out-2")
            )
            assert dispatched >= 1
            assert inner_wiring.wait_for_pending_updates(timeout=10.0)
        assert llm.daily_calls == ["daily_context_extract"]
        assert not _daily_warnings(caplog)
        assert _table_count(db_path, "daily_context") == 1
