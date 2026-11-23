"""Task 03: per-group short-term context keys/settings and the host policy seam.

The host consumes ``ViveMindGroupContextPolicy`` through the real contract registry, so the
session-key and shared-session answers below exercise the same path a live turn takes.

    PYTHONPATH=<vivemind>/plugins \
      scripts/run_tests.sh <vivemind>/tests/test_multiplatform_group_context.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("hermes_host_contracts")
pytest.importorskip("gateway.session")

from gateway.session import (  # noqa: E402
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
)
from hermes_host_contracts import (  # noqa: E402
    clear_all_contract_registrations,
    register_group_context_policy,
)
from gateway.config import Platform  # noqa: E402
from sanae_mem.governance.cognitive_context import ViveMindInboundMergePolicy  # noqa: E402
from sanae_mem.runtime.group_context import (  # noqa: E402
    ViveMindGroupContextPolicy,
    conversation_scope_key,
    group_feature_enabled,
    group_scope_key,
)
from sanae_mem.governance.source_context import ChatScope  # noqa: E402


def _scope(*, platform="slack", chat_type="group", chat_id="C1", thread_id="",
           parent="", scope_id="T1", actor="U-A"):
    return ChatScope(
        "public", platform, actor, "", "m1", chat_type, chat_id, thread_id,
        scope_id=scope_id, parent_chat_id=parent, profile="p1",
    )


def _config(**overrides):
    return {"group_context": {"defaults": {}, "overrides": overrides}}


@pytest.fixture(autouse=True)
def _reset_contracts():
    clear_all_contract_registrations()
    yield
    clear_all_contract_registrations()


def test_group_key_is_escaped_per_segment_and_thread_uses_parent():
    base = group_scope_key(_scope(chat_id="C 1/x"))
    assert base == "p1/slack/T1/C%201%2Fx"
    thread = _scope(chat_id="T-9", thread_id="T-9", parent="C 1/x")
    assert group_scope_key(thread) == "p1/slack/T1/C%201%2Fx"
    assert conversation_scope_key(thread) == "p1/slack/T1/C%201%2Fx/threads/T-9"
    assert conversation_scope_key(_scope()) == "p1/slack/T1/C1"


def test_untrusted_domains_have_no_key_and_resolve_false():
    assert group_scope_key(_scope(platform="discord", scope_id="")) is None
    assert group_scope_key(_scope(platform="slack", scope_id="")) is None
    assert group_scope_key(_scope(platform="qqbot", scope_id="")) is None
    assert group_scope_key(_scope(platform="telegram")) is None
    assert group_feature_enabled(_config(), _scope(platform="telegram"), "short_term_shared") is False
    thread = _scope(chat_id="T-9", thread_id="T-9", parent="")
    assert group_scope_key(thread) is None


def test_private_and_unknown_scenes_never_get_a_group_key():
    """A bound Slack DM has an id and a workspace; neither makes it a group."""
    dm = ChatScope("private", "slack", "U-A", "", "m1", "dm", "D1", scope_id="T1", profile="p1")
    assert group_scope_key(dm) is None
    assert conversation_scope_key(dm) is None
    unknown = ChatScope("unknown", "slack", "U-A", "", "m1", "group", "C1", scope_id="T1")
    assert group_scope_key(unknown) is None
    # A public visibility with a private transport type cannot manufacture a group either.
    mismatched = ChatScope("public", "slack", "U-A", "", "m1", "dm", "C1", scope_id="T1")
    assert group_scope_key(mismatched) is None


def test_named_profile_and_scope_are_separate_key_namespaces():
    key = group_scope_key(_scope())
    named = group_scope_key(_scope()._replace(profile="yuki"))
    other_app = group_scope_key(_scope(platform="qqbot", scope_id="APP2", chat_id="G1"))
    this_app = group_scope_key(_scope(platform="qqbot", scope_id="APP1", chat_id="G1"))
    assert len({key, named, other_app, this_app}) == 4
    assert named.startswith("yuki/")
    assert this_app == "p1/qqbot/APP1/G1"
    config = {"group_context": {"defaults": {"short_term_shared": False},
                                "overrides": {named: {"short_term_shared": True}}}}
    assert group_feature_enabled(config, _scope()._replace(profile="yuki"), "short_term_shared") is True
    assert group_feature_enabled(config, _scope(), "short_term_shared") is False


def test_a_synthetic_reply_thread_is_not_its_own_conversation():
    """The host's reply landing spot resolves to the group it was opened in."""
    synthetic = _scope(chat_id="TH1", thread_id="TH1", parent="CH1")._replace(
        platform="discord", scope_id="G1", synthetic_reply_thread=True)
    assert group_scope_key(synthetic) == "p1/discord/G1/CH1"
    assert conversation_scope_key(synthetic) == "p1/discord/G1/CH1"
    real = synthetic._replace(synthetic_reply_thread=False)
    assert group_scope_key(real) == "p1/discord/G1/CH1"
    assert conversation_scope_key(real) == "p1/discord/G1/CH1/threads/TH1"


def test_feature_resolution_is_per_key_and_per_feature():
    key = group_scope_key(_scope())
    config = {"group_context": {
        "defaults": {"short_term_shared": False, "long_term_memory": True},
        "overrides": {key: {"short_term_shared": True, "long_term_memory": None}},
    }}
    assert group_feature_enabled(config, _scope(), "short_term_shared") is True
    assert group_feature_enabled(config, _scope(), "long_term_memory") is True
    assert group_feature_enabled(config, _scope(chat_id="C2"), "short_term_shared") is False
    assert group_feature_enabled({"group_context": {"defaults": {"short_term_shared": True}}},
                                 _scope(), "short_term_shared") is True
    assert group_feature_enabled({"group_context": {"overrides": {key: {"short_term_shared": False}}}},
                                 _scope(), "short_term_shared") is False


def _slack_source(sender, *, thread_id=None):
    return SessionSource(
        platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id=sender,
        scope_id="T1", thread_id=thread_id, parent_chat_id="C1" if thread_id else None,
    )


def test_host_uses_the_vivemind_ruling_for_keys_and_sharing():
    config = {"group_context": {"defaults": {"short_term_shared": True}}}
    config["group_context"]["overrides"] = {group_scope_key(_scope(platform="slack", scope_id="T1")): {
        "short_term_shared": True}}
    register_group_context_policy(ViveMindGroupContextPolicy(lambda: config))
    a, b = _slack_source("U-A"), _slack_source("U-B")
    assert build_session_key(a) == build_session_key(b)
    assert is_shared_multi_user_session(a) is True

    config["group_context"]["defaults"]["short_term_shared"] = False
    # The already-resolved event keeps its admitted scope; a NEW event follows the new mode.
    assert build_session_key(a) == "agent:main:slack:group:T1:C1:~shared"
    fresh_a, fresh_b = _slack_source("U-A"), _slack_source("U-B")
    assert build_session_key(fresh_a) != build_session_key(fresh_b)
    assert is_shared_multi_user_session(fresh_a) is False
    # Real threads stay isolated from each other and from the channel.
    t1 = build_session_key(_slack_source("U-A", thread_id="T-1"))
    t2 = build_session_key(_slack_source("U-A", thread_id="T-2"))
    assert len({build_session_key(fresh_a), t1, t2}) == 3


def test_dm_and_governed_unknown_scenes_are_not_shared():
    register_group_context_policy(ViveMindGroupContextPolicy(lambda: _config()))
    policy = ViveMindGroupContextPolicy(lambda: _config())
    dm = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm",
                       scene_chat_type="dm", user_id="U-A", scope_id="T1")
    assert policy(dm) is None
    # A known group whose domain is missing cannot be scoped: members stay isolated (never the
    # host's shared-thread default).
    unscoped = SessionSource(platform=Platform.DISCORD, chat_id="C1", chat_type="group", user_id="U-A")
    assert policy(unscoped) is False
    # QQ's app id is its domain; without it the group id is only unique inside an app we cannot see.
    assert policy(SessionSource(platform=Platform.QQBOT, chat_id="G1", chat_type="group",
                                user_id="M1")) is False
    assert policy(SessionSource(platform=Platform.QQBOT, chat_id="G1", chat_type="group",
                                user_id="M1", scope_id="APP1")) is False


def test_qq_group_without_a_display_name_is_a_shared_group():
    config = {"group_context": {"defaults": {"short_term_shared": True}}}
    register_group_context_policy(ViveMindGroupContextPolicy(lambda: config))
    qq = SessionSource(platform=Platform.QQBOT, chat_id="G1", chat_type="group",
                       user_id="MEMBER-OPENID", scope_id="APP1")
    assert is_shared_multi_user_session(qq) is True
    assert build_session_key(qq) == "agent:main:qqbot:group:G1:~scope:APP1:~shared"


def test_persisted_native_source_round_trip_keeps_the_ruling_and_the_thread_kind():
    config = {"group_context": {"defaults": {"short_term_shared": True}}}
    register_group_context_policy(ViveMindGroupContextPolicy(lambda: config))
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="TH1", chat_type="group", user_id="U-A",
        thread_id="TH1", parent_chat_id="CH1", scope_id="G1", synthetic_reply_thread=True)
    assert is_shared_multi_user_session(source) is True
    admitted = build_session_key(source)
    restored = SessionSource.from_native_dict(source.to_native_dict())
    assert (restored.group_history_shared, restored.synthetic_reply_thread) == (True, True)
    # The mode flips after admission; the queued row still resolves to the key it was admitted on.
    config["group_context"]["defaults"]["short_term_shared"] = False
    assert build_session_key(restored) == admitted == "agent:main:discord:group:CH1:~scope:G1:~shared"
    assert is_shared_multi_user_session(restored) is True


class _Event:
    def __init__(self, source, text="hi", **facts):
        self.source = source
        self.text = text
        for name, value in facts.items():
            setattr(self, name, value)

    def get_command(self):
        return None


def test_merge_policy_covers_trusted_slack_discord_dms_only():
    policy = ViveMindInboundMergePolicy()
    slack_dm = _Event(SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm",
                                    scene_chat_type="dm", user_id="U-A", scope_id="T1"))
    assert policy.should_debounce_event(slack_dm, "slack", "dm", busy=False) is True
    assert policy.can_merge_events(slack_dm, slack_dm, "slack", "dm", busy=False) is True
    assert policy.debounce_window_seconds("slack", "dm", busy=False) == 5.0

    mpim = _Event(SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm",
                                scene_chat_type="mpim", user_id="U-A", scope_id="T1"))
    # The route reads as a 1:1 but the platform's own scene fact is a group DM: the policy declines
    # this input (False) — it does not decline the route.
    assert policy.should_debounce_event(mpim, "slack", "dm", busy=False) is False
    # The route the host actually derives for that group DM is outside the policy.
    assert policy.should_debounce_event(mpim, "slack", "mpim", busy=False) is None
    unscoped = _Event(SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm",
                                    scene_chat_type="dm", user_id="U-A"))
    # No workspace scope: an apparent 1:1 the platform did not confirm.
    assert policy.should_debounce_event(unscoped, "slack", "dm", busy=False) is False

    discord_user = SessionSource(platform=Platform.DISCORD, chat_id="D1", chat_type="dm", user_id="U-A")
    discord = _Event(discord_user)
    assert policy.can_merge_events(discord, discord, "discord", "dm", busy=False) is True
    group = _Event(SessionSource(platform=Platform.DISCORD, chat_id="C1", chat_type="group", user_id="U-A"))
    assert policy.should_debounce_event(group, "discord", "group", busy=False) is None

    quoted_a = _Event(discord_user, reply_to_message_id="m1")
    quoted_b = _Event(discord_user, reply_to_message_id="m2")
    assert policy.can_merge_events(quoted_a, quoted_a, "discord", "dm", busy=False) is True
    assert policy.can_merge_events(quoted_a, quoted_b, "discord", "dm", busy=False) is False
    command = _Event(discord_user, text="/model")
    assert policy.can_merge_events(discord, command, "discord", "dm", busy=False) is False
