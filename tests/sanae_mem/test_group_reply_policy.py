"""ViveMind's group-reply policy: independent switch, per-group override, start cadence (Task 05)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sanae_mem.runtime.group_context import group_scope_key
from sanae_mem.runtime.group_reply import (
    DEFAULT_INTERVAL_SECONDS,
    ViveMindGroupReplyPolicy,
    unmentioned_interval_seconds,
)


@dataclass(frozen=True)
class _Decision:
    group_key: str
    conversation_key: str
    allow_unmentioned_reply: bool
    unmentioned_interval_seconds: float


def _scope(platform="slack", chat_id="C1", thread=None, profile=None, chat_type="group"):
    scope = SimpleNamespace(
        platform=platform, chat_id=chat_id, chat_type=chat_type, scope_id="T1" if platform == "slack" else "G1",
        thread_id=thread, parent_chat_id=chat_id if thread else None, synthetic_reply_thread=False,
        visibility="public", profile=profile,
    )
    return scope


def _source(platform="slack", chat_id="C1", thread=None, profile=None):
    """Every attribute ``group_context``'s trusted-envelope reshape reads."""
    return SimpleNamespace(
        platform=platform,
        chat_type="group",
        scene_chat_type=None,
        chat_id=chat_id,
        thread_id=thread or "",
        parent_chat_id=chat_id if thread else "",
        scope_id="T1" if platform == "slack" else "G1",
        user_id="U-A",
        user_id_alt="",
        message_id="111.1",
        reply_to_message_id="",
        occurred_at="",
        inbound_coverage="unknown",
        is_bot=False,
        thread_visibility="unknown",
        profile=profile or "",
        synthetic_reply_thread=False,
    )


def _policy(config):
    return ViveMindGroupReplyPolicy(lambda: config, _Decision)


def _config(**overrides):
    return {"group_context": {"defaults": dict(overrides)}}


def test_the_switch_defaults_off_and_the_interval_defaults_to_thirty_seconds():
    decision = _policy({})(_source())
    assert decision.allow_unmentioned_reply is False
    assert decision.unmentioned_interval_seconds == 30.0
    assert DEFAULT_INTERVAL_SECONDS == 30.0


def test_enabling_one_switch_leaves_the_other_group_features_alone():
    config = {
        "group_context": {
            "defaults": {"allow_unmentioned_reply": True},
            "overrides": {},
        }
    }
    decision = _policy(config)(_source())
    assert decision.allow_unmentioned_reply is True
    assert "short_term_shared" not in config["group_context"]["defaults"]


def test_a_per_group_boolean_beats_the_global_default_both_ways():
    config = {
        "group_context": {
            "defaults": {"allow_unmentioned_reply": False, "unmentioned_interval_seconds": 45},
            "overrides": {},
        }
    }
    policy = _policy(config)
    quiet = _source(chat_id="C-QUIET")
    assert policy(quiet).allow_unmentioned_reply is False
    assert policy(quiet).unmentioned_interval_seconds == 45
    config["group_context"]["overrides"][group_scope_key(_scope(chat_id="C-LOUD"))] = {
        "allow_unmentioned_reply": True,
        "unmentioned_interval_seconds": 0,
    }
    loud = policy(_source(chat_id="C-LOUD"))
    assert loud.allow_unmentioned_reply is True
    assert loud.unmentioned_interval_seconds == 0.0, "0 is a configured value, not a missing one"


def test_intervals_are_validated_without_an_invented_cap():
    for stated, expected in (
        (-1, 30.0), ("5", 30.0), (None, 30.0), (float("nan"), 30.0), (float("inf"), 30.0),
        (0, 0.0), (7.5, 7.5), (86400, 86400.0),
    ):
        scope = _scope()
        config = {"group_context": {"defaults": {"unmentioned_interval_seconds": stated}}}
        assert unmentioned_interval_seconds(config, scope) == expected, stated
    assert not math.isnan(unmentioned_interval_seconds(
        {"group_context": {"defaults": {"unmentioned_interval_seconds": float("nan")}}}, _scope()))


def test_group_and_conversation_keys_reuse_the_03_resolver():
    decision = _policy(_config())(_source(chat_id="C1", thread="222.2"))
    group = _scope(chat_id="C1")
    thread = _scope(chat_id="C1", thread="222.2")
    assert decision.group_key == group_scope_key(group)
    assert decision.conversation_key.endswith("/threads/222.2")
    assert _policy(_config())(_source(chat_id="C1")).conversation_key == group_scope_key(group)


def test_two_threads_of_one_group_share_a_group_key_but_not_a_conversation():
    policy = _policy(_config(allow_unmentioned_reply=True))
    a = policy(_source(thread="222.2"))
    b = policy(_source(thread="333.3"))
    assert a.group_key == b.group_key
    assert a.conversation_key != b.conversation_key


def test_profiles_never_share_a_group_scope():
    policy = _policy(_config(allow_unmentioned_reply=True))
    a = policy(_source(profile="alice"))
    b = policy(_source(profile="bob"))
    assert a.group_key != b.group_key


def test_scenes_the_plugin_does_not_govern_decline():
    assert _policy(_config(allow_unmentioned_reply=True))(_source(platform="telegram")) is None
    assert _policy(_config(allow_unmentioned_reply=True))(
        _source(platform="discord", chat_id="CH1")) is not None


@pytest.mark.parametrize("platform", ["qq", "slack", "discord"])
def test_every_managed_platform_is_answered(platform):
    decision = _policy(_config(allow_unmentioned_reply=True))(
        _source(platform=platform, chat_id="C1"))
    assert decision is not None and decision.group_key and decision.conversation_key
