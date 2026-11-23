"""Task 01 replay into ViveMind: the host's bound turn facts become a ``ChatScope``.

The scope is read through the real chain -- ``gateway.session_context.set_session_vars`` (what a
live turn binds) → ``hermes_host_contracts.current_turn_envelope`` → ``current_chat_scope``. No
envelope is constructed by hand here, so a change in either side is visible.

Run from the Hermes worktree with ViveMind's provider on ``PYTHONPATH``.

    PYTHONPATH=<vivemind>/plugins \
      scripts/run_tests.sh <vivemind>/tests/test_multiplatform_chat_scope.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("hermes_host_contracts")
pytest.importorskip("gateway.session_context")

from gateway.session_context import clear_session_vars, set_session_vars  # noqa: E402
from sanae_mem.governance.source_context import current_chat_scope  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402


@pytest.fixture
def bind():
    """Bind the session facts a turn binds, restoring the context afterwards."""
    tokens = []

    def _bind(**facts) -> None:
        tokens.extend(set_session_vars(**facts))

    yield _bind
    if tokens:
        clear_session_vars(tokens)


def _slack_dm(scope_id: str, user: str = "U_SAME") -> dict:
    return dict(platform="slack", chat_type="dm", chat_id="D_CHAN", user_id=user,
                scope_id=scope_id, session_key=f"slack:{scope_id}:dm:D_CHAN", message_id="m1")


def test_slack_dm_is_private_and_scoped_to_its_workspace(bind):
    bind(**_slack_dm("T_ONE"))
    scope = current_chat_scope()

    assert (scope.visibility, scope.chat_type) == ("private", "dm")
    assert scope.scope_id == "T_ONE"
    assert scope.owner, "a workspace-scoped DM still yields its owner"
    assert scope.owner == _owner_key("slack", user_id="U_SAME", scope_id="T_ONE")


def test_same_shaped_slack_id_in_another_workspace_is_another_person(bind):
    bind(**_slack_dm("T_ONE"))
    first = current_chat_scope().owner
    clear_session_vars([])
    bind(**_slack_dm("T_TWO"))
    second = current_chat_scope().owner

    assert first and second and first != second


def test_slack_dm_without_a_workspace_gets_no_private_owner(bind):
    facts = _slack_dm("")
    bind(**facts)
    scope = current_chat_scope()

    assert scope.visibility == "unknown"
    assert scope.owner == ""


def test_slack_channel_id_prefix_does_not_decide_the_scene(bind):
    """The platform-stated chat_type decides; a ``D…``/``G…`` id is not a classification."""
    bind(**_slack_dm("T_ONE", user="U_SAME") | {"chat_id": "G_MPIM"})
    assert current_chat_scope().visibility == "private"

    clear_session_vars([])
    bind(platform="slack", chat_type="group", chat_id="D_LOOKING", user_id="U_SAME",
         scope_id="T_ONE", session_key="slack:T_ONE:group:D_LOOKING", message_id="m2")
    assert current_chat_scope().visibility == "public"


def test_recovered_slack_scene_without_trusted_type_stays_unknown(bind):
    """A row the host could not classify (scene unknown) keeps its facts but no private owner."""
    bind(platform="slack", chat_type="unknown", chat_id="G_MPIM", user_id="U_SAME",
         user_id_alt="", scope_id="T_ONE", parent_chat_id="C_PARENT", message_id="m3",
         occurred_at="2024-05-01T00:00:00+00:00", session_key="slack:T_ONE:old:G_MPIM")
    scope = current_chat_scope()

    assert (scope.visibility, scope.owner) == ("unknown", "")
    assert (scope.chat_id, scope.actor, scope.scope_id) == ("G_MPIM", "U_SAME", "T_ONE")
    assert scope.occurred_at == "2024-05-01T00:00:00+00:00"


def test_thread_visibility_is_carried_and_unknown_forms_fall_back(bind):
    bind(platform="discord", chat_type="thread", chat_id="800", parent_chat_id="700",
         user_id="42", scope_id="1", thread_id="800", message_id="m4",
         session_key="discord:thread:800:42", thread_visibility="narrower")
    assert current_chat_scope().thread_visibility == "narrower"

    clear_session_vars([])
    bind(platform="discord", chat_type="thread", chat_id="800", parent_chat_id="700",
         user_id="42", scope_id="1", thread_id="800", message_id="m5",
         session_key="discord:thread:800:42", thread_visibility="inheritedish")
    assert current_chat_scope().thread_visibility == "unknown"


def test_slack_maintainer_sender_id_is_not_cross_workspace_proof():
    """A bare Slack sender never earns the maintainer surface until Task 02 binds accounts."""
    from sanae_mem.governance import cognitive_context as cognitive
    from types import SimpleNamespace

    config = {"memory_identity": {"platform": "slack", "user_id": "U_SAME"}}
    envelope = SimpleNamespace(
        platform="slack", chat_type="dm", chat_id="D_CHAN", sender_id="U_SAME",
        sender_id_alt="", scope_id="T_ONE", session_key="slack:T_ONE:dm:D_CHAN",
        message_id="m6")
    assert cognitive.classify_turn_surface(envelope, config) == "private_external"

    envelope.scope_id = ""
    assert cognitive.classify_turn_surface(envelope, config) == "restricted_public"


def test_slack_mpim_is_public_but_keeps_the_speaker(bind):
    bind(platform="slack", chat_type="group", chat_id="G_MPIM", user_id="U_SAME",
         scope_id="T_ONE", session_key="slack:T_ONE:group:G_MPIM:U_SAME", message_id="m2")
    scope = current_chat_scope()

    assert (scope.visibility, scope.owner) == ("public", "")
    assert (scope.actor, scope.scope_id) == ("U_SAME", "T_ONE")


def test_omitted_envelope_never_upgrades_an_unknown_scene_to_private(bind):
    """The compat entry (no envelope) reads routing fields from the live ChatScope, but the
    scene it already decided is authoritative: QQ without a session key stays unknown."""
    from sanae_mem.governance import cognitive_context as cognitive

    config = {"memory_identity": {"platform": "qqbot", "user_id": "U_OPENID"}}
    bind(platform="qqbot", chat_type="dm", chat_id="U_OPENID", user_id="U_OPENID",
         scope_id="102000001", message_id="m7")
    assert current_chat_scope().visibility == "unknown"
    assert cognitive.classify_turn_surface(None, config) == "restricted_public"

    clear_session_vars([])
    bind(platform="qqbot", chat_type="dm", chat_id="U_OPENID", user_id="U_OPENID",
         scope_id="102000001", session_key="qqbot:dm:U_OPENID", message_id="m8")
    assert cognitive.classify_turn_surface(None, config) == "private_maintainer"


def test_qq_private_owner_key_keeps_its_existing_namespace(bind):
    """QQ private owners already depend on the scope-free key; adding app_id must not move it."""
    bind(platform="qqbot", chat_type="dm", chat_id="U_OPENID", user_id="U_OPENID",
         scope_id="102000001", session_key="qqbot:dm:U_OPENID", message_id="m3")
    scope = current_chat_scope()

    assert scope.visibility == "private"
    assert scope.scope_id == "102000001"
    assert scope.owner == _owner_key("qqbot", user_id="U_OPENID")


def test_qq_group_at_keeps_mention_only_coverage(bind):
    bind(platform="qqbot", chat_type="group", chat_id="G_OPENID", user_id="M_OPENID",
         scope_id="102000001", session_key="qqbot:group:G_OPENID:M_OPENID",
         message_id="m4", inbound_coverage="mention_only")
    scope = current_chat_scope()

    assert (scope.visibility, scope.owner) == ("public", "")
    assert scope.inbound_coverage == "mention_only"


def test_discord_thread_keeps_its_own_address_and_parent(bind):
    bind(platform="discord", chat_type="thread", chat_id="800", parent_chat_id="700",
         user_id="42", scope_id="1", thread_id="800",
         session_key="discord:thread:800:42", message_id="900")
    scope = current_chat_scope()

    assert (scope.visibility, scope.owner) == ("public", "")
    assert (scope.chat_id, scope.parent_chat_id) == ("800", "700")


def test_unknown_bot_state_is_not_reported_as_human(bind):
    bind(platform="discord", chat_type="dm", chat_id="55", user_id="42",
         session_key="discord:dm:55:42", message_id="m5")
    assert current_chat_scope().is_bot is None

    clear_session_vars([])
    bind(platform="discord", chat_type="dm", chat_id="55", user_id="42",
         session_key="discord:dm:55:42", message_id="m5", is_bot="0")
    assert current_chat_scope().is_bot is False
