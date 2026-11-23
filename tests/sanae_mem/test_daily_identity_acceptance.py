from test_original_hermes import clear_session_vars, reset_session_vars, set_session_vars

from gateway import session_context
from sanae_mem.governance.source_context import current_chat_scope


def test_native_private_sender_binding_is_required(monkeypatch):
    route = dict(platform="telegram", chat_type="dm", chat_id="synthetic-chat",
                 user_id="synthetic-user", user_id_alt="synthetic-alt", session_id="synthetic-session")
    reset_session_vars()
    for field, value in route.items():
        monkeypatch.setenv("HERMES_SESSION_" + ("ID" if field == "session_id" else field.upper()), value)
    monkeypatch.setenv("HERMES_SESSION_KEY", "synthetic-from-environment")
    assert current_chat_scope().visibility == "unknown"

    def read_scope(**extra):
        tokens = set_session_vars(**route, session_key="synthetic-bound-key", **extra)
        try:
            return current_chat_scope()
        finally:
            clear_session_vars(tokens)

    scope = read_scope()
    assert (scope.visibility, scope.actor, scope.actor_alt, scope.session_key) == (
        "private", "synthetic-user", "synthetic-alt", "synthetic-bound-key",
    )
    if hasattr(session_context, "scoped_turn_actor"):
        assert read_scope(actor_id="").visibility == "unknown"
        foreign = read_scope(actor_id="synthetic-second-actor")
        assert foreign.actor == "synthetic-second-actor" and not foreign.actor_alt
