from ..infra.defaults import (
    DEFAULT_PARTNER_ROLE as DEFAULT_PARTNER_ROLE,
    DEFAULT_PERSONA_NAME as DEFAULT_PERSONA_NAME,
)

CONFIG_SCHEMA = [
    {
        "key": "shadow",
        "description": (
            "Manual operator switch: write memories but suppress recall "
            "(never engaged automatically)"
        ),
        "type": "boolean",
        "default": False,
    },
    {
        "key": "read_gate",
        "description": "Recall gate",
        "default": "explicit",
        "choices": ["explicit", "scheduled"],
    },
    {
        "key": "inject_max_chars",
        "description": "Maximum recalled context characters",
        "type": "integer",
        "default": 400,
        "minimum": 1,
        "maximum": 400,
    },
    {
        "key": "daily_scheduled_asks",
        "description": "Daily scheduled recall limit",
        "type": "integer",
        "default": 2,
        "minimum": 0,
    },
    {
        "key": "gap_continue",
        "description": "Offer one natural continuation cue after a long gap or on a new session",
        "type": "boolean",
        "default": True,
    },
    {
        "key": "gap_threshold_hours",
        "description": "Hours of inactivity before offering a continuation cue",
        "type": "number",
        "default": 2,
        "minimum": 1,
    },
    {
        "key": "emotion_ttl_days",
        "description": "TTL days for emotion/relation auto-captured events",
        "type": "integer",
        "default": 90,
        "minimum": 1,
    },
    {
        "key": "expiry_review_days",
        "description": "Days an expired short event remains available for recognition",
        "type": "integer",
        "default": 30,
        "minimum": 1,
    },
    {
        "key": "expiring_importance_decay",
        "description": "Daily importance multiplier for expired short events",
        "type": "number",
        "default": 0.9,
        "minimum": 0,
        "maximum": 1,
    },
    {
        "key": "expiring_importance_floor",
        "description": "Minimum importance retained while an expired short event awaits recognition",
        "type": "number",
        "default": 0.4,
        "minimum": 0,
        "maximum": 1,
    },
    {
        "key": "distill_threshold",
        "description": "Minimum importance for daily distillation",
        "type": "number",
        "default": 0.6,
        "minimum": 0,
        "maximum": 1,
    },
    {
        "key": "distill_daily_cap",
        "description": "Maximum distilled memories per day",
        "type": "integer",
        "default": 2,
        "minimum": 0,
    },
    {
        "key": "semantic_backend",
        "description": "Semantic index backend",
        "default": "chroma",
        "choices": ["chroma", "sqlite"],
    },
    {
        "key": "data_dir",
        "description": "Operational override for the default <home>/sanae-mem data directory",
        "type": "string",
        "default": "",
    },
    {
        "key": "memory_identity",
        "description": (
            "Trusted fixed maintainer identity for daily context. Deployment copies the "
            "existing platform/user_id/user_id_alt identity; user_id_env may name an "
            "existing environment variable. Empty means daily context is disabled; "
            "there is no first/current-sender fallback."
        ),
        "type": "object",
        "default": {},
        "properties": {
            "platform": {"type": "string", "default": ""},
            "user_id": {"type": "string", "default": ""},
            "user_id_alt": {"type": "string", "default": ""},
            "user_id_env": {"type": "string", "default": ""},
            "scope_id": {"type": "string", "default": ""},
        },
    },
    {
        "key": "maintainer_account_bindings",
        "description": (
            "Trusted list of the maintainer's own accounts on Slack, Discord, QQ or QQBot. "
            "Each entry needs platform and user_id (or user_id_alt); Slack also needs its "
            "workspace scope_id, and QQ/QQBot needs the adapter's App ID in scope_id (app_id is "
            "accepted too). Discord user IDs are global. profile defaults to the active/default "
            "profile. Bound accounts read and write the same private memory as memory_identity "
            "in a verified one-to-one chat; in a group they only mark the same person. Nicknames "
            "and message text never bind an account."
        ),
        "type": "array",
        "default": [],
        "items": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "default": ""},
                "user_id": {"type": "string", "default": ""},
                "user_id_alt": {"type": "string", "default": ""},
                "scope_id": {"type": "string", "default": ""},
                "app_id": {"type": "string", "default": ""},
                "profile": {"type": "string", "default": ""},
            },
        },
    },
    {
        "key": "chroma_site_packages",
        "description": "Operational override for the Chroma dependency search path",
        "type": "string",
        "default": "",
    },
    {
        "key": "nightly_journal",
        "description": "Write one private MEMORY.md entry at 23:00 Asia/Shanghai (requires memory_identity)",
        "type": "boolean",
        "default": False,
    },
    {
        "key": "free_think",
        "description": (
            "Run private reflections daily at 08:00, 16:00 and 22:00 Asia/Shanghai "
            "with web and memory search and local-only delivery (requires memory_identity)"
        ),
        "type": "boolean",
        "default": False,
    },
    {
        "key": "persona_name",
        "description": "Persona display name used in auxiliary extraction and inner-state prompts",
        "type": "string",
        "default": DEFAULT_PERSONA_NAME,
    },
    {
        "key": "partner_name",
        "description": "Optional display name of the primary partner/user in auxiliary prompts",
        "type": "string",
        "default": "",
    },
    {
        "key": "partner_role",
        "description": "Relationship role of the primary partner/user in auxiliary prompts",
        "type": "string",
        "default": DEFAULT_PARTNER_ROLE,
    },
    {
        "key": "partner_pronoun",
        "description": "Third-person pronoun for the primary partner/user in first-person thoughts",
        "default": "他",
        "choices": ["他", "她"],
    },
    {
        "key": "group_context",
        "description": (
            "Per-group short-term context and memory switches: defaults plus per-group "
            "overrides keyed by profile/platform/scope_id/chat_id. "
            "unmentioned_interval_seconds is the per-group minimum wait between non-addressed "
            "replies (default 30; 0 means no wait); it is independent of the three switches."
        ),
        "type": "object",
        "default": {
            "defaults": {
                "short_term_shared": False,
                "long_term_memory": False,
                "allow_unmentioned_reply": False,
                "unmentioned_interval_seconds": 30,
            },
            "overrides": {},
        },
    },
]
