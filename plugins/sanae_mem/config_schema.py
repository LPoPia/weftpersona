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
        },
    },
    {
        "key": "chroma_site_packages",
        "description": "Operational override for the Chroma dependency search path",
        "type": "string",
        "default": "",
    },
    {
        "key": "persona_name",
        "description": "Persona display name used in auxiliary extraction and inner-state prompts",
        "type": "string",
        "default": "早苗",
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
        "default": "挚友",
    },
    {
        "key": "partner_pronoun",
        "description": "Third-person pronoun for the primary partner/user in first-person thoughts",
        "default": "他",
        "choices": ["他", "她"],
    },
]
