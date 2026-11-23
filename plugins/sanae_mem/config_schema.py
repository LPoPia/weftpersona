"""Public configuration schema facade re-exporting ``CONFIG_SCHEMA``."""

from __future__ import annotations

from .runtime.config_schema import CONFIG_SCHEMA as CONFIG_SCHEMA
from .infra.defaults import (
    DEFAULT_PARTNER_ROLE as DEFAULT_PARTNER_ROLE,
    DEFAULT_PERSONA_NAME as DEFAULT_PERSONA_NAME,
)
