"""Read the configured memory account independently of daily or chat ownership policy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryIdentity:
    platform: str
    user_id: str
    user_id_alt: str
    scope_id: str


def read_memory_identity(
    config: Mapping[str, object] | None, environ: Mapping[str, str],
) -> MemoryIdentity:
    raw = (config or {}).get("memory_identity")
    if not isinstance(raw, dict):
        return MemoryIdentity("", "", "", "")
    platform = str(raw.get("platform") or "").strip()
    user_id = str(raw.get("user_id") or "").strip()
    user_id_alt = str(raw.get("user_id_alt") or "").strip()
    env_name = str(raw.get("user_id_env") or "").strip()
    if env_name:
        user_id = str(environ.get(env_name, "") or "").strip()
    scope_id = str(raw.get("scope_id") or raw.get("app_id") or "").strip()
    return MemoryIdentity(platform, user_id, user_id_alt, scope_id)


def daily_memory_identity(config: Mapping[str, object] | None) -> tuple[str, str, str]:
    identity = read_memory_identity(config, os.environ)
    return identity.platform, identity.user_id, identity.user_id_alt
