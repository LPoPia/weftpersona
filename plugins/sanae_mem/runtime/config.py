from pathlib import Path
from typing import Any, Mapping

from ..infra.host_bridge import load_host_config


def live_plugin_config(home: Path, *, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bound HOME overrides explicit constructor defaults; missing keys keep those defaults."""
    plugins = load_host_config(home).get("plugins")
    configured = plugins.get("sanae_mem") if isinstance(plugins, dict) else None
    return {**(defaults or {}), **(configured if isinstance(configured, dict) else {})}
