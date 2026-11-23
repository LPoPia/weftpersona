from __future__ import annotations

import re
from pathlib import Path


def runtime_footer_pattern(home: Path, platform: str, model: str = "") -> re.Pattern | None:
    """Describe configured transport footer fields without importing gateway internals."""
    try:
        from ..infra.host_bridge import load_host_config

        config = load_host_config(home)
        display = config.get("display") or {}
        platform_config = (display.get("platforms") or {}).get(platform) or {}
        footer = {"enabled": False, "fields": ["model", "context_pct", "cwd"]}
        for section in (display.get("runtime_footer"), platform_config.get("runtime_footer")):
            if not isinstance(section, dict):
                continue
            if "enabled" in section:
                footer["enabled"] = bool(section.get("enabled"))
            if isinstance(section.get("fields"), list) and section["fields"]:
                footer["fields"] = [str(item) for item in section["fields"]]
        if not footer["enabled"]:
            return None
    except (ImportError, OSError, ValueError, TypeError, RuntimeError, AttributeError):
        return None

    parts = []
    for field in footer["fields"]:
        required = False
        if field == "model" and model:
            value = str(model).rsplit("/", 1)[-1]
            if "\n" in value or "\r" in value:
                continue
            pattern, required = re.escape(value), True
        elif field == "model":
            # When the receipt has no model label, the footer still contains one.
            pattern = r"[A-Za-z0-9_.:@/+\-]+"
        elif field == "cwd":
            try:
                from ..infra.host_bridge import resolve_task_cwd

                cwd = str(resolve_task_cwd()[0])
                home_path = str(home.expanduser().resolve())
                if cwd.startswith(home_path):
                    cwd = "~" + cwd[len(home_path):]
                pattern, required = re.escape(cwd), bool(cwd)
            except (ImportError, OSError, RuntimeError, TypeError):
                continue
        elif field == "context_pct":
            pattern = r"(?:100|[1-9]?[0-9])%"
        elif field == "latency":
            pattern, required = r"(?:<1s|[1-9]s|[1-5][0-9]s|[1-9][0-9]*m[0-5][0-9]s)", True
        elif field == "served_model":
            # The proxy may supply the served model after the generation request.
            pattern = r"[A-Za-z0-9_.:@/+\-]+ → [A-Za-z0-9_.:@/+\-]+"
        else:
            continue
        part = rf"(?:(?:(?<=^)| · ){pattern})"
        parts.append(part if required else part + "?")
    return re.compile("".join(parts)) if parts else None


def clean_receipt_text(
    delivered_text: str, clean_text: str = "", footer_pattern: re.Pattern | None = None,
) -> str:
    """Prefer the transport's clean body; otherwise strip only a matched final footer line."""
    if str(clean_text or "").strip():
        return str(clean_text).strip()
    text = str(delivered_text or "").strip()
    if footer_pattern is None:
        return text
    separator = text.rfind("\n\n")
    if separator < 0:
        return text
    footer = text[separator + 2:]
    if "\n" in footer or "\r" in footer or not footer_pattern.fullmatch(footer):
        return text
    return text[:separator].rstrip()
