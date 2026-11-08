from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK = "sanae_extract"
DAILY_TASK = "sanae_daily_flash"


def register(ctx: Any) -> Any:
    """Register the existing slot and obtain the host's configured LLM client."""
    ctx.register_auxiliary_task(
        TASK,
        display_name="早苗记忆抽取",
        description="每日高价值闲聊记忆抽取与概括改写",
        defaults={"provider": "opencode-go", "model": "deepseek-v4-pro", "timeout": 60},
    )
    llm_ctx = ctx
    if not hasattr(llm_ctx, "llm") and hasattr(llm_ctx, "_plugin_context"):
        llm_ctx = llm_ctx._plugin_context()
    return llm_ctx.llm


def register_daily(ctx: Any) -> None:
    """Daily extraction and inner state share this separately configured slot."""
    ctx.register_auxiliary_task(
        DAILY_TASK,
        display_name="早苗每日记忆与内心延续 Flash",
        description="整理每日记忆，并在回复送达后续写当下的心里话",
        defaults={"provider": "deepseek", "model": "deepseek-flash",
                  "base_url": "https://api.deepseek.com/v1",
                  "api_mode": "chat_completions",
                  "reasoning_effort": "none", "timeout": 25},
    )


def complete_simple(
    llm: Any, messages: list[dict], *, temperature: float, max_tokens: int,
    timeout: float, failure_message: str, purpose: Optional[str] = None,
) -> Optional[Any]:
    """Call the Flash slot directly; Hermes resolves its model and transport."""
    if llm is None:
        return None
    kwargs = dict(task=DAILY_TASK, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if purpose is not None:
        kwargs["purpose"] = purpose
    try:
        return llm.complete(messages, **kwargs)
    except Exception as exc:
        logger.warning("sanae_mem: %s (%s)", failure_message, type(exc).__name__)
        return None


def complete_complex(
    llm: Any, messages: list[dict], *, temperature: float, max_tokens: int,
    timeout: float, failure_message: str, purpose: Optional[str] = None,
) -> Optional[Any]:
    """Call sanae_extract directly, preserving its parameters and failure behavior."""
    if llm is None:
        return None
    kwargs = dict(task=TASK, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if purpose is not None:
        kwargs["purpose"] = purpose
    try:
        return llm.complete(messages, **kwargs)
    except Exception as exc:
        logger.warning("sanae_mem: %s (%s)", failure_message, type(exc).__name__)
        return None
