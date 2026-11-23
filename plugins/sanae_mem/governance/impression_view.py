"""Prompt-facing rendering for partner impressions."""

from __future__ import annotations

from ..infra.defaults import DEFAULT_PARTNER_ROLE
from ..storage.impression_store import Impression, _parse_sections


def render_impression(
    item: Impression | str,
    *,
    is_maintainer: bool = True,
    partner_name: str = "",
    partner_role: str = DEFAULT_PARTNER_ROLE,
) -> str:
    if not item:
        return ""
    if isinstance(item, Impression):
        baseline, recent, text = item.baseline, item.recent, item.text
    else:
        text = str(item).strip()
        baseline, recent = _parse_sections(text)

    if not text and not baseline and not recent:
        return ""

    if is_maintainer:
        name_or_role = partner_name or partner_role or "他"
        header = f"[我最近眼里的{name_or_role}]"
        note = "这是我在相处中形成、也会修正的私人印象。它不是对方此刻的心情，也不必当面说出来。"
    else:
        header = "[我对当前对话者的印象]"
        note = "这是我在正常相处中形成、也会随着了解慢慢修正的印象，不是对方此刻的反应，也不必当面说出来。"

    body_lines = []
    if baseline and recent:
        body_lines.append(f"【长期底色】{baseline}")
        body_lines.append(f"【阶段近况】{recent}")
    elif baseline:
        body_lines.append(f"【长期底色】{baseline}")
    elif recent:
        body_lines.append(f"【阶段近况】{recent}")
    else:
        body_lines.append(text)

    body = "\n".join(body_lines)
    return f"{header}\n{note}\n{body}"
