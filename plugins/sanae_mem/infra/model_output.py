from __future__ import annotations

import json
from typing import Any


def _load_braced_json(text: str) -> Any | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    return json.loads(text[start:end + 1])
