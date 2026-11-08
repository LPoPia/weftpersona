"""FOLLOWUP 已退役；保留空投影与拒绝旧写入的兼容入口。"""

from typing import Any, Dict, List, Optional


def active_items(home: Any) -> List[Any]:
    return []


def projection_lines(home: Any, *, limit: int = 5) -> List[str]:
    return []


def open_item(home: Any, summary: str, *, evidence: str = "", condition: str = "",
              invalid_if: str = "", review_on: str = "", today: Optional[str] = None) -> Any:
    raise ValueError("followup_retired")


def update_item(home: Any, item_id: str, *, expected_sha256: Optional[str] = None,
                summary: Optional[str] = None, today: Optional[str] = None,
                **changes: str) -> Dict[str, Any]:
    raise ValueError("followup_retired")


def close_item(home: Any, item_id: str, outcome: str, *, result: str = "",
               expected_sha256: Optional[str] = None, today: Optional[str] = None) -> Dict[str, Any]:
    raise ValueError("followup_retired")
