from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .textutil import hermes_home as _resolve_hermes_home
except ImportError:
    from textutil import hermes_home as _resolve_hermes_home

logger = logging.getLogger(__name__)

_BLOCK_MESSAGE = "这条先别写：内存余量不够，先把同类旧条目合并或删一条再写。"


def _guard_log_path() -> Path:
    return Path(_resolve_hermes_home()) / "sanae-mem" / "guard.log"


def log_guard_block(reason: str, target: str, **numbers: Any) -> None:
    """记录拦截原因码与数字，不记录正文；日志失败不影响原判定。"""
    try:
        path = _guard_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        detail = " ".join("%s=%s" % (k, v) for k, v in sorted(numbers.items()))
        line = "%s | %s | target=%s | %s\n" % (
            datetime.now().astimezone().isoformat(timespec="seconds"), reason, target, detail)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass


def _apply_operation(entries: List[str], operation: Dict[str, Any]) -> bool:
    action = operation.get("action")
    content = operation.get("content")
    if content is None:
        content = operation.get("new_text")
    old_text = operation.get("old_text")

    if action == "add" and isinstance(content, str) and content.strip():
        value = content.strip()
        if value not in entries:
            entries.append(value)
        return True
    if action == "merge":
        text = operation.get("text")
        if text is None:
            text = content
        if not isinstance(text, str) or not text.strip():
            return False
        index = operation.get("index")
        find = operation.get("find") or old_text or ""
        if isinstance(index, int) and not isinstance(index, bool):
            if not (1 <= index <= len(entries)):
                return False
            entries[index - 1] = text.strip()
            return True
        if isinstance(find, str) and find.strip():
            matches = [i for i, e in enumerate(entries) if find.strip() in e]
            if len(matches) != 1:
                return False
            entries[matches[0]] = text.strip()
            return True
        return False
    if action not in {"replace", "remove"} or not isinstance(old_text, str) or not old_text.strip():
        return False

    matches = [index for index, entry in enumerate(entries) if old_text.strip() in entry]
    if len(matches) != 1:
        return False
    index = matches[0]
    if action == "remove":
        del entries[index]
        return True
    if not isinstance(content, str) or not content.strip():
        return False
    entries[index] = content.strip()
    return True


def _projected_chars(args: Dict[str, Any], store: Any) -> Optional[tuple[int, int]]:
    target = args.get("target", "memory")
    if target not in {"memory", "user"}:
        return None
    entries = list(store._entries_for(target))

    operations = args.get("operations")
    if operations is not None:
        if not isinstance(operations, list) or not operations:
            return None
        for operation in operations:
            if not isinstance(operation, dict) or not _apply_operation(entries, operation):
                return None
    else:
        if not _apply_operation(entries, args):
            return None

    from tools.memory_tool import ENTRY_DELIMITER

    projected = len(ENTRY_DELIMITER.join(entries))
    return projected, int(store._char_limit(target) * 0.975)


_MEM_ENTRY_SOFT_CAP = 200
_MEM_DUP_RATIO = 0.6
_LONG_FORM_PREFIX = "【长条目】"
_LONG_REMIND = "这条偏长了，像经过不像结论——看要不要收成一句。"
_DUP_REMIND = "旧条里有条很像的（开头是「{head}……」），下回整理时并一并。"


def _tri_shingles(s: str) -> set:
    import re as _re
    t = _re.sub(r"[\s，,。；;：:、（）()「」『』【】\[\]]", "", s)
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


def _entry_policy_reminders(content: str, target: str, skip: str = "") -> List[str]:
    """返回粒度提醒，不阻止写入；skip 用于跳过包含该原文的既有条目。

    读取既有条目失败时，仅返回已生成的超长提醒。
    """
    notes: List[str] = []
    c = (content or "").strip()
    if not c:
        return notes
    if len(c) > _MEM_ENTRY_SOFT_CAP and not c.startswith(_LONG_FORM_PREFIX):
        log_guard_block("entry_too_long", target, chars=len(c), cap=_MEM_ENTRY_SOFT_CAP)
        notes.append(_LONG_REMIND)
    try:
        from tools.memory_tool import load_on_disk_store
        entries = list(load_on_disk_store()._entries_for(target))
    except Exception:
        return notes
    mine = _tri_shingles(c)
    if not mine:
        return notes
    for i, e in enumerate(entries):
        if skip and skip in e:
            continue
        if e == c:
            continue
        other = _tri_shingles(e)
        if not other:
            continue
        inter = len(mine & other)
        ratio = inter / max(1, min(len(mine), len(other)))
        if ratio >= _MEM_DUP_RATIO:
            log_guard_block("entry_duplicate", target, overlap=round(ratio, 2), chars=len(c), index=i)
            notes.append(_DUP_REMIND.format(head=e[:14]))
            break
    return notes


def remind_memory_result(tool_name: str = "", args: Any = None, result: Any = None, **_: Any) -> Optional[str]:
    """写入成功后，对超长或近重复内容追加粒度提醒，不撤销写入。

    字符串返回值替换原工具结果，None 保持原结果；异常返回 None。"""
    if tool_name != "memory" or not isinstance(args, dict) or not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
        if not isinstance(payload, dict) or not payload.get("success"):
            return None
        target = str(args.get("target", "memory"))
        if target not in {"memory", "user"}:
            return None
        bodies: List[Tuple[str, str]] = []
        ops = args.get("operations")
        if isinstance(ops, list) and ops:
            for op in ops:
                if isinstance(op, dict) and str(op.get("action")) in ("add", "replace"):
                    bodies.append((str(op.get("content") or op.get("new_text") or ""),
                                   str(op.get("old_text") or "")))
        elif str(args.get("action")) in ("add", "replace"):
            bodies.append((str(args.get("content") or args.get("new_text") or ""),
                           str(args.get("old_text") or "")))
        notes: List[str] = []
        for body, old_text in bodies:
            notes.extend(_entry_policy_reminders(body, target, skip=old_text.strip()))
        if not notes:
            return None
        payload["reminder"] = "；".join(notes)
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return None


FILE_WRITE_TOOLS = frozenset({"write_file", "patch"})
_PATH_KEYS = ("path", "file_path", "filepath", "filename", "target_path", "output_path", "dest", "destination")
CONTROLLED_MEMORY_FILES = (
    ("memories/MEMORY.md", "MEMORY"),
    ("memories/USER.md", "USER"),
    ("memories/FOLLOWUP.md", "FOLLOWUP"),
    ("memories/OPS-NOTES.md", "OPS-NOTES"),
    ("BEHAVIOR.md", "BEHAVIOR"),
)
CONTROLLED_WRITE_MESSAGE = (
    "这个文件不直接落盘：写记忆请走 memory 工具的固定目标"
    "（memory／user／ops，迁移用 migrate）；"
    "BEHAVIOR.md 只能走人工／授权批次，不在模型写面内。"
)


def _real_path(value: Any) -> Optional[Path]:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        if text.startswith("~"):
            text = os.path.expanduser(text)
        return Path(os.path.realpath(text))
    except Exception:
        return None


def _controlled_roots() -> List[Path]:
    found: List[Path] = []
    try:
        found.append(Path(_resolve_hermes_home()).resolve())
    except Exception:
        pass
    try:
        override = (os.environ.get("HERMES_HOME_OVERRIDE") or "").strip()
        if override:
            candidate = Path(override).expanduser().resolve()
            if candidate not in found:
                found.append(candidate)
    except Exception:
        pass
    return found


def _controlled_memory_target(value: Any) -> Optional[str]:
    resolved = _real_path(value)
    if resolved is None:
        return "unknown"
    for root in _controlled_roots():
        for relative, logical in CONTROLLED_MEMORY_FILES:
            candidate = _real_path(os.path.join(str(root), relative))
            if candidate is None:
                continue
            if os.path.normcase(str(resolved)) == os.path.normcase(str(candidate)):
                return logical
    return None


def guard_controlled_memory_write(tool_name: str = "", args: Any = None,
                                  session_id: str = "", **_: Any) -> Optional[Dict[str, str]]:
    """按真实路径（含 symlink）限制 write_file/patch 对受控记忆文件的写入。"""
    try:
        name = str(tool_name or "")
        if name not in FILE_WRITE_TOOLS:
            return None
        targets = [args[k] for k in _PATH_KEYS if isinstance(args, dict) and k in args]
        if not targets:
            return {"action": "block", "message": CONTROLLED_WRITE_MESSAGE}
        for value in targets:
            logical = _controlled_memory_target(value)
            if logical is not None:
                return {"action": "block", "message": CONTROLLED_WRITE_MESSAGE}
        return None
    except Exception:
        if str(tool_name or "") in FILE_WRITE_TOOLS:
            return {"action": "block", "message": CONTROLLED_WRITE_MESSAGE}
        return None


def guard_memory_write(tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> Optional[Dict[str, str]]:
    """拦截常驻记忆容量超限写入；粒度问题由写入后的结果提醒处理。异常放行。"""
    if tool_name != "memory" or not isinstance(args, dict):
        return None
    try:
        from tools.memory_tool import load_on_disk_store

        estimate = _projected_chars(args, load_on_disk_store())
        if estimate is not None and estimate[0] > estimate[1]:
            log_guard_block("batch_over_limit", str(args.get("target", "memory")),
                            projected=estimate[0], limit=estimate[1])
            return {"action": "block", "message": _BLOCK_MESSAGE}
        return None
    except Exception as exc:
        logger.warning("sanae_mem: memory write guard failed open (%s)", type(exc).__name__)
        return None
