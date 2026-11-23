from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path


@contextmanager
def file_lock(path: Path):
    """Hold the same lock across processes until the file handle closes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def fsync_directory(path: Path) -> None:
    if os.name != "nt":
        with suppress(OSError):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous = target.stat()
    except FileNotFoundError:
        previous = None
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            if previous is not None:
                if os.name == "posix":
                    with suppress(PermissionError):
                        os.fchown(handle.fileno(), previous.st_uid, previous.st_gid)
                os.chmod(temporary, stat.S_IMODE(previous.st_mode))
            os.fsync(handle.fileno())
        if os.name == "nt":
            for attempt in range(8):
                try:
                    os.replace(temporary, target)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        else:
            os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_write(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(data, ensure_ascii=True, indent=2)
    atomic_write_text(path, text)
