from __future__ import annotations

import importlib
import logging
import site
import sqlite3
import sys
from collections.abc import Callable
from threading import Lock, RLock
from datetime import datetime
from pathlib import Path
from typing import List, Protocol, TypedDict, runtime_checkable

from ..infra.words import _COLLECTION
from .sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)
_CHROMA_FALLBACK_LOCKS_GUARD = Lock()
_CHROMA_FALLBACK_LOCKS: dict[Path, Lock] = {}


def _chroma_fallback_lock(data_dir: Path) -> Lock:
    resolved = Path(data_dir).resolve()
    with _CHROMA_FALLBACK_LOCKS_GUARD:
        lock = _CHROMA_FALLBACK_LOCKS.get(resolved)
        if lock is None:
            lock = Lock()
            _CHROMA_FALLBACK_LOCKS[resolved] = lock
        return lock


def add_chroma_site_packages(data_dir: Path, configured: str) -> None:
    candidates = [Path(configured).expanduser()] if configured else []
    venv = data_dir / "venv"
    candidates.extend(
        [
            venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
            venv / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            site.addsitedir(str(candidate))


class _VectorMatches(TypedDict, total=False):
    ids: list[list[str]] | None


class _VectorRecords(TypedDict):
    ids: list[str]
    documents: list[str] | None


class VectorCollection(Protocol):
    def upsert(
        self, *, ids: list[str], documents: list[str], embeddings: list[list[float]],
        metadatas: list[dict[str, str | int | float]],
    ) -> None: ...

    def query(
        self, *, query_embeddings: list[list[float]], n_results: int,
        where: dict[str, dict[str, list[int]]] | None = None,
    ) -> _VectorMatches: ...

    def get(self, *, ids: list[str]) -> _VectorRecords: ...

    def count(self) -> int: ...

    def delete(self, *, ids: list[str]) -> None: ...


@runtime_checkable
class _ClosableChromaClient(Protocol):
    def close(self) -> None: ...


def _release_chroma_client(client: object) -> None:
    if isinstance(client, _ClosableChromaClient):
        client.close()


def chroma_backend_errors() -> tuple[type[Exception], ...]:
    try:
        errors = importlib.import_module("chromadb.errors")
    except ImportError:
        return (OSError, sqlite3.OperationalError)
    return (OSError, sqlite3.OperationalError, errors.InternalError, errors.NotFoundError,
            errors.InvalidDimensionException, errors.InvalidArgumentError)


def is_chroma_backend_failure(exc: Exception) -> bool:
    if isinstance(exc, (OSError, sqlite3.OperationalError)):
        return True
    errors = importlib.import_module("chromadb.errors")
    if isinstance(exc, errors.InvalidArgumentError):
        # The Rust backend reports an incompatible on-disk collection through this type.
        return str(exc).startswith("Collection expecting embedding with dimension of ")
    return isinstance(exc, (errors.InternalError, errors.NotFoundError, errors.InvalidDimensionException))


def _open_vectors(data_dir: Path, configured: str) -> tuple[object, VectorCollection] | None:
    try:
        add_chroma_site_packages(data_dir, configured)
        chromadb = importlib.import_module("chromadb")
    except (ImportError, OSError) as exc:
        logger.warning("sanae_mem: Chroma unavailable; using SQLite vectors (%s)", type(exc).__name__)
        return None

    try:
        chroma_path = data_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            collection = client.get_or_create_collection(
                name=_COLLECTION, metadata={"hnsw:space": "cosine"},
            )
        except BaseException:
            _release_chroma_client(client)
            raise
    except chroma_backend_errors() as exc:
        if not is_chroma_backend_failure(exc):
            raise
        logger.warning("sanae_mem: Chroma unavailable; using SQLite vectors (%s)", type(exc).__name__)
        return None
    return client, collection


def _upsert_vector(
    collection: VectorCollection, event_id: int, raw: str, topic: str,
    importance: float, ts: str, vector: list[float],
) -> None:
    try:
        collection.upsert(
            ids=[f"evt-{event_id}"],
            documents=[raw],
            embeddings=[vector],
            metadatas=[{"event_id": event_id, "ts": ts, "topic": topic, "importance": importance}],
        )
    except chroma_backend_errors() as exc:
        if not is_chroma_backend_failure(exc):
            raise
        logger.warning("sanae_mem: Chroma write failed; SQLite retained event (%s)", type(exc).__name__)


def upsert_chroma_event_record(
    data_dir: Path,
    event_id: int,
    raw: str,
    topic: str,
    importance: float,
    ts: str,
    vector: List[float],
    *,
    chroma_site_packages: str,
) -> None:
    resolved_dir = data_dir.resolve()
    with _chroma_fallback_lock(resolved_dir):
        vectors = _open_vectors(resolved_dir, chroma_site_packages)
        if vectors is not None:
            client, collection = vectors
            try:
                _upsert_vector(collection, event_id, raw, topic, importance, ts, vector)
            finally:
                _release_chroma_client(client)


class Storage:
    def __init__(
        self, sqlite: SqliteFacade, lock: RLock, *, current_data_dir: Callable[[], Path],
        chroma_enabled: bool, chroma_site_packages: str,
    ) -> None:
        self._sqlite = sqlite
        self._lock = lock
        self._current_data_dir = current_data_dir
        self._chroma_enabled = chroma_enabled
        self._chroma_site_packages = chroma_site_packages
        self._vectors: tuple[object, VectorCollection] | None = None

    def reconfigure(self, *, chroma_enabled: bool, chroma_site_packages: str) -> None:
        with self._lock:
            self._chroma_enabled = chroma_enabled
            self._chroma_site_packages = chroma_site_packages

    def open_sqlite(self, data_dir: Path, *, archive_dir: Path) -> None:
        from .database import open_database

        db_path = data_dir / "sanae.db"
        conn = open_database(db_path, archive_dir=archive_dir)
        self._sqlite.attach(conn, db_path=db_path)

    @staticmethod
    def backup_paths(data_dir: Path) -> List[str]:
        return [str(data_dir / "sanae.db"), str(data_dir / "chroma"),
                str(data_dir / "memes")]

    def ensure_vectors(self) -> None:
        with self._lock:
            if self._chroma_enabled and self._sqlite.is_open and self._vectors is None:
                self._vectors = _open_vectors(self._current_data_dir(), self._chroma_site_packages)

    def current_collection(self) -> VectorCollection | None:
        vectors = self._vectors
        return vectors[1] if vectors is not None else None

    def upsert_vector(
        self, event_id: int, raw: str, topic: str, importance: float, ts: str, vector: list[float],
    ) -> None:
        collection = self.current_collection()
        if collection is not None:
            _upsert_vector(collection, event_id, raw, topic, importance, ts, vector)

    def write_sample(self, text: str) -> None:
        date = datetime.now().astimezone().date().isoformat()
        path = self._current_data_dir() / "sample.log"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("sanae_mem: sample log read failed (%s)", type(exc).__name__)
            return
        if sum(1 for line in lines if line.startswith(date)) < 3:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{date}\t{text[:80]}\n")

    def close(self) -> None:
        self._sqlite.close()
        with self._lock:
            vectors = self._vectors
            self._vectors = None
            if vectors is not None:
                _release_chroma_client(vectors[0])
