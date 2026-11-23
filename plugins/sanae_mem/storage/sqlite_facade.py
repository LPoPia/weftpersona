from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Iterator, List, Literal, Mapping, Optional, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_BUSY_TIMEOUT_MS = 10000
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0

SqlParams = Sequence[Any] | Mapping[str, Any]
ConnectionMode = Literal["ro", "rw", "rwc"]


def configure_connection(
    conn: sqlite3.Connection,
    *,
    readonly: bool = False,
    wal: bool = True,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Apply standard PRAGMAs and Row factory to a SQLite connection."""
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    if readonly:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA query_only=ON")
        return conn
    if wal:
        for attempt in range(10):
            try:
                row = conn.execute("PRAGMA journal_mode").fetchone()
                mode = str(row[0]).lower() if row is not None else ""
                if mode == "wal":
                    break
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" not in msg and "busy" not in msg) or attempt == 9:
                    raise
                time.sleep(0.02 * (attempt + 1))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def connect_readonly(
    path: Path | str,
    *,
    timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    check_same_thread: bool = False,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a read-only SQLite connection with busy_timeout, foreign_keys, and query_only."""
    database_path = Path(path).resolve()
    conn = sqlite3.connect(
        database_path.as_uri() + "?mode=ro",
        uri=True,
        timeout=timeout,
        check_same_thread=check_same_thread,
    )
    try:
        configure_connection(
            conn,
            readonly=True,
            wal=False,
            busy_timeout_ms=busy_timeout_ms,
        )
    except BaseException:
        conn.close()
        raise
    return conn


def connect_readwrite(
    path: Path | str,
    *,
    timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    check_same_thread: bool = False,
    create: bool = True,
    wal: bool = True,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection with busy_timeout, WAL, and foreign_keys."""
    database_path = Path(path)
    if create:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(database_path),
            timeout=timeout,
            check_same_thread=check_same_thread,
        )
    else:
        conn = sqlite3.connect(
            database_path.resolve().as_uri() + "?mode=rw",
            uri=True,
            timeout=timeout,
            check_same_thread=check_same_thread,
        )
    try:
        configure_connection(
            conn,
            readonly=False,
            wal=wal,
            busy_timeout_ms=busy_timeout_ms,
        )
    except BaseException:
        conn.close()
        raise
    return conn


def open_sqlite_connection(
    path: Path | str,
    *,
    mode: ConnectionMode = "rwc",
    timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    check_same_thread: bool = False,
    wal: bool = True,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Unified read-only / read-write SQLite connection constructor."""
    if mode == "ro":
        return connect_readonly(
            path,
            timeout=timeout,
            check_same_thread=check_same_thread,
            busy_timeout_ms=busy_timeout_ms,
        )
    if mode == "rw":
        return connect_readwrite(
            path,
            timeout=timeout,
            check_same_thread=check_same_thread,
            create=False,
            wal=wal,
            busy_timeout_ms=busy_timeout_ms,
        )
    if mode == "rwc":
        return connect_readwrite(
            path,
            timeout=timeout,
            check_same_thread=check_same_thread,
            create=True,
            wal=wal,
            busy_timeout_ms=busy_timeout_ms,
        )
    raise ValueError(f"unsupported SQLite connection mode: {mode!r}")


class _ThreadReadState(threading.local):
    ro_scope: Optional[tuple[sqlite3.Connection, threading.RLock]] = None


class SqliteFacade:
    """Thread-safe SQLite access facade pairing an RLock with one connection.

    Re-entrant writes on the same thread use SQLite ``SAVEPOINT`` so an inner
    transaction never commits or rolls back an outer active transaction.
    ``thread_readonly()`` binds an independent read-only connection to the
    calling thread so background hot-path readers never share ``_conn`` or
    contend on the writer lock.
    """

    def __init__(
        self,
        *,
        lock: Optional[threading.RLock] = None,
        conn: Optional[sqlite3.Connection] = None,
        conn_getter: Optional[Callable[[], Optional[sqlite3.Connection]]] = None,
        conn_setter: Optional[Callable[[Optional[sqlite3.Connection]], None]] = None,
        db_path: Optional[Path | str] = None,
    ) -> None:
        self._lock: threading.RLock = lock if lock is not None else threading.RLock()
        self._meta_lock: threading.Lock = threading.Lock()
        self._owned_conn: Optional[sqlite3.Connection] = conn
        self._conn_getter = conn_getter
        self._conn_setter = conn_setter
        self._db_path: Optional[Path] = Path(db_path).resolve() if db_path is not None else None
        if self._db_path is None and conn is not None:
            self._db_path = self._inspect_db_path(conn)
        self._meta_open: bool = conn is not None
        self._meta_db_path: Optional[Path] = self._db_path
        self._tx_depth: int = 0
        self._sp_seq: int = 0
        self._thread_state = _ThreadReadState()

    @staticmethod
    def _inspect_db_path(conn: sqlite3.Connection) -> Optional[Path]:
        for row in conn.execute("PRAGMA database_list").fetchall():
            if str(row[1]) == "main":
                file_path = str(row[2] or "").strip()
                if file_path:
                    return Path(file_path).resolve()
                break
        return None

    def _current_conn(self) -> Optional[sqlite3.Connection]:
        if self._conn_getter is not None:
            return self._conn_getter()
        return self._owned_conn

    def _set_conn(self, conn: Optional[sqlite3.Connection]) -> None:
        self._owned_conn = conn
        if self._conn_setter is not None:
            self._conn_setter(conn)

    def _require_conn(self) -> sqlite3.Connection:
        conn = self._current_conn()
        if conn is None:
            raise RuntimeError("SQLite connection is not open")
        return conn

    def _resolve_db_path(self, conn: Optional[sqlite3.Connection]) -> Optional[Path]:
        if self._db_path is not None:
            return self._db_path
        if conn is None:
            return None
        resolved = self._inspect_db_path(conn)
        if resolved is not None:
            self._db_path = resolved
        return resolved

    @property
    def is_open(self) -> bool:
        if self._thread_state.ro_scope is not None:
            return True
        with self._meta_lock:
            if self._meta_open:
                return True
        if self._conn_getter is not None and self._lock.acquire(blocking=False):
            try:
                return self._current_conn() is not None
            finally:
                self._lock.release()
        return False

    @property
    def connection(self) -> Optional[sqlite3.Connection]:
        ro_scope = self._thread_state.ro_scope
        if ro_scope is not None:
            return ro_scope[0]
        with self._lock:
            return self._current_conn()

    @property
    def db_path(self) -> Optional[Path]:
        with self._meta_lock:
            if self._meta_db_path is not None:
                return self._meta_db_path
        with self._lock:
            resolved = self._resolve_db_path(self._current_conn())
        if resolved is not None:
            with self._meta_lock:
                if self._meta_open and self._meta_db_path is None:
                    self._meta_db_path = resolved
        return resolved

    @property
    def tx_depth(self) -> int:
        with self._lock:
            return self._tx_depth

    def attach(
        self,
        conn: Optional[sqlite3.Connection],
        *,
        db_path: Optional[Path | str] = None,
    ) -> None:
        with self._lock:
            self._set_conn(conn)
            resolved = Path(db_path).resolve() if db_path is not None else (
                self._inspect_db_path(conn) if conn is not None else None
            )
            self._db_path = resolved
            self._tx_depth = 0
            with self._meta_lock:
                self._meta_open = conn is not None
                self._meta_db_path = resolved

    def close(self) -> None:
        with self._meta_lock:
            self._meta_open = False
            self._meta_db_path = None
        with self._lock:
            conn = self._current_conn()
            if conn is not None:
                try:
                    if conn.in_transaction:
                        conn.rollback()
                finally:
                    conn.close()
                    self._set_conn(None)
            self._db_path = None
            self._tx_depth = 0

    def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        ro_scope = self._thread_state.ro_scope
        if ro_scope is not None:
            ro_conn, ro_lock = ro_scope
            with ro_lock:
                return fn(ro_conn)
        with self._lock:
            return fn(self._require_conn())

    def read_optional(self, fn: Callable[[Optional[sqlite3.Connection]], T]) -> T:
        ro_scope = self._thread_state.ro_scope
        if ro_scope is not None:
            ro_conn, ro_lock = ro_scope
            with ro_lock:
                return fn(ro_conn)
        with self._lock:
            return fn(self._current_conn())

    def fetchone(self, sql: str, params: SqlParams = ()) -> Optional[sqlite3.Row]:
        return self.read(lambda conn: conn.execute(sql, params).fetchone())

    def fetchall(self, sql: str, params: SqlParams = ()) -> List[sqlite3.Row]:
        return self.read(lambda conn: conn.execute(sql, params).fetchall())

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Acquire the facade's reentrant lock without opening a transaction."""
        with self._lock:
            yield

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self._thread_state.ro_scope is not None:
            raise RuntimeError("cannot open write transaction inside read-only thread scope")
        with self._lock:
            conn = self._require_conn()
            nested = self._tx_depth > 0 or conn.in_transaction
            if nested:
                self._sp_seq += 1
                savepoint = f"_sanae_sp_{self._sp_seq}"
                conn.execute(f"SAVEPOINT {savepoint}")
                self._tx_depth += 1
                try:
                    yield conn
                except BaseException:
                    self._tx_depth -= 1
                    if conn.in_transaction:
                        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
                else:
                    self._tx_depth -= 1
                    try:
                        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    except BaseException:
                        if conn.in_transaction:
                            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                        raise
            else:
                conn.execute("BEGIN IMMEDIATE")
                self._tx_depth += 1
                try:
                    yield conn
                except BaseException:
                    self._tx_depth -= 1
                    if conn.in_transaction:
                        conn.rollback()
                    raise
                else:
                    self._tx_depth -= 1
                    if conn.in_transaction:
                        try:
                            conn.commit()
                        except BaseException:
                            if conn.in_transaction:
                                conn.rollback()
                            raise

    def write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        with self.transaction() as conn:
            return fn(conn)

    def execute(self, sql: str, params: SqlParams = ()) -> sqlite3.Cursor:
        with self.transaction() as conn:
            return conn.execute(sql, params)

    @contextmanager
    def thread_readonly(
        self,
        *,
        deadline: Optional[float] = None,
    ) -> Iterator[Optional[sqlite3.Connection]]:
        if self._thread_state.ro_scope is not None:
            yield self._thread_state.ro_scope[0]
            return
        with self._meta_lock:
            meta_open = self._meta_open
            db_path = self._meta_db_path
        if not meta_open:
            if self._conn_getter is not None:
                with self._lock:
                    conn = self._require_conn()
                    db_path = self._resolve_db_path(conn)
            else:
                raise RuntimeError("SQLite connection is not open")
        if db_path is None or not db_path.is_file():
            yield None
            return
        ro_conn = connect_readonly(db_path)
        if deadline is not None:
            ro_conn.set_progress_handler(
                lambda: 1 if time.monotonic() >= deadline else 0,
                5000,
            )
        ro_lock = threading.RLock()
        self._thread_state.ro_scope = (ro_conn, ro_lock)
        try:
            yield ro_conn
        finally:
            self._thread_state.ro_scope = None
            if deadline is not None:
                ro_conn.set_progress_handler(None, 0)
            ro_conn.close()
