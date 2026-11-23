from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ..infra.source_descriptors import receipt_tables
from ..storage.sqlite_facade import connect_readonly


@dataclass(frozen=True)
class SourceReceipt:
    delivered_at: float
    label: str
    text: str


def read_source_receipts(
    data_dir: Path | str | None, owners: tuple[str, ...], since: float, through: float,
) -> list[SourceReceipt]:
    """Read historical source sends from ViveMind's own database."""
    if data_dir is None or not owners:
        return []
    db = Path(data_dir).resolve() / "sanae.db"
    if not db.is_file():
        return []
    receipts: list[SourceReceipt] = []
    with closing(connect_readonly(db)) as conn:
        for layout in receipt_tables():
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (layout.table,)).fetchone() is None:
                continue
            placeholders = ",".join("?" for _ in owners)
            rows = conn.execute(
                f"SELECT at, visibility, text FROM {layout.table} "
                f"WHERE owner_key IN ({placeholders}) AND at>? AND at<=?",
                (*owners, since, through),
            )
            receipts.extend(
                SourceReceipt(stamp, layout.private_label if visibility == "private"
                              else layout.public_label, text)
                for stamp, visibility, text in rows
            )
    return receipts
