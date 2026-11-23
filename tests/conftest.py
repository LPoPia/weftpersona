"""Shared pytest path setup for ViveMind standalone and host-coupled suites."""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
PLUGIN_ROOT = REPO_ROOT / "plugins"

for _entry in (str(TESTS_ROOT), str(PLUGIN_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
