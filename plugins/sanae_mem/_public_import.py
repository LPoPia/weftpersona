"""Make the public package importable from an installed plugin's physical path."""

from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence


class _PublicPackageFinder(MetaPathFinder):
    def __init__(self, parent: Path) -> None:
        self.parent = parent

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != "sanae_mem" or path is not None:
            return None
        return PathFinder.find_spec(fullname, [str(self.parent)], target)


def import_public_package(plugin_dir: Path) -> None:
    # The host loads under its own namespace and keeps the plugin parent off
    # sys.path. Let Python perform one ordinary import under the public name.
    finder = _PublicPackageFinder(plugin_dir.parent)
    sys.meta_path.append(finder)
    try:
        import_module("sanae_mem")
    finally:
        sys.meta_path.remove(finder)
