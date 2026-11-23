from datetime import datetime as datetime, timedelta as timedelta
from pathlib import Path

from ._public_import import import_public_package

if __name__ != "sanae_mem":
    import_public_package(Path(__file__).resolve().parent)

from .bilibili_history import BILIBILI_SOURCE
from .infra.source_descriptors import install_builtin_sources

install_builtin_sources(BILIBILI_SOURCE)

from .governance.source_intake import _bind_source_intake_worker, submit_source_material
from .plugin_entry import SanaeMemProvider, register
from . import integrations as integrations
from .domains.memes import known_inbound_images
from .infra.textutil import _hash_embed as _hash_embed, _owner_key as _owner_key

_bind_source_intake_worker(SanaeMemProvider)

__all__ = ["SanaeMemProvider", "known_inbound_images", "register", "submit_source_material"]
