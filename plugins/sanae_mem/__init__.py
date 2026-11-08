"""Sanae's local, fail-open memory provider for Hermes.

Implements the MemoryProvider contract (see agent.memory_provider); ships
SanaeMemProvider + register(). Implementation lives in sibling modules."""

from datetime import datetime, timedelta

try:
    from .plugin_entry import SanaeMemProvider, register
    from .memes import known_inbound_images
    from .textutil import _hash_embed, _owner_key
except ImportError:
    from plugin_entry import SanaeMemProvider, register
    from memes import known_inbound_images
    from textutil import _hash_embed, _owner_key

__all__ = ["SanaeMemProvider", "known_inbound_images", "register"]
