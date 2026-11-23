from sanae_mem.governance.runtime_registry import resolve_active_provider as resolve_active_provider
from .config_schema import CONFIG_SCHEMA as CONFIG_SCHEMA
from .infra.host_bridge import PluginRegistration
from .domains.memes import known_inbound_images as known_inbound_images
from .provider import SanaeMemProvider as SanaeMemProvider
from .runtime.hooks import register_runtime_hooks
from .governance.source_intake import submit_source_material as submit_source_material


def register(ctx: PluginRegistration[SanaeMemProvider]) -> None:
    provider = SanaeMemProvider()
    ctx.register_memory_provider(provider)
    register_runtime_hooks(ctx, provider)
