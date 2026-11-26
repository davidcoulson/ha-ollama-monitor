"""Monitor an Ollama server from Home Assistant."""

from __future__ import annotations

import importlib
import importlib.util
import sys

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_URL, Platform
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    issue_registry as ir,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.service import async_register_admin_service
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import BASE_PRELOAD_PLATFORMS

from .api import OllamaClient
from .const import CONF_CHECK_UPDATES, DEFAULT_CHECK_UPDATES, DOMAIN, ISSUE_UNREACHABLE
from .coordinator import (
    OllamaConfigEntry,
    OllamaCoordinator,
    OllamaRuntimeData,
    ReleaseCoordinator,
)
from .entity import model_identifier, server_device_info, server_identifier
from .services import async_setup_services

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.EVENT,
    Platform.SENSOR,
    Platform.UPDATE,
]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def _import_code(package: str) -> None:
    """Import the new code in the import executor, not the event loop.

    After an unload HA still lists "<domain>.<platform>" as loaded, so it
    would import the new modules in the event loop (and warn about it);
    once they are in sys.modules its own import is a cache hit.
    """
    module = importlib.import_module(package)
    # The new module's list, plus what HA preloads with a package (diagnostics, ...).
    for name in (*module.PLATFORMS, *BASE_PRELOAD_PLATFORMS):
        if importlib.util.find_spec(f"{package}.{name}") is not None:
            importlib.import_module(f"{package}.{name}")


def _forget_code(hass: HomeAssistant, package: str, domain: str) -> None:
    """Drop a package's modules so the next setup imports them from disk again.

    Home Assistant keeps its own copy of each imported integration module
    (loader.py: hass.data["components"], keyed "<domain>" and
    "<domain>.<platform>") next to Python's sys.modules; both have to go.
    """
    for cache, root in (
        (sys.modules, package),
        (hass.data.get("components", {}), domain),
        (hass.data.get("missing_platforms", {}), domain),
    ):
        for name in [n for n in cache if n == root or n.startswith(root + ".")]:
            del cache[name]
    importlib.invalidate_caches()  # or a file added since the last import may not be found


def _register_reload_code(hass: HomeAssistant) -> None:
    """Dev: ollama_monitor.reload_code loads the Python files from disk without a restart."""

    async def _reload_code(call: ServiceCall) -> None:
        # Disabled and ignored entries are never set up, so they are not ours to reload.
        entries = hass.config_entries.async_entries(
            DOMAIN, include_ignore=False, include_disabled=False
        )
        for entry in entries:
            if not await hass.config_entries.async_unload(entry.entry_id):
                raise HomeAssistantError(
                    f"{DOMAIN} did not unload; restart Home Assistant instead"
                )
        _forget_code(hass, __name__, DOMAIN)
        try:
            await hass.async_add_import_executor_job(_import_code, __name__)
            component = sys.modules[__name__]
            await component.async_setup(hass, {})
        except Exception as err:
            # This handler stays registered, so the action works again once the code is fixed.
            raise HomeAssistantError(
                f"{DOMAIN} is unloaded: the new code did not load ({err}). "
                f"Fix it and call {DOMAIN}.reload_code again"
            ) from err
        failed = [e.title for e in entries if not await hass.config_entries.async_setup(e.entry_id)]
        if failed:
            raise HomeAssistantError(f"{DOMAIN} did not set up again for {failed}; see the log")

    async_register_admin_service(hass, DOMAIN, "reload_code", _reload_code)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the integration's actions."""
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: OllamaConfigEntry) -> bool:
    """Set up Ollama Monitor from a config entry."""
    _register_reload_code(hass)
    client = OllamaClient(async_get_clientsession(hass), entry.data[CONF_URL])
    coordinator = OllamaCoordinator(hass, entry, client)
    release_coordinator = (
        ReleaseCoordinator(hass, entry)
        if entry.options.get(CONF_CHECK_UPDATES, DEFAULT_CHECK_UPDATES)
        else None
    )
    entry.runtime_data = OllamaRuntimeData(client, coordinator, release_coordinator)

    await coordinator.async_config_entry_first_refresh()
    if release_coordinator is not None:
        # GitHub being unreachable must not block monitoring the local server.
        entry.async_create_background_task(
            hass, release_coordinator.async_refresh(), f"{DOMAIN} release check"
        )

    dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id, **server_device_info(entry)
    )
    async_sync_devices(hass, entry)
    entry.async_on_unload(
        coordinator.async_add_listener(lambda: async_sync_devices(hass, entry))
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


@callback
def async_sync_devices(hass: HomeAssistant, entry: OllamaConfigEntry) -> None:
    """Keep the server version current and remove devices of vanished models."""
    coordinator = entry.runtime_data.coordinator
    if not coordinator.last_update_success or coordinator.data is None:
        return
    registry = dr.async_get(hass)
    if (
        server := registry.async_get_device_by_identifier(
            server_identifier(entry), entry.entry_id
        )
    ) and server.sw_version != coordinator.data.version:
        registry.async_update_device(server.id, sw_version=coordinator.data.version)

    current = {model_identifier(entry, name) for name in coordinator.data.models}
    for child in dr.async_child_entries_for_config_entry(registry, entry.entry_id):
        if not child.identifiers & current:
            registry.async_remove_device(child.id)
    # Model devices of versions before 0.3 are full devices; current ones are
    # converted to child devices when their entities are added.
    server_id = server_identifier(entry)
    for device in dr.async_entries_for_config_entry(registry, entry.entry_id):
        if server_id in device.identifiers or device.identifiers & current:
            continue
        registry.async_update_device(device.id, remove_config_entry_id=entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: OllamaConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: OllamaConfigEntry) -> None:
    """Remove the stored activity history of a deleted entry."""
    await OllamaCoordinator.async_remove_store_for(hass, entry.entry_id)
    ir.async_delete_issue(hass, DOMAIN, f"{ISSUE_UNREACHABLE}_{entry.entry_id}")


async def async_remove_config_entry_device(
    hass: HomeAssistant, entry: OllamaConfigEntry, device: dr.AnyDeviceEntry
) -> bool:
    """Allow deleting model devices that are no longer on the server."""
    if server_identifier(entry) in device.identifiers:
        return False
    # No model inventory while the entry is not loaded, so nothing can be removed.
    if entry.state is not ConfigEntryState.LOADED:
        return False
    data = entry.runtime_data.coordinator.data
    return not any(
        model_identifier(entry, name) in device.identifiers for name in data.models
    )
