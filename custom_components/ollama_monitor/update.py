"""Ollama release update entity."""

from __future__ import annotations

from homeassistant.components.update import (
    UpdateEntity,
    UpdateEntityDescription,
    UpdateEntityFeature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import OllamaConfigEntry, ReleaseCoordinator
from .entity import server_device_info

PARALLEL_UPDATES = 0

SERVER_UPDATE = UpdateEntityDescription(key="update")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the update entity when release checks are enabled."""
    if (release_coordinator := entry.runtime_data.release_coordinator) is None:
        return
    async_add_entities([OllamaUpdateEntity(entry, release_coordinator)])


class OllamaUpdateEntity(CoordinatorEntity[ReleaseCoordinator], UpdateEntity):
    """Compare the server version with the latest Ollama release.

    Ollama is installed outside Home Assistant (Docker, system service), so this
    entity only notifies; it cannot install.
    """

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = UpdateEntityFeature.RELEASE_NOTES
    _attr_title = "Ollama"
    entity_description = SERVER_UPDATE

    def __init__(
        self, entry: OllamaConfigEntry, coordinator: ReleaseCoordinator
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._server = entry.runtime_data.coordinator
        self._attr_unique_id = f"{entry.entry_id}_update"
        self._attr_device_info = server_device_info(entry)

    async def async_added_to_hass(self) -> None:
        """Also follow the server version."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self._server.async_add_listener(self._handle_coordinator_update)
        )

    @property
    def available(self) -> bool:
        """Return True once both versions are known."""
        return (
            super().available
            and self.coordinator.data is not None
            and self._server.last_update_success
        )

    @property
    def installed_version(self) -> str | None:
        """Return the version running on the server."""
        return self._server.data.version

    @property
    def latest_version(self) -> str | None:
        """Return the latest released version."""
        return self.coordinator.data.version

    @property
    def release_url(self) -> str | None:
        """Return the release page."""
        return self.coordinator.data.url

    async def async_release_notes(self) -> str | None:
        """Return the release notes."""
        return self.coordinator.data.notes
