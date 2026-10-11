"""Base entities for Ollama Monitor."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import ModelState, OllamaConfigEntry, OllamaCoordinator


def server_identifier(entry: OllamaConfigEntry) -> tuple[str, str]:
    """Return the device identifier of the server device."""
    return (DOMAIN, entry.entry_id)


def model_identifier(entry: OllamaConfigEntry, model: str) -> tuple[str, str]:
    """Return the device identifier of a model device."""
    return (DOMAIN, f"{entry.entry_id}_model_{model}")


def server_device_info(entry: OllamaConfigEntry) -> DeviceInfo:
    """Return the device info of the server device."""
    coordinator = entry.runtime_data.coordinator
    return DeviceInfo(
        identifiers={server_identifier(entry)},
        name=entry.title,
        manufacturer="Ollama",
        model="Ollama server",
        sw_version=coordinator.data.version if coordinator.data else None,
        configuration_url=coordinator.client.url,
    )


class OllamaEntity(CoordinatorEntity[OllamaCoordinator]):
    """An entity of the Ollama server device."""

    _attr_has_entity_name = True

    def __init__(
        self, coordinator: OllamaCoordinator, description: EntityDescription
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self.entity_description = description
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = server_device_info(entry)


class OllamaModelEntity(CoordinatorEntity[OllamaCoordinator]):
    """An entity of a model device."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: OllamaCoordinator,
        description: EntityDescription,
        model: str,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self.entity_description = description
        self.model = model
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_model_{model}_{description.key}"
        state = coordinator.data.models[model]
        self._attr_device_info = DeviceInfo(
            identifiers={model_identifier(entry, model)},
            name=model,
            manufacturer="Ollama",
            model=state.device_model,
            sw_version=state.digest[:12] if state.digest else None,
            via_device_id=dr.async_get_device_id_by_identifier(
                coordinator.hass,
                server_identifier(entry),
                config_entry_id=entry.entry_id,
            ),
        )

    @property
    def available(self) -> bool:
        """Return True while the model exists on the server."""
        return super().available and self.model in self.coordinator.data.models

    @property
    def model_state(self) -> ModelState:
        """Return the current state of the model."""
        return self.coordinator.data.models[self.model]


@callback
def async_add_model_entities(
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
    factory: Callable[[OllamaCoordinator, str], Iterable[Entity]],
) -> None:
    """Add entities for every model now and whenever new models appear.

    Entities of models that disappear are removed together with their device
    (see `async_sync_devices`); forgetting them here lets them come back.
    """
    coordinator = entry.runtime_data.coordinator
    known: set[str] = set()

    @callback
    def _async_check() -> None:
        if not coordinator.last_update_success or coordinator.data is None:
            return
        current = set(coordinator.data.models)
        if new := current - known:
            async_add_entities(
                entity
                for model in sorted(new)
                for entity in factory(coordinator, model)
            )
        known.clear()
        known.update(current)

    _async_check()
    entry.async_on_unload(coordinator.async_add_listener(_async_check))
