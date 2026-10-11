"""Model change events for Ollama Monitor."""

from __future__ import annotations

from homeassistant.components.event import EventEntity, EventEntityDescription
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import ATTR_MODEL, EVENT_TYPES
from .coordinator import OllamaConfigEntry, OllamaCoordinator, OllamaData
from .entity import OllamaEntity

PARALLEL_UPDATES = 0

MODEL_EVENT = EventEntityDescription(
    key="model_event", translation_key="model_event", event_types=EVENT_TYPES
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the event entity."""
    async_add_entities([OllamaModelEventEntity(entry.runtime_data.coordinator)])


class OllamaModelEventEntity(OllamaEntity, EventEntity):
    """Fires when models are loaded, unloaded, installed or removed."""

    def __init__(self, coordinator: OllamaCoordinator) -> None:
        """Initialize the entity."""
        super().__init__(coordinator, MODEL_EVENT)
        # Events of the snapshot that existed before this entity was added are
        # history, not news.
        self._handled: OllamaData | None = coordinator.data

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fire the events detected by the latest poll."""
        data = self.coordinator.data
        # Failed polls notify listeners with the previous snapshot; only fire once.
        events = data.events if data is not self._handled else ()
        self._handled = data
        # Write the availability change on its own first, so that triggers using
        # `not_from: unavailable` still see an event fired right after an outage.
        self.async_write_ha_state()
        for event in events:
            self._trigger_event(event.event_type, {ATTR_MODEL: event.model})
            self.async_write_ha_state()
