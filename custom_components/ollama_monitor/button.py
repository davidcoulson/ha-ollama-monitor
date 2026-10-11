"""Buttons for Ollama Monitor."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import OllamaConfigEntry, OllamaCoordinator
from .entity import OllamaModelEntity, async_add_model_entities
from .services import async_unload_model

PARALLEL_UPDATES = 1

UNLOAD = ButtonEntityDescription(key="unload", translation_key="unload")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the buttons."""

    def _model_entities(
        coordinator: OllamaCoordinator, model: str
    ) -> list[OllamaUnloadButton]:
        return [OllamaUnloadButton(coordinator, UNLOAD, model)]

    async_add_model_entities(entry, async_add_entities, _model_entities)


class OllamaUnloadButton(OllamaModelEntity, ButtonEntity):
    """Unload a model from memory."""

    @property
    def available(self) -> bool:
        """Return True only while the model is loaded."""
        return super().available and self.model_state.loaded is not None

    async def async_press(self) -> None:
        """Unload the model."""
        await async_unload_model(self.coordinator.config_entry, self.model)
