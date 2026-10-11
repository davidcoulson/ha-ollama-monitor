"""Binary sensors for Ollama Monitor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.components.binary_sensor.const import BinarySensorDeviceClass
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import ATTR_ACTIVE_MODELS, ATTR_LAST_ACTIVE_MODEL, ATTR_LAST_ACTIVITY
from .coordinator import ModelState, OllamaConfigEntry, OllamaCoordinator, OllamaData
from .entity import OllamaEntity, OllamaModelEntity, async_add_model_entities

PARALLEL_UPDATES = 0


@dataclass(frozen=True, kw_only=True)
class OllamaBinarySensorEntityDescription(BinarySensorEntityDescription):
    """Describes a binary sensor of the server device."""

    value_fn: Callable[[OllamaData], bool]
    attributes_fn: Callable[[OllamaData], Mapping[str, Any]] | None = None


@dataclass(frozen=True, kw_only=True)
class OllamaModelBinarySensorEntityDescription(BinarySensorEntityDescription):
    """Describes a binary sensor of a model device."""

    value_fn: Callable[[ModelState], bool]


def _activity_attributes(data: OllamaData) -> dict[str, Any]:
    """Return which models are active and which one was used last."""
    last = data.last_active_model
    return {
        ATTR_ACTIVE_MODELS: data.active_models,
        ATTR_LAST_ACTIVE_MODEL: last.name if last else None,
        ATTR_LAST_ACTIVITY: last.last_used if last else None,
    }


REACHABLE = BinarySensorEntityDescription(
    key="reachable",
    translation_key="reachable",
    device_class=BinarySensorDeviceClass.CONNECTIVITY,
    entity_category=EntityCategory.DIAGNOSTIC,
)

SERVER_BINARY_SENSORS: tuple[OllamaBinarySensorEntityDescription, ...] = (
    OllamaBinarySensorEntityDescription(
        key="active",
        translation_key="active",
        device_class=BinarySensorDeviceClass.RUNNING,
        value_fn=lambda data: data.active,
        attributes_fn=_activity_attributes,
    ),
    OllamaBinarySensorEntityDescription(
        key="cpu_offload",
        translation_key="cpu_offload",
        value_fn=lambda data: data.cpu_offload,
    ),
)

MODEL_BINARY_SENSORS: tuple[OllamaModelBinarySensorEntityDescription, ...] = (
    OllamaModelBinarySensorEntityDescription(
        key="loaded",
        translation_key="loaded",
        value_fn=lambda state: state.loaded is not None,
    ),
    OllamaModelBinarySensorEntityDescription(
        key="model_active",
        translation_key="model_active",
        device_class=BinarySensorDeviceClass.RUNNING,
        value_fn=lambda state: state.active,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the binary sensors."""
    coordinator = entry.runtime_data.coordinator
    async_add_entities(
        [
            OllamaReachableBinarySensor(coordinator, REACHABLE),
            *(
                OllamaBinarySensor(coordinator, description)
                for description in SERVER_BINARY_SENSORS
            ),
        ]
    )

    def _model_entities(
        coordinator: OllamaCoordinator, model: str
    ) -> list[OllamaModelBinarySensor]:
        return [
            OllamaModelBinarySensor(coordinator, description, model)
            for description in MODEL_BINARY_SENSORS
        ]

    async_add_model_entities(entry, async_add_entities, _model_entities)


class OllamaReachableBinarySensor(OllamaEntity, BinarySensorEntity):
    """Whether the last poll reached the server.

    Always available, so automations can react to the server going down.
    """

    @property
    def available(self) -> bool:
        """Return True; this entity reports the reachability itself."""
        return True

    @property
    def is_on(self) -> bool:
        """Return True if the server answered the last poll."""
        return self.coordinator.last_update_success


class OllamaBinarySensor(OllamaEntity, BinarySensorEntity):
    """A binary sensor of the server device."""

    entity_description: OllamaBinarySensorEntityDescription

    @property
    def is_on(self) -> bool:
        """Return the state."""
        return self.entity_description.value_fn(self.coordinator.data)

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return extra attributes."""
        if self.entity_description.attributes_fn is None:
            return None
        return self.entity_description.attributes_fn(self.coordinator.data)


class OllamaModelBinarySensor(OllamaModelEntity, BinarySensorEntity):
    """A binary sensor of a model device."""

    entity_description: OllamaModelBinarySensorEntityDescription

    @property
    def is_on(self) -> bool:
        """Return the state."""
        return self.entity_description.value_fn(self.model_state)
