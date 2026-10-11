"""Sensors for Ollama Monitor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfInformation
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import ATTR_MODELS
from .coordinator import ModelState, OllamaConfigEntry, OllamaCoordinator, OllamaData
from .entity import OllamaEntity, OllamaModelEntity, async_add_model_entities

PARALLEL_UPDATES = 0

type SensorValue = str | int | float | datetime | None


@dataclass(frozen=True, kw_only=True)
class OllamaSensorEntityDescription(SensorEntityDescription):
    """Describes a sensor of the server device."""

    value_fn: Callable[[OllamaData], SensorValue]
    attributes_fn: Callable[[OllamaData], Mapping[str, Any]] | None = None


@dataclass(frozen=True, kw_only=True)
class OllamaModelSensorEntityDescription(SensorEntityDescription):
    """Describes a sensor of a model device."""

    value_fn: Callable[[ModelState], SensorValue]


def _data_size(key: str, **kwargs: Any) -> dict[str, Any]:
    """Return the common settings of a data size sensor."""
    return {
        "key": key,
        "translation_key": key,
        "device_class": SensorDeviceClass.DATA_SIZE,
        "native_unit_of_measurement": UnitOfInformation.BYTES,
        "suggested_unit_of_measurement": UnitOfInformation.GIGABYTES,
        "suggested_display_precision": 2,
        "state_class": SensorStateClass.MEASUREMENT,
        **kwargs,
    }


SERVER_SENSORS: tuple[OllamaSensorEntityDescription, ...] = (
    OllamaSensorEntityDescription(
        key="loaded_models",
        translation_key="loaded_models",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: len(data.loaded_names),
        attributes_fn=lambda data: {ATTR_MODELS: data.loaded_names},
    ),
    OllamaSensorEntityDescription(
        key="installed_models",
        translation_key="installed_models",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.installed_count,
    ),
    OllamaSensorEntityDescription(
        **_data_size("vram_used"), value_fn=lambda data: data.vram_used
    ),
    OllamaSensorEntityDescription(
        **_data_size("memory_used"), value_fn=lambda data: data.memory_used
    ),
    OllamaSensorEntityDescription(
        key="gpu_share",
        translation_key="gpu_share",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        value_fn=lambda data: data.gpu_share,
    ),
    OllamaSensorEntityDescription(
        **_data_size(
            "disk_usage",
            entity_category=EntityCategory.DIAGNOSTIC,
        ),
        value_fn=lambda data: data.disk_usage,
    ),
    OllamaSensorEntityDescription(
        key="last_activity",
        translation_key="last_activity",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda data: data.last_activity,
    ),
    OllamaSensorEntityDescription(
        key="version",
        translation_key="version",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: data.version,
    ),
)


MODEL_SENSORS: tuple[OllamaModelSensorEntityDescription, ...] = (
    OllamaModelSensorEntityDescription(
        **_data_size("model_memory"),
        value_fn=lambda state: state.loaded.size if state.loaded else 0,
    ),
    OllamaModelSensorEntityDescription(
        **_data_size("model_vram"),
        value_fn=lambda state: state.loaded.size_vram if state.loaded else 0,
    ),
    OllamaModelSensorEntityDescription(
        key="model_gpu_share",
        translation_key="model_gpu_share",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        value_fn=lambda state: state.loaded.gpu_share if state.loaded else None,
    ),
    OllamaModelSensorEntityDescription(
        key="context_length",
        translation_key="context_length",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda state: state.loaded.context_length if state.loaded else None,
    ),
    OllamaModelSensorEntityDescription(
        key="unloads_at",
        translation_key="unloads_at",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda state: state.loaded.unloads_at if state.loaded else None,
    ),
    OllamaModelSensorEntityDescription(
        key="last_used",
        translation_key="last_used",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda state: state.last_used,
    ),
    OllamaModelSensorEntityDescription(
        **_data_size(
            "size_on_disk",
            state_class=None,
            entity_category=EntityCategory.DIAGNOSTIC,
            entity_registry_enabled_default=False,
        ),
        value_fn=lambda state: state.installed.size if state.installed else None,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the sensors."""
    coordinator = entry.runtime_data.coordinator
    async_add_entities(
        OllamaSensor(coordinator, description) for description in SERVER_SENSORS
    )

    def _model_entities(
        coordinator: OllamaCoordinator, model: str
    ) -> list[OllamaModelSensor]:
        return [
            OllamaModelSensor(coordinator, description, model)
            for description in MODEL_SENSORS
        ]

    async_add_model_entities(entry, async_add_entities, _model_entities)


class OllamaSensor(OllamaEntity, SensorEntity):
    """A sensor of the server device."""

    entity_description: OllamaSensorEntityDescription

    @property
    def native_value(self) -> SensorValue:
        """Return the state."""
        return self.entity_description.value_fn(self.coordinator.data)

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return extra attributes."""
        if self.entity_description.attributes_fn is None:
            return None
        return self.entity_description.attributes_fn(self.coordinator.data)


class OllamaModelSensor(OllamaModelEntity, SensorEntity):
    """A sensor of a model device."""

    entity_description: OllamaModelSensorEntityDescription

    @property
    def native_value(self) -> SensorValue:
        """Return the state."""
        return self.entity_description.value_fn(self.model_state)
