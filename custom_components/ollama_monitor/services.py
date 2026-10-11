"""Actions for Ollama Monitor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import probatio

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .api import OllamaConnectionError, OllamaError, OllamaModelNotFoundError
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_MODEL,
    CONF_ALLOW_MANAGEMENT,
    DEFAULT_ALLOW_MANAGEMENT,
    DOMAIN,
    LOGGER,
    SERVICE_DELETE_MODEL,
    SERVICE_PULL_MODEL,
    SERVICE_UNLOAD_MODEL,
)
from .coordinator import OllamaConfigEntry

SERVICE_SCHEMA = probatio.Schema(
    {
        probatio.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        probatio.Required(ATTR_MODEL): probatio.All(
            cv.string, probatio.Strip, probatio.Length(min=1)
        ),
    }
)


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the actions."""
    hass.services.async_register(
        DOMAIN, SERVICE_UNLOAD_MODEL, _async_handle_unload, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_PULL_MODEL, _async_handle_pull, schema=SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_DELETE_MODEL, _async_handle_delete, schema=SERVICE_SCHEMA
    )


def _get_entry(call: ServiceCall) -> OllamaConfigEntry:
    entry_id: str = call.data[ATTR_CONFIG_ENTRY_ID]
    entry = call.hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_found",
            translation_placeholders={"entry_id": entry_id},
        )
    if entry.state is not ConfigEntryState.LOADED:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_loaded",
            translation_placeholders={"title": entry.title},
        )
    return entry


def _require_management(entry: OllamaConfigEntry) -> None:
    if not entry.options.get(CONF_ALLOW_MANAGEMENT, DEFAULT_ALLOW_MANAGEMENT):
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="management_disabled",
            translation_placeholders={"title": entry.title},
        )


def _raise_action_error(err: OllamaError, model: str) -> None:
    """Translate an API error into the right Home Assistant exception."""
    if isinstance(err, OllamaModelNotFoundError):
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="model_not_found",
            translation_placeholders={"model": model},
        ) from err
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key=(
            "action_cannot_connect"
            if isinstance(err, OllamaConnectionError)
            else "action_failed"
        ),
        translation_placeholders={"model": model, "error": str(err)},
    ) from err


async def async_unload_model(entry: OllamaConfigEntry, model: str) -> None:
    """Unload a model and refresh the data."""
    try:
        await entry.runtime_data.client.async_unload_model(model)
    except OllamaError as err:
        _raise_action_error(err, model)
    await entry.runtime_data.coordinator.async_request_refresh()


async def _async_handle_unload(call: ServiceCall) -> None:
    await async_unload_model(_get_entry(call), call.data[ATTR_MODEL])


async def _async_handle_delete(call: ServiceCall) -> None:
    entry = _get_entry(call)
    _require_management(entry)
    model: str = call.data[ATTR_MODEL]
    try:
        await entry.runtime_data.client.async_delete_model(model)
    except OllamaError as err:
        _raise_action_error(err, model)
    await entry.runtime_data.coordinator.async_request_refresh()


async def _async_handle_pull(call: ServiceCall) -> None:
    """Start a pull; the download continues in the background.

    The first progress message is awaited so that an unknown model name or an
    unreachable server is reported to the caller right away.
    """
    entry = _get_entry(call)
    _require_management(entry)
    model: str = call.data[ATTR_MODEL]
    progress = entry.runtime_data.client.async_pull_model(model)
    try:
        first = await anext(progress, None)
    except OllamaError as err:
        _raise_action_error(err, model)
    if first is None:
        await entry.runtime_data.coordinator.async_request_refresh()
        return
    LOGGER.info("Pulling %s on %s", model, entry.title)
    entry.async_create_background_task(
        call.hass,
        _async_finish_pull(entry, model, progress, first),
        f"{DOMAIN} pull {model}",
    )


async def _async_finish_pull(
    entry: OllamaConfigEntry,
    model: str,
    progress: AsyncIterator[dict[str, Any]],
    last: dict[str, Any],
) -> None:
    try:
        async for message in progress:
            last = message
            LOGGER.debug("Pull %s on %s: %s", model, entry.title, message)
    except OllamaError as err:
        LOGGER.error("Pulling %s on %s failed: %s", model, entry.title, err)
        return
    LOGGER.info("Pulling %s on %s finished: %s", model, entry.title, last.get("status"))
    await entry.runtime_data.coordinator.async_request_refresh()
