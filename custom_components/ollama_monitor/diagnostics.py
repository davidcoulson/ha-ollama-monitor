"""Diagnostics for Ollama Monitor."""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant

from .coordinator import OllamaConfigEntry

TO_REDACT = {CONF_URL, "parent_model"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: OllamaConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    runtime = entry.runtime_data
    release = runtime.release_coordinator
    return {
        "entry": {
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": dict(entry.options),
        },
        "last_update_success": runtime.coordinator.last_update_success,
        "api_responses": async_redact_data(runtime.client.last_responses, TO_REDACT),
        "activity": runtime.coordinator.diagnostics(),
        "latest_release": (
            release.data.version if release is not None and release.data else None
        ),
    }
