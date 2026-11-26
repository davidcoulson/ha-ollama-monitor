"""The dev-only reload_code action swaps in the Python files on disk."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from custom_components.ollama_monitor.const import DOMAIN
from homeassistant.config_entries import ConfigEntries, ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.loader import Integration


async def test_reload_code_replaces_the_modules(hass, init_integration) -> None:
    entry = init_integration
    old = {n: m for n, m in sys.modules.items() if n.startswith("custom_components.ollama_monitor")}
    assert hass.services.has_service(DOMAIN, "reload_code")

    await hass.services.async_call(DOMAIN, "reload_code", {}, blocking=True)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert sys.modules["custom_components.ollama_monitor"] is not old["custom_components.ollama_monitor"]
    assert sys.modules["custom_components.ollama_monitor.const"] is not old["custom_components.ollama_monitor.const"]
    assert hass.services.has_service(DOMAIN, "reload_code")
    assert hass.services.has_service(DOMAIN, "unload_model")


@pytest.mark.parametrize(
    ("target", "result"),
    [
        (ConfigEntries, "async_unload"),
        (Integration, "async_get_component"),
        (ConfigEntries, "async_setup"),
    ],
)
async def test_reload_code_reports_failures(hass, init_integration, target, result) -> None:
    kwargs = {"return_value": False}
    if target is Integration:  # unload needs the old module; only the fresh import fails
        real = Integration.async_get_component

        async def kwargs_wrapper(self):
            if "custom_components.ollama_monitor" not in sys.modules:
                raise ImportError("boom")
            return await real(self)

        kwargs = {"new": kwargs_wrapper}
    with patch.object(target, result, **kwargs), pytest.raises(HomeAssistantError):
        await hass.services.async_call(DOMAIN, "reload_code", {}, blocking=True)
    # the action survives a failed reload, so it can be called again once the code is fixed
    assert hass.services.has_service(DOMAIN, "reload_code")
