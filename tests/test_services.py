"""Tests for the unload, pull and delete actions."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ollama_monitor.const import (
    CONF_ALLOW_MANAGEMENT,
    DOMAIN,
    SERVICE_DELETE_MODEL,
    SERVICE_PULL_MODEL,
    SERVICE_UNLOAD_MODEL,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from .conftest import FakeOllama

ENTRY_ID = "01TESTENTRY"


async def _call(hass: HomeAssistant, service: str, model: str = "qwen3:8b") -> None:
    await hass.services.async_call(
        DOMAIN,
        service,
        {"config_entry_id": ENTRY_ID, "model": model},
        blocking=True,
    )
    await hass.async_block_till_done(wait_background_tasks=True)


@pytest.fixture
def management(options: dict[str, Any]) -> dict[str, Any]:
    """Enable model management."""
    options[CONF_ALLOW_MANAGEMENT] = True
    return options


async def test_unload_action(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """Unload works without enabling model management."""
    await _call(hass, SERVICE_UNLOAD_MODEL, " qwen3:30b ")
    assert fake_ollama.calls("POST", "/api/generate") == [
        {"model": "qwen3:30b", "keep_alive": 0, "stream": False}
    ]


async def test_unknown_entry(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """An unknown or foreign entry is rejected."""
    other = MockConfigEntry(domain="other", entry_id="OTHER")
    other.add_to_hass(hass)
    for entry_id in ("missing", "OTHER"):
        with pytest.raises(ServiceValidationError) as err:
            await hass.services.async_call(
                DOMAIN,
                SERVICE_UNLOAD_MODEL,
                {"config_entry_id": entry_id, "model": "x"},
                blocking=True,
            )
        assert err.value.translation_key == "entry_not_found"


async def test_entry_not_loaded(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """A not-loaded entry is rejected."""
    await hass.config_entries.async_unload(init_integration.entry_id)
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, SERVICE_UNLOAD_MODEL)
    assert err.value.translation_key == "entry_not_loaded"


@pytest.mark.parametrize("service", [SERVICE_PULL_MODEL, SERVICE_DELETE_MODEL])
async def test_management_disabled(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    service: str,
) -> None:
    """Pull and delete are refused unless enabled in the options."""
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, service)
    assert err.value.translation_key == "management_disabled"
    assert not fake_ollama.calls("POST", "/api/pull")
    assert not fake_ollama.calls("DELETE", "/api/delete")


@pytest.mark.usefixtures("management")
async def test_delete(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """Delete removes the model and refreshes."""
    fake_ollama.installed = [
        m for m in fake_ollama.installed if m["name"] != "qwen3:8b"
    ]
    await _call(hass, SERVICE_DELETE_MODEL)
    assert fake_ollama.calls("DELETE", "/api/delete") == [{"model": "qwen3:8b"}]
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_vram") is None


@pytest.mark.usefixtures("management")
async def test_delete_errors(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """Delete errors are translated."""
    fake_ollama.response("DELETE", "/api/delete", 404, {"error": "not found"})
    with pytest.raises(ServiceValidationError) as err:
        await _call(hass, SERVICE_DELETE_MODEL, "nope")
    assert err.value.translation_key == "model_not_found"

    fake_ollama.responses.clear()
    fake_ollama.fail = True
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, SERVICE_DELETE_MODEL)
    assert err.value.translation_key == "action_cannot_connect"


@pytest.mark.usefixtures("management")
async def test_pull(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pull finishes in the background and refreshes the models."""
    fake_ollama.installed.append({"name": "granite4:tiny-h", "size": 1})
    await _call(hass, SERVICE_PULL_MODEL, "granite4:tiny-h")
    assert fake_ollama.calls("POST", "/api/pull") == [
        {"model": "granite4:tiny-h", "stream": True}
    ]
    assert "Pulling granite4:tiny-h on Ollama (10.2.6.50) finished: success" in (
        caplog.text
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_granite4_tiny_h_vram") is not None


@pytest.mark.usefixtures("management")
async def test_pull_empty_stream(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """A pull that returns nothing is an error."""
    fake_ollama.pull_lines = []
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, SERVICE_PULL_MODEL)
    assert err.value.translation_key == "action_failed"


@pytest.mark.usefixtures("management")
async def test_pull_ends_early(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stream that ends without success is logged as a failure."""
    fake_ollama.pull_lines = [b'{"status":"pulling manifest"}\n']
    await _call(hass, SERVICE_PULL_MODEL)
    assert "Pulling qwen3:8b on Ollama (10.2.6.50) failed" in caplog.text
    assert "finished" not in caplog.text


@pytest.mark.usefixtures("management")
async def test_pull_fails_immediately(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """An error in the first message is raised to the caller."""
    fake_ollama.pull_lines = [b'{"error":"pull model manifest: file does not exist"}\n']
    with pytest.raises(HomeAssistantError) as err:
        await _call(hass, SERVICE_PULL_MODEL, "nope:1b")
    assert err.value.translation_key == "action_failed"
    assert "file does not exist" in str(err.value)


@pytest.mark.usefixtures("management")
async def test_pull_fails_later(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An error after the download started is logged."""
    fake_ollama.pull_lines = [
        b'{"status":"pulling manifest"}\n',
        b'{"error":"disk full"}\n',
    ]
    await _call(hass, SERVICE_PULL_MODEL)
    assert "Pulling qwen3:8b on Ollama (10.2.6.50) failed: disk full" in caplog.text
