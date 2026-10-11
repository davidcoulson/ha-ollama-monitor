"""Tests for the update, button and event entities and diagnostics."""

from __future__ import annotations

from datetime import timedelta
from http import HTTPStatus
from typing import Any

from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)
from pytest_homeassistant_custom_component.typing import (
    ClientSessionGenerator,
    WebSocketGenerator,
)

from custom_components.ollama_monitor.const import CONF_CHECK_UPDATES
from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN, SERVICE_PRESS
from homeassistant.const import (
    ATTR_ENTITY_ID,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from .conftest import FakeOllama

UPDATE_ENTITY = "update.ollama_10_2_6_50"


async def _poll(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, seconds: int = 11
) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)


async def test_update_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    hass_ws_client: WebSocketGenerator,
) -> None:
    """The update entity compares the server with the latest release."""
    await hass.async_block_till_done(wait_background_tasks=True)
    state = hass.states.get(UPDATE_ENTITY)
    assert state.state == STATE_ON
    assert state.attributes["installed_version"] == "0.40.2"
    assert state.attributes["latest_version"] == "0.41.0"
    assert state.attributes["release_url"].endswith("/v0.41.0")
    assert state.attributes["title"] == "Ollama"

    client = await hass_ws_client(hass)
    await client.send_json_auto_id(
        {"type": "update/release_notes", "entity_id": UPDATE_ENTITY}
    )
    result = await client.receive_json()
    assert result["result"] == "## What's changed\n- Faster"


async def test_update_entity_follows_server(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """Upgrading the server clears the update; an outage makes it unavailable."""
    await hass.async_block_till_done(wait_background_tasks=True)
    fake_ollama.version = "0.41.0"
    await _poll(hass, frozen_time)
    assert hass.states.get(UPDATE_ENTITY).state == STATE_OFF

    fake_ollama.fail = True
    await _poll(hass, frozen_time)
    assert hass.states.get(UPDATE_ENTITY).state == STATE_UNAVAILABLE


@pytest.mark.parametrize(
    ("status", "release"),
    [
        (HTTPStatus.FORBIDDEN, {"message": "rate limited"}),
        (HTTPStatus.OK, {"name": "no tag"}),
        (HTTPStatus.OK, ["not", "a", "dict"]),
    ],
)
async def test_update_entity_github_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    status: HTTPStatus,
    release: Any,
) -> None:
    """GitHub failures leave monitoring working and the update unavailable."""
    fake_ollama.release_status = status
    fake_ollama.release = release
    mock_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get(UPDATE_ENTITY).state == STATE_UNAVAILABLE
    assert hass.states.get("sensor.ollama_10_2_6_50_version").state == "0.40.2"


async def test_update_entity_optional_fields(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """A release without URL or notes is still usable."""
    fake_ollama.release = {"tag_name": "0.40.2"}
    mock_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    state = hass.states.get(UPDATE_ENTITY)
    assert state.state == STATE_OFF
    assert state.attributes["release_url"] is None


@pytest.mark.parametrize("options", [{CONF_CHECK_UPDATES: False}])
async def test_update_entity_disabled(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Without release checks there is no update entity."""
    assert hass.states.get(UPDATE_ENTITY) is None


async def test_unload_button(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
) -> None:
    """The unload button sends keep_alive 0; it is unavailable when unloaded."""
    assert hass.states.get("button.qwen3_30b_unload").state == STATE_UNKNOWN
    assert hass.states.get("button.qwen3_8b_unload").state == STATE_UNAVAILABLE

    await hass.services.async_call(
        BUTTON_DOMAIN,
        SERVICE_PRESS,
        {ATTR_ENTITY_ID: "button.qwen3_30b_unload"},
        blocking=True,
    )
    assert fake_ollama.calls("POST", "/api/generate") == [
        {"model": "qwen3:30b", "keep_alive": 0, "stream": False}
    ]


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (
            lambda fake: fake.response("POST", "/api/generate", 500, {"error": "x"}),
            "The Ollama server refused the request for qwen3:30b: x",
        ),
        (
            lambda fake: fake.response(
                "POST", "/api/generate", 404, {"error": "model not found"}
            ),
            "The model qwen3:30b was not found on the server",
        ),
    ],
)
async def test_unload_button_errors(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    setup: Any,
    error: str,
) -> None:
    """Server errors become translated Home Assistant errors."""
    setup(fake_ollama)
    with pytest.raises(HomeAssistantError, match=error):
        await hass.services.async_call(
            BUTTON_DOMAIN,
            SERVICE_PRESS,
            {ATTR_ENTITY_ID: "button.qwen3_30b_unload"},
            blocking=True,
        )


async def test_unload_button_connection_error(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
) -> None:
    """An unreachable server is reported."""
    fake_ollama.fail = True
    with pytest.raises(HomeAssistantError) as err:
        await hass.services.async_call(
            BUTTON_DOMAIN,
            SERVICE_PRESS,
            {ATTR_ENTITY_ID: "button.qwen3_30b_unload"},
            blocking=True,
        )
    assert not isinstance(err.value, ServiceValidationError)
    assert err.value.translation_key == "action_cannot_connect"


async def test_event_entity_initial_state(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The first poll is a baseline and fires nothing."""
    state = hass.states.get("event.ollama_10_2_6_50_model_event")
    assert state.state == STATE_UNKNOWN
    assert state.attributes["event_types"] == [
        "model_loaded",
        "model_unloaded",
        "model_installed",
        "model_removed",
    ]


async def test_diagnostics(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    hass_client: ClientSessionGenerator,
) -> None:
    """Diagnostics include the raw responses with the URL redacted."""
    await hass.async_block_till_done(wait_background_tasks=True)
    diag = await get_diagnostics_for_config_entry(hass, hass_client, init_integration)
    assert diag["entry"]["data"] == {"url": "**REDACTED**"}
    assert diag["last_update_success"] is True
    assert diag["latest_release"] == "0.41.0"
    assert set(diag["api_responses"]) == {"/api/version", "/api/ps", "/api/tags"}
    tags = diag["api_responses"]["/api/tags"]["models"]
    assert tags[1]["details"]["parent_model"] == "**REDACTED**"
    assert diag["activity"] == {
        "last_used": {},
        "expires_at": {"qwen3:30b": "2026-10-11T02:11:57.453437+00:00"},
        "unreachable_since": None,
    }


@pytest.mark.parametrize("options", [{CONF_CHECK_UPDATES: False}])
async def test_diagnostics_without_release(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    hass_client: ClientSessionGenerator,
) -> None:
    """Diagnostics work without release checks and while unreachable."""
    fake_ollama.loaded[0]["expires_at"] = None
    await _poll(hass, frozen_time)
    fake_ollama.fail = True
    await _poll(hass, frozen_time)
    diag = await get_diagnostics_for_config_entry(hass, hass_client, init_integration)
    assert diag["latest_release"] is None
    assert diag["last_update_success"] is False
    assert diag["activity"]["unreachable_since"] is not None
    assert diag["activity"]["expires_at"] == {"qwen3:30b": None}
