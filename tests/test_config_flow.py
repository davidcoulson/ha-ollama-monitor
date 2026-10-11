"""Tests for the config, reconfigure and options flows."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ollama_monitor.config_flow import (
    InvalidUrlError,
    normalize_url,
)
from custom_components.ollama_monitor.const import (
    CONF_ACTIVITY_WINDOW,
    CONF_ALLOW_MANAGEMENT,
    CONF_CHECK_UPDATES,
    CONF_SCAN_INTERVAL,
    DOMAIN,
)
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from .conftest import URL_BASE, FakeOllama


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.2.6.50", "http://10.2.6.50:11434"),
        ("  ollama.local  ", "http://ollama.local:11434"),
        ("10.2.6.50:8080", "http://10.2.6.50:8080"),
        ("http://10.2.6.50:11434/", "http://10.2.6.50:11434"),
        ("https://AI.Example.com", "https://ai.example.com"),
        ("https://u:p@proxy/ollama/?x=1#f", "https://proxy/ollama"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    """URLs are normalized."""
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", ["ftp://host", "http://", "http://[::1"])
def test_normalize_url_invalid(raw: str) -> None:
    """Unusable URLs are rejected."""
    with pytest.raises(InvalidUrlError):
        normalize_url(raw)


async def test_user_flow(hass: HomeAssistant, fake_ollama: FakeOllama) -> None:
    """A reachable server creates an entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.ollama_monitor.async_setup_entry", return_value=True
    ) as mock_setup:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: "10.2.6.50"}
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Ollama (10.2.6.50)"
    assert result["data"] == {CONF_URL: URL_BASE}
    assert result["result"].unique_id == URL_BASE
    assert len(mock_setup.mock_calls) == 1


@pytest.mark.parametrize(
    ("setup", "url", "error"),
    [
        (
            lambda fake: setattr(fake, "fail", True),
            URL_BASE,
            {"base": "cannot_connect"},
        ),
        (
            lambda fake: fake.response("GET", "/api/version", 200, "Ollama"),
            URL_BASE,
            {"base": "not_ollama"},
        ),
        (lambda fake: None, "ftp://x", {CONF_URL: "invalid_url"}),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    fake_ollama: FakeOllama,
    setup: Any,
    url: str,
    error: dict[str, str],
) -> None:
    """Errors are shown and the flow can recover."""
    setup(fake_ollama)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: url}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == error

    fake_ollama.fail = False
    fake_ollama.responses.clear()
    with patch("custom_components.ollama_monitor.async_setup_entry", return_value=True):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: URL_BASE}
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_user_flow_unknown_error(
    hass: HomeAssistant, fake_ollama: FakeOllama
) -> None:
    """Unexpected exceptions are reported as unknown."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    with patch(
        "custom_components.ollama_monitor.config_flow.OllamaClient.async_get_version",
        side_effect=RuntimeError,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: URL_BASE}
        )
    assert result["errors"] == {"base": "unknown"}


async def test_user_flow_already_configured(
    hass: HomeAssistant, fake_ollama: FakeOllama, mock_config_entry: MockConfigEntry
) -> None:
    """The same server cannot be added twice."""
    mock_config_entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: "10.2.6.50"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reconfigure(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """Reconfigure changes the URL, unique ID and default title."""
    result = await init_integration.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    # Same server, written differently: allowed and keeps the entry.
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: "10.2.6.50"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert init_integration.data[CONF_URL] == URL_BASE
    assert init_integration.title == "Ollama (10.2.6.50)"


async def test_reconfigure_new_url(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """A new URL updates data, unique ID and the default title."""
    result = await init_integration.start_reconfigure_flow(hass)
    with patch(
        "custom_components.ollama_monitor.config_flow.OllamaClient.async_get_version",
        return_value="0.40.2",
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: "http://10.2.6.51:11434"}
        )
        await hass.async_block_till_done()
    assert result["reason"] == "reconfigure_successful"
    assert init_integration.data[CONF_URL] == "http://10.2.6.51:11434"
    assert init_integration.unique_id == "http://10.2.6.51:11434"
    assert init_integration.title == "Ollama (10.2.6.51)"


async def test_reconfigure_keeps_custom_title(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """A title the user chose is not overwritten."""
    hass.config_entries.async_update_entry(init_integration, title="GPU box")
    result = await init_integration.start_reconfigure_flow(hass)
    with patch(
        "custom_components.ollama_monitor.config_flow.OllamaClient.async_get_version",
        return_value="0.40.2",
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: "http://10.2.6.51:11434"}
        )
        await hass.async_block_till_done()
    assert result["reason"] == "reconfigure_successful"
    assert init_integration.title == "GPU box"


async def test_reconfigure_errors_and_conflict(
    hass: HomeAssistant, init_integration: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """Errors keep the form; another entry's server aborts."""
    MockConfigEntry(
        domain=DOMAIN,
        unique_id="http://10.2.6.52:11434",
        data={CONF_URL: "http://10.2.6.52:11434"},
    ).add_to_hass(hass)
    result = await init_integration.start_reconfigure_flow(hass)

    fake_ollama.fail = True
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: URL_BASE}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}

    with patch(
        "custom_components.ollama_monitor.config_flow.OllamaClient.async_get_version",
        return_value="0.40.2",
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: "http://10.2.6.52:11434"}
        )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert init_integration.data[CONF_URL] == URL_BASE


async def test_options_flow(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Options are validated, saved and reload the entry."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)
    assert result["type"] is FlowResultType.FORM

    bad = {
        CONF_SCAN_INTERVAL: 30,
        CONF_ACTIVITY_WINDOW: 10,
        CONF_CHECK_UPDATES: False,
        CONF_ALLOW_MANAGEMENT: True,
    }
    result = await hass.config_entries.options.async_configure(result["flow_id"], bad)
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_ACTIVITY_WINDOW: "activity_window_too_short"}

    good = {**bad, CONF_ACTIVITY_WINDOW: 120.0}
    result = await hass.config_entries.options.async_configure(result["flow_id"], good)
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert init_integration.options == {
        CONF_SCAN_INTERVAL: 30,
        CONF_ACTIVITY_WINDOW: 120,
        CONF_CHECK_UPDATES: False,
        CONF_ALLOW_MANAGEMENT: True,
    }
    coordinator = init_integration.runtime_data.coordinator
    assert coordinator.update_interval is not None
    assert coordinator.update_interval.total_seconds() == 30
    assert init_integration.runtime_data.release_coordinator is None
