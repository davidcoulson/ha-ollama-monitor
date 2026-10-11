"""Fixtures for Ollama Monitor tests."""

from __future__ import annotations

from copy import deepcopy
from http import HTTPStatus
import json
from typing import Any

import aiohttp
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)
from yarl import URL

from custom_components.ollama_monitor.const import (
    CONF_ACTIVITY_WINDOW,
    CONF_ALLOW_MANAGEMENT,
    CONF_CHECK_UPDATES,
    CONF_SCAN_INTERVAL,
    DOMAIN,
    GITHUB_LATEST_RELEASE_URL,
)
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant

URL_BASE = "http://10.2.6.50:11434"

# Shapes taken from the live server (Ollama 0.40.2) during development.
QWEN3_30B_PS: dict[str, Any] = {
    "name": "qwen3:30b",
    "model": "qwen3:30b",
    "size": 19675558390,
    "digest": "ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen3moe",
        "families": ["qwen3moe"],
        "parameter_size": "30.5B",
        "quantization_level": "Q4_K_M",
    },
    "expires_at": "2026-10-11T02:11:57.453437592Z",
    "size_vram": 6973858774,
    "context_length": 16384,
    "runner": "llamacpp",
}

QWEN3_8B_PS: dict[str, Any] = {
    "name": "qwen3:8b",
    "model": "qwen3:8b",
    "size": 6000000000,
    "digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
    "details": {
        "family": "qwen3",
        "parameter_size": "8.2B",
        "quantization_level": "Q4_K_M",
    },
    "expires_at": "2026-10-11T02:15:00Z",
    "size_vram": 6000000000,
    "context_length": 4096,
}

TAGS: list[dict[str, Any]] = [
    {
        "name": "qwen3:30b",
        "model": "qwen3:30b",
        "modified_at": "2026-10-11T01:48:14.504196095Z",
        "size": 18556699314,
        "digest": "ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen3moe",
            "families": ["qwen3moe"],
            "parameter_size": "30.5B",
            "quantization_level": "Q4_K_M",
            "context_length": 262144,
            "embedding_length": 2048,
        },
        "capabilities": ["completion", "tools", "thinking"],
    },
    {
        "name": "qwen3-embedding:0.6b",
        "model": "qwen3-embedding:0.6b",
        "modified_at": "2026-10-11T01:13:46.554325891Z",
        "size": 639150858,
        "digest": "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d",
        "details": {
            "parent_model": "/Users/ollama/.ollama/models/blobs/sha256-0650",
            "family": "qwen3",
            "parameter_size": "595.78M",
            "quantization_level": "Q8_0",
            "context_length": 32768,
        },
        "capabilities": ["embedding"],
    },
    {
        "name": "qwen3:8b",
        "model": "qwen3:8b",
        "modified_at": "2026-10-11T01:13:34.69768182Z",
        "size": 5225388164,
        "digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        "details": {
            "family": "qwen3",
            "parameter_size": "8.2B",
            "quantization_level": "Q4_K_M",
        },
    },
]

RELEASE = {
    "tag_name": "v0.41.0",
    "html_url": "https://github.com/ollama/ollama/releases/tag/v0.41.0",
    "body": "## What's changed\n- Faster",
}


class FakeOllama:
    """A mutable fake Ollama server served through aioclient_mock."""

    def __init__(self) -> None:
        """Initialize with one loaded model."""
        self.version = "0.40.2"
        self.loaded: list[dict[str, Any]] = [deepcopy(QWEN3_30B_PS)]
        self.installed: list[dict[str, Any]] = deepcopy(TAGS)
        self.fail = False
        self.release: dict[str, Any] | None = deepcopy(RELEASE)
        self.release_status = HTTPStatus.OK
        self.requests: list[tuple[str, str, Any]] = []
        self.responses: dict[tuple[str, str], tuple[int, Any]] = {}
        self.pull_lines: list[bytes] = [
            b'{"status":"pulling manifest"}\n',
            b'{"status":"success"}\n',
        ]

    def register(self, aioclient_mock: AiohttpClientMocker) -> None:
        """Register all endpoints."""
        for method, path in (
            ("get", "/api/version"),
            ("get", "/api/ps"),
            ("get", "/api/tags"),
            ("post", "/api/generate"),
            ("delete", "/api/delete"),
            ("post", "/api/pull"),
        ):
            aioclient_mock.request(
                method, f"{URL_BASE}{path}", side_effect=self._handle
            )
        aioclient_mock.get(GITHUB_LATEST_RELEASE_URL, side_effect=self._handle_github)

    def response(self, method: str, path: str, status: int, body: Any) -> None:
        """Override the next answers of one endpoint."""
        self.responses[(method.upper(), path)] = (status, body)

    async def _handle(
        self, method: str, url: URL, data: Any
    ) -> AiohttpClientMockResponse:
        method = method.upper()
        path = url.path
        self.requests.append((method, path, data))
        if self.fail:
            return AiohttpClientMockResponse(
                method, url, exc=aiohttp.ClientConnectionError()
            )
        if (method, path) in self.responses:
            status, body = self.responses[(method, path)]
            return AiohttpClientMockResponse(
                method,
                url,
                status=status,
                text=body if isinstance(body, str) else json.dumps(body),
            )
        if path == "/api/version":
            body = {"version": self.version}
        elif path == "/api/ps":
            body = {"models": self.loaded}
        elif path == "/api/tags":
            body = {"models": self.installed}
        elif path == "/api/pull":
            return AiohttpClientMockResponse(
                method, url, text=b"".join(self.pull_lines).decode()
            )
        else:
            body = {}
        return AiohttpClientMockResponse(method, url, text=json.dumps(body))

    async def _handle_github(
        self, method: str, url: URL, data: Any
    ) -> AiohttpClientMockResponse:
        return AiohttpClientMockResponse(
            method, url, status=self.release_status, text=json.dumps(self.release)
        )

    def calls(self, method: str, path: str) -> list[Any]:
        """Return the bodies of the requests made to an endpoint."""
        return [data for m, p, data in self.requests if (m, p) == (method, path)]


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations in all tests."""


@pytest.fixture
def fake_ollama(aioclient_mock: AiohttpClientMocker) -> FakeOllama:
    """Return a fake Ollama server."""
    fake = FakeOllama()
    fake.register(aioclient_mock)
    return fake


@pytest.fixture
def options() -> dict[str, Any]:
    """Return the entry options."""
    return {
        CONF_SCAN_INTERVAL: 10,
        CONF_ACTIVITY_WINDOW: 60,
        CONF_CHECK_UPDATES: True,
        CONF_ALLOW_MANAGEMENT: False,
    }


@pytest.fixture
def mock_config_entry(options: dict[str, Any]) -> MockConfigEntry:
    """Return a config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Ollama (10.2.6.50)",
        unique_id=URL_BASE,
        data={CONF_URL: URL_BASE},
        options=options,
        entry_id="01TESTENTRY",
    )


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: Any,
) -> MockConfigEntry:
    """Set up the integration."""
    mock_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry


@pytest.fixture
def frozen_time(freezer: Any) -> Any:
    """Freeze time shortly before the sample expiry times."""
    freezer.move_to("2026-10-11T02:06:00+00:00")
    return freezer
