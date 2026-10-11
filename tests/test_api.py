"""Tests for the Ollama API client."""

from __future__ import annotations

from datetime import UTC, datetime

import aiohttp
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.ollama_monitor.api import (
    LoadedModel,
    ModelDetails,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    parse_installed_models,
    parse_loaded_models,
    parse_timestamp,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .conftest import QWEN3_30B_PS, TAGS, URL_BASE


@pytest.fixture
def client(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> OllamaClient:
    """Return a client using the mocked session."""
    return OllamaClient(async_get_clientsession(hass), f"{URL_BASE}/")


def _loaded(size: int, size_vram: int, expires: datetime | None) -> LoadedModel:
    return LoadedModel(
        name="m",
        digest=None,
        size=size,
        size_vram=size_vram,
        expires_at=expires,
        context_length=None,
        details=ModelDetails(None, None, None),
    )


def test_parse_loaded_models() -> None:
    """Parse a live /api/ps payload."""
    [model] = parse_loaded_models({"models": [QWEN3_30B_PS]})
    assert model.name == "qwen3:30b"
    assert model.size == 19675558390
    assert model.size_vram == 6973858774
    assert model.context_length == 16384
    assert model.gpu_share == 35.4
    assert model.expires_at == datetime(2026, 10, 11, 2, 11, 57, 453437, tzinfo=UTC)
    assert model.details == ModelDetails("qwen3moe", "30.5B", "Q4_K_M")


def test_parse_loaded_models_missing_fields() -> None:
    """Older servers omit fields; bad values are ignored."""
    [model] = parse_loaded_models(
        {
            "models": [
                {
                    "name": "old",
                    "size": True,
                    "expires_at": "not a date",
                    "details": None,
                }
            ]
        }
    )
    assert model.size == 0
    assert model.size_vram == 0
    assert model.expires_at is None
    assert model.context_length is None
    assert model.gpu_share is None
    assert model.unloads_at is None
    assert model.details == ModelDetails(None, None, None)


def test_parse_loaded_models_bad_expiry_type() -> None:
    """A non-string expiry is ignored."""
    [model] = parse_loaded_models({"models": [{"name": "m", "expires_at": 5}]})
    assert model.expires_at is None


@pytest.mark.parametrize("value", [5, None, "not a date", "2026-13-45T00:00:00Z"])
def test_parse_timestamp_invalid(value: object) -> None:
    """Invalid timestamps parse to None."""
    assert parse_timestamp(value) is None


def test_unloads_at() -> None:
    """A keep-forever expiry is reported as None."""
    soon = datetime(2026, 10, 11, 2, 0, tzinfo=UTC)
    assert _loaded(1, 1, soon).unloads_at == soon
    assert _loaded(1, 1, datetime(2318, 1, 1, tzinfo=UTC)).unloads_at is None


def test_gpu_share_caps_at_100() -> None:
    """VRAM larger than the model size reports 100 %."""
    assert _loaded(10, 20, None).gpu_share == 100.0


def test_parse_installed_models() -> None:
    """Parse a live /api/tags payload."""
    models = parse_installed_models({"models": [*TAGS, {"name": "bare"}]})
    assert [m.name for m in models] == [
        "qwen3:30b",
        "qwen3-embedding:0.6b",
        "qwen3:8b",
        "bare",
    ]
    assert models[0].capabilities == ("completion", "tools", "thinking")
    assert models[0].context_length == 262144
    assert models[0].modified_at is not None
    assert models[2].capabilities == ()
    assert models[3].size == 0
    assert models[3].details == ModelDetails(None, None, None)


def test_parse_installed_models_bad_capabilities() -> None:
    """Non-string capabilities are dropped."""
    [model] = parse_installed_models(
        {"models": [{"name": "m", "capabilities": ["tools", 3]}]}
    )
    assert model.capabilities == ("tools",)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"models": None},
        {"x": []},
        {"models": [QWEN3_30B_PS, "junk"]},
        {"models": [QWEN3_30B_PS, {"name": ""}]},
        {"models": [{"model": "m"}]},
    ],
)
def test_parse_invalid(payload: object) -> None:
    """A payload without a complete, usable model list is rejected."""
    with pytest.raises(OllamaResponseError):
        parse_loaded_models(payload)
    with pytest.raises(OllamaResponseError):
        parse_installed_models(payload)


async def test_get_endpoints(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """GET endpoints parse and remember responses; the base URL is normalized."""
    aioclient_mock.get(f"{URL_BASE}/api/version", json={"version": "0.40.2"})
    aioclient_mock.get(f"{URL_BASE}/api/ps", json={"models": [QWEN3_30B_PS]})
    aioclient_mock.get(f"{URL_BASE}/api/tags", json={"models": TAGS})
    assert client.url == URL_BASE
    assert await client.async_get_version() == "0.40.2"
    assert len(await client.async_get_loaded_models()) == 1
    assert len(await client.async_get_installed_models()) == 3
    assert set(client.last_responses) == {"/api/version", "/api/ps", "/api/tags"}


async def test_subpath_kept(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A reverse-proxy sub path is preserved."""
    aioclient_mock.get("https://proxy/ollama/api/version", json={"version": "1"})
    client = OllamaClient(async_get_clientsession(hass), "https://proxy/ollama")
    assert await client.async_get_version() == "1"


@pytest.mark.parametrize("payload", [{"version": ""}, ["x"], {}])
async def test_version_invalid(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker, payload: object
) -> None:
    """A version response without a version is rejected."""
    aioclient_mock.get(f"{URL_BASE}/api/version", json=payload)
    with pytest.raises(OllamaResponseError):
        await client.async_get_version()


async def test_invalid_json(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """Non-JSON answers are response errors."""
    aioclient_mock.get(f"{URL_BASE}/api/version", text="Ollama is running")
    with pytest.raises(OllamaResponseError):
        await client.async_get_version()


@pytest.mark.parametrize("exc", [aiohttp.ClientConnectionError(), TimeoutError()])
async def test_connection_errors(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker, exc: Exception
) -> None:
    """Network failures are connection errors."""
    aioclient_mock.get(f"{URL_BASE}/api/version", exc=exc)
    with pytest.raises(OllamaConnectionError):
        await client.async_get_version()


async def test_http_errors(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """HTTP errors carry the server's message and status."""
    aioclient_mock.post(
        f"{URL_BASE}/api/generate", status=404, json={"error": "model 'x' not found"}
    )
    with pytest.raises(OllamaModelNotFoundError, match="model 'x' not found") as err:
        await client.async_unload_model("x")
    assert err.value.status == 404

    aioclient_mock.clear_requests()
    aioclient_mock.post(f"{URL_BASE}/api/generate", status=500, text="boom")
    with pytest.raises(OllamaResponseError, match="HTTP 500"):
        await client.async_unload_model("x")

    aioclient_mock.clear_requests()
    aioclient_mock.post(f"{URL_BASE}/api/generate", status=400, json=["odd"])
    with pytest.raises(OllamaResponseError, match="HTTP 400"):
        await client.async_unload_model("x")


async def test_unload(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """Unloading sends keep_alive 0."""
    aioclient_mock.post(f"{URL_BASE}/api/generate", json={"done_reason": "unload"})
    await client.async_unload_model("qwen3:8b")
    assert aioclient_mock.mock_calls[-1][2] == {
        "model": "qwen3:8b",
        "keep_alive": 0,
        "stream": False,
    }


async def test_delete(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """Deleting sends the model name and maps errors."""
    aioclient_mock.delete(f"{URL_BASE}/api/delete", text="")
    await client.async_delete_model("qwen3:8b")
    assert aioclient_mock.mock_calls[-1][2] == {"model": "qwen3:8b"}

    aioclient_mock.clear_requests()
    aioclient_mock.delete(
        f"{URL_BASE}/api/delete", status=404, json={"error": "model 'x' not found"}
    )
    with pytest.raises(OllamaModelNotFoundError):
        await client.async_delete_model("x")

    aioclient_mock.clear_requests()
    aioclient_mock.delete(f"{URL_BASE}/api/delete", exc=aiohttp.ClientError())
    with pytest.raises(OllamaConnectionError):
        await client.async_delete_model("x")


async def test_pull(client: OllamaClient, aioclient_mock: AiohttpClientMocker) -> None:
    """Pull yields progress and skips blank lines."""
    aioclient_mock.post(
        f"{URL_BASE}/api/pull",
        text='{"status":"pulling manifest"}\n\n{"status":"success"}\n',
    )
    messages = [m async for m in client.async_pull_model("qwen3:8b")]
    assert messages == [{"status": "pulling manifest"}, {"status": "success"}]
    assert aioclient_mock.mock_calls[-1][2] == {"model": "qwen3:8b", "stream": True}


async def test_pull_errors(
    client: OllamaClient, aioclient_mock: AiohttpClientMocker
) -> None:
    """Pull errors in the stream, bad lines and network failures are raised."""
    aioclient_mock.post(
        f"{URL_BASE}/api/pull",
        text='{"status":"pulling manifest"}\n{"error":"file does not exist"}\n',
    )
    progress = client.async_pull_model("nope")
    assert await anext(progress) == {"status": "pulling manifest"}
    with pytest.raises(OllamaResponseError, match="file does not exist"):
        await anext(progress)

    for body in ("", '{"status":"pulling manifest"}\n', "[1]\n"):
        aioclient_mock.clear_requests()
        aioclient_mock.post(f"{URL_BASE}/api/pull", text=body)
        with pytest.raises(OllamaResponseError):
            [m async for m in client.async_pull_model("x")]

    aioclient_mock.clear_requests()
    aioclient_mock.post(f"{URL_BASE}/api/pull", text="garbage\n")
    with pytest.raises(OllamaResponseError, match="Invalid progress"):
        await anext(client.async_pull_model("x"))

    aioclient_mock.clear_requests()
    aioclient_mock.post(f"{URL_BASE}/api/pull", status=500, text="")
    with pytest.raises(OllamaResponseError):
        await anext(client.async_pull_model("x"))

    aioclient_mock.clear_requests()
    aioclient_mock.post(f"{URL_BASE}/api/pull", exc=TimeoutError())
    with pytest.raises(OllamaConnectionError):
        await anext(client.async_pull_model("x"))
