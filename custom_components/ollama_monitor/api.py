"""Minimal asynchronous client for the Ollama HTTP API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

import aiohttp

from homeassistant.util import dt as dt_util

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
# A pull can stay silent for a long time while blobs are verified, and the whole
# download can take hours, so only bound the time between chunks.
PULL_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=900)

# Ollama reports keep_alive < 0 ("forever") as an expiry centuries away.
_FOREVER_AFTER_YEARS = 100


class OllamaError(Exception):
    """Base error for Ollama API failures."""


class OllamaConnectionError(OllamaError):
    """The server could not be reached or timed out."""


class OllamaResponseError(OllamaError):
    """The server answered with an error or an unexpected payload."""

    def __init__(self, message: str, status: int | None = None) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.status = status


class OllamaModelNotFoundError(OllamaResponseError):
    """The requested model does not exist on the server."""


@dataclass(frozen=True, slots=True)
class ModelDetails:
    """Static details shared by loaded and installed models."""

    family: str | None
    parameter_size: str | None
    quantization_level: str | None

    @classmethod
    def from_api(cls, raw: Any) -> ModelDetails:
        """Build from the `details` object of an API response."""
        details: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        return cls(
            family=_opt_str(details.get("family")),
            parameter_size=_opt_str(details.get("parameter_size")),
            quantization_level=_opt_str(details.get("quantization_level")),
        )


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model currently loaded in memory (`/api/ps`)."""

    name: str
    digest: str | None
    size: int
    size_vram: int
    expires_at: datetime | None
    context_length: int | None
    details: ModelDetails

    @property
    def gpu_share(self) -> float | None:
        """Return the percentage of the model held in VRAM."""
        if self.size <= 0:
            return None
        return round(min(self.size_vram, self.size) / self.size * 100, 1)

    @property
    def unloads_at(self) -> datetime | None:
        """Return when the model will be unloaded, None if kept forever."""
        if self.expires_at is None:
            return None
        if self.expires_at.year - dt_util.utcnow().year > _FOREVER_AFTER_YEARS:
            return None
        return self.expires_at


@dataclass(frozen=True, slots=True)
class InstalledModel:
    """A model installed on the server (`/api/tags`)."""

    name: str
    digest: str | None
    size: int
    modified_at: datetime | None
    context_length: int | None
    capabilities: tuple[str, ...]
    details: ModelDetails


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp; return None for anything invalid."""
    if not isinstance(value, str):
        return None
    try:
        return dt_util.parse_datetime(value)
    except ValueError:
        return None


def _model_entries(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(
        models := payload.get("models"), list
    ):
        raise OllamaResponseError("Unexpected response: missing model list")
    return [
        item
        for item in models
        if isinstance(item, Mapping) and _opt_str(item.get("name"))
    ]


def parse_loaded_models(payload: Any) -> list[LoadedModel]:
    """Parse an `/api/ps` response."""
    return [
        LoadedModel(
            name=item["name"],
            digest=_opt_str(item.get("digest")),
            size=_opt_int(item.get("size")) or 0,
            size_vram=_opt_int(item.get("size_vram")) or 0,
            expires_at=parse_timestamp(item.get("expires_at")),
            context_length=_opt_int(item.get("context_length")),
            details=ModelDetails.from_api(item.get("details")),
        )
        for item in _model_entries(payload)
    ]


def parse_installed_models(payload: Any) -> list[InstalledModel]:
    """Parse an `/api/tags` response."""
    result: list[InstalledModel] = []
    for item in _model_entries(payload):
        raw_details = item.get("details")
        details = raw_details if isinstance(raw_details, Mapping) else {}
        capabilities = item.get("capabilities")
        result.append(
            InstalledModel(
                name=item["name"],
                digest=_opt_str(item.get("digest")),
                size=_opt_int(item.get("size")) or 0,
                modified_at=parse_timestamp(item.get("modified_at")),
                context_length=_opt_int(details.get("context_length")),
                capabilities=tuple(
                    cap
                    for cap in (capabilities if isinstance(capabilities, list) else [])
                    if isinstance(cap, str)
                ),
                details=ModelDetails.from_api(details),
            )
        )
    return result


class OllamaClient:
    """Talk to one Ollama server."""

    def __init__(self, session: aiohttp.ClientSession, url: str) -> None:
        """Initialize the client."""
        self._session = session
        self._base = url.rstrip("/")
        self.last_responses: dict[str, Any] = {}

    @property
    def url(self) -> str:
        """Return the server base URL."""
        return self._base

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        try:
            async with self._session.request(
                method, f"{self._base}{path}", json=payload, timeout=REQUEST_TIMEOUT
            ) as resp:
                await _raise_for_status(resp)
                try:
                    return await resp.json(content_type=None)
                except ValueError as err:
                    raise OllamaResponseError(
                        f"Invalid JSON from {path}", resp.status
                    ) from err
        except (TimeoutError, aiohttp.ClientError) as err:
            raise OllamaConnectionError(
                f"Error talking to Ollama at {self._base}: {err!r}"
            ) from err

    async def _get(self, path: str) -> Any:
        data = await self._request("GET", path)
        self.last_responses[path] = data
        return data

    async def async_get_version(self) -> str:
        """Return the server version."""
        data = await self._get("/api/version")
        if not isinstance(data, Mapping) or not (
            version := _opt_str(data.get("version"))
        ):
            raise OllamaResponseError("Unexpected response: missing version")
        return version

    async def async_get_loaded_models(self) -> list[LoadedModel]:
        """Return the models currently loaded in memory."""
        return parse_loaded_models(await self._get("/api/ps"))

    async def async_get_installed_models(self) -> list[InstalledModel]:
        """Return the models installed on the server."""
        return parse_installed_models(await self._get("/api/tags"))

    async def async_unload_model(self, model: str) -> None:
        """Unload a model by asking for it with keep_alive 0 (`ollama stop`)."""
        await self._request(
            "POST",
            "/api/generate",
            {"model": model, "keep_alive": 0, "stream": False},
        )

    async def async_delete_model(self, model: str) -> None:
        """Delete an installed model."""
        try:
            async with self._session.delete(
                f"{self._base}/api/delete",
                json={"model": model},
                timeout=REQUEST_TIMEOUT,
            ) as resp:
                await _raise_for_status(resp)
        except (TimeoutError, aiohttp.ClientError) as err:
            raise OllamaConnectionError(
                f"Error talking to Ollama at {self._base}: {err!r}"
            ) from err

    async def async_pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Pull a model and yield the progress messages as they arrive.

        Ollama reports failures that happen after the download started as an
        `error` message in the stream; they are raised as OllamaResponseError.
        """
        try:
            async with self._session.post(
                f"{self._base}/api/pull",
                json={"model": model, "stream": True},
                timeout=PULL_TIMEOUT,
            ) as resp:
                await _raise_for_status(resp)
                async for raw_line in resp.content:
                    if not (line := raw_line.strip()):
                        continue
                    try:
                        message = json.loads(line)
                    except ValueError as err:
                        raise OllamaResponseError(
                            "Invalid progress message from /api/pull"
                        ) from err
                    if not isinstance(message, dict):
                        continue
                    if error := message.get("error"):
                        raise OllamaResponseError(str(error))
                    yield message
        except (TimeoutError, aiohttp.ClientError) as err:
            raise OllamaConnectionError(
                f"Error talking to Ollama at {self._base}: {err!r}"
            ) from err


async def _raise_for_status(resp: aiohttp.ClientResponse) -> None:
    if resp.status < 400:
        return
    message = f"HTTP {resp.status}"
    try:
        body = await resp.json(content_type=None)
    except ValueError:
        pass
    else:
        if isinstance(body, Mapping) and (error := _opt_str(body.get("error"))):
            message = error
    if resp.status == 404:
        raise OllamaModelNotFoundError(message, resp.status)
    raise OllamaResponseError(message, resp.status)
