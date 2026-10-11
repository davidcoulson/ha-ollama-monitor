"""Data update coordinators for Ollama Monitor."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import (
    InstalledModel,
    LoadedModel,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    parse_timestamp,
)
from .const import (
    CONF_ACTIVITY_WINDOW,
    CONF_SCAN_INTERVAL,
    DEFAULT_ACTIVITY_WINDOW,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    EVENT_MODEL_INSTALLED,
    EVENT_MODEL_LOADED,
    EVENT_MODEL_REMOVED,
    EVENT_MODEL_UNLOADED,
    GITHUB_LATEST_RELEASE_URL,
    ISSUE_UNREACHABLE,
    LOGGER,
    RELEASE_CHECK_INTERVAL,
    STORAGE_VERSION,
    UNREACHABLE_ISSUE_AFTER,
)

type OllamaConfigEntry = ConfigEntry[OllamaRuntimeData]

SAVE_DELAY = 30


@dataclass(slots=True)
class OllamaRuntimeData:
    """Runtime data stored on the config entry."""

    client: OllamaClient
    coordinator: OllamaCoordinator
    release_coordinator: ReleaseCoordinator | None


@dataclass(frozen=True, slots=True)
class ModelState:
    """Everything known about one model on the server."""

    name: str
    installed: InstalledModel | None
    loaded: LoadedModel | None
    last_used: datetime | None
    active: bool

    @property
    def device_model(self) -> str | None:
        """Return a short description for the device registry."""
        if (source := self.installed or self.loaded) is None:
            return None
        details = source.details
        parts = [
            part
            for part in (
                details.family,
                details.parameter_size,
                details.quantization_level,
            )
            if part
        ]
        return " · ".join(parts) or None

    @property
    def digest(self) -> str | None:
        """Return the model digest."""
        source = self.installed or self.loaded
        return source.digest if source else None


@dataclass(frozen=True, slots=True)
class ModelEvent:
    """A change detected between two polls."""

    event_type: str
    model: str


@dataclass(frozen=True, slots=True)
class OllamaData:
    """Snapshot of the server produced by one poll."""

    version: str
    models: dict[str, ModelState]
    last_activity: datetime | None
    events: tuple[ModelEvent, ...] = field(default_factory=tuple)

    @property
    def loaded(self) -> list[LoadedModel]:
        """Return the loaded models."""
        return [state.loaded for state in self.models.values() if state.loaded]

    @property
    def loaded_names(self) -> list[str]:
        """Return the names of the loaded models."""
        return [name for name, state in self.models.items() if state.loaded]

    @property
    def installed_count(self) -> int:
        """Return the number of installed models."""
        return sum(1 for state in self.models.values() if state.installed)

    @property
    def disk_usage(self) -> int:
        """Return the disk space used by installed models in bytes."""
        return sum(
            state.installed.size for state in self.models.values() if state.installed
        )

    @property
    def vram_used(self) -> int:
        """Return the VRAM used by loaded models in bytes."""
        return sum(model.size_vram for model in self.loaded)

    @property
    def memory_used(self) -> int:
        """Return the memory (VRAM + RAM) used by loaded models in bytes."""
        return sum(model.size for model in self.loaded)

    @property
    def gpu_share(self) -> float | None:
        """Return the share of loaded model memory held in VRAM."""
        if (memory := self.memory_used) <= 0:
            return None
        return round(min(self.vram_used, memory) / memory * 100, 1)

    @property
    def cpu_offload(self) -> bool:
        """Return True if any loaded model does not fit in VRAM."""
        return any(model.size_vram < model.size for model in self.loaded)

    @property
    def active(self) -> bool:
        """Return True if any model is active."""
        return any(state.active for state in self.models.values())

    @property
    def active_models(self) -> list[str]:
        """Return the active models, most recently used first."""
        return [state.name for state in self._by_recent_use() if state.active]

    @property
    def last_active_model(self) -> ModelState | None:
        """Return the most recently used model, active or not."""
        return next(iter(self._by_recent_use()), None)

    def _by_recent_use(self) -> list[ModelState]:
        """Return the models with a known last use, newest first (ties by name)."""
        used = sorted(
            (
                (state.last_used, state)
                for state in self.models.values()
                if state.last_used is not None
            ),
            key=lambda pair: pair[1].name,
        )
        # Stable sort: models used at the same moment stay in name order.
        used.sort(key=lambda pair: pair[0], reverse=True)
        return [state for _, state in used]


class OllamaCoordinator(DataUpdateCoordinator[OllamaData]):
    """Poll the Ollama server and derive activity from the changes."""

    config_entry: OllamaConfigEntry

    def __init__(
        self, hass: HomeAssistant, entry: OllamaConfigEntry, client: OllamaClient
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title}",
            update_interval=timedelta(
                seconds=entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
            ),
        )
        self.client = client
        self._activity_window = timedelta(
            seconds=entry.options.get(CONF_ACTIVITY_WINDOW, DEFAULT_ACTIVITY_WINDOW)
        )
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}"
        )
        self._baseline = False
        # Set after a failed poll: changes seen on recovery happened at an unknown
        # time during the outage and must not count as current activity.
        self._recovering = False
        self._expires: dict[str, datetime | None] = {}
        self._installed_names: set[str] = set()
        self._last_used: dict[str, datetime] = {}
        self._unreachable_since: datetime | None = None
        self._unsaved = False

    async def _async_setup(self) -> None:
        """Restore the activity history."""
        stored = await self._store.async_load() or {}
        raw_last_used = stored.get("last_used")
        if isinstance(raw_last_used, Mapping):
            for name, value in raw_last_used.items():
                if parsed := parse_timestamp(value):
                    self._last_used[name] = parsed

    async def _async_update_data(self) -> OllamaData:
        """Fetch the server state."""
        try:
            version, loaded, installed = await asyncio.gather(
                self.client.async_get_version(),
                self.client.async_get_loaded_models(),
                self.client.async_get_installed_models(),
            )
        except OllamaError as err:
            self._track_unreachable()
            self._recovering = True
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key=(
                    "cannot_connect"
                    if isinstance(err, OllamaConnectionError)
                    else "invalid_response"
                ),
                translation_placeholders={"url": self.client.url, "error": str(err)},
            ) from err

        if self._unreachable_since is not None:
            self._unreachable_since = None
            ir.async_delete_issue(
                self.hass, DOMAIN, f"{ISSUE_UNREACHABLE}_{self.config_entry.entry_id}"
            )
        return self._process(version, loaded, installed)

    def _track_unreachable(self) -> None:
        now = dt_util.utcnow()
        if self._unreachable_since is None:
            self._unreachable_since = now
        if now - self._unreachable_since >= UNREACHABLE_ISSUE_AFTER:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                f"{ISSUE_UNREACHABLE}_{self.config_entry.entry_id}",
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key=ISSUE_UNREACHABLE,
                translation_placeholders={
                    "title": self.config_entry.title,
                    "url": self.client.url,
                },
            )

    def _process(
        self,
        version: str,
        loaded: list[LoadedModel],
        installed: list[InstalledModel],
    ) -> OllamaData:
        now = dt_util.utcnow()
        installed_by_name = {model.name: model for model in installed}
        by_digest = {model.digest: model.name for model in installed if model.digest}
        loaded_by_name: dict[str, LoadedModel] = {}
        for model in loaded:
            # Match a loaded model to its installed name, falling back to the digest
            # when the server reports a different spelling of the same model.
            name = model.name
            if name not in installed_by_name and model.digest in by_digest:
                name = by_digest[model.digest]
            loaded_by_name[name] = model

        events: list[ModelEvent] = []
        if self._baseline:
            events.extend(
                ModelEvent(EVENT_MODEL_INSTALLED, name)
                for name in installed_by_name
                if name not in self._installed_names
            )
            events.extend(
                ModelEvent(EVENT_MODEL_REMOVED, name)
                for name in self._installed_names
                if name not in installed_by_name
            )
            events.extend(
                ModelEvent(EVENT_MODEL_UNLOADED, name)
                for name in self._expires
                if name not in loaded_by_name
            )
            for name, model in loaded_by_name.items():
                if name not in self._expires:
                    events.append(ModelEvent(EVENT_MODEL_LOADED, name))
                    if not self._recovering:
                        self._mark_used(name, now)
                elif self._expires[name] != model.expires_at and not self._recovering:
                    self._mark_used(name, now)

        self._baseline = True
        self._recovering = False
        self._expires = {
            name: model.expires_at for name, model in loaded_by_name.items()
        }
        self._installed_names = set(installed_by_name)

        models = {
            name: ModelState(
                name=name,
                installed=installed_by_name.get(name),
                loaded=loaded_by_name.get(name),
                last_used=self._last_used.get(name),
                active=(
                    name in self._last_used
                    and now - self._last_used[name] < self._activity_window
                ),
            )
            for name in sorted(installed_by_name.keys() | loaded_by_name.keys())
        }
        if stale := self._last_used.keys() - models.keys():
            for name in stale:
                del self._last_used[name]
            self._schedule_save()

        return OllamaData(
            version=version,
            models=models,
            last_activity=max(self._last_used.values(), default=None),
            events=tuple(events),
        )

    def _mark_used(self, name: str, when: datetime) -> None:
        self._last_used[name] = when
        self._schedule_save()

    def _schedule_save(self) -> None:
        self._unsaved = True
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    async def async_shutdown(self) -> None:
        """Write pending activity history before the entry unloads or reloads."""
        await super().async_shutdown()
        if self._unsaved:
            await self._store.async_save(self._data_to_save())

    def _data_to_save(self) -> dict[str, Any]:
        self._unsaved = False
        return {
            "last_used": {
                name: value.isoformat() for name, value in self._last_used.items()
            }
        }

    @staticmethod
    async def async_remove_store_for(hass: HomeAssistant, entry_id: str) -> None:
        """Remove the persisted activity history of a config entry."""
        await Store[dict[str, Any]](
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}"
        ).async_remove()

    def diagnostics(self) -> dict[str, Any]:
        """Return internal state for diagnostics."""
        return {
            "last_used": {
                name: value.isoformat() for name, value in self._last_used.items()
            },
            "expires_at": {
                name: value.isoformat() if value else None
                for name, value in self._expires.items()
            },
            "unreachable_since": (
                self._unreachable_since.isoformat() if self._unreachable_since else None
            ),
        }


@dataclass(frozen=True, slots=True)
class Release:
    """The latest Ollama release on GitHub."""

    version: str
    url: str | None
    notes: str | None


class ReleaseCoordinator(DataUpdateCoordinator[Release]):
    """Check GitHub for the latest Ollama release."""

    config_entry: OllamaConfigEntry

    def __init__(self, hass: HomeAssistant, entry: OllamaConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title} releases",
            update_interval=RELEASE_CHECK_INTERVAL,
        )
        self._session = async_get_clientsession(hass)

    async def _async_update_data(self) -> Release:
        """Fetch the latest release."""
        try:
            async with self._session.get(
                GITHUB_LATEST_RELEASE_URL,
                headers={"Accept": "application/vnd.github+json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (TimeoutError, aiohttp.ClientError, ValueError) as err:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="release_check_failed",
                translation_placeholders={"error": str(err)},
            ) from err
        tag = data.get("tag_name") if isinstance(data, dict) else None
        if not isinstance(tag, str) or not tag:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="release_check_failed",
                translation_placeholders={"error": "missing tag_name"},
            )
        url = data.get("html_url")
        notes = data.get("body")
        return Release(
            version=tag.removeprefix("v"),
            url=url if isinstance(url, str) else None,
            notes=notes if isinstance(notes, str) else None,
        )
