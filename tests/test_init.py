"""Tests for setup, polling, activity detection and dynamic devices."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any

from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.ollama_monitor import async_remove_config_entry_device
from custom_components.ollama_monitor.api import ModelDetails
from custom_components.ollama_monitor.const import DOMAIN, ISSUE_UNREACHABLE
from custom_components.ollama_monitor.coordinator import ModelState
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import Event, EventStateChangedData, HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    issue_registry as ir,
)
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

from .conftest import QWEN3_8B_PS, FakeOllama

EVENT_ENTITY = "event.ollama_10_2_6_50_model_event"
ISSUE_ID = f"{ISSUE_UNREACHABLE}_01TESTENTRY"


async def _poll(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, seconds: int = 11
) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)


def _bump(fake: FakeOllama, name: str, expires: str) -> None:
    for model in fake.loaded:
        if model["name"] == name:
            model["expires_at"] = expires


def _record_events(hass: HomeAssistant) -> list[tuple[str, str]]:
    fired: list[tuple[str, str]] = []

    def _listener(event: Event[EventStateChangedData]) -> None:
        old, new = event.data["old_state"], event.data["new_state"]
        # A real event changes the state (its timestamp); availability flips
        # restore the previous timestamp and are not events.
        if (
            new is not None
            and old is not None
            and STATE_UNAVAILABLE not in (old.state, new.state)
            and old.state != new.state
        ):
            state = new
            fired.append((state.attributes["event_type"], state.attributes["model"]))

    async_track_state_change_event(hass, [EVENT_ENTITY], _listener)
    return fired


async def test_setup_and_unload(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entry sets up, creates devices and unloads cleanly."""
    assert init_integration.state is ConfigEntryState.LOADED
    registry = dr.async_get(hass)
    server = registry.async_get_device_by_identifier(
        (DOMAIN, "01TESTENTRY"), "01TESTENTRY"
    )
    assert server is not None
    assert server.sw_version == "0.40.2"
    assert server.manufacturer == "Ollama"
    model = registry.async_get_child_device_by_identifier(
        (DOMAIN, "01TESTENTRY_model_qwen3:30b"), "01TESTENTRY"
    )
    assert model is not None
    assert model.parent_device_id == server.id
    assert model.name == "qwen3:30b"
    # Model details live in diagnostic sensors on the model device.
    prefix = "sensor.ollama_10_2_6_50_qwen3_30b"
    assert hass.states.get(f"{prefix}_family").state == "qwen3moe"
    assert hass.states.get(f"{prefix}_parameter_size").state == "30.5B"
    assert hass.states.get(f"{prefix}_quantization").state == "Q4_K_M"
    entity = er.async_get(hass).async_get(f"{prefix}_quantization")
    assert entity is not None
    assert entity.entity_category is er.EntityCategory.DIAGNOSTIC
    assert entity.device_id == model.id

    assert await hass.config_entries.async_unload(init_integration.entry_id)
    assert init_integration.state is ConfigEntryState.NOT_LOADED


async def test_setup_retry(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, fake_ollama: FakeOllama
) -> None:
    """An unreachable server retries setup."""
    fake_ollama.fail = True
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_activity_detection(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """A changed expires_at marks the model as used."""
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
        == "unknown"
    )
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_30b_active").state
        == STATE_OFF
    )

    # No change: no activity.
    await _poll(hass, frozen_time)
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_active").state == STATE_OFF

    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:20:00Z")
    await _poll(hass, frozen_time)
    used = dt_util.utcnow()
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_30b_active").state
        == STATE_ON
    )
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_active").state == STATE_ON
    assert dt_util.parse_datetime(
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
    ) == used.replace(microsecond=0)
    assert hass.states.get("sensor.ollama_10_2_6_50_last_activity").state == (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
    )

    # The activity window (60 s) passes without further requests.
    for _ in range(6):
        await _poll(hass, frozen_time)
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_30b_active").state
        == STATE_OFF
    )
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_active").state == STATE_OFF
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
        != "unknown"
    )


async def test_load_unload_events(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """Loading and unloading fire events and update the model entities."""
    fired = _record_events(hass)
    fake_ollama.loaded = [deepcopy(QWEN3_8B_PS)]
    await _poll(hass, frozen_time)

    assert fired == [("model_unloaded", "qwen3:30b"), ("model_loaded", "qwen3:8b")]
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_8b_loaded").state
        == STATE_ON
    )
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_30b_loaded").state
        == STATE_OFF
    )
    # Loading alone is not use.
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_8b_active").state
        == STATE_OFF
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_last_used").state == "unknown"
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_vram").state == "0.0"
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_gpu_share").state
        == "unknown"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_gpu_share").state == "100.0"
    )
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_cpu_offload").state == (
        STATE_OFF
    )
    loaded = hass.states.get("sensor.ollama_10_2_6_50_loaded_models")
    assert loaded.state == "1"
    assert loaded.attributes["models"] == ["qwen3:8b"]

    # A failed poll must not re-fire the previous events.
    fake_ollama.fail = True
    await _poll(hass, frozen_time)
    fake_ollama.fail = False
    await _poll(hass, frozen_time)
    assert len(fired) == 2


async def test_dynamic_model_devices(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    entity_registry: er.EntityRegistry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Installed and removed models add and remove devices without a reload."""
    fired = _record_events(hass)
    fake_ollama.installed.append(
        {
            "name": "granite4:tiny-h",
            "size": 4230995221,
            "digest": "566b72",
            "details": {"family": "granitehybrid", "parameter_size": "6.9B"},
        }
    )
    fake_ollama.installed = [
        m for m in fake_ollama.installed if m["name"] != "qwen3-embedding:0.6b"
    ]
    await _poll(hass, frozen_time)

    assert fired == [
        ("model_installed", "granite4:tiny-h"),
        ("model_removed", "qwen3-embedding:0.6b"),
    ]
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_granite4_tiny_h_vram").state == "0.0"
    )
    assert (
        hass.states.get("button.ollama_10_2_6_50_granite4_tiny_h_unload").state
        == STATE_UNAVAILABLE
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_embedding_0_6b_vram") is None
    assert (
        entity_registry.async_get("sensor.ollama_10_2_6_50_qwen3_embedding_0_6b_vram")
        is None
    )
    assert (
        device_registry.async_get_child_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_qwen3-embedding:0.6b"), "01TESTENTRY"
        )
        is None
    )
    assert (
        device_registry.async_get_child_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_granite4:tiny-h"), "01TESTENTRY"
        )
        is not None
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_installed_models").state == "3"

    # The removed model comes back.
    fake_ollama.installed.append({"name": "qwen3-embedding:0.6b", "size": 1})
    await _poll(hass, frozen_time)
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_embedding_0_6b_vram").state
        == "0.0"
    )
    assert init_integration.state is ConfigEntryState.LOADED


async def test_loaded_model_matching(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Loaded models are matched by digest; unknown ones get their own device."""
    alias = deepcopy(QWEN3_8B_PS)
    alias["name"] = "registry.example/qwen3:8b"
    stranger = {
        "name": "mystery:1b",
        "size": 100,
        "size_vram": 50,
        "details": {},
        "expires_at": "2318-01-01T00:00:00Z",
    }
    fake_ollama.loaded = [alias, stranger]
    await _poll(hass, frozen_time)

    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_8b_loaded").state
        == STATE_ON
    )
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_mystery_1b_loaded").state
        == STATE_ON
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_mystery_1b_unloads_at").state
        == "unknown"
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_mystery_1b_size_on_disk") is None
    device = device_registry.async_get_child_device_by_identifier(
        (DOMAIN, "01TESTENTRY_model_mystery:1b"), "01TESTENTRY"
    )
    assert device is not None
    assert hass.states.get("sensor.ollama_10_2_6_50_mystery_1b_family").state == (
        "unknown"
    )
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_cpu_offload").state == (
        STATE_ON
    )


async def test_unreachable_and_repair(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    issue_registry: ir.IssueRegistry,
) -> None:
    """Entities go unavailable; a repair appears after 5 minutes and clears."""
    fake_ollama.fail = True
    await _poll(hass, frozen_time)
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_vram").state
        == STATE_UNAVAILABLE
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_version").state == (
        STATE_UNAVAILABLE
    )
    reachable = hass.states.get("binary_sensor.ollama_10_2_6_50_reachable")
    assert reachable.state == STATE_OFF
    assert issue_registry.async_get_issue(DOMAIN, ISSUE_ID) is None

    await _poll(hass, frozen_time, 300)
    issue = issue_registry.async_get_issue(DOMAIN, ISSUE_ID)
    assert issue is not None
    assert issue.translation_placeholders == {
        "title": "Ollama (10.2.6.50)",
        "url": "http://10.2.6.50:11434",
    }

    fake_ollama.fail = False
    await _poll(hass, frozen_time)
    assert issue_registry.async_get_issue(DOMAIN, ISSUE_ID) is None
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_reachable").state == (
        STATE_ON
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_vram").state
        != STATE_UNAVAILABLE
    )


async def test_invalid_response(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed answer makes the entities unavailable."""
    fake_ollama.response("GET", "/api/tags", 200, {"oops": True})
    await _poll(hass, frozen_time)
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_vram").state
        == STATE_UNAVAILABLE
    )
    assert "Unexpected response from the Ollama server" in caplog.text


async def test_version_change_updates_device(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    device_registry: dr.DeviceRegistry,
) -> None:
    """The server device follows the server version."""
    fake_ollama.version = "0.41.0"
    await _poll(hass, frozen_time)
    server = device_registry.async_get_device_by_identifier(
        (DOMAIN, "01TESTENTRY"), "01TESTENTRY"
    )
    assert server is not None
    assert server.sw_version == "0.41.0"


async def test_activity_persisted(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    hass_storage: dict[str, Any],
) -> None:
    """Last-used times are saved and dropped when a model disappears."""
    key = f"{DOMAIN}.01TESTENTRY"
    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:30:00Z")
    await _poll(hass, frozen_time)
    await _poll(hass, frozen_time, 31)
    assert set(hass_storage[key]["data"]["last_used"]) == {"qwen3:30b"}

    fake_ollama.loaded = []
    fake_ollama.installed = [
        m for m in fake_ollama.installed if m["name"] != "qwen3:30b"
    ]
    await _poll(hass, frozen_time)
    await _poll(hass, frozen_time, 31)
    assert hass_storage[key]["data"]["last_used"] == {}


async def test_activity_restored(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    hass_storage: dict[str, Any],
) -> None:
    """Stored last-used times are restored; junk is ignored."""
    hass_storage[f"{DOMAIN}.01TESTENTRY"] = {
        "version": 1,
        "key": f"{DOMAIN}.01TESTENTRY",
        "data": {
            "last_used": {
                "qwen3:8b": "2026-10-10T20:00:00+00:00",
                "bad": 5,
                "worse": "not a date",
                "worst": "2026-13-45T00:00:00Z",
            }
        },
    }
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_last_used").state == (
        "2026-10-10T20:00:00+00:00"
    )


async def test_activity_restore_ignores_bad_store(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    hass_storage: dict[str, Any],
) -> None:
    """A store without a mapping is ignored."""
    hass_storage[f"{DOMAIN}.01TESTENTRY"] = {
        "version": 1,
        "key": f"{DOMAIN}.01TESTENTRY",
        "data": {"last_used": ["x"]},
    }
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.LOADED


async def test_remove_entry_removes_store(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """Deleting the entry deletes the activity history."""
    hass_storage[f"{DOMAIN}.01TESTENTRY"] = {"version": 1, "data": {}}
    await hass.config_entries.async_remove(init_integration.entry_id)
    await hass.async_block_till_done()
    assert f"{DOMAIN}.01TESTENTRY" not in hass_storage


async def test_stale_devices_removed_on_setup(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Devices of models deleted while Home Assistant was down are removed."""
    mock_config_entry.add_to_hass(hass)
    server = device_registry.async_get_or_create(
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY")},
    )
    # A child device (0.3+) and a full device (older versions) of vanished models.
    device_registry.async_get_or_create_child(
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY_model_gone:1b")},
        parent_device_id=server.id,
    )
    device_registry.async_get_or_create(
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY_model_old:1b")},
    )
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert (
        device_registry.async_get_child_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_gone:1b"), "01TESTENTRY"
        )
        is None
    )
    assert (
        device_registry.async_get_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_old:1b"), "01TESTENTRY"
        )
        is None
    )


async def test_remove_config_entry_device(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Only devices of vanished models can be deleted by hand."""
    server = device_registry.async_get_device_by_identifier(
        (DOMAIN, "01TESTENTRY"), "01TESTENTRY"
    )
    model = device_registry.async_get_child_device_by_identifier(
        (DOMAIN, "01TESTENTRY_model_qwen3:8b"), "01TESTENTRY"
    )
    assert server is not None
    stale = device_registry.async_get_or_create_child(
        config_entry_id=init_integration.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY_model_gone:1b")},
        parent_device_id=server.id,
    )
    assert model is not None
    assert not await async_remove_config_entry_device(hass, init_integration, server)
    assert not await async_remove_config_entry_device(hass, init_integration, model)
    assert await async_remove_config_entry_device(hass, init_integration, stale)


async def test_sensor_values(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Server and model sensors report the expected values and units."""
    vram = hass.states.get("sensor.ollama_10_2_6_50_vram_used")
    assert vram.state == "6.973858774"
    assert vram.attributes["unit_of_measurement"] == "GB"
    assert vram.attributes["device_class"] == "data_size"
    assert hass.states.get("sensor.ollama_10_2_6_50_model_memory").state == (
        "19.67555839"
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_gpu_share").state == "35.4"
    assert hass.states.get("sensor.ollama_10_2_6_50_disk_usage").state == (
        "24.421238336"
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_installed_models").state == "3"
    assert hass.states.get("sensor.ollama_10_2_6_50_version").state == "0.40.2"
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_cpu_offload").state == (
        STATE_ON
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_context_length").state
        == "16384"
    )
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_unloads_at").state == (
        "2026-10-11T02:11:57+00:00"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_memory").state
        == "19.67555839"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_gpu_share").state == "35.4"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_context_length").state
        == "unknown"
    )


async def test_server_gpu_share_without_models(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """With nothing loaded the GPU share is unknown."""
    fake_ollama.loaded = []
    await _poll(hass, frozen_time)
    assert hass.states.get("sensor.ollama_10_2_6_50_gpu_share").state == "unknown"
    assert hass.states.get("sensor.ollama_10_2_6_50_vram_used").state == "0.0"


async def test_size_on_disk_disabled_by_default(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Size on disk is created disabled."""
    entry = entity_registry.async_get("sensor.ollama_10_2_6_50_qwen3_8b_size_on_disk")
    assert entry is not None
    assert entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION


async def test_size_on_disk_enabled(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    entity_registry: er.EntityRegistry,
) -> None:
    """Size on disk reports the installed size once enabled."""
    fake_ollama.loaded.append(
        {"name": "mystery:1b", "size": 1, "size_vram": 1, "details": {}}
    )
    await _poll(hass, frozen_time)
    for entity_id in (
        "sensor.ollama_10_2_6_50_qwen3_8b_size_on_disk",
        "sensor.ollama_10_2_6_50_mystery_1b_size_on_disk",
    ):
        entity_registry.async_update_entity(entity_id, disabled_by=None)
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_size_on_disk").state
        == "5.225388164"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_mystery_1b_size_on_disk").state
        == "unknown"
    )


def test_model_state_without_source() -> None:
    """A model state without data has empty details."""
    state = ModelState(
        name="x", installed=None, loaded=None, last_used=None, active=False
    )
    assert state.details == ModelDetails(None, None, None)


async def test_activity_flushed_on_reload(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    hass_storage: dict[str, Any],
) -> None:
    """A reload writes pending activity at once, so the new coordinator sees it."""
    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:30:00Z")
    await _poll(hass, frozen_time)
    used = hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
    assert f"{DOMAIN}.01TESTENTRY" not in hass_storage

    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()
    assert set(hass_storage[f"{DOMAIN}.01TESTENTRY"]["data"]["last_used"]) == {
        "qwen3:30b"
    }
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state == used

    # Nothing pending: unloading does not write again.
    hass_storage.pop(f"{DOMAIN}.01TESTENTRY")
    await hass.config_entries.async_unload(init_integration.entry_id)
    assert f"{DOMAIN}.01TESTENTRY" not in hass_storage


async def test_malformed_row_keeps_devices(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    entity_registry: er.EntityRegistry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """A row without a name rejects the poll instead of removing devices."""
    entity_registry.async_update_entity(
        "sensor.ollama_10_2_6_50_qwen3_8b_vram", name="My renamed sensor"
    )
    fake_ollama.installed = [
        m for m in fake_ollama.installed if m["name"] != "qwen3:8b"
    ] + [{"model": "qwen3:8b", "size": 1}]
    await _poll(hass, frozen_time)

    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_vram").state
        == STATE_UNAVAILABLE
    )
    assert (
        device_registry.async_get_child_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_qwen3:8b"), "01TESTENTRY"
        )
        is not None
    )
    entry = entity_registry.async_get("sensor.ollama_10_2_6_50_qwen3_8b_vram")
    assert entry is not None
    assert entry.name == "My renamed sensor"


async def test_outage_changes_are_not_activity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """Changes that happened during an outage are not stamped as current use."""
    fired = _record_events(hass)
    fake_ollama.fail = True
    await _poll(hass, frozen_time)

    # During the outage qwen3:30b served a request and qwen3:8b was loaded.
    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:40:00Z")
    fake_ollama.loaded.append(deepcopy(QWEN3_8B_PS))
    await _poll(hass, frozen_time, 180)
    fake_ollama.fail = False
    await _poll(hass, frozen_time)

    assert fired == [("model_loaded", "qwen3:8b")]
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_30b_last_used").state
        == "unknown"
    )
    assert (
        hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_last_used").state == "unknown"
    )
    assert hass.states.get("binary_sensor.ollama_10_2_6_50_active").state == STATE_OFF

    # The next change after recovery counts again.
    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:45:00Z")
    await _poll(hass, frozen_time)
    assert (
        hass.states.get("binary_sensor.ollama_10_2_6_50_qwen3_30b_active").state
        == STATE_ON
    )


async def test_active_model_reporting(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
) -> None:
    """The server reports which models are serving requests, newest first."""
    active = "binary_sensor.ollama_10_2_6_50_active"
    active_model = "sensor.ollama_10_2_6_50_active_model"
    state = hass.states.get(active)
    assert state.attributes["active_models"] == []
    assert state.attributes["last_active_model"] is None
    assert state.attributes["last_activity"] is None
    assert hass.states.get(active_model).state == "unknown"
    assert hass.states.get(active_model).attributes["active_models"] == []

    # Loaded but idle is not active.
    await _poll(hass, frozen_time)
    assert hass.states.get(active_model).state == "unknown"

    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:20:00Z")
    await _poll(hass, frozen_time)
    first_use = dt_util.utcnow()
    state = hass.states.get(active)
    assert state.state == STATE_ON
    assert state.attributes["active_models"] == ["qwen3:30b"]
    assert state.attributes["last_active_model"] == "qwen3:30b"
    assert state.attributes["last_activity"] == first_use
    assert hass.states.get(active_model).state == "qwen3:30b"

    # Loading a second model (e.g. a preload) does not displace the active one.
    fake_ollama.loaded.append(deepcopy(QWEN3_8B_PS))
    await _poll(hass, frozen_time)
    state = hass.states.get(active)
    assert state.attributes["active_models"] == ["qwen3:30b"]
    assert state.attributes["last_active_model"] == "qwen3:30b"
    assert state.attributes["last_activity"] == first_use
    assert hass.states.get(active_model).state == "qwen3:30b"

    # Once it serves a request, the newest one leads.
    _bump(fake_ollama, "qwen3:8b", "2026-10-11T02:20:30Z")
    await _poll(hass, frozen_time)
    second_use = dt_util.utcnow()
    state = hass.states.get(active)
    assert state.attributes["active_models"] == ["qwen3:8b", "qwen3:30b"]
    assert state.attributes["last_active_model"] == "qwen3:8b"
    assert state.attributes["last_activity"] == second_use
    model_state = hass.states.get(active_model)
    assert model_state.state == "qwen3:8b"
    assert model_state.attributes["active_models"] == ["qwen3:8b", "qwen3:30b"]

    # Both used in the same poll: ordered by name.
    _bump(fake_ollama, "qwen3:30b", "2026-10-11T02:21:00Z")
    _bump(fake_ollama, "qwen3:8b", "2026-10-11T02:21:00Z")
    await _poll(hass, frozen_time)
    assert hass.states.get(active).attributes["active_models"] == [
        "qwen3:30b",
        "qwen3:8b",
    ]
    assert hass.states.get(active_model).state == "qwen3:30b"

    # The activity window passes: nothing is active, the last model is remembered.
    for _ in range(6):
        await _poll(hass, frozen_time)
    state = hass.states.get(active)
    assert state.state == STATE_OFF
    assert state.attributes["active_models"] == []
    assert state.attributes["last_active_model"] == "qwen3:30b"
    assert hass.states.get(active_model).state == "unknown"


async def test_upgrade_converts_model_devices_in_place(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fake_ollama: FakeOllama,
    frozen_time: FrozenDateTimeFactory,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    area_registry: ar.AreaRegistry,
) -> None:
    """Model devices from 0.2 become child devices without losing anything."""
    mock_config_entry.add_to_hass(hass)
    office = area_registry.async_create("Office")
    rack = area_registry.async_create("Rack")
    # Registry state as left by 0.2: full model devices linked with via_device.
    server = device_registry.async_get_or_create(
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY")},
        name="Ollama (10.2.6.50)",
    )
    old_model = device_registry.async_get_or_create(
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(DOMAIN, "01TESTENTRY_model_qwen3:8b")},
        name="qwen3:8b",
        manufacturer="Ollama",
        model="qwen3 · 8.2B · Q4_K_M",
        sw_version="500a1f067a9f",
        via_device_id=server.id,
    )
    device_registry.async_update_device(
        old_model.id, name_by_user="Small Qwen", area_id=office.id
    )
    old_vram = entity_registry.async_get_or_create(
        "sensor",
        DOMAIN,
        "01TESTENTRY_model_qwen3:8b_model_vram",
        config_entry=mock_config_entry,
        device_id=old_model.id,
        suggested_object_id="qwen3_8b_vram",
    )
    entity_registry.async_update_entity(
        old_vram.entity_id, name="Qwen VRAM", area_id=rack.id
    )
    old_loaded = entity_registry.async_get_or_create(
        "binary_sensor",
        DOMAIN,
        "01TESTENTRY_model_qwen3:8b_loaded",
        config_entry=mock_config_entry,
        device_id=old_model.id,
        suggested_object_id="qwen3_8b_loaded",
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    assert old_vram.entity_id == "sensor.qwen3_8b_vram"
    # Upgrading means a restart: the registry no longer treats these devices as
    # registered by the running integration.
    device_registry.async_config_entry_unloaded(mock_config_entry.entry_id)

    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Same device id, now a child of the server; user settings kept.
    child = device_registry.async_get_child_device_by_identifier(
        (DOMAIN, "01TESTENTRY_model_qwen3:8b"), "01TESTENTRY"
    )
    assert child is not None
    assert child.id == old_model.id
    assert child.parent_device_id == server.id
    assert child.name == "qwen3:8b"
    assert child.name_by_user == "Small Qwen"
    assert child.area_id == office.id
    assert (
        device_registry.async_get_device_by_identifier(
            (DOMAIN, "01TESTENTRY_model_qwen3:8b"), "01TESTENTRY"
        )
        is None
    )

    # Entities keep entity_id (and so their history), overrides and device link.
    vram = entity_registry.async_get("sensor.qwen3_8b_vram")
    assert vram is not None
    assert vram.unique_id == old_vram.unique_id
    assert vram.name == "Qwen VRAM"
    assert vram.area_id == rack.id
    assert vram.device_id == child.id
    # (Created here without the GB display unit a real 0.2 install has.)
    assert hass.states.get("sensor.qwen3_8b_vram").state == "0"
    loaded = entity_registry.async_get(old_loaded.entity_id)
    assert loaded is not None
    assert loaded.disabled_by is er.RegistryEntryDisabler.USER
    assert loaded.device_id == child.id
    assert entity_registry.async_get_entity_id(
        "sensor", DOMAIN, "01TESTENTRY_model_qwen3:8b_model_vram"
    ) == ("sensor.qwen3_8b_vram")
    assert hass.states.get("sensor.ollama_10_2_6_50_qwen3_8b_vram") is None
    # New entities of the converted device join it.
    new_entity_id = entity_registry.async_get_entity_id(
        "sensor", DOMAIN, "01TESTENTRY_model_qwen3:8b_quantization"
    )
    assert new_entity_id is not None
    new_entity = entity_registry.async_get(new_entity_id)
    assert new_entity is not None
    assert hass.states.get(new_entity_id).state == "Q4_K_M"
    assert new_entity.device_id == child.id

    # Removing the model from the server still cleans up the converted device.
    fake_ollama.installed = [
        m for m in fake_ollama.installed if m["name"] != "qwen3:8b"
    ]
    await _poll(hass, frozen_time)
    assert device_registry.async_get(child.id) is None
    assert entity_registry.async_get("sensor.qwen3_8b_vram") is None
