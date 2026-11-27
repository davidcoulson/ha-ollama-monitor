"""The dev-only reload_code action swaps in the Python files on disk."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from custom_components.ollama_monitor import _CODE_VERSION, _clear_marketplace_restart
from custom_components.ollama_monitor.const import DOMAIN
from homeassistant.config_entries import ConfigEntries, ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir


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
        (importlib, "import_module"),
        (ConfigEntries, "async_setup"),
    ],
)
async def test_reload_code_reports_failures(hass, init_integration, target, result) -> None:
    kwargs = {"return_value": False}
    if target is importlib:  # unload needs the old module; only the fresh import fails
        real = importlib.import_module

        def kwargs_wrapper(name, *a, **k):
            if name == "custom_components.ollama_monitor" and name not in sys.modules:
                raise ImportError("boom")
            return real(name, *a, **k)

        kwargs = {"new": kwargs_wrapper}
    with patch.object(target, result, **kwargs), pytest.raises(HomeAssistantError):
        await hass.services.async_call(DOMAIN, "reload_code", {}, blocking=True)
    # the action survives a failed reload, so it can be called again once the code is fixed
    assert hass.services.has_service(DOMAIN, "reload_code")


async def test_reload_clears_only_the_running_versions_restart_repair(hass, monkeypatch) -> None:
    monkeypatch.setitem(_clear_marketplace_restart.__globals__, "_CODE_VERSION", "1.2.3")
    for domain, issue_id, issue_domain in (
        ("marketplace", "restart_required_1_tags/1.2.3", DOMAIN),
        ("marketplace", "restart_required_2_tags/v1.2.3", DOMAIN),
        ("marketplace", "restart_required_3_tags/1.2.4", DOMAIN),
        ("marketplace", "restart_required_4_tags/1.2.3", "ssh"),
        ("marketplace", "critical_x", DOMAIN),
        (DOMAIN, "restart_required_x", DOMAIN),
    ):
        ir.async_create_issue(
            hass, domain, issue_id, is_fixable=False, issue_domain=issue_domain,
            severity=ir.IssueSeverity.WARNING, translation_key="x",
        )
    _clear_marketplace_restart(hass)
    assert set(ir.async_get(hass).issues) == {
        ("marketplace", "restart_required_3_tags/1.2.4"),
        ("marketplace", "restart_required_4_tags/1.2.3"),
        ("marketplace", "critical_x"),
        (DOMAIN, "restart_required_x"),
    }


def test_code_version_is_the_manifest_version() -> None:
    manifest = Path(__file__).parents[1] / "custom_components" / "ollama_monitor" / "manifest.json"
    assert _CODE_VERSION == json.loads(manifest.read_text())["version"]
