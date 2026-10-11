"""Constants for the Ollama Monitor integration."""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Final

DOMAIN: Final = "ollama_monitor"
LOGGER = logging.getLogger(__package__)

DEFAULT_PORT: Final = 11434

CONF_SCAN_INTERVAL: Final = "scan_interval"
CONF_ACTIVITY_WINDOW: Final = "activity_window"
CONF_CHECK_UPDATES: Final = "check_updates"
CONF_ALLOW_MANAGEMENT: Final = "allow_management"

DEFAULT_SCAN_INTERVAL: Final = 10
MIN_SCAN_INTERVAL: Final = 5
MAX_SCAN_INTERVAL: Final = 300
DEFAULT_ACTIVITY_WINDOW: Final = 60
MIN_ACTIVITY_WINDOW: Final = 10
MAX_ACTIVITY_WINDOW: Final = 3600
DEFAULT_CHECK_UPDATES: Final = True
DEFAULT_ALLOW_MANAGEMENT: Final = False

RELEASE_CHECK_INTERVAL: Final = timedelta(hours=6)
GITHUB_LATEST_RELEASE_URL: Final = (
    "https://api.github.com/repos/ollama/ollama/releases/latest"
)

# Raise a repair issue once the server has been unreachable this long.
UNREACHABLE_ISSUE_AFTER: Final = timedelta(minutes=5)
ISSUE_UNREACHABLE: Final = "server_unreachable"

ATTR_MODEL: Final = "model"
ATTR_MODELS: Final = "models"
ATTR_ACTIVE_MODELS: Final = "active_models"
ATTR_LAST_ACTIVE_MODEL: Final = "last_active_model"
ATTR_LAST_ACTIVITY: Final = "last_activity"
ATTR_CONFIG_ENTRY_ID: Final = "config_entry_id"

SERVICE_UNLOAD_MODEL: Final = "unload_model"
SERVICE_PULL_MODEL: Final = "pull_model"
SERVICE_DELETE_MODEL: Final = "delete_model"

EVENT_MODEL_LOADED: Final = "model_loaded"
EVENT_MODEL_UNLOADED: Final = "model_unloaded"
EVENT_MODEL_INSTALLED: Final = "model_installed"
EVENT_MODEL_REMOVED: Final = "model_removed"
EVENT_TYPES: Final = [
    EVENT_MODEL_LOADED,
    EVENT_MODEL_UNLOADED,
    EVENT_MODEL_INSTALLED,
    EVENT_MODEL_REMOVED,
]

STORAGE_VERSION: Final = 1
