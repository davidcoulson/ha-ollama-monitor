"""Config flow for Ollama Monitor."""

from __future__ import annotations

from typing import Any

import probatio
from yarl import URL

from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_URL
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api import OllamaClient, OllamaConnectionError, OllamaError
from .const import (
    CONF_ACTIVITY_WINDOW,
    CONF_ALLOW_MANAGEMENT,
    CONF_CHECK_UPDATES,
    CONF_SCAN_INTERVAL,
    DEFAULT_ACTIVITY_WINDOW,
    DEFAULT_ALLOW_MANAGEMENT,
    DEFAULT_CHECK_UPDATES,
    DEFAULT_PORT,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    LOGGER,
    MAX_ACTIVITY_WINDOW,
    MAX_SCAN_INTERVAL,
    MIN_ACTIVITY_WINDOW,
    MIN_SCAN_INTERVAL,
)
from .coordinator import OllamaConfigEntry

URL_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.URL))


class InvalidUrlError(ValueError):
    """The URL cannot be used."""


def normalize_url(raw: str) -> str:
    """Return a canonical server URL.

    A bare host (no scheme) gets `http://` and Ollama's default port.
    """
    raw = raw.strip()
    has_scheme = "://" in raw
    try:
        url = URL(raw if has_scheme else f"http://{raw}")
    except ValueError as err:
        raise InvalidUrlError from err
    if url.scheme not in ("http", "https") or not url.host:
        raise InvalidUrlError
    if not has_scheme and url.explicit_port is None:
        url = url.with_port(DEFAULT_PORT)
    url = url.with_query(None).with_fragment(None).with_user(None)
    return str(url).rstrip("/")


def default_title(url: str) -> str:
    """Return the default entry title for a URL."""
    return f"Ollama ({URL(url).host})"


def _user_schema(url: str | None = None) -> probatio.Schema:
    return probatio.Schema(
        {
            probatio.Required(
                CONF_URL, default=url or probatio.UNDEFINED
            ): URL_SELECTOR,
        }
    )


class OllamaMonitorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ollama Monitor."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: OllamaConfigEntry) -> OptionsFlowHandler:
        """Return the options flow."""
        return OptionsFlowHandler()

    async def _async_validate(
        self, user_input: dict[str, Any], errors: dict[str, str]
    ) -> tuple[str, str] | None:
        """Validate the URL; return (url, version) or None with errors set."""
        try:
            url = normalize_url(user_input[CONF_URL])
        except InvalidUrlError:
            errors[CONF_URL] = "invalid_url"
            return None
        client = OllamaClient(async_get_clientsession(self.hass), url)
        try:
            version = await client.async_get_version()
        except OllamaConnectionError:
            errors["base"] = "cannot_connect"
        except OllamaError:
            errors["base"] = "not_ollama"
        except Exception:
            LOGGER.exception("Unexpected error validating %s", url)
            errors["base"] = "unknown"
        else:
            return url, version
        return None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the server URL."""
        errors: dict[str, str] = {}
        if user_input is not None and (
            result := await self._async_validate(user_input, errors)
        ):
            url, _version = result
            await self.async_set_unique_id(url)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=default_title(url), data={CONF_URL: url}
            )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                _user_schema(), user_input or {}
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the server URL."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None and (
            result := await self._async_validate(user_input, errors)
        ):
            url, _version = result
            await self.async_set_unique_id(url)
            if url != entry.unique_id:
                self._abort_if_unique_id_configured()
            title = (
                default_title(url)
                if entry.title == default_title(entry.data[CONF_URL])
                else entry.title
            )
            return self.async_update_reload_and_abort(
                entry,
                unique_id=url,
                title=title,
                data_updates={CONF_URL: url},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_user_schema(
                user_input[CONF_URL] if user_input else entry.data[CONF_URL]
            ),
            errors=errors,
        )


class OptionsFlowHandler(OptionsFlowWithReload):
    """Handle the options."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the options."""
        errors: dict[str, str] = {}
        if user_input is not None and int(user_input[CONF_ACTIVITY_WINDOW]) < int(
            user_input[CONF_SCAN_INTERVAL]
        ):
            errors[CONF_ACTIVITY_WINDOW] = "activity_window_too_short"
        elif user_input is not None:
            return self.async_create_entry(
                data={
                    CONF_SCAN_INTERVAL: int(user_input[CONF_SCAN_INTERVAL]),
                    CONF_ACTIVITY_WINDOW: int(user_input[CONF_ACTIVITY_WINDOW]),
                    CONF_CHECK_UPDATES: user_input[CONF_CHECK_UPDATES],
                    CONF_ALLOW_MANAGEMENT: user_input[CONF_ALLOW_MANAGEMENT],
                }
            )
        options = {**self.config_entry.options, **(user_input or {})}
        schema = probatio.Schema(
            {
                probatio.Required(
                    CONF_SCAN_INTERVAL,
                    default=options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=MIN_SCAN_INTERVAL,
                        max=MAX_SCAN_INTERVAL,
                        step=1,
                        unit_of_measurement="s",
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                probatio.Required(
                    CONF_ACTIVITY_WINDOW,
                    default=options.get(CONF_ACTIVITY_WINDOW, DEFAULT_ACTIVITY_WINDOW),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=MIN_ACTIVITY_WINDOW,
                        max=MAX_ACTIVITY_WINDOW,
                        step=1,
                        unit_of_measurement="s",
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                probatio.Required(
                    CONF_CHECK_UPDATES,
                    default=options.get(CONF_CHECK_UPDATES, DEFAULT_CHECK_UPDATES),
                ): BooleanSelector(),
                probatio.Required(
                    CONF_ALLOW_MANAGEMENT,
                    default=options.get(
                        CONF_ALLOW_MANAGEMENT, DEFAULT_ALLOW_MANAGEMENT
                    ),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)
