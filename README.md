# Ollama Monitor for Home Assistant

A Home Assistant custom integration that **watches** an [Ollama](https://ollama.com) server:
which models are loaded, how much VRAM they use, whether they spill over to the CPU,
when they were last used and when they will be unloaded.

Home Assistant's built-in `ollama` integration turns Ollama models into conversation
agents; it does not report what the server is doing. Ollama Monitor fills that gap and
can sit alongside it (they use different domains: `ollama` vs `ollama_monitor`).

Everything is configured in the UI. There is no YAML configuration.

## Feature set

Ollama's HTTP API has no authentication, no metrics endpoint and no request counters,
so the integration polls `/api/version`, `/api/ps` and `/api/tags` and derives
everything else from changes between polls.

### Devices

- **One device per Ollama server** (manufacturer *Ollama*, software version = the
  server's version, link to the server URL).
- **One device per model** (installed or loaded), attached to the server device
  via `via_device`. The device model shows family, parameter count and
  quantization (for example `qwen3 · 8.2B · Q4_K_M`).

Model devices and their entities are created and removed automatically as models are
installed, loaded, unloaded and deleted, without reloading the integration. A model
device that disappears from the server is removed; stale devices can also be deleted
by hand from the device page.

### Server entities

| Entity | Type | Notes |
| --- | --- | --- |
| Reachable | binary sensor (connectivity, diagnostic) | Stays available while the server is down, so it can drive automations |
| Active | binary sensor (running) | On when any model handled a request within the *activity window* |
| CPU offload | binary sensor | On when any loaded model does not fit completely in VRAM (slow inference) |
| Loaded models | sensor | Count; the `models` attribute lists their names |
| Installed models | sensor | Count |
| VRAM used | sensor (data size) | Sum of `size_vram` over loaded models |
| Model memory | sensor (data size) | Sum of `size` over loaded models (VRAM + system RAM) |
| GPU share | sensor (%) | VRAM used / model memory, the server-wide CPU/GPU split |
| Disk usage | sensor (data size, diagnostic) | Total size of installed models |
| Last activity | sensor (timestamp) | Most recent detected request on any model |
| Version | sensor (diagnostic) | Ollama server version |
| Model event | event | `model_loaded`, `model_unloaded`, `model_installed`, `model_removed` with a `model` attribute |
| *(server name)* | update (optional) | Compares the server version with the latest GitHub release of `ollama/ollama`; shows release notes. Read-only: Ollama runs in Docker or as a system service, so updates are installed outside Home Assistant |

### Model entities

| Entity | Type | Notes |
| --- | --- | --- |
| Loaded | binary sensor | Model is in memory |
| Active | binary sensor (running) | Model handled a request within the activity window |
| Memory | sensor (data size) | `size` while loaded, 0 when not loaded |
| VRAM | sensor (data size) | `size_vram` while loaded, 0 when not loaded |
| GPU share | sensor (%) | `size_vram / size`, the GPU half of what `ollama ps` shows as e.g. `64%/36% CPU/GPU`; unknown when not loaded |
| Context length | sensor (diagnostic) | Context window the model was loaded with |
| Unloads at | sensor (timestamp) | `expires_at`; unknown when not loaded or when kept alive forever |
| Last used | sensor (timestamp) | Last detected request; survives restarts |
| Size on disk | sensor (data size, diagnostic, disabled by default) | From `/api/tags` |
| Unload | button | Frees the model's memory (`keep_alive: 0`); available only while loaded |

### Actions

| Action | Guarded | Description |
| --- | --- | --- |
| `ollama_monitor.unload_model` | no | Unload a model by name |
| `ollama_monitor.pull_model` | yes | Download (or update) a model; runs in the background and reports failures in the log |
| `ollama_monitor.delete_model` | yes | Delete an installed model |

Guarded actions are refused unless **Allow model management** is enabled in the
integration's options (off by default).

### How activity is detected

Every request to a loaded model resets its `expires_at` to *now + keep_alive*. When a
poll sees a new `expires_at` (or a model that was not loaded before), the model counts
as used at that moment. Consequences:

- Precision is one poll interval (10 s by default).
- Ollama only resets `expires_at` when a request **finishes**, so a long generation
  shows up when it completes.
- Requests to models that are not kept loaded (`keep_alive: 0`) are only seen if a
  poll happens while they are loaded.

### Repairs and diagnostics

- If the server stays unreachable for 5 minutes a repair issue is raised; it clears
  itself when the server answers again.
- *Download diagnostics* returns the options, the last API responses and the activity
  state, with the server URL redacted.

## Installation

### HACS (custom repository)

1. In HACS, open the menu, choose **Custom repositories** and add this repository's
   URL with category **Integration**.
2. Install **Ollama Monitor** and restart Home Assistant.

### Manual

Copy `custom_components/ollama_monitor` into your Home Assistant `config/custom_components`
folder and restart Home Assistant.

## Setup

1. **Settings → Devices & services → Add integration → Ollama Monitor**.
2. Enter the server URL, for example `http://10.2.6.50:11434`. A bare host such as
   `10.2.6.50` is accepted and expanded to `http://10.2.6.50:11434`.
3. The integration checks that `/api/version` answers before creating the entry. Each
   server can be added once.

To change the URL later use **Reconfigure** on the integration entry.

### Options

| Option | Default | Description |
| --- | --- | --- |
| Polling interval | 10 s | How often `/api/ps`, `/api/tags` and `/api/version` are read (5–300 s) |
| Activity window | 60 s | How long a model counts as *Active* after a detected request |
| Check GitHub for new Ollama releases | on | Creates the update entity; polls the GitHub API every 6 hours |
| Allow model management | off | Enables the pull and delete actions |

## Removal

Delete the integration entry under **Settings → Devices & services**. No data is
left on the Ollama server; the stored activity history is removed with the entry.

## Development

```bash
uv venv --python 3.14
uv pip install -r requirements_test.txt
pytest
ruff check . && ruff format --check . && mypy
```

## License

MIT
