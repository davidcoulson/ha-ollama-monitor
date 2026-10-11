# Ollama Monitor for Home Assistant

> **TL;DR:** See what your local AI server is doing, right in Home Assistant.
>
> - **What's running:** which AI models are loaded right now, and when each was last used.
> - **Is it fast:** how much graphics-card memory they use, and a warning you can automate on
>   when a model doesn't fit and slows down by running partly on the CPU.
> - **What's installed:** every model gets its own device, plus disk usage and events when
>   models are loaded, unloaded, installed or removed.
> - **Light housekeeping:** unload a model with one action. Downloading and deleting models
>   is available but off until you allow it.
>
> It only watches. It doesn't replace Home Assistant's built-in Ollama chat integration and
> works alongside it. Set it up in the UI by entering your Ollama server's address.

A Home Assistant custom integration that **watches** an [Ollama](https://ollama.com) server:
which models are loaded, how much VRAM they use, whether they spill over to the CPU,
when they were last used and when they will be unloaded.

Home Assistant's built-in `ollama` integration turns Ollama models into conversation
agents; it does not report what the server is doing. Ollama Monitor fills that gap and
can sit alongside it (they use different domains: `ollama` vs `ollama_monitor`).

Everything is configured in the UI. There is no YAML configuration.

**Requirements:** Home Assistant 2026.10 or newer and any Ollama server reachable over
HTTP(S) (developed and tested against Ollama 0.40.2; fields that older servers do not
report, such as the context length, show as unknown).

## Feature set

Ollama's HTTP API has no authentication, no metrics endpoint and no request counters,
so the integration polls `/api/version`, `/api/ps` and `/api/tags` and derives
everything else from changes between polls.

### Devices

- **One device per Ollama server** (manufacturer *Ollama*, software version = the
  server's version, link to the server URL).
- **One sub-device per model** (installed or loaded), named by the model tag (for
  example `qwen3:8b`) and shown under the server device. Family, parameter size and
  quantization are diagnostic sensors on the model device.

Model devices and their entities are created and removed automatically as models are
installed, loaded, unloaded and deleted, without reloading the integration. A model
device that disappears from the server is removed; stale devices can also be deleted
by hand from the device page.

### Server entities

| Entity | Type | Notes |
| --- | --- | --- |
| Reachable | binary sensor (connectivity, diagnostic) | Stays available while the server is down, so it can drive automations |
| Active | binary sensor (running) | On when any model handled a request within the *activity window*. Attributes: `active_models` (most recently used first), `last_active_model` and its `last_activity` time |
| Active model | sensor | The model that most recently served a request, while it is still within the activity window; unknown when idle. The `active_models` attribute lists every active model when more than one is busy |
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
| Family | sensor (diagnostic) | e.g. `qwen3` |
| Parameter size | sensor (diagnostic) | e.g. `8.2B` |
| Quantization | sensor (diagnostic) | e.g. `Q4_K_M` |
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
poll sees a new `expires_at` for a model that was already loaded, the model counts as
used at that moment. Loading alone does not count, because a preload (a request with an
empty prompt) looks the same; while a model's first request is still running Ollama
reports a moving `expires_at`, so that request is picked up on the next poll.
Consequences:

- Precision is one poll interval (10 s by default).
- Ollama only resets `expires_at` when a request **finishes**, so a long generation
  shows up when it completes.
- Requests to models that are not kept loaded (`keep_alive: 0`) are only seen if a
  poll happens while they are loaded.
- A request that loads a model and finishes between two polls is not counted (it is
  indistinguishable from a preload); the next request to that model is.

### Repairs and diagnostics

- If the server stays unreachable for 5 minutes a repair issue is raised; it clears
  itself when the server answers again.
- *Download diagnostics* returns the options, the last API responses and the activity
  state, with the server URL redacted.

### Upgrading from 0.2 or earlier

Model devices used to be separate devices linked to the server. From 0.3 they are
sub-devices of the server. Home Assistant converts the existing model devices in place
on the first start after the update: device and entity IDs, names, areas, labels,
disabled entities and history are kept. Models added afterwards get entity IDs that
start with the server name (for example `sensor.ollama_10_2_6_50_qwen3_8b_vram`).

**Going back to 0.2 is not supported by simply reinstalling it.** Once 0.3 has converted
the model devices, 0.2 cannot register its model entities on them (Home Assistant
refuses to treat a sub-device as a regular device). Before updating, let HACS create a
backup (or take one yourself). To roll back, restore that backup. Without a backup,
delete the Ollama Monitor entry under **Settings → Devices & services**, install 0.2,
restart and add the server again; all devices and entities are recreated, but entity
customisations and the stored *Last used* history are lost.

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

## Use cases and examples

- Put *VRAM used*, *GPU share* and the loaded model list next to your GPU's own
  sensors (for example the Unraid integration's GPU utilisation and memory) to see
  which model is holding the card.
- Get told when a model spills over to system RAM, which makes it several times slower:

  ```yaml
  triggers:
    - trigger: state
      entity_id: binary_sensor.ollama_10_2_6_50_cpu_offload
      to: "on"
      for: "00:01:00"
  actions:
    - action: notify.notify
      data:
        message: >
          Ollama is running {{ state_attr('sensor.ollama_10_2_6_50_loaded_models', 'models') | join(', ') }}
          partly on the CPU ({{ states('sensor.ollama_10_2_6_50_gpu_share') }} % on the GPU).
  ```

- React to a model being loaded (the event entity's `event_type` and `model` attributes):

  ```yaml
  triggers:
    - trigger: state
      entity_id: event.ollama_10_2_6_50_model_event
      not_from: unavailable
  conditions:
    - condition: state
      entity_id: event.ollama_10_2_6_50_model_event
      attribute: event_type
      state: model_loaded
  actions:
    - action: logbook.log
      data:
        name: Ollama
        message: "Loaded {{ trigger.to_state.attributes.model }}"
  ```

- Show which model is answering right now: an entity card with
  `sensor.ollama_10_2_6_50_active_model`, or a template such as
  `{{ state_attr('binary_sensor.ollama_10_2_6_50_active', 'last_active_model') }}`.
- Free the GPU for something else, e.g. before a game or a transcode job:
  `action: ollama_monitor.unload_model` with the server and model, or press the model's
  **Unload** button.

## Data updates

The server is polled every *polling interval* (default 10 s) with three small GET
requests. The GitHub release check runs every 6 hours. Pull and delete are only ever
sent when you call those actions.

## Known limitations

- Ollama exposes no request counters, queue length or token rates, so *Active* and
  *Last used* are inferred (see above) and cannot count requests.
- GPU utilisation, temperature and total VRAM are not available from Ollama; use your
  GPU or host integration for those.
- Ollama has no authentication; anyone who can reach the URL can also unload, pull and
  delete models. The integration only does that when you ask it to.
- The update entity only notifies. Update Ollama the way you installed it (Docker image,
  package manager).

## Troubleshooting

- **Failed to connect**: Ollama listens on `127.0.0.1` by default. Set
  `OLLAMA_HOST=0.0.0.0` (in Docker: publish port 11434) and check that
  `curl http://<host>:11434/api/version` works from the Home Assistant host.
- **The server answered, but not like an Ollama server**: the URL points at something
  else, or a reverse proxy rewrites the API paths.
- **The update entity is unavailable**: GitHub could not be reached or rate-limited the
  request; it retries every 6 hours. Turn the option off if you do not need it.
- Download the diagnostics from the integration entry and attach them to an issue.

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

The integration's icon and logo (shown by Home Assistant from
`custom_components/ollama_monitor/brand/`) are original artwork. The SVG sources are in
`assets/brand/`; `python3 assets/brand/render.py` regenerates the PNGs (needs Google
Chrome and ImageMagick).

## License

MIT
