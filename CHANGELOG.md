# Changelog

## 0.3.1

- New icon and logo: Ollama's llama with a gauge badge, in light and dark variants
  (`icon`, `dark_icon`, `logo`, `dark_logo`, each with `@2x`). The llama is Ollama's
  trademark, used to identify the monitored product; the badge marks this as an
  unofficial monitor, distinct from Home Assistant's core `ollama` integration.

## 0.3.0

- Models are sub-devices of the server device, named by their tag.
- Diagnostic Family, Parameter size and Quantization sensors on each model device.
- Existing model devices are converted in place on the first start after the update;
  take a backup first, 0.2 cannot run on converted devices.

## 0.2.1

- Integration icon and logo.

## 0.2.0

- Active model sensor; `active_models`, `last_active_model` and `last_activity`
  attributes on the Active binary sensor.
- Loading a model no longer counts as use.

## 0.1.0

- First release.
