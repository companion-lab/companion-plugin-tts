# Third-party notices

This plugin's code is licensed under the MIT License (see [LICENSE](./LICENSE)).
Third-party models, libraries, and assets bundled with or fetched by this
plugin retain their own licenses, listed below.

## Supertonic-3 (text-to-speech)

- **What:** the ONNX model weights (fetched by `scripts/download-models.sh`),
  plus the config JSON and voice-style embeddings committed under
  `assets/supertonic/`. `F6.json` and `F7.json` are derivative voice
  embeddings created in this project from Supertonic's F1–M5 embeddings.
- **Source:** <https://huggingface.co/Supertone/supertonic-3>.
- **License:** BigScience **Open RAIL-M** — a permissive-style license with
  use-based restrictions. It is _not_ MIT; redistribution is allowed but the
  restrictions must be passed on. The full text is vendored at
  [`assets/supertonic/LICENSE`](assets/supertonic/LICENSE) and applies to
  everything in that directory, including the derived `F6.json` / `F7.json`.
  Distributions of this plugin ship these files together with the weights
  under the same terms.
