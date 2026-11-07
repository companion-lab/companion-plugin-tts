#!/usr/bin/env bash
set -euo pipefail

HF_REPO="https://huggingface.co/Supertone/supertonic-3/resolve/main"
ASSETS_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/supertonic"

ONNX_FILES=(
  "onnx/duration_predictor.onnx"
  "onnx/text_encoder.onnx"
  "onnx/vector_estimator.onnx"
  "onnx/vocoder.onnx"
  "onnx/tts.json"
  "onnx/unicode_indexer.json"
)

VOICE_STYLES=(F1 F2 F3 F4 F5 M1 M2 M3 M4 M5)

echo "Downloading Supertonic models, configs, and voice styles..."

mkdir -p "$ASSETS_DIR/onnx" "$ASSETS_DIR/voice_styles"

# tts.json (inference config) and unicode_indexer.json (text -> id mapping)
# are required by the backend at init; voice_styles/*.json provide the
# built-in voices. All come from the same HF repo as the ONNX weights.
for f in "${ONNX_FILES[@]}"; do
  dest="$ASSETS_DIR/$f"
  if [ -s "$dest" ]; then
    echo "  skip $f (already exists)"
    continue
  fi
  echo "  get $f"
  curl -fSL "$HF_REPO/$f" -o "$dest"
done

for v in "${VOICE_STYLES[@]}"; do
  dest="$ASSETS_DIR/voice_styles/$v.json"
  if [ -s "$dest" ]; then
    echo "  skip voice_styles/$v.json (already exists)"
    continue
  fi
  echo "  get voice_styles/$v.json"
  curl -fSL "$HF_REPO/voice_styles/$v.json" -o "$dest"
done

echo "Done. Models ready at $ASSETS_DIR/"
