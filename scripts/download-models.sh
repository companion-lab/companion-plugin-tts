#!/usr/bin/env bash
set -euo pipefail

HF_REPO="https://huggingface.co/Supertone/supertonic-3/resolve/main"
ASSETS_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/supertonic"

ONNX_FILES=(
  "onnx/duration_predictor.onnx"
  "onnx/text_encoder.onnx"
  "onnx/vector_estimator.onnx"
  "onnx/vocoder.onnx"
)

echo "Downloading Supertonic ONNX models..."

mkdir -p "$ASSETS_DIR/onnx"

for f in "${ONNX_FILES[@]}"; do
  dest="$ASSETS_DIR/$f"
  if [ -f "$dest" ] && [ "$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null)" -gt 1000000 ]; then
    echo "  skip $f (already exists)"
    continue
  fi
  echo "  get $f"
  curl -fSL "$HF_REPO/$f" -o "$dest"
done

echo "Done. Models ready at $ASSETS_DIR/"
