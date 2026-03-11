#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs"
RESULTS_DIR="$ROOT_DIR/results"
TMP_DIR="$RESULTS_DIR/tmp_configs"

mkdir -p "$RESULTS_DIR" "$TMP_DIR"

MODEL_NAME=$(python - <<PY
import yaml
from pathlib import Path
with open(Path("$CONFIG_DIR") / "model_config.yaml", "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["model_name"])
PY
)

for LANG in ptbr en; do
  python "$ROOT_DIR/src/data_loader.py" \
    --data-config "$CONFIG_DIR/data_config.yaml" \
    --model-name "$MODEL_NAME" \
    --language "$LANG"

done

for QUANT in bf16 int8 fp8; do
  TMP_QUANT="$TMP_DIR/quant_${QUANT}.yaml"
  python - <<PY
import yaml
from pathlib import Path
src = Path("$CONFIG_DIR/quant_config.yaml")
raw = yaml.safe_load(src.read_text(encoding="utf-8"))
raw["quantization"] = "$QUANT"
raw["activation_capture"] = True
Path("$TMP_QUANT").write_text(yaml.safe_dump(raw), encoding="utf-8")
PY
  for LANG in ptbr en; do
    python "$ROOT_DIR/src/analysis.py" \
      --model-config "$CONFIG_DIR/model_config.yaml" \
      --data-config "$CONFIG_DIR/data_config.yaml" \
      --quant-config "$TMP_QUANT" \
      --output-dir "$RESULTS_DIR/${QUANT}_${LANG}_outliers" \
      --language "$LANG"
  done

done
