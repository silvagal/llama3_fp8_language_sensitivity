#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python src/data_loader.py \
  --data-config configs/data_config.yaml \
  --model-name meta-llama/Meta-Llama-3-8B \
  --language ptbr

python src/data_loader.py \
  --data-config configs/data_config.yaml \
  --model-name meta-llama/Meta-Llama-3-8B \
  --language en

bash scripts/01_run_perplexity_gap.sh
bash scripts/02_analyze_outliers.sh
python scripts/03_generate_final_results.py
