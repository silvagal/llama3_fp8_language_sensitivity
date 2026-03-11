# Lost in Quantization: Activation Outliers Explain Language-Specific FP8 Sensitivity in Llama-3

This repository contains the experimental pipeline used in the study **"Lost in Quantization: Activation Outliers Explain Language-Specific FP8 Sensitivity in Llama-3"** (submitted to **PROPOR**).

## Project overview

This project evaluates how low-precision inference affects Llama-3-8B in two languages:

- English (`en`)
- Brazilian Portuguese (`ptbr`)

It compares three inference regimes:

- **BF16** (reference baseline)
- **INT8** (outlier-aware quantization)
- **FP8 E4M3** (naive casting stress test, without calibration/scaling)

The pipeline:

1. Downloads and preprocesses text from Project Gutenberg.
2. Builds deterministic token chunks (2048 tokens, 64 chunks, fixed seed).
3. Computes perplexity (and optional KL divergence).
4. Captures attention activation outliers and generates histograms.
5. Consolidates outputs into a final report file.

## Main results

From the reported experiments:

- **BF16** is the reference.
- **INT8** preserved perplexity for both languages.
- **Naive FP8** degraded both languages, with stronger relative degradation in English.

### Perplexity summary

| Method | English PPL | English Δ | PT-BR PPL | PT-BR Δ |
|---|---:|---:|---:|---:|
| BF16 (Ref) | 3.45 | - | 13.25 | - |
| INT8 | 3.45 | +0.00 | 13.25 | +0.00 |
| FP8 (E4M3, naive) | 4.08 | +0.63 | 13.77 | +0.52 |

Relative increase under naive FP8:

- English: **+18%**
- PT-BR: **+3.9%**

### Outlier observations

- English showed rarer but more extreme activation spikes (max >35.50).
- PT-BR showed denser moderately high activations, but lower absolute maximum (31.88).
- This supports the interpretation that naive FP8 is especially sensitive to rare extreme spikes.

## Repository structure

- `src/data_loader.py`: data download, cleaning, tokenization, chunk generation.
- `src/evaluation.py`: perplexity and optional KL divergence.
- `src/analysis.py`: activation outlier capture and histogram generation.
- `scripts/01_run_perplexity_gap.sh`: runs BF16/INT8/FP8 perplexity experiments.
- `scripts/02_analyze_outliers.sh`: runs outlier analyses for BF16/INT8/FP8.
- `scripts/03_generate_final_results.py`: merges outputs into a consolidated summary.
- `run_all_experiments.bash`: end-to-end execution.

## How to run

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Optional) set Hugging Face token

If the model access requires authentication:

```bash
export HF_TOKEN="<your_token>"
```

The code also checks `HUGGINGFACEHUB_API_TOKEN` and `HUGGINGFACE_HUB_TOKEN`.

### 3) Run full pipeline

```bash
bash run_all_experiments.bash
```

### 4) Run stages separately (optional)

```bash
bash scripts/01_run_perplexity_gap.sh
bash scripts/02_analyze_outliers.sh
python scripts/03_generate_final_results.py
```

## Where results are saved

All outputs are written under `results/`:

- Perplexity/KL JSON files:
  - `results/bf16_en_ppl.json`
  - `results/bf16_ptbr_ppl.json`
  - `results/int8_en_ppl.json`
  - `results/int8_ptbr_ppl.json`
  - `results/fp8_en_ppl.json`
  - `results/fp8_ptbr_ppl.json`
- Outlier artifacts (per quantization and language):
  - `results/<quant>_<lang>_outliers/outliers_<lang>.json`
  - `results/<quant>_<lang>_outliers/hist_<lang>.png`
- Consolidated final report:
  - `results/final_quantization_results`

## Notes

- The FP8 setup here is a **naive casting stress test**, not an optimized FP8 recipe.
- The study is intentionally limited to one model and a literary-domain English/PT-BR comparison.
- This repository intentionally excludes personal contact details and author-identifying information in documentation and code.
