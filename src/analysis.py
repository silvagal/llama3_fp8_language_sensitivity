"""Activation outlier analysis and visualization."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

try:
    from quant_linguistics.src.data_loader import chunks_to_batches, get_config as get_data_config, load_chunks
    from quant_linguistics.src.models import ModelConfig, load_model
    from quant_linguistics.src.quantization import (
        QuantizationConfig,
        autocast_context,
        is_cublaslt_error,
        is_fp8_error,
    )
except ModuleNotFoundError:
    from data_loader import chunks_to_batches, get_config as get_data_config, load_chunks
    from models import ModelConfig, load_model
    from quantization import QuantizationConfig, autocast_context, is_cublaslt_error, is_fp8_error

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    model_config: ModelConfig
    data_config_path: str
    quant_config: QuantizationConfig
    output_dir: Path
    language_override: str | None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze activation outliers.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", choices=["ptbr", "en"], help="Override language")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    model_raw = load_yaml(args.model_config)
    quant_raw = load_yaml(args.quant_config)
    model_config = ModelConfig(
        model_name=model_raw["model_name"],
        trust_remote_code=model_raw["trust_remote_code"],
        device_map=model_raw["device_map"],
        use_auth_token=model_raw["use_auth_token"],
        revision=model_raw["revision"],
        max_memory_gb=model_raw["max_memory_gb"],
    )
    quant_config = QuantizationConfig(
        quantization=quant_raw["quantization"],
        compute_kl=quant_raw["compute_kl"],
        reference_quantization=quant_raw["reference_quantization"],
        activation_capture=quant_raw["activation_capture"],
    )
    return AnalysisConfig(
        model_config=model_config,
        data_config_path=args.data_config,
        quant_config=quant_config,
        output_dir=Path(args.output_dir),
        language_override=args.language,
    )


def load_batches(data_config_path: str, language_override: str | None, tokenizer_name: str) -> List[torch.Tensor]:
    dummy_args = argparse.Namespace(data_config=data_config_path, language=language_override, model_name=tokenizer_name)
    data_config = get_data_config(dummy_args)
    chunks_path = Path(data_config.processed_dir) / f"{data_config.language}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunks at {chunks_path}. Run data_loader.py first to prepare dataset."
        )
    chunks = load_chunks(chunks_path)
    return chunks_to_batches(chunks)


def find_attention_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    modules: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if "self_attn" in name or name.endswith("attn"):
            modules.append((name, module))
    return modules


def capture_outliers(
    model: torch.nn.Module, batches: List[torch.Tensor], quantization: str
) -> Dict[str, List[float]]:
    outliers: Dict[str, List[float]] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            max_val = tensor.detach().abs().max().item()
            outliers.setdefault(layer_name, []).append(max_val)

        return hook

    for name, module in find_attention_modules(model):
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in batches:
            batch = batch.to(model.device)
            with autocast_context(quantization):
                model(batch)

    for handle in handles:
        handle.remove()

    return outliers


def plot_histogram(values: List[float], title: str, output_path: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(values, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Max |activation|")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = build_config(args)

    batches = load_batches(config.data_config_path, config.language_override, config.model_config.model_name)

    quant_config = config.quant_config
    model = load_model(config.model_config, quant_config)
    try:
        outliers = capture_outliers(model, batches, quant_config.quantization)
    except Exception as exc:
        if quant_config.quantization.lower() not in {"int8", "fp8"} or not (
            is_cublaslt_error(exc) or is_fp8_error(exc)
        ):
            raise
        LOGGER.warning("%s failed during outlier capture (%s). Falling back to BF16.", quant_config.quantization, exc)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        quant_config = QuantizationConfig(
            quantization="bf16",
            compute_kl=quant_config.compute_kl,
            reference_quantization=quant_config.reference_quantization,
            activation_capture=quant_config.activation_capture,
        )
        model = load_model(config.model_config, quant_config)
        outliers = capture_outliers(model, batches, quant_config.quantization)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outliers_path = config.output_dir / f"outliers_{config.language_override or 'default'}.json"
    outliers_path.write_text(json.dumps(outliers, indent=2), encoding="utf-8")

    flattened = [value for layer_vals in outliers.values() for value in layer_vals]
    hist_path = config.output_dir / f"hist_{config.language_override or 'default'}.png"
    plot_histogram(flattened, "Activation Outliers", hist_path)

    LOGGER.info("Saved outliers to %s", outliers_path)
    LOGGER.info("Saved histogram to %s", hist_path)


if __name__ == "__main__":
    main()
