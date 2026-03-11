"""Perplexity and KL evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.nn import functional as F

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
class EvalConfig:
    model_config: ModelConfig
    data_config_path: str
    quant_config: QuantizationConfig
    output_path: Path
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
    parser = argparse.ArgumentParser(description="Evaluate perplexity and KL divergence.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", choices=["ptbr", "en"], help="Override language")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
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
    return EvalConfig(
        model_config=model_config,
        data_config_path=args.data_config,
        quant_config=quant_config,
        output_path=Path(args.output),
        language_override=args.language,
    )


def compute_perplexity(model: torch.nn.Module, batches: Iterable[torch.Tensor], quantization: str) -> float:
    losses: List[float] = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(model.device)
            with autocast_context(quantization):
                outputs = model(batch, labels=batch)
                loss = outputs.loss
            losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    return math.exp(mean_loss)


def compute_kl_divergence(
    logits_p: torch.Tensor, logits_q: torch.Tensor
) -> torch.Tensor:
    prob_p = F.log_softmax(logits_p, dim=-1)
    prob_q = F.log_softmax(logits_q, dim=-1)
    return F.kl_div(prob_q, prob_p, reduction="batchmean", log_target=True)


def run_kl(
    model_p: torch.nn.Module,
    model_q: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    quant_p: str,
    quant_q: str,
) -> float:
    kl_values: List[float] = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(model_p.device)
            with autocast_context(quant_p):
                logits_p = model_p(batch).logits
            with autocast_context(quant_q):
                logits_q = model_q(batch).logits
            kl = compute_kl_divergence(logits_p, logits_q)
            kl_values.append(kl.item())
    return float(np.mean(kl_values))


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


def main() -> None:
    configure_logging()
    args = parse_args()
    config = build_config(args)

    batches = load_batches(config.data_config_path, config.language_override, config.model_config.model_name)

    quant_config = config.quant_config
    model = load_model(config.model_config, quant_config)
    try:
        ppl = compute_perplexity(model, batches, quant_config.quantization)
    except Exception as exc:
        if quant_config.quantization.lower() not in {"int8", "fp8"} or not (
            is_cublaslt_error(exc) or is_fp8_error(exc)
        ):
            raise
        LOGGER.warning("%s failed during evaluation (%s). Falling back to BF16.", quant_config.quantization, exc)
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
        ppl = compute_perplexity(model, batches, quant_config.quantization)

    results: Dict[str, float] = {"perplexity": ppl}

    if quant_config.compute_kl:
        reference_config = QuantizationConfig(
            quantization=quant_config.reference_quantization,
            compute_kl=False,
            reference_quantization=quant_config.reference_quantization,
            activation_capture=False,
        )
        reference_model = load_model(config.model_config, reference_config)
        try:
            kl = run_kl(
                reference_model,
                model,
                batches,
                reference_config.quantization,
                quant_config.quantization,
            )
        except Exception as exc:
            if (
                not (is_cublaslt_error(exc) or is_fp8_error(exc))
                or (
                    reference_config.quantization.lower() not in {"int8", "fp8"}
                    and quant_config.quantization.lower() not in {"int8", "fp8"}
                )
            ):
                raise
            LOGGER.warning(
                "%s failed during KL computation (%s). Falling back to BF16.",
                "INT8/FP8",
                exc,
            )
            reload_model = quant_config.quantization.lower() in {"int8", "fp8"}
            reload_reference = reference_config.quantization.lower() in {"int8", "fp8"}
            if reload_model:
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
                ppl = compute_perplexity(model, batches, quant_config.quantization)
                results["perplexity"] = ppl
            if reload_reference:
                del reference_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                reference_config = QuantizationConfig(
                    quantization="bf16",
                    compute_kl=False,
                    reference_quantization="bf16",
                    activation_capture=False,
                )
                reference_model = load_model(config.model_config, reference_config)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            kl = run_kl(reference_model, model, batches, reference_config.quantization, quant_config.quantization)
        results["kl_divergence"] = kl

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation to %s", config.output_path)


if __name__ == "__main__":
    main()
