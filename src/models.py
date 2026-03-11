"""Model loading utilities with quantization wrappers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from quant_linguistics.src.hf_utils import get_hf_token
    from quant_linguistics.src.quantization import QuantizationConfig, load_quantized_model
except ModuleNotFoundError:
    from hf_utils import get_hf_token
    from quantization import QuantizationConfig, load_quantized_model

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    trust_remote_code: bool
    device_map: str
    use_auth_token: bool
    revision: str
    max_memory_gb: int


def load_tokenizer(model_name: str) -> AutoTokenizer:
    token = get_hf_token()
    if token:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    dtype: Optional[torch.dtype] = None,
) -> AutoModelForCausalLM:
    LOGGER.info("Loading model %s with quantization=%s", model_config.model_name, quant_config.quantization)
    dtype = resolve_dtype(dtype, quant_config.quantization)
    max_memory = build_max_memory(model_config.device_map, model_config.max_memory_gb)
    model = load_quantized_model(
        model_name=model_config.model_name,
        quant_config=quant_config,
        dtype=dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
        max_memory=max_memory,
        use_auth_token=model_config.use_auth_token,
    )
    return model


def resolve_dtype(requested: Optional[torch.dtype], quantization: str) -> torch.dtype:
    if requested is not None:
        return requested
    quantization = (quantization or "").lower()
    if quantization == "int8":
        if not torch.cuda.is_available():
            LOGGER.warning("INT8 requested without CUDA; falling back to float32.")
            return torch.float32
        if torch.cuda.is_bf16_supported():
            LOGGER.info("INT8 enabled with BF16 support; using float16 for compute stability.")
        return torch.float16
    if quantization == "bf16":
        return preferred_mixed_precision()
    if quantization == "fp8":
        return preferred_mixed_precision()
    return preferred_mixed_precision()


def preferred_mixed_precision() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        LOGGER.warning("BF16 not supported on this GPU; using float16.")
        return torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def build_max_memory(device_map: str, max_memory_gb: int) -> Optional[dict]:
    if max_memory_gb <= 0:
        return None
    memory = f"{max_memory_gb}GiB"
    device_map = (device_map or "").lower()
    if device_map in {"cpu", "mps"}:
        return {device_map: memory}
    if device_map.startswith("cuda"):
        index = 0
        if ":" in device_map:
            try:
                index = int(device_map.split(":", 1)[1])
            except ValueError:
                index = 0
        return {index: memory}
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count:
            return {idx: memory for idx in range(device_count)}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"mps": memory}
    return {"cpu": memory}
