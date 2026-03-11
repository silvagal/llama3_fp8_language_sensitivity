"""Quantization logic for BF16, INT8, and FP8."""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoModelForCausalLM

try:
    from quant_linguistics.src.hf_utils import get_hf_token
except ModuleNotFoundError:
    from hf_utils import get_hf_token
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantizationConfig:
    quantization: str
    compute_kl: bool
    reference_quantization: str
    activation_capture: bool


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def ensure_module(module_name: str) -> None:
    if not module_available(module_name):
        raise RuntimeError(f"Required module '{module_name}' is not available.")


def load_quantized_model(
    model_name: str,
    quant_config: QuantizationConfig,
    dtype: torch.dtype,
    device_map: str,
    trust_remote_code: bool,
    revision: str,
    max_memory: Optional[Dict[Union[int, str], str]] = None,
    use_auth_token: bool = False,
) -> AutoModelForCausalLM:
    quantization = quant_config.quantization.lower()
    max_memory = normalize_max_memory(max_memory)
    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "max_memory": max_memory,
    }
    token = get_hf_token()
    if token:
        model_kwargs["use_auth_token"] = token
    elif use_auth_token:
        model_kwargs["use_auth_token"] = True

    if quantization == "int8":
        ensure_module("bitsandbytes")
        if not int8_supported():
            LOGGER.warning(
                "INT8 requested but bitsandbytes/cublasLt not available; falling back to BF16."
            )
            quantization = "bf16"
        else:
            int8_kwargs = dict(model_kwargs)
            int8_kwargs["load_in_8bit"] = True
            int8_kwargs["device_map"] = "auto"
            LOGGER.info("Using bitsandbytes INT8 quantization.")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **int8_kwargs)
            except Exception as exc:
                if is_cublaslt_error(exc):
                    LOGGER.warning(
                        "bitsandbytes INT8 failed (%s); falling back to BF16.",
                        exc,
                    )
                    quantization = "bf16"
                else:
                    raise
            else:
                model.eval()
                return model
    elif quantization == "fp8":
        if not fp8_supported():
            LOGGER.warning("FP8 requested but GPU/torch does not support it; falling back to BF16.")
            quantization = "bf16"
        else:
            LOGGER.info("FP8 enabled: casting weights to FP8 where supported.")
    elif quantization == "bf16":
        LOGGER.info("Using BF16 baseline.")
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if quantization == "fp8":
        model = cast_model_fp8(model, keep_dtype=dtype)

    model.eval()
    return model


def int8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from bitsandbytes import cextension
        from bitsandbytes.cuda_specs import get_cuda_specs
    except Exception:
        return False
    lib = getattr(cextension, "lib", None)
    if lib is None or not getattr(lib, "compiled_with_cuda", False):
        return False
    try:
        specs = get_cuda_specs()
    except Exception:
        return False
    return bool(specs and specs.has_cublaslt)


def is_cublaslt_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cublaslt" in message or "igemmlt" in message


def is_fp8_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "float8" in message or "fp8" in message


def normalize_max_memory(
    max_memory: Optional[Dict[Union[int, str], Any]]
) -> Optional[Dict[Union[int, str], Any]]:
    if not max_memory or not isinstance(max_memory, dict):
        return max_memory
    normalized: Dict[Any, Any] = {}
    for key, value in max_memory.items():
        new_key = key
        if isinstance(key, str):
            lowered = key.lower()
            if lowered == "cuda":
                new_key = 0
            elif lowered.startswith("cuda:"):
                index = lowered.split(":", 1)[1]
                try:
                    new_key = int(index)
                except ValueError:
                    new_key = 0
            elif lowered.isdigit():
                new_key = int(lowered)
        if new_key in normalized and isinstance(key, str) and isinstance(new_key, int):
            continue
        normalized[new_key] = value
    return normalized


def cast_model_fp8(model: AutoModelForCausalLM, keep_dtype: torch.dtype) -> AutoModelForCausalLM:
    fp8_dtype = torch.float8_e4m3fn
    linear_weight_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for param_name, _ in module.named_parameters(recurse=False):
                if param_name != "weight":
                    continue
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                linear_weight_names.add(full_name)

    for name, param in model.named_parameters():
        if not param.is_floating_point():
            LOGGER.debug("Skipping non-float param %s", name)
            continue
        if name in linear_weight_names:
            param.data = param.data.to(fp8_dtype)
        elif param.dtype != keep_dtype:
            param.data = param.data.to(keep_dtype)
    return model


def fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return False
    for idx in range(device_count):
        major, minor = torch.cuda.get_device_capability(idx)
        if (major, minor) < (9, 0):
            return False
    return True


def autocast_context(quantization: str):
    if not torch.cuda.is_available():
        return nullcontext()
    if quantization == "fp8":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if quantization == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)
