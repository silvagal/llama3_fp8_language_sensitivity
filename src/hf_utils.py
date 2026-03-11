"""Hugging Face auth helpers."""
from __future__ import annotations

import os
from typing import Optional

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def get_hf_token() -> Optional[str]:
    for env_var in TOKEN_ENV_VARS:
        token = os.getenv(env_var)
        if token:
            return token
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return None
    return HfFolder.get_token()
