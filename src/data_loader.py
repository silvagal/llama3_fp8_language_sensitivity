"""Dataset utilities for Project Gutenberg corpora."""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
import torch
import yaml
from transformers import AutoTokenizer

try:
    from quant_linguistics.src.hf_utils import get_hf_token
except ModuleNotFoundError:
    from hf_utils import get_hf_token
LOGGER = logging.getLogger(__name__)

GUTENBERG_HEADER = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_FOOTER = "*** END OF THE PROJECT GUTENBERG EBOOK"
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class DataConfig:
    language: str
    ptbr_url: str
    en_url: str
    cache_dir: str
    processed_dir: str
    sequence_length: int
    max_characters: int
    num_chunks: int
    seed: int


@dataclass(frozen=True)
class DatasetArtifacts:
    raw_path: Path
    processed_path: Path
    chunks_path: Path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess Gutenberg data.")
    parser.add_argument("--data-config", required=True, help="Path to YAML data config.")
    parser.add_argument("--model-name", required=True, help="Tokenizer model name.")
    parser.add_argument("--language", choices=["ptbr", "en"], help="Override language.")
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> DataConfig:
    raw_config = load_yaml(args.data_config)
    language = args.language or raw_config["language"]
    base_dir = Path(args.data_config).resolve().parent
    cache_dir = Path(raw_config["cache_dir"])
    processed_dir = Path(raw_config["processed_dir"])
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir
    if not processed_dir.is_absolute():
        processed_dir = base_dir / processed_dir
    return DataConfig(
        language=language,
        ptbr_url=raw_config["ptbr_url"],
        en_url=raw_config["en_url"],
        cache_dir=str(cache_dir),
        processed_dir=str(processed_dir),
        sequence_length=raw_config["sequence_length"],
        max_characters=raw_config["max_characters"],
        num_chunks=raw_config["num_chunks"],
        seed=raw_config["seed"],
    )


def resolve_artifacts(config: DataConfig) -> DatasetArtifacts:
    cache_dir = Path(config.cache_dir)
    processed_dir = Path(config.processed_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{config.language}.txt"
    processed_path = processed_dir / f"{config.language}_clean.txt"
    chunks_path = processed_dir / f"{config.language}_chunks.json"
    return DatasetArtifacts(raw_path=raw_path, processed_path=processed_path, chunks_path=chunks_path)


def download_text(url: str, destination: Path) -> None:
    LOGGER.info("Downloading %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")


def strip_gutenberg(text: str) -> str:
    start_idx = text.find(GUTENBERG_HEADER)
    if start_idx != -1:
        start_line_end = text.find("\n", start_idx)
        if start_line_end != -1:
            text = text[start_line_end + 1 :]
        else:
            text = text[start_idx:]
    end_idx = text.find(GUTENBERG_FOOTER)
    if end_idx != -1:
        text = text[:end_idx]
    return text


def redact_personal_info(text: str) -> str:
    """Redact explicit personal identifiers that may appear in source corpora."""
    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)


def clean_text(text: str) -> str:
    cleaned = strip_gutenberg(text)
    cleaned = redact_personal_info(cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def tokenize_chunks(
    tokenizer: AutoTokenizer, text: str, sequence_length: int, num_chunks: int
) -> List[List[int]]:
    tokens = tokenizer(text, return_tensors=None, add_special_tokens=False).input_ids
    if len(tokens) < sequence_length:
        raise ValueError("Text too short for requested sequence length.")
    max_start = len(tokens) - sequence_length
    indices = np.linspace(0, max_start, num=num_chunks, dtype=int)
    return [tokens[i : i + sequence_length] for i in indices]


def prepare_dataset(config: DataConfig, model_name: str) -> Tuple[DatasetArtifacts, List[List[int]]]:
    set_seed(config.seed)
    artifacts = resolve_artifacts(config)
    if not artifacts.raw_path.exists():
        url = config.ptbr_url if config.language == "ptbr" else config.en_url
        download_text(url, artifacts.raw_path)
    raw_text = artifacts.raw_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)[: config.max_characters]
    artifacts.processed_path.write_text(cleaned, encoding="utf-8")
    token = get_hf_token()
    if token:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    chunks = tokenize_chunks(tokenizer, cleaned, config.sequence_length, config.num_chunks)
    artifacts.chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    LOGGER.info("Saved %d chunks to %s", len(chunks), artifacts.chunks_path)
    return artifacts, chunks


def load_chunks(chunks_path: Path) -> List[List[int]]:
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def chunks_to_batches(chunks: Iterable[List[int]]) -> List[torch.Tensor]:
    return [torch.tensor(chunk, dtype=torch.long).unsqueeze(0) for chunk in chunks]


def main() -> None:
    configure_logging()
    args = parse_args()
    config = get_config(args)
    try:
        prepare_dataset(config, args.model_name)
    except (requests.HTTPError, requests.ConnectionError, ValueError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
