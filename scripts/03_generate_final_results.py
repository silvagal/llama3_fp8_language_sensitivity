#!/usr/bin/env python3
"""Generate a consolidated summary file for quantization experiments."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def load_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return float("nan")
    if fraction <= 0:
        return min(values)
    if fraction >= 1:
        return max(values)
    values_sorted = sorted(values)
    index = int(round(fraction * (len(values_sorted) - 1)))
    return values_sorted[index]


def summarize_outliers(path: Path) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = [val for layer_vals in data.values() for val in layer_vals]
    stats = {
        "count": float(len(values)),
        "mean": mean(values) if values else float("nan"),
        "median": median(values) if values else float("nan"),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
    }
    layer_means = sorted(
        ((layer, mean(vals) if vals else float("nan")) for layer, vals in data.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return stats, layer_means[:5]


def format_float(value: float) -> str:
    if value != value:  # NaN check
        return "nan"
    return f"{value:.6f}"


def format_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    header_line = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator_line = "-|-".join("-" * widths[idx] for idx in range(len(headers)))
    lines.append(header_line)
    lines.append(separator_line)
    for row in rows:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return lines


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    results_dir = root_dir / "results"
    configs_dir = root_dir / "configs"
    output_path = results_dir / "final_quantization_results"

    model_cfg = load_simple_yaml(configs_dir / "model_config.yaml")
    data_cfg = load_simple_yaml(configs_dir / "data_config.yaml")

    ppl_files = sorted(results_dir.glob("*_ppl.json"))
    outlier_files = sorted(results_dir.glob("*_outliers/outliers_*.json"))
    hist_files = sorted(results_dir.glob("*_outliers/hist_*.png"))
    source_files = ppl_files + outlier_files + hist_files

    lines: List[str] = []
    lines.append("final_quantization_results")
    lines.append(f"generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"model_name: {model_cfg.get('model_name', 'unknown')}")
    lines.append(f"sequence_length: {data_cfg.get('sequence_length', 'unknown')}")
    lines.append(f"num_chunks: {data_cfg.get('num_chunks', 'unknown')}")
    lines.append(f"max_characters: {data_cfg.get('max_characters', 'unknown')}")
    lines.append(f"ptbr_url: {data_cfg.get('ptbr_url', 'unknown')}")
    lines.append(f"en_url: {data_cfg.get('en_url', 'unknown')}")
    lines.append("")

    if source_files:
        lines.append("[SOURCE_FILES]")
        headers = ["Path", "Bytes"]
        rows = [[str(path.relative_to(root_dir)), str(path.stat().st_size)] for path in source_files]
        lines.extend(format_table(headers, rows))
        lines.append("")

    ppl_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for path in ppl_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = path.stem.split("_")
        quant = parts[0] if parts else "unknown"
        lang = parts[1] if len(parts) > 1 else "unknown"
        ppl_map[(quant, lang)] = {
            "perplexity": float(payload.get("perplexity", float("nan"))),
            "kl_divergence": float(payload.get("kl_divergence", float("nan"))),
        }

    outlier_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    outlier_top: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for path in outlier_files:
        stats, top_layers = summarize_outliers(path)
        parent = path.parent.name
        parts = parent.split("_")
        quant = parts[0] if parts else "unknown"
        lang = parts[1] if len(parts) > 1 else "unknown"
        outlier_map[(quant, lang)] = stats
        outlier_top[(quant, lang)] = top_layers

    quant_order = ["bf16", "int8", "fp8"]
    lang_order = ["ptbr", "en"]

    def sort_key(item: Tuple[str, str]) -> Tuple[int, int, str, str]:
        quant, lang = item
        return (
            quant_order.index(quant) if quant in quant_order else len(quant_order),
            lang_order.index(lang) if lang in lang_order else len(lang_order),
            quant,
            lang,
        )

    if ppl_map:
        lines.append("[PERPLEXITY_KL_TABLE]")
        headers = ["Quantization", "Language", "Perplexity", "KL Divergence"]
        rows: List[List[str]] = []
        for key in sorted(ppl_map.keys(), key=sort_key):
            quant, lang = key
            ppl = ppl_map[key]["perplexity"]
            kl = ppl_map[key]["kl_divergence"]
            rows.append([quant, lang, format_float(ppl), format_float(kl)])
        lines.extend(format_table(headers, rows))
        lines.append("")

    if outlier_map:
        lines.append("[OUTLIER_STATS_TABLE]")
        headers = [
            "Quantization",
            "Language",
            "Count",
            "Mean",
            "Median",
            "P95",
            "P99",
            "Min",
            "Max",
        ]
        rows = []
        for key in sorted(outlier_map.keys(), key=sort_key):
            quant, lang = key
            stats = outlier_map[key]
            rows.append(
                [
                    quant,
                    lang,
                    str(int(stats["count"])),
                    format_float(stats["mean"]),
                    format_float(stats["median"]),
                    format_float(stats["p95"]),
                    format_float(stats["p99"]),
                    format_float(stats["min"]),
                    format_float(stats["max"]),
                ]
            )
        lines.extend(format_table(headers, rows))
        lines.append("")

        lines.append("[OUTLIER_TOP_LAYERS]")
        headers = ["Quantization", "Language", "Rank", "Layer", "Mean"]
        rows = []
        for key in sorted(outlier_top.keys(), key=sort_key):
            quant, lang = key
            for rank, (layer, layer_mean) in enumerate(outlier_top[key], start=1):
                rows.append([quant, lang, str(rank), layer, format_float(layer_mean)])
        lines.extend(format_table(headers, rows))
        lines.append("")

    if hist_files:
        lines.append("[HISTOGRAMS]")
        headers = ["Quantization", "Language", "Path"]
        rows = []
        for path in hist_files:
            parent = path.parent.name
            parts = parent.split("_")
            quant = parts[0] if parts else "unknown"
            lang = parts[1] if len(parts) > 1 else "unknown"
            rows.append([quant, lang, str(path.relative_to(root_dir))])
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        lines.extend(format_table(headers, rows))
        lines.append("")

    if ppl_files or outlier_files:
        lines.append("[RAW_METRICS_JSON]")
        for path in ppl_files + outlier_files:
            rel = path.relative_to(root_dir)
            lines.append(f"--- {rel}")
            lines.append(path.read_text(encoding="utf-8").rstrip())
            lines.append("")

    if not ppl_files and not outlier_files:
        lines.append("No results found in results directory.")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
