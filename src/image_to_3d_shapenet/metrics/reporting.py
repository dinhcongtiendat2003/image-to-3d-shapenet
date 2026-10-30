"""Incremental CSV/JSON/tables/plots for interruption-safe evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IDENTITY_FIELDS = {
    "sample_id", "model_id", "view_id", "category_id", "category_name",
    "configuration", "backend", "model_version", "status",
}
BOOLEAN_FIELDS = {"watertight"}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _finite_numbers(row: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in row.items():
        if key in IDENTITY_FIELDS or key in BOOLEAN_FIELDS:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    group_specs = (
        ("model", lambda row: (row["configuration"], row["backend"], row["model_id"])),
        ("category", lambda row: (row["configuration"], row["backend"], row["category_id"])),
        ("test_set", lambda row: (row["configuration"], row["backend"], "all")),
    )
    for level, key_function in group_specs:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(key_function(row), []).append(row)
        for (configuration, backend, group_id), values in sorted(groups.items()):
            numeric_names = sorted(set().union(*(_finite_numbers(value).keys() for value in values)))
            summary: dict[str, Any] = {
                "level": level,
                "group_id": group_id,
                "configuration": configuration,
                "backend": backend,
                "sample_count": len(values),
            }
            for name in numeric_names:
                available = [float(value[name]) for value in values if name in _finite_numbers(value)]
                summary[name] = sum(available) / len(available)
            summary["watertight_rate"] = sum(bool(value["watertight"]) for value in values) / len(values)
            summaries.append(summary)
    return summaries


def _tables(summary: list[dict[str, Any]]) -> tuple[str, str]:
    rows = [row for row in summary if row["level"] == "test_set"]
    columns = (
        "configuration", "backend", "sample_count", "chamfer_distance",
        "fscore_1_percent", "fscore_2_percent", "normal_consistency",
        "silhouette_iou", "edge_chamfer_distance", "dihedral_angle_error_degrees",
        "inference_time_seconds", "peak_vram_gb", "watertight_rate",
    )
    labels = ("Config", "Backend", "N", "Chamfer", "F@1%", "F@2%", "Normal", "IoU", "Edge CD", "Dihedral°", "Time(s)", "VRAM", "Watertight")
    markdown = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"]
    latex = [
        "\\begin{tabular}{llrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(labels) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        markdown.append("| " + " | ".join(values) + " |")
        latex.append(" & ".join(value.replace("_", "\\_") for value in values) + " \\\\")
    latex.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(markdown) + "\n", "\n".join(latex) + "\n"


def _bar_plot(path: Path, rows: list[dict[str, Any]], metric: str) -> None:
    width, height = 720, 360
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), metric, fill="black")
    values = [float(row.get(metric, 0.0)) for row in rows]
    maximum = max(values + [1e-12])
    bar_width = max(10, (width - 80) // max(len(rows), 1))
    for index, (row, value) in enumerate(zip(rows, values)):
        x0 = 50 + index * bar_width
        x1 = x0 + max(4, bar_width - 5)
        y1 = height - 45
        y0 = y1 - int((height - 90) * value / maximum)
        draw.rectangle((x0, y0, x1, y1), fill=(55, 105, 180))
        label = f"{row['configuration']}\n{row['backend']}"
        draw.text((x0, height - 40), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_incremental_reports(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    summary = summarize(rows) if rows else []
    _csv(output / "metrics_per_sample.csv", rows)
    if summary:
        _csv(output / "metrics_summary.csv", summary)
    _atomic_json(output / "failures.json", failures)
    _atomic_json(
        output / "metrics.json",
        {"metadata": metadata, "per_sample": rows, "summary": summary, "failures": failures},
    )
    markdown, latex = _tables(summary)
    _atomic_text(output / "benchmark_table.md", markdown)
    _atomic_text(output / "benchmark_table.tex", latex)
    test_rows = [row for row in summary if row["level"] == "test_set"]
    for metric in (
        "chamfer_distance", "fscore_1_percent", "silhouette_iou",
        "edge_chamfer_distance", "dihedral_angle_error_degrees",
    ):
        if test_rows:
            _bar_plot(output / "plots" / f"{metric}.png", test_rows, metric)
    return summary
