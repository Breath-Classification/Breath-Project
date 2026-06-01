from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PYTORCH_DIR = Path(__file__).resolve().parents[1]
ML_ROOT = PYTORCH_DIR.parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from EvaluateFineTuning import apply_layer_artifacts, load_manifest, load_model_for_eval  # noqa: E402
from scripts.convert_tens_to_model_input import (  # noqa: E402
    convert_predicted_rows,
    raw_to_pretrained,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot local fine-tuned model predictions next to phone predictions over time."
    )
    parser.add_argument("--ground-truth", required=True, type=Path, help="CSV with time,value,prediction,confidence,true_label.")
    parser.add_argument("--manifest", required=True, type=Path, help="Fine-tuning manifest with layer artifacts.")
    parser.add_argument("--model-file", default=None, type=Path, help="Optional explicit base TorchScript model path.")
    parser.add_argument("--output-csv", default=Path("data/NewData/layers/prediction_timeline.csv"), type=Path)
    parser.add_argument("--output-png", default=Path("data/NewData/layers/prediction_timeline.png"), type=Path)
    parser.add_argument("--output-html", default=Path("data/NewData/layers/prediction_timeline.html"), type=Path)
    parser.add_argument("--window-size", default=5, type=int)
    parser.add_argument("--moving-average-window", default=5, type=int)
    parser.add_argument("--normalization-range", default=150, type=int)
    parser.add_argument("--confidence-threshold", default=0.0, type=float)
    parser.add_argument("--block-size", default=30, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--no-plot", action="store_true", help="Only write CSV and JSON summary.")
    return parser.parse_args()


def read_ground_truth_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "value", "prediction", "confidence", "true_label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Ground-truth CSV is missing column(s): {', '.join(missing)}")
        return [row for row in reader if row]


def build_timeline_rows(
    rows: list[dict[str, str]],
    *,
    window_size: int,
    moving_average_window: int,
    normalization_range: int,
    confidence_threshold: float,
) -> list[dict[str, float | int | str]]:
    raw_rows = [
        [row["time"], row["value"], row["prediction"], row["confidence"]]
        for row in rows
    ]
    times, raw_values, phone_labels, confidences = convert_predicted_rows(raw_rows)
    values, _, aligned_times, aligned_confidences = raw_to_pretrained(
        times,
        raw_values,
        moving_average_window=moving_average_window,
        normalization_range=normalization_range,
        source_labels=phone_labels,
        source_confidences=confidences,
    )

    true_labels = [int(float(row["true_label"])) for row in rows]
    aligned_true_labels = true_labels[moving_average_window - 1:]
    aligned_true_labels = aligned_true_labels[-len(values):]
    aligned_phone_labels = phone_labels[moving_average_window - 1:]
    aligned_phone_labels = aligned_phone_labels[-len(values):]

    timeline_rows: list[dict[str, float | int | str]] = []
    for source_index in range(window_size, len(values)):
        label_position = source_index - window_size + window_size // 2
        if aligned_confidences is not None and aligned_confidences[label_position] < confidence_threshold:
            continue

        window = values[source_index - window_size:source_index]
        amplitude = abs(max(window) - min(window))
        timeline_rows.append(
            {
                "sequence_index": len(timeline_rows),
                "time_seconds": aligned_times[label_position],
                "normalized_value": values[label_position],
                "phone_prediction": aligned_phone_labels[label_position],
                "true_label": aligned_true_labels[label_position],
                "features": [*window, amplitude],
            }
        )

    return timeline_rows


def predict_blocks(
    model: torch.nn.Module,
    timeline_rows: list[dict[str, float | int | str]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    features = [row["features"] for row in timeline_rows]
    predictions: list[int] = []
    row_indexes: list[int] = []

    model.eval()
    with torch.no_grad():
        pending_blocks = []
        pending_indexes = []
        for end_index in range(block_size - 1, len(features)):
            block = features[end_index - block_size + 1:end_index + 1]
            pending_blocks.append(block)
            pending_indexes.append(end_index)

            if len(pending_blocks) == batch_size:
                predictions.extend(predict_batch(model, pending_blocks, device))
                row_indexes.extend(pending_indexes)
                pending_blocks = []
                pending_indexes = []

        if pending_blocks:
            predictions.extend(predict_batch(model, pending_blocks, device))
            row_indexes.extend(pending_indexes)

    return row_indexes, predictions


def predict_batch(model: torch.nn.Module, blocks: list[list[list[float]]], device: torch.device) -> list[int]:
    batch_x = torch.tensor(blocks, dtype=torch.float32, device=device)
    logits = model(batch_x)
    return [int(value) for value in torch.argmax(logits, dim=1).cpu().tolist()]


def attach_predictions(
    timeline_rows: list[dict[str, float | int | str]],
    row_indexes: list[int],
    predictions: list[int],
    column_name: str,
) -> None:
    for row_index, prediction in zip(row_indexes, predictions, strict=False):
        timeline_rows[row_index][column_name] = prediction - 1


def write_timeline_csv(path: Path, timeline_rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_index",
        "time_seconds",
        "normalized_value",
        "true_label",
        "phone_prediction",
        "local_base_prediction",
        "local_fine_tuned_prediction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in timeline_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def accuracy(rows: list[dict[str, float | int | str]], prediction_column: str) -> float:
    total = 0
    correct = 0
    for row in rows:
        if prediction_column not in row:
            continue
        total += 1
        correct += int(row[prediction_column]) == int(row["true_label"])
    return correct / total if total else 0.0


def plot_timeline(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib.pyplot as plt

    plotted_rows = [row for row in rows if "local_base_prediction" in row and "local_fine_tuned_prediction" in row]
    times = [float(row["time_seconds"]) for row in plotted_rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (signal_ax, labels_ax) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    signal_ax.plot(times, [float(row["normalized_value"]) for row in plotted_rows], color="black", linewidth=1)
    signal_ax.set_ylabel("normalized value")
    signal_ax.set_title("BreathSense predictions over time")

    labels_ax.step(times, [int(row["true_label"]) for row in plotted_rows], where="post", label="ground truth", linewidth=2)
    labels_ax.step(times, [int(row["phone_prediction"]) for row in plotted_rows], where="post", label="phone", alpha=0.75)
    labels_ax.step(times, [int(row["local_base_prediction"]) for row in plotted_rows], where="post", label="local base", alpha=0.75)
    labels_ax.step(
        times,
        [int(row["local_fine_tuned_prediction"]) for row in plotted_rows],
        where="post",
        label="local fine-tuned",
        alpha=0.75,
    )
    labels_ax.set_xlabel("time [s]")
    labels_ax.set_ylabel("label")
    labels_ax.set_yticks([-1, 0, 1, 2])
    labels_ax.grid(True, alpha=0.2)
    labels_ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_timeline_html(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    plotted_rows = [row for row in rows if "local_base_prediction" in row and "local_fine_tuned_prediction" in row]
    if not plotted_rows:
        raise ValueError("No rows with local predictions available for HTML plot.")

    width = 1400
    panel_height = 170
    padding_left = 64
    padding_right = 24
    padding_top = 42
    gap = 46
    padding_bottom = 58
    panels = [
        ("true_label", "Ground truth"),
        ("phone_prediction", "Phone prediction"),
        ("local_base_prediction", "Local base model"),
        ("local_fine_tuned_prediction", "Local fine-tuned model"),
    ]
    total_height = padding_top + len(panels) * panel_height + (len(panels) - 1) * gap + padding_bottom
    plot_width = width - padding_left - padding_right

    times = [float(row["time_seconds"]) for row in plotted_rows]
    start_time = min(times)
    end_time = max(times)
    time_range = end_time - start_time or 1.0
    label_colors = {
        -1: "#2563eb",
        0: "#16a34a",
        1: "#dc2626",
        2: "#f97316",
    }

    def x_for(time_value: float) -> float:
        return padding_left + ((time_value - start_time) / time_range) * plot_width

    def signal_y(value: float, panel_top: float) -> float:
        return panel_top + ((1.0 - value) / 2.0) * panel_height

    def panel_top(index: int) -> float:
        return padding_top + index * (panel_height + gap)

    def colored_segments(column: str, top: float) -> str:
        segments = []
        for previous, current in zip(plotted_rows, plotted_rows[1:], strict=False):
            label = int(previous[column])
            color = label_colors.get(label, "#6b7280")
            x1 = x_for(float(previous["time_seconds"]))
            y1 = signal_y(float(previous["normalized_value"]), top)
            x2 = x_for(float(current["time_seconds"]))
            y2 = signal_y(float(current["normalized_value"]), top)
            segments.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{color}" stroke-width="2.2" stroke-linecap="round" />'
            )
        return "\n".join(segments)

    def panel_markup(index: int, column: str, title: str) -> str:
        top = panel_top(index)
        mid = top + panel_height / 2
        bottom = top + panel_height
        return f"""
    <text x="{padding_left}" y="{top - 14:.2f}" font-weight="700">{title}</text>
    <line x1="{padding_left}" x2="{width - padding_right}" y1="{top:.2f}" y2="{top:.2f}" class="grid" />
    <line x1="{padding_left}" x2="{width - padding_right}" y1="{mid:.2f}" y2="{mid:.2f}" class="grid" />
    <line x1="{padding_left}" x2="{width - padding_right}" y1="{bottom:.2f}" y2="{bottom:.2f}" class="grid" />
    <text x="{padding_left - 14}" y="{top + 4:.2f}" text-anchor="end">1</text>
    <text x="{padding_left - 14}" y="{mid + 4:.2f}" text-anchor="end">0</text>
    <text x="{padding_left - 14}" y="{bottom + 4:.2f}" text-anchor="end">-1</text>
    {colored_segments(column, top)}
"""

    panels_markup = "\n".join(panel_markup(index, column, title) for index, (column, title) in enumerate(panels))
    legend_items = "\n".join(
        f'<span><i style="background:{color}"></i>{label}</span>'
        for label, color in [("-1 inhale", "#2563eb"), ("0 pause", "#16a34a"), ("1 exhale", "#dc2626"), ("2 hold/other", "#f97316")]
    )
    x_ticks = []
    for fraction in [0, 0.25, 0.5, 0.75, 1.0]:
        time_value = start_time + time_range * fraction
        x = x_for(time_value)
        x_ticks.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{padding_top}" y2="{total_height - padding_bottom}" class="grid" />'
            f'<text x="{x:.2f}" y="{total_height - 18}" text-anchor="middle">{time_value:.1f}s</text>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>BreathSense prediction timeline</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ padding: 24px; }}
    svg {{ width: 100%; height: auto; background: white; border: 1px solid #d1d5db; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    text {{ fill: #374151; font-size: 13px; }}
    .legend {{ display: flex; gap: 18px; margin: 12px 0 0; flex-wrap: wrap; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; font-size: 14px; }}
    .legend i {{ width: 22px; height: 3px; display: inline-block; }}
  </style>
</head>
<body>
<main>
  <h1>BreathSense prediction timeline</h1>
  <svg viewBox="0 0 {width} {total_height}" role="img">
    {''.join(x_ticks)}
    {panels_markup}
  </svg>
  <div class="legend">{legend_items}</div>
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_ground_truth_rows(args.ground_truth)
    manifest = load_manifest(args.manifest)
    timeline_rows = build_timeline_rows(
        rows,
        window_size=args.window_size,
        moving_average_window=args.moving_average_window,
        normalization_range=args.normalization_range,
        confidence_threshold=args.confidence_threshold,
    )
    if len(timeline_rows) < args.block_size:
        raise ValueError(f"Need at least {args.block_size} timeline rows, got {len(timeline_rows)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_count = len(timeline_rows[0]["features"])
    base_model = load_model_for_eval(
        manifest=manifest,
        model_file=args.model_file,
        feature_count=feature_count,
        device=device,
    )
    tuned_model = load_model_for_eval(
        manifest=manifest,
        model_file=args.model_file,
        feature_count=feature_count,
        device=device,
    )
    apply_layer_artifacts(
        model=tuned_model,
        manifest=manifest,
        manifest_path=args.manifest,
        device=device,
    )

    row_indexes, base_predictions = predict_blocks(
        base_model,
        timeline_rows,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    _, tuned_predictions = predict_blocks(
        tuned_model,
        timeline_rows,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    attach_predictions(timeline_rows, row_indexes, base_predictions, "local_base_prediction")
    attach_predictions(timeline_rows, row_indexes, tuned_predictions, "local_fine_tuned_prediction")

    write_timeline_csv(args.output_csv, timeline_rows)
    write_timeline_html(args.output_html, timeline_rows)
    png_path = None
    if not args.no_plot:
        try:
            plot_timeline(args.output_png, timeline_rows)
            png_path = str(args.output_png)
        except ModuleNotFoundError:
            png_path = None

    summary = {
        "created_at": datetime.now().isoformat(),
        "ground_truth_file": str(args.ground_truth),
        "manifest_file": str(args.manifest),
        "output_csv": str(args.output_csv),
        "output_html": str(args.output_html),
        "output_png": png_path,
        "rows": len(timeline_rows),
        "plotted_rows": sum(1 for row in timeline_rows if "local_base_prediction" in row),
        "phone_accuracy": accuracy(timeline_rows, "phone_prediction"),
        "local_base_accuracy": accuracy(timeline_rows, "local_base_prediction"),
        "local_fine_tuned_accuracy": accuracy(timeline_rows, "local_fine_tuned_prediction"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
