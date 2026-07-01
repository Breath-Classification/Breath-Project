from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_DIR / "data" / "Models" / "mobile_models" / "LSTMBASE_tens copy.pt"
DEFAULT_SEQUENCE = PROJECT_DIR / "data" / "NewData" / "sequence" / "Zuzia_cut_pretrained_sequence.txt"
DEFAULT_OUTPUT_PNG = PROJECT_DIR / "data" / "NewData" / "layers" / "zuzia_model_colored_predictions.png"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data" / "NewData" / "layers" / "zuzia_model_colored_predictions.csv"

MODEL_LABEL_OFFSET = 1
LABEL_COLORS = {
    -1: "#2563eb",  # inhale / wdech
    0: "#16a34a",   # pause / bezruch
    1: "#dc2626",   # exhale / wydech
    2: "#f97316",   # other / hold
}
LABEL_NAMES = {
    -1: "wdech",
    0: "pauza",
    1: "wydech",
    2: "inne",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a simple colored PNG with model predictions and true breathing labels."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, type=Path)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG, type=Path)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, type=Path)
    parser.add_argument("--block-size", default=30, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def read_sequence(path: Path) -> list[list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file does not exist: {path}")

    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append([float(value) for value in stripped.split(",")])

    if not rows:
        raise ValueError(f"Sequence file is empty: {path}")
    return rows


def display_label(model_label: int) -> int:
    return model_label - MODEL_LABEL_OFFSET


def signal_value(features: list[float]) -> float:
    if not features:
        return 0.0

    # The last feature is amplitude, the earlier values are the local signal window.
    window_values = features[:-1] if len(features) > 1 else features
    return window_values[len(window_values) // 2]


def predict_blocks(
    model: torch.nn.Module,
    rows: list[list[float]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, int | float | bool]]:
    features = [row[:-1] for row in rows]
    true_labels = [int(row[-1]) for row in rows]
    output_rows: list[dict[str, int | float | bool]] = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        pending_blocks: list[list[list[float]]] = []
        pending_indexes: list[int] = []

        for end_index in range(block_size - 1, len(features)):
            pending_blocks.append(features[end_index - block_size + 1:end_index + 1])
            pending_indexes.append(end_index)

            if len(pending_blocks) == batch_size:
                output_rows.extend(
                    predict_batch(model, pending_blocks, pending_indexes, features, true_labels, device)
                )
                pending_blocks = []
                pending_indexes = []

        if pending_blocks:
            output_rows.extend(
                predict_batch(model, pending_blocks, pending_indexes, features, true_labels, device)
            )

    return output_rows


def predict_batch(
    model: torch.nn.Module,
    blocks: list[list[list[float]]],
    indexes: list[int],
    features: list[list[float]],
    true_labels: list[int],
    device: torch.device,
) -> list[dict[str, int | float | bool]]:
    batch_x = torch.tensor(blocks, dtype=torch.float32, device=device)
    logits = model(batch_x)
    predictions = [int(value) for value in torch.argmax(logits, dim=1).cpu().tolist()]

    rows: list[dict[str, int | float | bool]] = []
    for source_index, prediction in zip(indexes, predictions, strict=False):
        true_label = true_labels[source_index]
        rows.append(
            {
                "sample_index": 0,
                "source_sequence_index": source_index,
                "signal_value": signal_value(features[source_index]),
                "true_label": display_label(true_label),
                "prediction": display_label(prediction),
                "correct": prediction == true_label,
            }
        )
    return rows


def renumber_rows(rows: list[dict[str, int | float | bool]]) -> None:
    for sample_index, row in enumerate(rows):
        row["sample_index"] = sample_index


def write_csv(path: Path, rows: list[dict[str, int | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "source_sequence_index",
                "signal_value",
                "true_label",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_colored_predictions(path: Path, rows: list[dict[str, int | float | bool]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not rows:
        raise ValueError("No prediction rows to plot.")

    indexes = [int(row["sample_index"]) for row in rows]
    values = [float(row["signal_value"]) for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    panels = [
        (axes[0], "prediction", "Predykcja modelu bazowego"),
        (axes[1], "true_label", "Prawda: Zuzia_cut_pretrained_sequence"),
    ]

    for axis, label_column, title in panels:
        for left, right in zip(range(len(rows) - 1), range(1, len(rows)), strict=False):
            label = int(rows[left][label_column])
            axis.plot(
                [indexes[left], indexes[right]],
                [values[left], values[right]],
                color=LABEL_COLORS.get(label, "#6b7280"),
                linewidth=2.0,
            )
        axis.set_title(title)
        axis.set_ylabel("normalized value")
        axis.grid(True, alpha=0.2)

    correct_indexes = [int(row["sample_index"]) for row in rows if bool(row["correct"])]
    wrong_indexes = [int(row["sample_index"]) for row in rows if not bool(row["correct"])]
    axes[0].scatter(
        wrong_indexes,
        [values[index] for index in wrong_indexes],
        color="#111827",
        s=12,
        alpha=0.75,
        label="bledna predykcja",
        zorder=3,
    )
    axes[0].set_xlabel("sample index")

    legend_items = [
        Line2D([0], [0], color=color, lw=3, label=f"{label}: {LABEL_NAMES[label]}")
        for label, color in LABEL_COLORS.items()
    ]
    legend_items.append(Line2D([0], [0], marker="o", color="#111827", lw=0, label="blad", markersize=5))
    axes[0].legend(handles=legend_items, loc="upper right")

    accuracy = len(correct_indexes) / len(rows)
    fig.suptitle(f"Zuzia model predictions, accuracy={accuracy:.3f}", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")

    device = torch.device(args.device)
    model = torch.jit.load(str(args.model), map_location=device)
    rows = read_sequence(args.sequence)
    prediction_rows = predict_blocks(
        model,
        rows,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    renumber_rows(prediction_rows)
    write_csv(args.output_csv, prediction_rows)
    plot_colored_predictions(args.output_png, prediction_rows)

    accuracy = sum(1 for row in prediction_rows if bool(row["correct"])) / len(prediction_rows)
    print(
        json.dumps(
            {
                "model": str(args.model),
                "sequence": str(args.sequence),
                "output_png": str(args.output_png),
                "output_csv": str(args.output_csv),
                "rows": len(prediction_rows),
                "accuracy": accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
