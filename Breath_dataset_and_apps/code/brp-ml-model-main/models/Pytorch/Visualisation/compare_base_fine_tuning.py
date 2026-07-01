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
EXPERIMENTS_DIR = PYTORCH_DIR / "Experiments"
for path in (PYTORCH_DIR, ML_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from EvaluateFineTuning import (  # noqa: E402
    apply_layer_artifacts,
    infer_ml_root,
    load_manifest,
    load_model_for_eval,
    path_from_manifest,
)
from data_loader import create_dataloaders  # noqa: E402


DEFAULT_MANIFEST = ML_ROOT / "data" / "NewData" / "layers" / "latest_manifest.json"
DEFAULT_OUTPUT_CSV = ML_ROOT / "data" / "NewData" / "layers" / "base_vs_fine_tuning.csv"
DEFAULT_OUTPUT_PNG = ML_ROOT / "data" / "NewData" / "layers" / "base_vs_fine_tuning.png"
LABEL_OFFSET = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare base and fine-tuned BreathSense models on a labelled sequence file. "
            "The plot highlights places where the fine-tuned/main model is correct and "
            "the base model is wrong."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--eval-data-txt", default=None, type=Path)
    parser.add_argument("--base-model", default=None, type=Path, help="Optional explicit base TorchScript model path.")
    parser.add_argument("--fine-model", default=None, type=Path, help="Optional explicit full fine-tuned TorchScript model path.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, type=Path)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG, type=Path)
    parser.add_argument("--dataset-type", default=None)
    parser.add_argument("--target", default=0, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--block-size", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device for inference. CPU is the default because some exported mobile TorchScript models create hidden states on CPU.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_existing_path(path_value: str | None, *, manifest_path: Path, ml_root: Path) -> Path | None:
    path = path_from_manifest(path_value, ml_root=ml_root)
    if path is not None and path.exists():
        return path

    if not path_value:
        return None

    relative_to_manifest = manifest_path.parent / path_value
    if relative_to_manifest.exists():
        return relative_to_manifest

    relative_to_root = ml_root / path_value
    if relative_to_root.exists():
        return relative_to_root

    return path


def resolve_eval_file(args: argparse.Namespace, manifest: dict, *, manifest_path: Path, ml_root: Path) -> Path:
    if args.eval_data_txt is not None:
        return args.eval_data_txt

    for key in ("test_data_txt", "source_sequence_file", "train_data_txt"):
        candidate = resolve_existing_path(manifest.get(key), manifest_path=manifest_path, ml_root=ml_root)
        if candidate is not None and candidate.exists():
            return candidate

    raise ValueError("Manifest does not point to an existing sequence file. Pass --eval-data-txt explicitly.")


def resolve_base_model_file(args: argparse.Namespace, manifest: dict, *, ml_root: Path) -> Path | None:
    if args.base_model is not None:
        return args.base_model

    manifest_model = path_from_manifest(manifest.get("base_model_file"), ml_root=ml_root)
    if manifest_model is not None and manifest_model.exists():
        return manifest_model

    model_name = manifest.get("model_name", "LSTMBASE_tens")
    candidates = [
        ml_root / "data" / "Models" / "mobile_models" / f"{model_name}.pt",
        ml_root / "models" / "mobile_models" / f"{model_name}.pt",
        ml_root.parent / "brp-app-main" / "android" / "app" / "src" / "main" / "assets" / f"{model_name}.pt",
        Path("../brp-app-main/android/app/src/main/assets") / f"{model_name}.pt",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_fine_model(
    *,
    args: argparse.Namespace,
    manifest: dict,
    manifest_path: Path,
    feature_count: int,
    device: torch.device,
):
    full_model_file = args.fine_model
    if full_model_file is None:
        model_file_name = manifest.get("model_file")
        if model_file_name:
            candidate = manifest_path.parent / model_file_name
            if candidate.exists():
                full_model_file = candidate

    if full_model_file is not None:
        model = torch.jit.load(str(full_model_file), map_location=device)
        model.eval()
        return model

    model = load_model_for_eval(
        manifest=manifest,
        model_file=resolve_base_model_file(args, manifest, ml_root=infer_ml_root(manifest_path)),
        feature_count=feature_count,
        device=device,
    )
    apply_layer_artifacts(
        model=model,
        manifest=manifest,
        manifest_path=manifest_path,
        device=device,
    )
    model.eval()
    return model


def predict_model(model, dataloader, device: torch.device) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    true_labels: list[int] = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_predictions = torch.argmax(logits, dim=1).cpu()
            if batch_y.ndim > 1:
                batch_y = batch_y[:, -1]
            predictions.extend(int(value) for value in batch_predictions.tolist())
            true_labels.extend(int(value) for value in batch_y.cpu().tolist())

    return predictions, true_labels


def read_sequence_features(sequence_file: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with sequence_file.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append([float(value) for value in stripped.split(",")])
    return rows


def signal_value_for_block(sequence_rows: list[list[float]], sample_index: int, block_size: int) -> float:
    if not sequence_rows:
        return 0.0

    source_index = min(sample_index + block_size - 1, len(sequence_rows) - 1)
    features = sequence_rows[source_index][:-1]
    if not features:
        return 0.0

    window_feature_count = max(1, len(features) - 1)
    center_index = min(window_feature_count // 2, len(features) - 1)
    return features[center_index]


def display_label(model_label: int) -> int:
    return model_label - LABEL_OFFSET


def build_rows(
    *,
    sequence_file: Path,
    block_size: int,
    base_predictions: list[int],
    fine_predictions: list[int],
    true_labels: list[int],
) -> list[dict[str, int | float | bool]]:
    sequence_rows = read_sequence_features(sequence_file)
    rows: list[dict[str, int | float | bool]] = []

    for index, (base_pred, fine_pred, true_label) in enumerate(
        zip(base_predictions, fine_predictions, true_labels, strict=False)
    ):
        base_correct = base_pred == true_label
        fine_correct = fine_pred == true_label
        rows.append(
            {
                "sample_index": index,
                "source_sequence_index": min(index + block_size - 1, max(0, len(sequence_rows) - 1)),
                "signal_value": signal_value_for_block(sequence_rows, index, block_size),
                "true_label": display_label(true_label),
                "base_prediction": display_label(base_pred),
                "fine_tuned_prediction": display_label(fine_pred),
                "base_correct": base_correct,
                "fine_tuned_correct": fine_correct,
                "fine_better": fine_correct and not base_correct,
                "base_better": base_correct and not fine_correct,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, int | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "source_sequence_index",
        "signal_value",
        "true_label",
        "base_prediction",
        "fine_tuned_prediction",
        "base_correct",
        "fine_tuned_correct",
        "fine_better",
        "base_better",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def contiguous_ranges(indexes: list[int]) -> list[tuple[int, int]]:
    if not indexes:
        return []

    ranges: list[tuple[int, int]] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


def plot_comparison(path: Path, rows: list[dict[str, int | float | bool]], summary: dict[str, float | int | str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No rows available for plotting.")

    indexes = [int(row["sample_index"]) for row in rows]
    signal = [float(row["signal_value"]) for row in rows]
    true_labels = [int(row["true_label"]) for row in rows]
    base_labels = [int(row["base_prediction"]) for row in rows]
    fine_labels = [int(row["fine_tuned_prediction"]) for row in rows]
    fine_better_indexes = [int(row["sample_index"]) for row in rows if row["fine_better"]]
    base_better_indexes = [int(row["sample_index"]) for row in rows if row["base_better"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (signal_ax, label_ax) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    for start, end in contiguous_ranges(fine_better_indexes):
        signal_ax.axvspan(start - 0.5, end + 0.5, color="#22c55e", alpha=0.18)
        label_ax.axvspan(start - 0.5, end + 0.5, color="#22c55e", alpha=0.18)
    for start, end in contiguous_ranges(base_better_indexes):
        signal_ax.axvspan(start - 0.5, end + 0.5, color="#ef4444", alpha=0.10)
        label_ax.axvspan(start - 0.5, end + 0.5, color="#ef4444", alpha=0.10)

    signal_ax.plot(indexes, signal, color="#111827", linewidth=1)
    signal_ax.scatter(
        fine_better_indexes,
        [signal[index] for index in fine_better_indexes],
        color="#16a34a",
        s=26,
        label="fine/main better",
        zorder=3,
    )
    signal_ax.set_ylabel("normalized signal")
    signal_ax.set_title(
        "Base vs fine-tuned/main model "
        f"(base acc {summary['base_accuracy']:.3f}, fine acc {summary['fine_tuned_accuracy']:.3f})"
    )
    signal_ax.grid(True, alpha=0.2)
    signal_ax.legend(loc="upper right")

    label_ax.step(indexes, true_labels, where="post", label="true", linewidth=2.2, color="#111827")
    label_ax.step(indexes, base_labels, where="post", label="base", alpha=0.85, color="#2563eb")
    label_ax.step(indexes, fine_labels, where="post", label="fine/main", alpha=0.85, color="#16a34a")
    label_ax.scatter(
        fine_better_indexes,
        [fine_labels[index] for index in fine_better_indexes],
        color="#16a34a",
        edgecolors="white",
        linewidths=0.8,
        s=34,
        zorder=4,
    )
    label_ax.set_xlabel("block/sample index")
    label_ax.set_ylabel("label")
    label_ax.set_yticks([-1, 0, 1, 2, 3])
    label_ax.grid(True, alpha=0.2)
    label_ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy(rows: list[dict[str, int | float | bool]], column: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row[column])) / len(rows)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    ml_root = infer_ml_root(args.manifest)
    eval_file = resolve_eval_file(args, manifest, manifest_path=args.manifest, ml_root=ml_root)
    block_size = args.block_size or int(manifest.get("block_size", 30))
    batch_size = args.batch_size or int(manifest.get("training", {}).get("batch_size", 16))
    dataset_type = args.dataset_type or manifest.get("dataset_type", "BlockDataset")
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "cuda":
        device = torch.device("cuda")

    _, eval_dataloader = create_dataloaders(
        transform=None,
        batch_size=batch_size,
        block_size=block_size,
        target=args.target,
        dataset_type=dataset_type,
        num_workers=args.num_workers,
        train_data_txt=str(eval_file),
        test_data_txt=str(eval_file),
    )
    first_batch_x, _ = next(iter(eval_dataloader))
    feature_count = int(first_batch_x.shape[-1])
    base_model_file = resolve_base_model_file(args, manifest, ml_root=ml_root)

    base_model = load_model_for_eval(
        manifest=manifest,
        model_file=base_model_file,
        feature_count=feature_count,
        device=device,
    )
    fine_model = load_fine_model(
        args=args,
        manifest=manifest,
        manifest_path=args.manifest,
        feature_count=feature_count,
        device=device,
    )

    base_predictions, true_labels = predict_model(base_model, eval_dataloader, device)
    fine_predictions, _ = predict_model(fine_model, eval_dataloader, device)
    rows = build_rows(
        sequence_file=eval_file,
        block_size=block_size,
        base_predictions=base_predictions,
        fine_predictions=fine_predictions,
        true_labels=true_labels,
    )
    write_csv(args.output_csv, rows)

    summary = {
        "created_at": datetime.now().isoformat(),
        "manifest_file": str(args.manifest),
        "eval_data_txt": str(eval_file),
        "base_model_file": str(base_model_file) if base_model_file is not None else None,
        "fine_model_file": str(args.fine_model) if args.fine_model is not None else str(args.manifest.parent / manifest["model_file"]) if manifest.get("model_file") else None,
        "output_csv": str(args.output_csv),
        "output_png": None if args.no_plot else str(args.output_png),
        "rows": len(rows),
        "base_accuracy": accuracy(rows, "base_correct"),
        "fine_tuned_accuracy": accuracy(rows, "fine_tuned_correct"),
        "accuracy_delta": accuracy(rows, "fine_tuned_correct") - accuracy(rows, "base_correct"),
        "fine_better_count": sum(1 for row in rows if row["fine_better"]),
        "base_better_count": sum(1 for row in rows if row["base_better"]),
    }

    if not args.no_plot:
        plot_comparison(args.output_png, rows, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
