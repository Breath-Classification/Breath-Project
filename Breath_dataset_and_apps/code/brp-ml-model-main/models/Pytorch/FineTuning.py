from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SEQUENCE_FILE = Path("data/NewData/sequence/concatenated.txt")
DEFAULT_LAYERS_DIR = Path("data/NewData/layers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal server-side personalization step. It reads pseudo-labelled "
            "sequence data and writes small layer artifacts plus a manifest."
        )
    )
    parser.add_argument("--sequence-file", default=DEFAULT_SEQUENCE_FILE, type=Path)
    parser.add_argument("--layers-dir", default=DEFAULT_LAYERS_DIR, type=Path)
    parser.add_argument("--model-name", default="unknown-model")
    parser.add_argument("--runtime", default="pytorch")
    parser.add_argument("--trained-layers", default="fc", help="Comma-separated layer list, e.g. fc or adapter,fc.")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def load_sequences(path: Path) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []

    if not path.exists():
        return features, labels

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split(",")]
        if len(values) < 2:
            continue
        features.append(values[:-1])
        labels.append(int(values[-1]))

    return features, labels


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_fc_artifact(features: list[list[float]], labels: list[int]) -> dict:
    feature_count = len(features[0]) if features else 0
    label_counts = Counter(labels)
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for feature_row, label in zip(features, labels, strict=False):
        grouped[label].append(feature_row)

    global_mean = [
        mean([row[index] for row in features])
        for index in range(feature_count)
    ] if features else []

    weights: dict[str, list[float]] = {}
    bias: dict[str, float] = {}
    for label, rows in sorted(grouped.items()):
        class_mean = [mean([row[index] for row in rows]) for index in range(feature_count)]
        weights[str(label)] = [round(class_mean[index] - global_mean[index], 6) for index in range(feature_count)]
        bias[str(label)] = round(math.log(label_counts[label] + 1), 6)

    return {
        "format": "breathsense.fc.v1",
        "feature_count": feature_count,
        "class_counts": dict(sorted(label_counts.items())),
        "weights": weights,
        "bias": bias,
    }


def build_adapter_artifact(features: list[list[float]]) -> dict:
    feature_count = len(features[0]) if features else 0
    feature_columns = [[row[index] for row in features] for index in range(feature_count)] if features else []

    return {
        "format": "breathsense.adapter.v1",
        "feature_count": feature_count,
        "scale": [round(std(column), 6) for column in feature_columns],
        "shift": [round(mean(column), 6) for column in feature_columns],
    }


def main() -> None:
    args = parse_args()
    args.layers_dir.mkdir(parents=True, exist_ok=True)
    features, labels = load_sequences(args.sequence_file)
    trained_layers = [layer.strip() for layer in args.trained_layers.split(",") if layer.strip()]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    layer_files: dict[str, str] = {}
    if "fc" in trained_layers:
        fc_path = args.layers_dir / f"{run_id}_fc.json"
        fc_path.write_text(json.dumps(build_fc_artifact(features, labels), indent=2), encoding="utf-8")
        layer_files["fc"] = fc_path.name

    if "adapter" in trained_layers:
        adapter_path = args.layers_dir / f"{run_id}_adapter.json"
        adapter_path.write_text(json.dumps(build_adapter_artifact(features), indent=2), encoding="utf-8")
        layer_files["adapter"] = adapter_path.name

    manifest = {
        "format": "breathsense.fine_tuning_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_name": args.model_name,
        "runtime": args.runtime,
        "trained_layers": trained_layers,
        "source_sequence_file": str(args.sequence_file),
        "sequence_rows": len(features),
        "layer_files": layer_files,
        "ready_for_download": bool(features) and bool(layer_files),
    }
    manifest_path = args.layers_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest_manifest_path = args.layers_dir / "latest_manifest.json"
    latest_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest_file": manifest_path.name}))


if __name__ == "__main__":
    main()
