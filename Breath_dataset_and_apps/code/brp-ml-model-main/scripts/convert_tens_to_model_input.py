from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/NewData")
DEFAULT_SEQUENCE_DIR = DEFAULT_INPUT_DIR / "sequence"
DEFAULT_CONCATENATED_NAME = "concatenated.txt"


@dataclass(frozen=True)
class CalibrationPoint:
    time: float | None
    value: float
    prediction: int
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert phone calibration rows in time,value,prediction,confidence format "
            "into tensometer model-input sequences."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV/TXT file with phone calibration rows.")
    parser.add_argument("--output-dir", default=DEFAULT_SEQUENCE_DIR, type=Path, help="Directory for sequence files.")
    parser.add_argument("--window-size", default=5, type=int, help="Number of tensometer values per sequence window.")
    parser.add_argument(
        "--confidence-threshold",
        default=0.8,
        type=float,
        help="Keep only rows whose confidence is greater than or equal to this value.",
    )
    parser.add_argument(
        "--concatenated-name",
        default=DEFAULT_CONCATENATED_NAME,
        help="Name of the aggregate sequence file inside output-dir.",
    )
    return parser.parse_args()


def load_points(path: Path) -> list[CalibrationPoint]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        return []

    sample = rows[0].lower()
    has_header = "value" in sample and ("prediction" in sample or "class" in sample)
    points: list[CalibrationPoint] = []

    if has_header:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                points.append(
                    CalibrationPoint(
                        time=parse_optional_float(row.get("time") or row.get("timestamp")),
                        value=parse_float(row.get("value")),
                        prediction=parse_prediction(row.get("prediction") or row.get("class")),
                        confidence=parse_optional_float(row.get("confidence"), default=1.0),
                    )
                )
        return points

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            cleaned = [cell.strip() for cell in row if cell.strip() != ""]
            if len(cleaned) >= 4:
                points.append(
                    CalibrationPoint(
                        time=parse_optional_float(cleaned[0]),
                        value=parse_float(cleaned[1]),
                        prediction=parse_prediction(cleaned[2]),
                        confidence=parse_optional_float(cleaned[3], default=1.0),
                    )
                )
            elif len(cleaned) >= 2:
                points.append(
                    CalibrationPoint(
                        time=None,
                        value=parse_float(cleaned[0]),
                        prediction=parse_prediction(cleaned[1]),
                        confidence=1.0,
                    )
                )

    return points


def parse_float(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing numeric value.")
    return float(value.strip())


def parse_optional_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value.strip() == "":
        return default
    return float(value.strip())


def parse_prediction(value: str | None) -> int:
    return int(float(parse_required(value, "prediction")))


def parse_required(value: str | None, name: str) -> str:
    if value is None or value.strip() == "":
        raise ValueError(f"Missing {name}.")
    return value.strip()


def normalize_window(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    if value_range == 0:
        return [0.0 for _ in values]
    return [(2 * (value - min_value)) / value_range - 1 for value in values]


def build_sequences(points: list[CalibrationPoint], window_size: int, confidence_threshold: float) -> list[list[float]]:
    high_confidence_points = [point for point in points if point.confidence >= confidence_threshold]
    sequences: list[list[float]] = []

    for index in range(window_size, len(high_confidence_points) + 1):
        window = high_confidence_points[index - window_size:index]
        values = [point.value for point in window]
        normalized_values = normalize_window(values)
        amplitude = max(normalized_values) - min(normalized_values)
        center_prediction = window[window_size // 2].prediction
        model_label = center_prediction + 1
        sequences.append([*normalized_values, amplitude, float(model_label)])

    return sequences


def write_sequences(path: Path, sequences: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sequence in sequences:
            file.write(",".join(f"{value:.6f}" for value in sequence) + "\n")


def rebuild_concatenated(output_dir: Path, concatenated_name: str) -> Path:
    concatenated_path = output_dir / concatenated_name
    sequence_files = sorted(
        path for path in output_dir.glob("*.txt")
        if path.name != concatenated_name and not path.name.endswith("_manifest.txt")
    )

    with concatenated_path.open("w", encoding="utf-8") as output:
        for sequence_file in sequence_files:
            for line in sequence_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    output.write(line.strip() + "\n")

    return concatenated_path


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(args.input)
    sequences = build_sequences(points, args.window_size, args.confidence_threshold)
    sequence_path = output_dir / f"{args.input.stem}_sequence.txt"
    write_sequences(sequence_path, sequences)
    concatenated_path = rebuild_concatenated(output_dir, args.concatenated_name)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.input),
        "sequence_file": str(sequence_path),
        "concatenated_file": str(concatenated_path),
        "window_size": args.window_size,
        "confidence_threshold": args.confidence_threshold,
        "input_rows": len(points),
        "sequence_rows": len(sequences),
    }
    manifest_path = output_dir / f"{args.input.stem}_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
