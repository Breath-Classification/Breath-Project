from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.pseudo_labeling import decide_pseudo_label
except ModuleNotFoundError:
    from pseudo_labeling import decide_pseudo_label


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "NewData"
DEFAULT_SEQUENCE_DIR = DEFAULT_INPUT_DIR / "sequence"
DEFAULT_PRETRAINED_DIR = DEFAULT_INPUT_DIR / "pretrained"
DEFAULT_CONCATENATED_NAME = "concatenated.txt"
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MOVING_AVERAGE = 5
DEFAULT_NORMALIZATION_RANGE = 150

RAW_TIME_VALUE = "raw_time_value"
RAW_TIME_VALUE_PREDICTION_CONFIDENCE = "raw_time_value_prediction_confidence"
PRETRAINED_VALUE_LABEL_TIME = "pretrained_value_label_time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert tensometer calibration data into model-input sequences. Supports "
            "time,value,prediction,confidence rows from the app, raw time,value rows, "
            "and pretrained value,label,time rows."
        )
    )
    parser.add_argument("--input", type=Path, help="Specific CSV/TXT calibration file to process.")
    parser.add_argument(
        "--filename",
        help="Optional file name from data/NewData. Kept for compatibility with the FT branch script.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_SEQUENCE_DIR, type=Path)
    parser.add_argument("--pretrained-dir", default=DEFAULT_PRETRAINED_DIR, type=Path)
    parser.add_argument("--window-size", default=DEFAULT_WINDOW_SIZE, type=int)
    parser.add_argument("--moving-average-window", default=DEFAULT_MOVING_AVERAGE, type=int)
    parser.add_argument("--normalization-range", default=DEFAULT_NORMALIZATION_RANGE, type=int)
    parser.add_argument(
        "--confidence-threshold",
        default=0.8,
        type=float,
        help="Keep pseudo-labelled sequences whose center confidence is greater than or equal to this value.",
    )
    parser.add_argument(
        "--pseudo-labelling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use prediction/confidence columns from app exports when available. "
            "Pass --no-pseudo-labelling to ignore them and treat the file as raw time,value data."
        ),
    )
    parser.add_argument("--save-pretrained", action="store_true")
    parser.add_argument("--concatenated-name", default=DEFAULT_CONCATENATED_NAME)
    return parser.parse_args()


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            stripped = [cell.strip() for cell in row if cell.strip() != ""]
            if not stripped:
                continue
            if is_header(stripped):
                continue
            rows.append(stripped)
    return rows


def is_header(row: list[str]) -> bool:
    lowered = ",".join(row).lower()
    return "value" in lowered and ("prediction" in lowered or "class" in lowered or "confidence" in lowered)


def detect_input_format(rows: list[list[str]]) -> str:
    if not rows:
        raise ValueError("Input file is empty.")

    first = rows[0]
    if len(first) < 2:
        raise ValueError("Each row must contain at least two comma-separated values.")

    if len(first) >= 4 and can_parse_time(first[0]) and can_parse_float(first[1]) and can_parse_float(first[2]) and can_parse_float(first[3]):
        return RAW_TIME_VALUE_PREDICTION_CONFIDENCE

    if can_parse_time(first[0]) and can_parse_float(first[1]) and not looks_like_pretrained(first):
        return RAW_TIME_VALUE

    if len(first) >= 3 and can_parse_float(first[0]) and can_parse_float(first[1]) and can_parse_float(first[2]):
        return PRETRAINED_VALUE_LABEL_TIME

    raise ValueError(
        "Unsupported input format. Expected time,value,prediction,confidence; "
        "time,value; or value,label,time[,confidence]."
    )


def looks_like_pretrained(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    if not all(can_parse_float(cell) for cell in row[:3]):
        return False
    label = float(row[1])
    return label in {-1.0, 0.0, 1.0, 2.0, 3.0}


def can_parse_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def can_parse_time(value: str) -> bool:
    if can_parse_float(value):
        return True
    try:
        parse_timestamp(value)
        return True
    except ValueError:
        return False


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


def parse_times(values: list[str]) -> list[float]:
    if all(can_parse_float(value) for value in values):
        first = float(values[0])
        return [float(value) - first for value in values]

    timestamps = [parse_timestamp(value) for value in values]
    start = min(timestamps)
    return [(timestamp - start).total_seconds() for timestamp in timestamps]


def convert_raw_rows(rows: list[list[str]]) -> tuple[list[float], list[float]]:
    times = parse_times([row[0] for row in rows])
    values = [float(row[1]) for row in rows]
    return times, values


def convert_predicted_rows(rows: list[list[str]]) -> tuple[list[float], list[float], list[int], list[float]]:
    times = parse_times([row[0] for row in rows])
    values = [float(row[1]) for row in rows]
    labels = [int(float(row[2])) for row in rows]
    confidences = [float(row[3]) for row in rows]
    return times, values, labels, confidences


def load_pretrained_rows(rows: list[list[str]]) -> tuple[list[float], list[int], list[float], list[float] | None]:
    values = [float(row[0]) for row in rows]
    labels = [int(float(row[1])) for row in rows]
    times = [float(row[2]) for row in rows]
    confidences = None
    if all(len(row) >= 4 for row in rows):
        confidences = [float(row[3]) for row in rows]
    return values, labels, times, confidences


def moving_average(values: list[float], window_size: int) -> list[float]:
    if window_size <= 1:
        return values[:]
    if len(values) < window_size:
        return []

    result: list[float] = []
    window_sum = sum(values[:window_size])
    result.append(window_sum / window_size)
    for index in range(window_size, len(values)):
        window_sum += values[index] - values[index - window_size]
        result.append(window_sum / window_size)
    return result


def normalize_window(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    if value_range == 0:
        return [0.0 for _ in values]
    return [(2 * (value - min_value)) / value_range - 1 for value in values]


def normalize(values: list[float], normalization_range: int) -> list[float]:
    normalized: list[float] = []
    for index in range(len(values)):
        window = values[max(0, index - normalization_range): index]
        if not window:
            continue
        normalized.append(normalize_window(window)[-1])
    return normalized


def monotonicity(values: list[float], threshold: float = 0.0079) -> list[int]:
    if not values:
        return []
    if len(values) == 1:
        return [0]

    gradients: list[float] = []
    for index in range(len(values)):
        if index == 0:
            derivative = values[1] - values[0]
        elif index == len(values) - 1:
            derivative = values[-1] - values[-2]
        else:
            derivative = (values[index + 1] - values[index - 1]) / 2
        gradients.append(derivative)

    labels = [0] * len(values)
    for index, derivative in enumerate(gradients):
        if abs(derivative) < threshold:
            labels[index] = 0
        elif derivative > 0:
            labels[index] = 1
        else:
            labels[index] = -1
    return labels


def raw_to_pretrained(
    times: list[float],
    raw_values: list[float],
    *,
    moving_average_window: int,
    normalization_range: int,
    source_labels: list[int] | None = None,
    source_confidences: list[float] | None = None,
) -> tuple[list[float], list[int], list[float], list[float] | None]:
    smoothed = moving_average(raw_values, moving_average_window)
    normalized = normalize(smoothed, normalization_range)
    if not normalized:
        raise ValueError("Not enough data to normalize the signal. Provide a longer recording.")

    aligned_times = times[moving_average_window - 1:]
    aligned_times = aligned_times[-len(normalized):]

    if source_labels is None:
        labels = monotonicity(normalized)
    else:
        labels = source_labels[moving_average_window - 1:]
        labels = labels[-len(normalized):]

    aligned_confidences = None
    if source_confidences is not None:
        aligned_confidences = source_confidences[moving_average_window - 1:]
        aligned_confidences = aligned_confidences[-len(normalized):]

    return normalized, labels, aligned_times, aligned_confidences


def build_sequences(
    values: list[float],
    labels: list[int],
    times: list[float] | None,
    confidences: list[float] | None,
    *,
    window_size: int,
    confidence_threshold: float,
) -> list[list[float]]:
    sequences: list[list[float]] = []
    for index in range(window_size, len(values)):
        position = index - window_size + window_size // 2
        label = labels[position]
        window = values[index - window_size:index]
        amplitude = abs(max(window) - min(window))
        
        decision = decide_pseudo_label(confidences,position,confidence_threshold,
                                       labels,times,window_size,amplitude,values)
        if decision == "keep":
            pass
        elif decision == "drop":
            continue
        elif isinstance(decision, tuple):
            tag, value = decision
            if tag == "relabeled":
                label = value

        model_label = label + 1
        sequences.append([*window, amplitude, float(model_label)])
    return sequences


def save_pretrained(
    path: Path,
    values: list[float],
    labels: list[int],
    times: list[float],
    confidences: list[float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for index, (value, label, time_value) in enumerate(zip(values, labels, times, strict=False)):
            if confidences is None:
                handle.write(f"{value},{label},{time_value}\n")
            else:
                handle.write(f"{value},{label},{time_value},{confidences[index]}\n")


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


def process_file(
    input_path: Path,
    *,
    output_dir: Path,
    pretrained_dir: Path,
    window_size: int,
    moving_average_window: int,
    normalization_range: int,
    confidence_threshold: float,
    pseudo_labelling: bool,
    save_pretrained_file: bool,
) -> dict:
    rows = read_rows(input_path)
    input_format = detect_input_format(rows)

    if input_format == RAW_TIME_VALUE:
        times, raw_values = convert_raw_rows(rows)
        values, labels, aligned_times, confidences = raw_to_pretrained(
            times,
            raw_values,
            moving_average_window=moving_average_window,
            normalization_range=normalization_range,
        )
    elif input_format == RAW_TIME_VALUE_PREDICTION_CONFIDENCE and pseudo_labelling:
        times, raw_values, source_labels, source_confidences = convert_predicted_rows(rows)
        values, labels, aligned_times, confidences = raw_to_pretrained(
            times,
            raw_values,
            moving_average_window=moving_average_window,
            normalization_range=normalization_range,
            source_labels=source_labels,
            source_confidences=source_confidences,
        )
    elif input_format == RAW_TIME_VALUE_PREDICTION_CONFIDENCE:
        times, raw_values, _, _ = convert_predicted_rows(rows)
        values, labels, aligned_times, confidences = raw_to_pretrained(
            times,
            raw_values,
            moving_average_window=moving_average_window,
            normalization_range=normalization_range,
        )
    else:
        values, labels, aligned_times, confidences = load_pretrained_rows(rows)

    pretrained_path = None
    if save_pretrained_file:
        pretrained_path = pretrained_dir / f"{input_path.stem}_pretrained.txt"
        save_pretrained(pretrained_path, values, labels, aligned_times, confidences)

    sequences = build_sequences(
        values,
        labels,
        aligned_times,
        confidences,
        window_size=window_size,
        confidence_threshold=confidence_threshold,
    )
    if not sequences:
        raise ValueError("Not enough high-confidence data to build model sequences. Provide a longer recording.")

    sequence_path = output_dir / f"{input_path.stem}_sequence.txt"
    write_sequences(sequence_path, sequences)

    return {
        "source_file": str(input_path),
        "input_format": input_format,
        "sequence_file": str(sequence_path),
        "pretrained_file": str(pretrained_path) if pretrained_path else None,
        "input_rows": len(rows),
        "pretrained_rows": len(values),
        "sequence_rows": len(sequences),
        "pseudo_labeled": confidences is not None and pseudo_labelling,
    }


def resolve_input_files(input_path: Path | None, input_dir: Path, filename: str | None) -> list[Path]:
    if input_path is not None:
        return [input_path]
    if filename:
        candidate = input_dir / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        return [candidate]
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".csv"}
    )


def main() -> None:
    args = parse_args()
    input_files = resolve_input_files(args.input, args.input_dir, args.filename)
    if not input_files:
        raise FileNotFoundError(f"No TXT or CSV files found in {args.input_dir}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = [
        process_file(
            input_file,
            output_dir=output_dir,
            pretrained_dir=args.pretrained_dir,
            window_size=args.window_size,
            moving_average_window=args.moving_average_window,
            normalization_range=args.normalization_range,
            confidence_threshold=args.confidence_threshold,
            pseudo_labelling=args.pseudo_labelling,
            save_pretrained_file=args.save_pretrained,
        )
        for input_file in input_files
    ]
    concatenated_path = rebuild_concatenated(output_dir, args.concatenated_name)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence_file": processed_files[-1]["sequence_file"],
        "concatenated_file": str(concatenated_path),
        "window_size": args.window_size,
        "moving_average_window": args.moving_average_window,
        "normalization_range": args.normalization_range,
        "confidence_threshold": args.confidence_threshold,
        "pseudo_labelling": args.pseudo_labelling,
        "files": processed_files,
        "input_rows": sum(file_info["input_rows"] for file_info in processed_files),
        "sequence_rows": sum(file_info["sequence_rows"] for file_info in processed_files),
    }
    manifest_path = output_dir / f"{Path(processed_files[-1]['source_file']).stem}_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
