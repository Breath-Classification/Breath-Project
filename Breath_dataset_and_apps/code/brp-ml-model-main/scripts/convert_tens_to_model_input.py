import argparse
import csv
from datetime import datetime
from pathlib import Path

from pseudo_labeling import label as apply_pseudo_labeling


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "NewData"
OUTPUT_DIR = DATA_DIR / "sequence"
PRETRAINED_DIR = DATA_DIR / "pretrained"
RAW_ISO_VALUE = "raw_iso_value"
RAW_ISO_VALUE_PREDICTION_CONFIDENCE = "raw_iso_value_prediction_confidence"
PRETRAINED_VALUE_LABEL_TIME = "pretrained_value_label_time"
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MOVING_AVERAGE = 5
DEFAULT_NORMALIZATION_RANGE = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Converts tensometer logs from the fixed NewData folder into the final "
            "sequence format accepted by the model."
        )
    )
    parser.add_argument(
        "--filename",
        help=(
            "Optional file name from data/NewData/. If omitted, all TXT and CSV files "
            "from that folder are processed."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of signal values in one model sequence for tensometer data.",
    )
    parser.add_argument(
        "--moving-average",
        type=int,
        default=DEFAULT_MOVING_AVERAGE,
        help="Moving average window used for raw tensometer data.",
    )
    parser.add_argument(
        "--normalization-range",
        type=int,
        default=DEFAULT_NORMALIZATION_RANGE,
        help="Backward-looking normalization range used for raw tensometer data.",
    )
    parser.add_argument(
        "--save-pretrained",
        action="store_true",
        help="Also save the intermediate value,label,time file to data/NewData/pretrained/.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


def read_rows(path: Path) -> list[list[str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            stripped = [cell.strip() for cell in row if cell.strip() != ""]
            if stripped:
                rows.append(stripped)
    return rows


def detect_input_format(rows: list[list[str]]) -> str:
    if not rows:
        raise ValueError("Input file is empty.")

    first = rows[0]
    if len(first) < 2:
        raise ValueError("Each row must contain at least two comma-separated values.")

    if len(first) >= 4:
        try:
            parse_timestamp(first[0])
            float(first[1])
            float(first[2])
            float(first[3])
            return RAW_ISO_VALUE_PREDICTION_CONFIDENCE
        except ValueError:
            pass

    try:
        parse_timestamp(first[0])
        float(first[1])
        return RAW_ISO_VALUE
    except ValueError:
        pass

    if len(first) >= 3:
        try:
            float(first[0])
            float(first[1])
            float(first[2])
            return PRETRAINED_VALUE_LABEL_TIME
        except ValueError:
            pass

    raise ValueError(
        "Unsupported input format. Expected 'timestamp,value', "
        "'timestamp,value,predicted_class,confidence' or 'value,label,time[,confidence]'."
    )


def convert_raw_to_seconds_and_values(rows: list[list[str]]) -> tuple[list[float], list[float]]:
    timestamps = [parse_timestamp(row[0]) for row in rows]
    values = [float(row[1]) for row in rows]
    start = min(timestamps)
    seconds = [(timestamp - start).total_seconds() for timestamp in timestamps]
    return seconds, values


def convert_predicted_rows(
    rows: list[list[str]],
) -> tuple[list[float], list[float], list[int], list[float]]:
    timestamps = [parse_timestamp(row[0]) for row in rows]
    values = [float(row[1]) for row in rows]
    labels = [int(float(row[2])) for row in rows]
    confidences = [float(row[3]) for row in rows]
    start = min(timestamps)
    seconds = [(timestamp - start).total_seconds() for timestamp in timestamps]
    return seconds, values, labels, confidences


def moving_average(values: list[float], window_size: int) -> list[float]:
    if window_size <= 1:
        return values[:]
    if len(values) < window_size:
        return []

    result = []
    window_sum = sum(values[:window_size])
    result.append(window_sum / window_size)

    for index in range(window_size, len(values)):
        window_sum += values[index] - values[index - window_size]
        result.append(window_sum / window_size)

    return result


def normalize_window(window: list[float]) -> list[float]:
    min_value = min(window)
    max_value = max(window)
    range_value = max_value - min_value
    if range_value == 0:
        return [0.0 for _ in window]
    return [-1 + 2 * (value - min_value) / range_value for value in window]


def normalize(values: list[float], normalization_range: int) -> list[float]:
    normalized_values = []
    for index in range(len(values)):
        window = values[max(0, index - normalization_range) : index]
        if not window:
            continue
        normalized_values.append(normalize_window(window)[-1])
    return normalized_values


def monotonicity(values: list[float], threshold: float = 0.0079) -> list[int]:
    if not values:
        return []
    if len(values) == 1:
        return [0]

    gradients = []
    for index in range(len(values)):
        if index == 0:
            derivative = values[1] - values[0]
        elif index == len(values) - 1:
            derivative = values[-1] - values[-2]
        else:
            derivative = (values[index + 1] - values[index - 1]) / 2
        gradients.append(derivative)

    tags = [0] * len(values)
    for index, derivative in enumerate(gradients):
        if abs(derivative) < threshold:
            tags[index] = 0
        elif derivative > 0:
            tags[index] = 1
        else:
            tags[index] = -1
    return tags


def raw_to_pretrained(
    seconds: list[float],
    raw_values: list[float],
    moving_average_window: int,
    normalization_range: int,
    source_labels: list[int] | None = None,
    source_confidences: list[float] | None = None,
) -> tuple[list[float], list[int], list[float], list[float] | None]:
    smoothed = moving_average(raw_values, moving_average_window)
    normalized = normalize(smoothed, normalization_range)

    if not normalized:
        raise ValueError(
            "Not enough data to normalize the signal. Provide a longer recording."
        )

    aligned_times = seconds[moving_average_window - 1 :]
    aligned_times = aligned_times[-len(normalized) :]

    if source_labels is None:
        labels = monotonicity(normalized)
    else:
        labels = source_labels[moving_average_window - 1 :]
        labels = labels[-len(normalized) :]

    aligned_confidences = None
    if source_confidences is not None:
        aligned_confidences = source_confidences[moving_average_window - 1 :]
        aligned_confidences = aligned_confidences[-len(normalized) :]

    return normalized, labels, aligned_times, aligned_confidences


def pretrained_to_sequences(
    values: list[float], labels: list[float], sequence_size: int
) -> list[list[float]]:
    sequences = []
    for index in range(sequence_size, len(values)):
        position = index - sequence_size + sequence_size // 2
        sequence = values[index - sequence_size : index]
        feature = abs(max(sequence) - min(sequence))
        label = labels[position] + 1
        sequences.append(sequence + [feature, label])
    return sequences


def pretrained_to_sequences_with_confidence(
    values: list[float],
    labels: list[float],
    confidences: list[float],
    sequence_size: int,
) -> list[list[float]]:
    sequences = []
    for index in range(sequence_size, len(values)):
        position = index - sequence_size + sequence_size // 2
        sequence = values[index - sequence_size : index]
        feature = abs(max(sequence) - min(sequence))
        label = labels[position] + 1
        confidence = confidences[position]
        sequences.append(sequence + [feature, label, confidence])
    return sequences


def save_pretrained(
    path: Path,
    values: list[float],
    labels: list[int],
    times: list[float],
    confidences: list[float] | None = None,
):
    with path.open("w", encoding="utf-8", newline="") as handle:
        if confidences is None:
            for value, label, time_value in zip(values, labels, times):
                handle.write(f"{value},{label},{time_value}\n")
            return

        for value, label, time_value, confidence in zip(values, labels, times, confidences):
            handle.write(f"{value},{label},{time_value},{confidence}\n")


def save_sequences(path: Path, sequences: list[list[float]]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sequence in sequences:
            handle.write(",".join(str(item) for item in sequence) + "\n")


def load_pretrained_rows(
    rows: list[list[str]],
) -> tuple[list[float], list[float], list[float], list[float] | None]:
    values = [float(row[0]) for row in rows]
    labels = [float(row[1]) for row in rows]
    times = [float(row[2]) for row in rows]
    confidences = None
    if all(len(row) >= 4 for row in rows):
        confidences = [float(row[3]) for row in rows]
    return values, labels, times, confidences


def process_file(
    input_path: Path,
    window_size: int,
    moving_average_window: int,
    normalization_range: int,
    save_pretrained_file: bool,
):
    rows = read_rows(input_path)
    input_format = detect_input_format(rows)

    confidences = None
    if input_format == RAW_ISO_VALUE:
        seconds, raw_values = convert_raw_to_seconds_and_values(rows)
        values, labels, times, confidences = raw_to_pretrained(
            seconds,
            raw_values,
            moving_average_window=moving_average_window,
            normalization_range=normalization_range,
        )
    elif input_format == RAW_ISO_VALUE_PREDICTION_CONFIDENCE:
        seconds, raw_values, source_labels, source_confidences = convert_predicted_rows(rows)
        values, labels, times, confidences = raw_to_pretrained(
            seconds,
            raw_values,
            moving_average_window=moving_average_window,
            normalization_range=normalization_range,
            source_labels=source_labels,
            source_confidences=source_confidences,
        )
    else:
        values, labels, times, confidences = load_pretrained_rows(rows)

    pretrained_path = None
    if save_pretrained_file:
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        pretrained_path = PRETRAINED_DIR / f"{input_path.stem}_pretrained.txt"
        save_pretrained(pretrained_path, values, labels, times, confidences)

    if confidences is not None:
        sequences = pretrained_to_sequences_with_confidence(
            values,
            labels,
            confidences,
            window_size,
        )
    else:
        sequences = pretrained_to_sequences(values, labels, window_size)

    if not sequences:
        raise ValueError(
            "Not enough data to build model sequences. Provide a longer recording."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_model_input.txt"
    save_sequences(output_path, sequences)

    pseudo_labeled = False
    if confidences is not None:
        apply_pseudo_labeling(output_path)
        pseudo_labeled = True

    return input_format, output_path, pretrained_path, len(sequences), pseudo_labeled


def resolve_input_files(filename: str | None) -> list[Path]:
    if filename:
        input_path = DATA_DIR / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]

    files = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".csv"}:
            continue
        files.append(path)
    return files


def main():
    args = parse_args()
    input_files = resolve_input_files(args.filename)
    if not input_files:
        raise FileNotFoundError(f"No TXT or CSV files found in {DATA_DIR}")

    for input_path in input_files:
        input_format, output_path, pretrained_path, sequence_count, pseudo_labeled = process_file(
            input_path=input_path,
            window_size=args.window_size,
            moving_average_window=args.moving_average,
            normalization_range=args.normalization_range,
            save_pretrained_file=args.save_pretrained,
        )
        print(f"Input file: {input_path}")
        print(f"Input format: {input_format}")
        print(f"Saved model-ready sequences to: {output_path}")
        if pretrained_path is not None:
            print(f"Saved pretrained data to: {pretrained_path}")
        if pseudo_labeled:
            print("Applied pseudo labeling to remove the confidence column from sequence output.")
        print(f"Number of sequences: {sequence_count}")
        print("-")


if __name__ == "__main__":
    main()
