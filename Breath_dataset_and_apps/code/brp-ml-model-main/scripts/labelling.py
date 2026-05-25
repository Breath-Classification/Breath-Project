from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import Button, Frame, Tk


COLOR_MAP = {
    -1: "red",
    0: "green",
    1: "blue",
    2: "orange",
    999: "black",
}

np = None
plt = None
SpanSelector = None


def load_plotting_backend() -> None:
    global np, plt, SpanSelector
    if np is not None and plt is not None and SpanSelector is not None:
        return

    try:
        import matplotlib
        import numpy

        matplotlib.use("TkAgg")
        from matplotlib import pyplot
        from matplotlib.widgets import SpanSelector as MatplotlibSpanSelector
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Manual labelling requires matplotlib and numpy. Install them in your "
            "Python environment, for example: python3 -m pip install matplotlib numpy"
        ) from exc

    np = numpy
    plt = pyplot
    SpanSelector = MatplotlibSpanSelector


BREATH_STATE = {
    -1: "BREATH OUT",
    0: "NO BREATH",
    1: "BREATH IN",
    2: "IN NO BREATH",
    999: "IGNORE",
}
MOVING_AVERAGE_WINDOW = 5
NORMALIZATION_RANGE = 150


@dataclass
class LabelRecord:
    time: str
    seconds: float
    value: float
    prediction: int
    confidence: float | None
    true_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual labelling tool for BreathSense tensometer files. Supports "
            "time,value,prediction,confidence and value,label,time formats."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input txt/csv file to label.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file. Defaults to <input_stem>_ground_truth.csv next to input.",
    )
    parser.add_argument(
        "--output-format",
        choices=["ground-truth", "pretrained"],
        default="ground-truth",
        help=(
            "ground-truth writes time,value,prediction,confidence,true_label. "
            "pretrained writes value,true_label,seconds."
        ),
    )
    parser.add_argument("--window-seconds", default=15.0, type=float)
    parser.add_argument("--step-seconds", default=10.0, type=float)
    return parser.parse_args()


def parse_time_values(values: list[str]) -> list[float]:
    if all(can_parse_float(value) for value in values):
        first = float(values[0])
        return [float(value) - first for value in values]

    timestamps = [parse_timestamp(value) for value in values]
    start = min(timestamps)
    return [(timestamp - start).total_seconds() for timestamp in timestamps]


def can_parse_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


def read_records(path: Path) -> list[LabelRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            return read_header_records(handle)
        return read_plain_records(handle)


def read_header_records(handle) -> list[LabelRecord]:
    reader = csv.DictReader(handle)
    fieldnames = {field.lower(): field for field in reader.fieldnames or []}
    required = {"time", "value"}
    if not required.issubset(fieldnames):
        raise ValueError("Header input must contain at least time and value columns.")

    rows = [row for row in reader if any((cell or "").strip() for cell in row.values())]
    raw_times = [row[fieldnames["time"]].strip() for row in rows]
    seconds = parse_time_values(raw_times)

    records = []
    for row, second in zip(rows, seconds, strict=False):
        prediction = parse_int(row.get(fieldnames.get("prediction", ""), "0") or "0")
        true_label_key = fieldnames.get("true_label") or fieldnames.get("label")
        true_label = parse_int(row.get(true_label_key, prediction) if true_label_key else prediction)
        confidence_key = fieldnames.get("confidence")
        confidence = parse_optional_float(row.get(confidence_key, "") if confidence_key else "")
        records.append(
            LabelRecord(
                time=row[fieldnames["time"]].strip(),
                seconds=second,
                value=float(row[fieldnames["value"]]),
                prediction=prediction,
                confidence=confidence,
                true_label=true_label,
            )
        )
    return records


def read_plain_records(handle) -> list[LabelRecord]:
    rows = []
    for row in csv.reader(handle):
        stripped = [cell.strip() for cell in row if cell.strip() != ""]
        if stripped:
            rows.append(stripped)
    if not rows:
        raise ValueError("Input file is empty.")

    if len(rows[0]) >= 4 and can_parse_timestamp_or_float(rows[0][0]):
        raw_times = [row[0] for row in rows]
        seconds = parse_time_values(raw_times)
        return [
            LabelRecord(
                time=row[0],
                seconds=second,
                value=float(row[1]),
                prediction=parse_int(row[2]),
                confidence=parse_optional_float(row[3]),
                true_label=parse_int(row[2]),
            )
            for row, second in zip(rows, seconds, strict=False)
        ]

    if len(rows[0]) >= 3:
        return [
            LabelRecord(
                time=row[2],
                seconds=float(row[2]),
                value=float(row[0]),
                prediction=parse_int(row[1]),
                confidence=parse_optional_float(row[3]) if len(row) >= 4 else None,
                true_label=parse_int(row[1]),
            )
            for row in rows
        ]

    raise ValueError("Expected time,value,prediction,confidence or value,label,time rows.")


def can_parse_timestamp_or_float(value: str) -> bool:
    if can_parse_float(value):
        return True
    try:
        parse_timestamp(value)
        return True
    except ValueError:
        return False


def parse_int(value) -> int:
    return int(float(value))


def parse_optional_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ground_truth.csv")


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


def normalize_window(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    if value_range == 0:
        return [0.0 for _ in values]
    return [(2 * (value - min_value)) / value_range - 1 for value in values]


def normalize(values: list[float], normalization_range: int) -> list[float]:
    normalized = []
    for index in range(len(values)):
        window = values[max(0, index - normalization_range): index]
        if not window:
            continue
        normalized.append(normalize_window(window)[-1])
    return normalized


def build_display_series(records: list[LabelRecord]) -> tuple[list[int], list[float], list[float]]:
    raw_values = [record.value for record in records]
    raw_seconds = [record.seconds for record in records]
    if not raw_values:
        return [], [], []

    # Old labelling files already store normalized values as value,label,time.
    if min(raw_values) >= -1.5 and max(raw_values) <= 1.5:
        return list(range(len(records))), raw_seconds, raw_values

    smoothed = moving_average(raw_values, MOVING_AVERAGE_WINDOW)
    normalized = normalize(smoothed, NORMALIZATION_RANGE)
    if not normalized:
        return list(range(len(records))), raw_seconds, raw_values

    start_index = MOVING_AVERAGE_WINDOW - 1
    aligned_indices = list(range(start_index, len(records)))
    aligned_indices = aligned_indices[-len(normalized):]
    aligned_seconds = [records[index].seconds for index in aligned_indices]
    return aligned_indices, aligned_seconds, normalized


def save_records(records: list[LabelRecord], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "pretrained":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for record in records:
                writer.writerow([record.value, record.true_label, record.seconds])
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "value", "prediction", "confidence", "true_label"])
        for record in records:
            writer.writerow(
                [
                    record.time,
                    record.value,
                    record.prediction,
                    "" if record.confidence is None else record.confidence,
                    record.true_label,
                ]
            )


class ManualLabeler:
    def __init__(
        self,
        records: list[LabelRecord],
        output_path: Path,
        output_format: str,
        window_seconds: float,
        step_seconds: float,
    ) -> None:
        load_plotting_backend()
        self.records = records
        self.output_path = output_path
        self.output_format = output_format
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.current_start = 0.0
        record_indices, seconds, values = build_display_series(records)
        self.record_indices = record_indices
        self.seconds = np.array(seconds)
        self.values = np.array(values)
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.scatter_indices: list[int] = []
        self.span_selector = SpanSelector(
            self.ax,
            self.on_span_select,
            "horizontal",
            useblit=True,
            props={"alpha": 0.25, "facecolor": "tab:gray"},
            interactive=True,
        )

    def run(self) -> None:
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.plot()
        plt.show()

    def visible_indices(self) -> list[int]:
        end = self.current_start + self.window_seconds
        return np.where((self.seconds >= self.current_start) & (self.seconds <= end))[0].tolist()

    def plot(self) -> None:
        indices = self.visible_indices()
        if not indices:
            return

        self.scatter_indices = indices
        self.ax.clear()
        labels = [self.records[self.record_indices[index]].true_label for index in indices]
        colors = [COLOR_MAP.get(label, "gray") for label in labels]
        x_values = self.seconds[indices]
        y_values = self.values[indices]

        self.ax.scatter(x_values, y_values, color=colors, picker=True, s=28)
        self.ax.plot(x_values, y_values, color="gray", alpha=0.45, linewidth=1)
        self.ax.set_xlim(self.current_start, self.current_start + self.window_seconds)
        self.ax.set_xlabel("Time [seconds]")
        self.ax.set_ylabel("Normalized tensometer value")
        self.ax.set_title(
            "Manual labelling: drag range or click point. "
            "Left/Right move, s save, q quit."
        )
        self.ax.grid(True, alpha=0.25)
        self.fig.canvas.draw_idle()

    def on_pick(self, event) -> None:
        if not event.ind:
            return
        visible_index = int(event.ind[0])
        if visible_index >= len(self.scatter_indices):
            return
        record_index = self.record_indices[self.scatter_indices[visible_index]]
        label = choose_label(self.records[record_index].true_label)
        self.assign_label(record_index, record_index + 1, label)

    def on_span_select(self, xmin: float, xmax: float) -> None:
        start = min(xmin, xmax)
        end = max(xmin, xmax)
        indices = np.where((self.seconds >= start) & (self.seconds <= end))[0]
        if len(indices) == 0:
            return
        start_record_index = self.record_indices[int(indices[0])]
        end_record_index = self.record_indices[int(indices[-1])]
        label = choose_label(self.records[start_record_index].true_label)
        self.assign_label(start_record_index, end_record_index + 1, label)

    def assign_label(self, start_index: int, end_index: int, label: int) -> None:
        for index in range(start_index, end_index):
            self.records[index].true_label = label
        save_records(self.records, self.output_path, self.output_format)
        self.plot()

    def on_key(self, event) -> None:
        max_start = max(0.0, float(self.seconds[-1]) - self.window_seconds)
        if event.key == "right":
            self.current_start = min(max_start, self.current_start + self.step_seconds)
        elif event.key == "left":
            self.current_start = max(0.0, self.current_start - self.step_seconds)
        elif event.key == "s":
            save_records(self.records, self.output_path, self.output_format)
            print(f"Saved labels to {self.output_path}")
        elif event.key == "q":
            save_records(self.records, self.output_path, self.output_format)
            plt.close(self.fig)
            return
        self.plot()


def choose_label(initial_label: int) -> int:
    result = {"label": initial_label}

    def select_label(label: int) -> None:
        result["label"] = label
        root.destroy()

    root = Tk()
    root.title("Choose label")

    frame = Frame(root)
    frame.pack(padx=10, pady=10)
    for label, color in COLOR_MAP.items():
        button = Button(
            frame,
            text=BREATH_STATE[label],
            command=lambda selected=label: select_label(selected),
            bg=color,
            width=12,
            highlightbackground=color,
        )
        button.pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", lambda: select_label(initial_label))
    root.wait_window()
    return result["label"]


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    if not records:
        raise ValueError("No records found in input file.")

    output_path = args.output or default_output_path(args.input)
    save_records(records, output_path, args.output_format)
    print(f"Initial labels saved to {output_path}")
    ManualLabeler(
        records=records,
        output_path=output_path,
        output_format=args.output_format,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    ).run()


if __name__ == "__main__":
    main()
