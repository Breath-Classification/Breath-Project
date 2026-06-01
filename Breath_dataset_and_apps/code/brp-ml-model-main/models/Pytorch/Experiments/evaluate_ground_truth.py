from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_LABELS = [-1, 0, 1, 2, 999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction labels against manually labelled ground truth."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV with prediction and true_label columns.")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--truth-column", default="true_label")
    parser.add_argument(
        "--ignore-label",
        action="append",
        default=[],
        type=int,
        help="Label to skip during evaluation. Can be passed multiple times, e.g. --ignore-label 999.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def read_label_pairs(
    path: Path,
    prediction_column: str,
    truth_column: str,
    ignored_labels: set[int],
) -> tuple[list[int], list[int], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        missing = [column for column in [prediction_column, truth_column] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        predictions = []
        truths = []
        skipped = 0
        for row in reader:
            if not row:
                continue
            prediction = parse_label(row[prediction_column])
            truth = parse_label(row[truth_column])
            if prediction in ignored_labels or truth in ignored_labels:
                skipped += 1
                continue
            predictions.append(prediction)
            truths.append(truth)

    if not predictions:
        raise ValueError("No evaluable rows found after applying ignored labels.")

    return predictions, truths, skipped


def parse_label(value: str) -> int:
    return int(float(value))


def evaluate(predictions: list[int], truths: list[int], skipped: int) -> dict:
    labels = sorted(set(DEFAULT_LABELS) | set(predictions) | set(truths))
    labels = [label for label in labels if label in set(predictions) or label in set(truths)]
    correct = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == truth)
    total = len(truths)

    per_class = {}
    for label in labels:
        tp = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == label and truth == label)
        fp = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == label and truth != label)
        fn = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction != label and truth == label)
        support = sum(1 for truth in truths if truth == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    confusion_matrix = {
        str(truth_label): {
            str(prediction_label): sum(
                1
                for prediction, truth in zip(predictions, truths, strict=False)
                if truth == truth_label and prediction == prediction_label
            )
            for prediction_label in labels
        }
        for truth_label in labels
    }

    return {
        "total": total,
        "skipped": skipped,
        "accuracy": safe_divide(correct, total),
        "macro_precision": mean([metrics["precision"] for metrics in per_class.values()]),
        "macro_recall": mean([metrics["recall"] for metrics in per_class.values()]),
        "macro_f1": mean([metrics["f1"] for metrics in per_class.values()]),
        "prediction_distribution": dict(sorted(Counter(predictions).items())),
        "truth_distribution": dict(sorted(Counter(truths).items())),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_text_report(metrics: dict) -> None:
    print(f"Rows evaluated: {metrics['total']}")
    print(f"Rows skipped: {metrics['skipped']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print()
    print("Class metrics:")
    for label, class_metrics in metrics["per_class"].items():
        print(
            f"  {label:>4}  "
            f"precision={class_metrics['precision']:.4f}  "
            f"recall={class_metrics['recall']:.4f}  "
            f"f1={class_metrics['f1']:.4f}  "
            f"support={class_metrics['support']}"
        )
    print()
    print("Confusion matrix: rows=true, columns=predicted")
    labels = list(metrics["confusion_matrix"].keys())
    print("true\\pred " + " ".join(f"{label:>6}" for label in labels))
    for truth_label in labels:
        row = metrics["confusion_matrix"][truth_label]
        print(f"{truth_label:>9} " + " ".join(f"{row[prediction_label]:>6}" for prediction_label in labels))


def main() -> None:
    args = parse_args()
    predictions, truths, skipped = read_label_pairs(
        args.input,
        args.prediction_column,
        args.truth_column,
        set(args.ignore_label),
    )
    metrics = evaluate(predictions, truths, skipped)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_text_report(metrics)


if __name__ == "__main__":
    main()
