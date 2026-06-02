from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTORCH_DIR = PROJECT_DIR / "models" / "Pytorch"
sys.path.append(str(PYTORCH_DIR))

from convert_tens_to_model_input import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MOVING_AVERAGE,
    DEFAULT_NORMALIZATION_RANGE,
    DEFAULT_WINDOW_SIZE,
    PRETRAINED_VALUE_LABEL_TIME,
    RAW_TIME_VALUE,
    RAW_TIME_VALUE_PREDICTION_CONFIDENCE,
    build_sequences,
    convert_predicted_rows,
    convert_raw_rows,
    detect_input_format,
    load_pretrained_rows,
    raw_to_pretrained,
    read_rows,
)

DEFAULT_FILENAME = ""
DEFAULT_BASE_MODEL = ""
DEFAULT_OUTPUT = None
STRATEGY = "none"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo labelling methods"
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, type=Path)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--strategy", default=STRATEGY, type=str)
    parser.add_argument("--window-size", default=DEFAULT_WINDOW_SIZE, type=int)
    parser.add_argument("--moving-average-window", default=DEFAULT_MOVING_AVERAGE, type=int)
    parser.add_argument("--normalization-range", default=DEFAULT_NORMALIZATION_RANGE, type=int)
    parser.add_argument("--confidence-threshold", default=0.8, type=float)
    parser.add_argument("--pseudo-labelling", action=argparse.BooleanOptionalAction, default=True)
    
    return parser.parse_args()


def resolve_filename(filename: Path) -> Path:
    if str(filename) == "":
        raise ValueError("Pass --filename with a CSV/TXT file to evaluate.")

    if filename.exists():
        return filename

    candidate = DEFAULT_INPUT_DIR / filename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Input file not found: {filename}")


def prepare_pretrained_data(
    filename: Path,
    moving_average_window: int,
    normalization_range: int,
    pseudo_labelling: bool,
):
    rows = read_rows(filename)
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
    elif input_format == PRETRAINED_VALUE_LABEL_TIME:
        values, labels, aligned_times, confidences = load_pretrained_rows(rows)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    return rows, input_format, values, labels, aligned_times, confidences


def run_strategy(args: argparse.Namespace): #decide_pseudo_labelling(strategy="XXx")
    print("test")
    filename = resolve_filename(args.filename)
    rows, input_format, values, labels, times, confidences = prepare_pretrained_data(
        filename,
        args.moving_average_window,
        args.normalization_range,
        args.pseudo_labelling,
    )
    sequences, decisions, relabeled = build_sequences(
        values,
        labels,
        times,
        confidences,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        pseudo_labelling=args.pseudo_labelling,
        strategy=args.strategy,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(filename),
        "input_format": input_format,
        "strategy": args.strategy,
        "input_rows": len(rows),
        "pretrained_rows": len(values),
        "sequence_rows": len(sequences),
        "decisions": decisions,
        "relabeled": relabeled,
    }


def visualize_decisions(result):#keep =? drop =? relabeled =? 1->0 =? itd
    print(f"File: {result['source_file']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Input rows: {result['input_rows']}")
    print(f"Pretrained rows: {result['pretrained_rows']}")
    print(f"Sequence rows: {result['sequence_rows']}")
    print("Decisions:")
    for name, count in result["decisions"].items():
        print(f"  {name}: {count}")
    print("Relabeled:")
    if result["relabeled"]:
        for change, count in result["relabeled"].items():
            print(f"  {change}: {count}")
    else:
        print("  none: 0")
    
def save_sequences(): #seqence.txt  input for model
    return

def fine_tuning_for_strategies(): # FineTuning.py
    return

def load_model(model_path: str | Path, device: torch.device | None = None):
    import torch

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    return model

def load_dataloaders(
    test_file: str | Path,
    batch_size: int = 16,
    block_size: int = 30,
    dataset_type: str = "BlockDataset",
    target: int = 0,
    num_workers: int = 0,
):
    from data_loader import create_dataloaders # type: ignore

    test_file = Path(test_file)

    if not test_file.exists():
        raise FileNotFoundError(f"Test file does not exist: {test_file}")

    return create_dataloaders(
        transform=None,
        batch_size=batch_size,
        block_size=block_size,
        target=target,
        dataset_type=dataset_type,
        num_workers=num_workers,
        train_data_txt=str(test_file),
        test_data_txt=str(test_file),
    )


def evaluate_model(model, test_loader, device: torch.device | None = None):
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            pred_classes = torch.argmax(y_pred, dim=1)

            all_preds.append(pred_classes.cpu())
            all_trues.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_trues = torch.cat(all_trues)

    accuracy = (all_preds == all_trues).float().mean().item()

    return {
        "accuracy": accuracy,
        "predictions": all_preds,
        "truth": all_trues,
    }
    

def compare_startegy_accuracy(models, strategies):

    for strategy in strategies:
        model = models[strategy]
        print(f"Strategy: {strategy} Accuracy: {model['accuracy']}")


def main() -> None:
    print("main")
    args = parse_args()
    result = run_strategy(args)
    visualize_decisions(result)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
