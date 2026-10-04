from __future__ import annotations

import argparse
import json
import subprocess
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
    write_sequences,
)

DEFAULT_FILENAME = ""
DEFAULT_BASE_MODEL = ""
DEFAULT_OUTPUT = None
STRATEGY = "none"
DEFAULT_STRATEGIES = "none,majority,isolated,slope,physical,cut"
NEW_DATA_DIR = DEFAULT_INPUT_DIR
DEFAULT_MODELS_DIR = NEW_DATA_DIR / "Models"
DEFAULT_TEST_DATA_DIR = NEW_DATA_DIR / "Test"
DEFAULT_EVALUATION_DATA_DIR = NEW_DATA_DIR / "sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo labelling methods"
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, type=Path)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--strategy", default=STRATEGY, type=str)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES, type=str)
    parser.add_argument("--window-size", default=DEFAULT_WINDOW_SIZE, type=int)
    parser.add_argument("--moving-average-window", default=DEFAULT_MOVING_AVERAGE, type=int)
    parser.add_argument("--normalization-range", default=DEFAULT_NORMALIZATION_RANGE, type=int)
    parser.add_argument("--confidence-threshold", default=0.8, type=float)
    parser.add_argument("--pseudo-labelling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-sequences", action="store_true")
    parser.add_argument("--sequence-output-dir", default=DEFAULT_INPUT_DIR / "strategy_sequences", type=Path)
    parser.add_argument("--run-fine-tuning", action="store_true")
    parser.add_argument("--fine-tuning-script", default=PYTORCH_DIR / "FineTuning.py", type=Path)
    parser.add_argument("--layers-dir", default=DEFAULT_INPUT_DIR / "layers" / "strategies", type=Path)
    parser.add_argument("--model-file", default=None, type=Path)
    parser.add_argument("--model-name", default="LSTMBASE_tens", type=str)
    parser.add_argument("--trained-layers", default="fc", type=str)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--block-size", default=30, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--test-data-txt", default=None, type=Path)
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run every selected strategy for every .pt model in --models-dir.",
    )
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, type=Path)
    parser.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA_DIR, type=Path)
    parser.add_argument("--evaluation-data-dir", default=DEFAULT_EVALUATION_DATA_DIR, type=Path)
    
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


def parse_strategies(args: argparse.Namespace) -> list[str]:
    selected = args.strategies if args.strategies else args.strategy
    return [strategy.strip() for strategy in selected.split(",") if strategy.strip()]


def run_strategy(
    args: argparse.Namespace,
    strategy: str,
    filename: Path,
    rows: list[list[str]],
    input_format: str,
    values: list[float],
    labels: list[int],
    times: list[float],
    confidences: list[float] | None,
): #decide_pseudo_labelling(strategy="XXx")
    sequences, decisions, relabeled = build_sequences(
        values,
        labels,
        times,
        confidences,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        pseudo_labelling=args.pseudo_labelling,
        strategy=strategy,
    )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(filename),
        "input_format": input_format,
        "strategy": strategy,
        "input_rows": len(rows),
        "pretrained_rows": len(values),
        "sequence_rows": len(sequences),
        "decisions": decisions,
        "relabeled": relabeled,
    }
    return result, sequences


def run_strategies(args: argparse.Namespace) -> list[tuple[dict, list[list[float]]]]:
    filename = resolve_filename(args.filename)
    rows, input_format, values, labels, times, confidences = prepare_pretrained_data(
        filename,
        args.moving_average_window,
        args.normalization_range,
        args.pseudo_labelling,
    )
    results = []
    for strategy in parse_strategies(args):
        result, sequences = run_strategy(
            args,
            strategy,
            filename,
            rows,
            input_format,
            values,
            labels,
            times,
            confidences,
        )
        results.append((result, sequences))
    return results


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
    
def save_sequences(path: Path, sequences: list[list[float]]) -> Path: #seqence.txt  input for model
    write_sequences(path, sequences)
    return path

def fine_tuning_for_strategies(args: argparse.Namespace, result: dict, sequence_file: Path) -> dict: # FineTuning.py
    return run_fine_tuning_for_strategy(args, result, sequence_file)


def save_strategy_sequences(
    result: dict,
    sequences: list[list[float]],
    output_dir: Path,
) -> Path:
    source_stem = Path(result["source_file"]).stem
    strategy = result["strategy"]
    sequence_path = output_dir / f"{source_stem}_{strategy}_sequence.txt"
    save_sequences(sequence_path, sequences)
    result["sequence_file"] = str(sequence_path)
    return sequence_path


def run_fine_tuning_for_strategy(args: argparse.Namespace, result: dict, sequence_file: Path) -> dict:
    if result["strategy"] == "none":
        return evaluate_base_model_for_strategy(args, result, sequence_file)

    run_id = f"{Path(result['source_file']).stem}_{result['strategy']}"
    command = [
        sys.executable,
        str(args.fine_tuning_script),
        "--train-data-txt",
        str(sequence_file),
        "--test-data-txt",
        str(args.test_data_txt or sequence_file),
        "--layers-dir",
        str(args.layers_dir),
        "--run-id",
        run_id,
        "--model-name",
        args.model_name,
        "--trained-layers",
        args.trained_layers,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--block-size",
        str(args.block_size),
        "--device",
        args.device,
    ]
    if args.model_file is not None:
        command.extend(["--model-file", str(args.model_file)])

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print("FineTuning.py failed.")
        if completed.stdout.strip():
            print("STDOUT:")
            print(completed.stdout)
        if completed.stderr.strip():
            print("STDERR:")
            print(completed.stderr)
        raise RuntimeError(f"Fine tuning failed for strategy: {result['strategy']}")

    output = json.loads(completed.stdout.strip().splitlines()[-1])
    result["fine_tuning"] = {
        "run_id": run_id,
        "command": command,
        "manifest_file": output.get("manifest_file"),
        "test_accuracy": output.get("training", {}).get("test_accuracy"),
        "final_loss": output.get("training", {}).get("final_loss"),
    }
    return result["fine_tuning"]


def evaluate_base_model_for_strategy(args: argparse.Namespace, result: dict, sequence_file: Path) -> dict:
    import torch

    eval_file = args.test_data_txt or sequence_file
    if args.model_file is None:
        raise ValueError("Pass --model-file to evaluate unchanged base model for strategy none.")

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    _, test_loader = load_dataloaders(
        eval_file,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    base_model = load_model(args.model_file, device=device)
    metrics = evaluate_model(base_model, test_loader, device=device)

    result["fine_tuning"] = {
        "run_id": f"{Path(result['source_file']).stem}_{result['strategy']}",
        "trained": False,
        "model_file": str(args.model_file),
        "test_data_txt": str(eval_file),
        "test_accuracy": metrics["accuracy"],
        "final_loss": None,
    }
    return result["fine_tuning"]

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


def evaluate_strategies(args: argparse.Namespace) -> list[dict]:
    strategy_results = run_strategies(args)
    output_results = []

    if args.save_sequences or args.run_fine_tuning:
        args.sequence_output_dir.mkdir(parents=True, exist_ok=True)

    for result, sequences in strategy_results:
        visualize_decisions(result)
        if args.save_sequences or args.run_fine_tuning:
            sequence_file = save_strategy_sequences(result, sequences, args.sequence_output_dir)
            print(f"Sequence file: {sequence_file}")
            if args.run_fine_tuning:
                try:
                    fine_tuning = run_fine_tuning_for_strategy(args, result, sequence_file)
                    print(f"Fine tuning accuracy: {fine_tuning['test_accuracy']}")
                except RuntimeError as error:
                    result["fine_tuning"] = {"error": str(error)}
                    print(error)
        output_results.append(result)

    return output_results


def find_evaluation_file(model_path: Path, evaluation_dir: Path) -> Path:
    suffix = "_pretrained_sequence.txt"
    model_name = model_path.stem.lower()
    files = sorted(evaluation_dir.glob(f"*{suffix}"))
    exact_matches = [file for file in files if file.name.lower() == f"{model_name}{suffix}"]
    matches = exact_matches or [file for file in files if file.stem.lower().startswith(model_name)]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one evaluation sequence for {model_path.name} in {evaluation_dir}, found {len(matches)}."
        )

    return matches[0]


def find_source_file(evaluation_file: Path, test_data_dir: Path) -> Path:
    source_stem = evaluation_file.name.removesuffix("_pretrained_sequence.txt").lower()
    matches = sorted(
        file for file in test_data_dir.glob("*.txt") if file.stem.lower() == source_stem
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source file for {evaluation_file.name} in {test_data_dir}, found {len(matches)}."
        )

    return matches[0]


def print_all_models_summary(model_results: list[dict], strategies: list[str]) -> None:
    print("\n===== ALL MODELS / ALL STRATEGIES =====")
    print("Model".ljust(12) + " ".join(strategy.rjust(11) for strategy in strategies))

    for model_result in model_results:
        if "error" in model_result:
            print(f"{model_result['model_name']:<12} ERROR: {model_result['error']}")
            continue

        accuracies = {
            result["strategy"]: result.get("fine_tuning", {}).get("test_accuracy")
            for result in model_result["results"]
        }
        values = [
            f"{accuracies[strategy] * 100:10.2f}%" if accuracies.get(strategy) is not None else "      ERROR"
            for strategy in strategies
        ]
        print(f"{model_result['model_name']:<12}" + " ".join(values))


def evaluate_all_models(args: argparse.Namespace) -> list[dict]:
    model_paths = sorted(args.models_dir.glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No .pt models found in {args.models_dir}")

    all_results = []
    for model_path in model_paths:
        print(f"\n===== {model_path.name} =====")
        try:
            evaluation_file = find_evaluation_file(model_path, args.evaluation_data_dir)
            source_file = find_source_file(evaluation_file, args.test_data_dir)
            model_args = argparse.Namespace(**vars(args))
            model_args.filename = source_file
            model_args.test_data_txt = evaluation_file
            model_args.model_file = model_path
            model_args.save_sequences = True
            model_args.run_fine_tuning = True
            model_args.sequence_output_dir = args.sequence_output_dir / "all_models" / model_path.stem
            model_args.layers_dir = args.layers_dir / "all_models" / model_path.stem

            results = evaluate_strategies(model_args)
            all_results.append(
                {
                    "model_name": model_path.stem,
                    "model_file": str(model_path),
                    "source_file": str(source_file),
                    "evaluation_file": str(evaluation_file),
                    "results": results,
                }
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"{model_path.name}: skipped ({error})")
            all_results.append({"model_name": model_path.stem, "model_file": str(model_path), "error": str(error)})

    return all_results


def write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.all_models:
        model_results = evaluate_all_models(args)
        output_file = args.output_file or args.sequence_output_dir / "all_models" / "fine_tuning_summary.json"
        write_output(
            output_file,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "strategies": parse_strategies(args),
                "models": model_results,
            },
        )
        print_all_models_summary(model_results, parse_strategies(args))
        print(f"\nFull results saved to: {output_file}")
        return

    output_results = evaluate_strategies(args)
    if args.output_file is not None:
        write_output(
            args.output_file,
            {"created_at": datetime.now(timezone.utc).isoformat(), "results": output_results},
        )


if __name__ == "__main__":
    main()
