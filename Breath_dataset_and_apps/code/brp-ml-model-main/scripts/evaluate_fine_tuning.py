import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTORCH_DIR = PROJECT_DIR / "models" / "Pytorch"
sys.path.append(str(PYTORCH_DIR))

from data_loader import create_dataloaders # type: ignore
import torch

DATA_DIR = PROJECT_DIR / "data"
BASE_MODEL_PATH = DATA_DIR / "Models" / "mobile_models" / "LSTMBASE_tens copy.pt"
FINE_TUNED_MODEL_PATH = DATA_DIR / "NewData" / "Models"
LABELLED_FILE_PATH = DATA_DIR / "NewData" / "sequence"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Evaluate base and fine tuned model"
    )
    
    parser.add_argument("--base_model", default=BASE_MODEL_PATH, type=Path)
    parser.add_argument(
        "--fine_tuned_model",
        default=FINE_TUNED_MODEL_PATH,
        type=Path,
        help="Directory containing fine-tuned .pt models.",
    )
    parser.add_argument(
        "--labelled_file",
        default=LABELLED_FILE_PATH,
        type=Path,
        help="Directory containing *_pretrained_sequence.txt test files.",
    )
    
    return parser.parse_args()

def load_model(model_path: str | Path, device: torch.device | None = None):
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

def calculate_statistics(base_accuracies, fine_accuracies):
    base_accuracies = np.array(base_accuracies)
    fine_accuracies = np.array(fine_accuracies)

    # Mean accuracy
    base_mean = np.mean(base_accuracies)
    fine_mean = np.mean(fine_accuracies)

    # Standard deviation
    base_std = np.std(base_accuracies, ddof=1) if len(base_accuracies) > 1 else 0.0
    fine_std = np.std(fine_accuracies, ddof=1) if len(fine_accuracies) > 1 else 0.0

    # Improvement for each run, in percentage points
    improvements = (fine_accuracies - base_accuracies) * 100

    # Mean improvement
    mean_improvement = np.mean(improvements)

    return {
        "base_mean": base_mean * 100,
        "base_std": base_std * 100,
        "fine_mean": fine_mean * 100,
        "fine_std": fine_std * 100,
        "mean_improvement": mean_improvement,
    }


def find_test_file(model_path: Path, test_dir: Path) -> Path:
    """Return the NewData sequence file belonging to a fine-tuned model."""
    model_name = model_path.stem.lower()
    matches = sorted(
        test_file
        for test_file in test_dir.glob("*_pretrained_sequence.txt")
        if test_file.stem.lower().startswith(model_name)
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one test file for {model_path.name} in {test_dir}, found {len(matches)}."
        )

    return matches[0]

def evaluate_model(model, test_loader, device: torch.device | None = None):
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
    
def main():
    args = parse_args()
    fine_model_paths = sorted(args.fine_tuned_model.glob("*.pt"))

    if not fine_model_paths:
        raise FileNotFoundError(f"No fine-tuned .pt models found in {args.fine_tuned_model}")

    # Saved LSTM TorchScript models create their hidden state on CPU internally.
    # Evaluating on CUDA would therefore mix CUDA inputs with CPU hidden tensors.
    device = torch.device("cpu")
    base_model = load_model(args.base_model, device)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned models: {args.fine_tuned_model}\n")

    base_accuracies = []
    fine_accuracies = []

    for fine_path in fine_model_paths:
        try:
            test_file = find_test_file(fine_path, args.labelled_file)
            _, test_loader = load_dataloaders(test_file)
            fine_model = load_model(fine_path, device)

            base_metrics = evaluate_model(base_model, test_loader, device)
            fine_metrics = evaluate_model(fine_model, test_loader, device)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"{fine_path.name}: skipped ({error})")
            continue

        base_acc = base_metrics["accuracy"]
        fine_acc = fine_metrics["accuracy"]

        base_accuracies.append(base_acc)
        fine_accuracies.append(fine_acc)

        print(
            f"{fine_path.name} on {test_file.name}: "
            f"Base: {base_acc * 100:.2f}% | "
            f"Fine: {fine_acc * 100:.2f}% | "
            f"Improvement: {(fine_acc - base_acc) * 100:+.2f} pp"
        )

    if not base_accuracies:
        raise RuntimeError("No model could be evaluated.")

    statistics = calculate_statistics(base_accuracies, fine_accuracies)

    print("\n===== FINAL RESULTS =====")

    print(
        f"Baseline: "
        f"{statistics['base_mean']:.2f} ± "
        f"{statistics['base_std']:.2f}%"
    )

    print(
        f"Fine-tuned: "
        f"{statistics['fine_mean']:.2f} ± "
        f"{statistics['fine_std']:.2f}%"
    )

    print(
        f"Mean Improvement: "
        f"{statistics['mean_improvement']:+.2f} pp"
    )


if __name__ == "__main__":
    main()
