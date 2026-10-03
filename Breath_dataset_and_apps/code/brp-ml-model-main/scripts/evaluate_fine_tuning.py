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

BASE_MODEL_PATH = ""
FINE_TUNED_MODEL_PATH =""
LABELLED_FILE_PATH = ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Evaluate base and fine tuned model"
    )
    
    parser.add_argument("--base_model", default=BASE_MODEL_PATH, type=Path)
    parser.add_argument("--fine_tuned_model", default=FINE_TUNED_MODEL_PATH, type=Path)
    parser.add_argument("--labelled_file", default=LABELLED_FILE_PATH, type=Path)
    
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
    base_std = np.std(base_accuracies, ddof=1)
    fine_std = np.std(fine_accuracies, ddof=1)

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
    
    _, test_loader = load_dataloaders(args.labelled_file)

    base_model_paths = [
        "path/to/baseline_1.pt",
        "path/to/baseline_2.pt",
        "path/to/baseline_3.pt",
        "path/to/baseline_4.pt",
        "path/to/baseline_5.pt",
    ]

    fine_model_paths = [
        "path/to/fine_1.pt",
        "path/to/fine_2.pt",
        "path/to/fine_3.pt",
        "path/to/fine_4.pt",
        "path/to/fine_5.pt",
    ]

    base_accuracies = []
    fine_accuracies = []

    for base_path, fine_path in zip(
        base_model_paths,
        fine_model_paths
    ):
        base_model = load_model(base_path)
        fine_model = load_model(fine_path)

        base_metrics = evaluate_model(
            base_model,
            test_loader
        )

        fine_metrics = evaluate_model(
            fine_model,
            test_loader
        )

        base_acc = base_metrics["accuracy"]
        fine_acc = fine_metrics["accuracy"]

        base_accuracies.append(base_acc)
        fine_accuracies.append(fine_acc)

        print(
            f"Base: {base_acc * 100:.2f}% | "
            f"Fine: {fine_acc * 100:.2f}% | "
            f"Improvement: {(fine_acc - base_acc) * 100:+.2f} pp"
        )

    # Mean
    base_mean = np.mean(base_accuracies)
    fine_mean = np.mean(fine_accuracies)

    # Standard deviation
    base_std = np.std(base_accuracies, ddof=1)
    fine_std = np.std(fine_accuracies, ddof=1)

    # Mean improvement
    improvements = [
        (fine - base) * 100
        for base, fine in zip(
            base_accuracies,
            fine_accuracies
        )
    ]

    mean_improvement = np.mean(improvements)

    print("\n===== FINAL RESULTS =====")

    print(
        f"Baseline: "
        f"{base_mean * 100:.2f} ± "
        f"{base_std * 100:.2f}%"
    )

    print(
        f"Fine-tuned: "
        f"{fine_mean * 100:.2f} ± "
        f"{fine_std * 100:.2f}%"
    )

    print(
        f"Mean Improvement: "
        f"{mean_improvement:+.2f} pp"
    )


if __name__ == "__main__":
    main()