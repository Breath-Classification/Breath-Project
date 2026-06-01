import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    
    _,test_loader = load_dataloaders(args.labelled_file)
    
    base_model = load_model(args.base_model)
    fine_model = load_model(args.fine_tuned_model)
    
    base_metrics = evaluate_model(base_model,test_loader)
    fine_metrics = evaluate_model(fine_model, test_loader)
    
    print("BASE:", base_metrics["accuracy"])
    print("FINE:", fine_metrics["accuracy"])


if __name__ == "__main__":
    main()