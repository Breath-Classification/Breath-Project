import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from convert_tens_to_model_input import build_sequences
from data_loader import create_dataloaders # type: ignore
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTORCH_DIR = PROJECT_DIR / "models" / "Pytorch"
sys.path.append(str(PYTORCH_DIR))

DEFAULT_FILENAME = ""
DEFAULT_BASE_MODEL = ""
DEFAULT_OUTPUT = ""
STRATEGY = ""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo labelling methods"
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, type=Path)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--strategy", default=STRATEGY, type=str)
    
    return parser.parse_args()



def run_strategy(strategy): #decide_pseudo_labelling(strategy="XXx")
    _,decisions,relabeled = build_sequences()
    return decisions, relabeled


def visualize_decisions(decisions, relabeled):#keep =? drop =? relabeled =? 1->0 =? itd
    print(decisions)
    conf_matrix(relabeled)
    
def save_sequences(): #seqence.txt  input for model
    return

def fine_tuning_for_strategies(): # FineTuning.py
    return

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
    

def compare_startegy_accuracy(models, strategies):

    for strategy in strategies:
        model = models[strategy]
        print(f"Strategy: {strategy} Accuracy: {model["accuracy"]}")