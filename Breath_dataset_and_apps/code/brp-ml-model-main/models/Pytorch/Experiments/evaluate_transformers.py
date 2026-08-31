"""Evaluate the best Transformer variants on the TENS and ACC test sets.

The two ``*.txt`` files in ``models/Transformers`` describe hyperparameters,
not model weights.  Pass the matching ``.pth`` files with the command-line
options below.  A checkpoint is classified from its state-dict, so the report
also says whether it is a plain Transformer or Transformer + CNN + CRF.

Example (run from ``models/Pytorch``)::

    python Experiments/evaluate_transformers.py \
        --transformer-checkpoint models/saved_models/optuna/BlockDataset/FocalLossAdaptive/Transformer_best.pth \
        --tccr-checkpoint /path/to/Transformer_CNN_CRF_best.pth

Use ``--tens-data`` and ``--acc-data`` to evaluate different test files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

# The script is normally launched as ``python Experiments/evaluate_transformers.py``.
# In that form Python adds ``Experiments`` rather than ``Pytorch`` to sys.path.
PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

import torch
from torch.utils.data import DataLoader, Dataset

from models.Transformers.transformer import Transformer
from models.Transformers.transformer_CNN_CRF import Transformer_CNN_CRF
from scripts.error_tolerance import RR_error, acceptable_error


ROOT = PYTORCH_DIR.parents[1]  # brp-ml-model-main
MODEL_DIR = ROOT / "models" / "Pytorch" / "models" / "Transformers"
EPSILON = 2


@dataclass(frozen=True)
class BestTrial:
    name: str
    variant: str
    parameters: dict[str, float | int]


class BlockDataset(Dataset):
    """Sliding windows for the plain, one-label-per-window Transformer."""

    def __init__(self, file_path: Path, block_size: int) -> None:
        features, labels = read_rows(file_path)
        if len(features) < block_size:
            raise ValueError(f"{file_path}: fewer rows than block_size={block_size}")
        self.features = torch.stack(
            [features[index : index + block_size] for index in range(len(features) - block_size + 1)]
        )
        self.labels = labels[block_size - 1 :]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class SequenceBlockDataset(Dataset):
    """Non-overlapping sequences used by Transformer + CNN + CRF."""

    def __init__(self, file_path: Path, block_size: int) -> None:
        features, labels = read_rows(file_path)
        count = len(features) // block_size
        if count == 0:
            raise ValueError(f"{file_path}: fewer rows than block_size={block_size}")
        usable = count * block_size
        features, labels = features[:usable], labels[:usable]
        # This derivative is part of the project's original SequenceBlockDataset.
        derivative = torch.zeros_like(features)
        derivative[1:] = features[1:] - features[:-1]
        features = torch.cat((features, derivative), dim=1)
        self.features = features.reshape(count, block_size, -1)
        self.labels = labels.reshape(count, block_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", type=Path, help="Weights of the plain Transformer (.pth).")
    parser.add_argument("--tccr-checkpoint", type=Path, help="Weights of Transformer + CNN + CRF (.pth).")
    parser.add_argument("--tens-data", type=Path, default=ROOT / "data/pretrained/tens_sequence/tens_test.txt")
    parser.add_argument("--acc-data", type=Path, default=ROOT / "data/pretrained/acc_sequence/acc_test.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_best_trial(file_path: Path, variant: str) -> BestTrial:
    text = file_path.read_text(encoding="utf-8")
    match = re.search(r"Najlepsze hiperparametry:\s*(\{.*\})", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Cannot read hyperparameters from {file_path}")
    parameters = ast.literal_eval(match.group(1))
    return BestTrial(file_path.stem, variant, parameters)


def apply_checkpoint_config(trial: BestTrial, checkpoint: Path) -> BestTrial:
    """Prefer settings stored with the weights over a possibly newer TXT file."""
    config_path = checkpoint.with_suffix(".json")
    if not config_path.is_file():
        return trial
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = dict(trial.parameters)
    # These fields are recorded by save_model() and alter data/model shape.
    for key in ("block_size", "hidden_units", "dropout", "num_layers", "dim_feedforward", "nhead"):
        if key in config:
            parameters[key] = config[key]
    print(f"{trial.name}: using checkpoint configuration from {config_path.name}")
    return replace(trial, parameters=parameters)


def read_rows(file_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append([float(value) for value in line.split(",")])
            except ValueError as error:
                raise ValueError(f"{file_path}:{line_number}: invalid numeric row") from error
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"{file_path}: expected features followed by a label")
    data = torch.tensor(rows, dtype=torch.float32)
    return data[:, :-1], data[:, -1].long()


def infer_variant(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("conv.") for key in state_dict) or any(key.startswith("crf.") for key in state_dict):
        return "Transformer + CNN + CRF"
    return "Transformer"


def create_model(trial: BestTrial, input_size: int, output_size: int) -> torch.nn.Module:
    params = trial.parameters
    d_model = int(params["head_dim"]) * int(params["nhead"])
    common = dict(
        input_shape=input_size,
        hidden_units=int(params["hidden_units"]),
        output_shape=output_size,
        d_model=d_model,
        dropout=float(params["dropout"]),
        num_layrer=int(params["num_layers"]),
        dim_feedforward=int(params["dim_feedforward"]),
        nhead=int(params["nhead"]),
    )
    if trial.variant == "tccr":
        return Transformer_CNN_CRF(**common)
    return Transformer(**common)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> str:
    state_dict = read_state_dict(path, device)
    detected = infer_variant(state_dict)
    expected = "Transformer + CNN + CRF" if hasattr(model, "crf") else "Transformer"
    if detected != expected:
        raise ValueError(f"{path} is {detected}, but its TXT file describes {expected}")
    model.load_state_dict(state_dict)
    return detected


def read_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location=device, weights_only=True)
    state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} does not contain a PyTorch state_dict")
    return state_dict


def checkpoint_input_size(state_dict: dict[str, torch.Tensor], is_tccr: bool) -> int:
    key = "conv.0.weight" if is_tccr else "embed.weight"
    if key not in state_dict:
        raise ValueError(f"Checkpoint has no {key}; cannot determine its input shape")
    return int(state_dict[key].shape[1])


def predict(model: torch.nn.Module, dataloader: DataLoader, is_crf: bool, device: torch.device) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.inference_mode():
        for features, labels in dataloader:
            features = features.to(device)
            output = model(features)
            if is_crf:
                batch_predictions = model.crf.decode(output)
                predictions.extend(value for sequence in batch_predictions for value in sequence)
            else:
                predictions.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(labels.reshape(-1).tolist())
    return predictions, targets


def metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    if not targets:
        raise ValueError("Dataset has no samples to evaluate")
    accuracy = sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)
    epsilon_accuracy = sum(
        prediction == target or acceptable_error(predictions, targets, index, EPSILON)
        for index, (prediction, target) in enumerate(zip(predictions, targets))
    ) / len(targets)
    true_transitions = sum(left != right for left, right in zip(targets, targets[1:]))
    predicted_transitions = sum(left != right for left, right in zip(predictions, predictions[1:]))
    transition_accuracy = true_transitions / predicted_transitions if predicted_transitions else 0.0
    return {
        "accuracy": accuracy,
        "epsilon_accuracy": epsilon_accuracy,
        "transition_accuracy": transition_accuracy,
        "cycle_accuracy": calculate_cycle_accuracy(predictions, targets),
    }


def calculate_cycle_accuracy(predictions: list[int], targets: list[int]) -> float:
    correct_cycles = total_cycles = 0
    index = 0
    while index < len(targets):
        if targets[index] != 2:
            index += 1
            continue
        cycle_targets: list[int] = []
        cycle_predictions: list[int] = []
        accurate = True
        left_inhale = False
        for end in range(index + 1, len(targets)):
            cycle_targets.append(targets[end])
            cycle_predictions.append(predictions[end])
            if targets[end] != predictions[end] and not acceptable_error(predictions, targets, end, EPSILON):
                accurate = False
            left_inhale |= targets[end] != 2
            if targets[end] == 2 and left_inhale:
                total_cycles += 1
                if accurate or RR_error(cycle_predictions, cycle_targets, 1):
                    correct_cycles += 1
                index = end + 1
                break
        else:
            index = len(targets)
    return correct_cycles / total_cycles if total_cycles else 0.0


def report(model_name: str, variant: str, dataset_name: str, values: dict[str, float]) -> None:
    print(f"{model_name} | {variant} | {dataset_name}")
    print("  " + "  ".join(f"{name}={value:.2%}" for name, value in values.items()))


def evaluate(trial: BestTrial, checkpoint: Path, datasets: Iterable[tuple[str, Path]], batch_size: int, device: torch.device) -> None:
    state_dict = read_state_dict(checkpoint, device)
    expected_input_size = checkpoint_input_size(state_dict, trial.variant == "tccr")
    for dataset_name, dataset_path in datasets:
        raw_features, _ = read_rows(dataset_path)
        dataset: Dataset
        if trial.variant == "tccr":
            dataset = SequenceBlockDataset(dataset_path, int(trial.parameters["block_size"]))
        else:
            dataset = BlockDataset(dataset_path, int(trial.parameters["block_size"]))
        # All project Transformer checkpoints are trained for classes 0..3.
        # Do not infer this from a test split: a split may omit one class.
        output_size = 4
        input_size = len(raw_features[0]) * (2 if trial.variant == "tccr" else 1)
        if input_size != expected_input_size:
            print(
                f"{trial.name} | {dataset_name}: skipped "
                f"(checkpoint expects {expected_input_size} input features, dataset supplies {input_size}; "
                "use weights trained for this sensor)"
            )
            continue
        model = create_model(trial, input_size, output_size).to(device)
        detected_variant = load_checkpoint(checkpoint, model, device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions, targets = predict(model, dataloader, trial.variant == "tccr", device)
        report(trial.name, detected_variant, dataset_name, metrics(predictions, targets))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    trials = (
        (read_best_trial(MODEL_DIR / "Transformer.txt", "transformer"), args.transformer_checkpoint),
        (read_best_trial(MODEL_DIR / "TCCR.txt", "tccr"), args.tccr_checkpoint),
    )
    datasets = (("TENS", args.tens_data), ("ACC", args.acc_data))
    for trial, checkpoint in trials:
        if checkpoint is None:
            print(f"{trial.name}: skipped (pass its checkpoint with --{'tccr' if trial.variant == 'tccr' else 'transformer'}-checkpoint)")
            continue
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluate(apply_checkpoint_config(trial, checkpoint), checkpoint, datasets, args.batch_size, device)


if __name__ == "__main__":
    main()
