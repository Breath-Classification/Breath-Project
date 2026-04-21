from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from data_loader import create_dataloaders


DEFAULT_TRAIN_DATA_FILE = Path("../../data/pretrained/tens_sequence/tens_concatenated.txt")
DEFAULT_TEST_DATA_FILE = Path("../../data/pretrained/tens_sequence/tens_test.txt")
DEFAULT_LAYERS_DIR = Path("data/NewData/layers")
DEFAULT_BLOCK_SIZE = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 8
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DATASET_TYPE = "BlockDataset"
DEFAULT_TARGET = 0
DEFAULT_NUM_WORKERS = 4


class InputAdapter(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.projection = nn.Linear(feature_count, feature_count)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(feature_count))
            self.projection.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FineTuningModel(nn.Module):
    def __init__(self, base_model: nn.Module, input_adapter: nn.Module | None = None):
        super().__init__()
        self.input_adapter = input_adapter
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        return self.base_model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune selected layers for the mobile PyTorch breathing model. "
            "The base model is frozen; only adapter and/or fc parameters are updated."
        )
    )
    parser.add_argument("--sequence-file", default=None, type=Path, help="Optional fallback: use one sequence file for both train and test.")
    parser.add_argument("--train-data-txt", default=DEFAULT_TRAIN_DATA_FILE, type=Path)
    parser.add_argument("--test-data-txt", default=DEFAULT_TEST_DATA_FILE, type=Path)
    parser.add_argument("--dataset-type", default=DEFAULT_DATASET_TYPE)
    parser.add_argument("--target", default=DEFAULT_TARGET, type=int)
    parser.add_argument("--num-workers", default=DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument("--layers-dir", default=DEFAULT_LAYERS_DIR, type=Path)
    parser.add_argument("--model-name", default="LSTMBASE_tens")
    parser.add_argument("--runtime", default="pytorch")
    parser.add_argument("--trained-layers", default="fc", help="Comma-separated layer list, e.g. fc or adapter,fc.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-file", default=None, type=Path)
    parser.add_argument("--block-size", default=DEFAULT_BLOCK_SIZE, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--learning-rate", default=DEFAULT_LEARNING_RATE, type=float)
    return parser.parse_args()


def parse_layers(trained_layers: str) -> list[str]:
    layers: list[str] = []
    for layer in [layer.strip().lower() for layer in trained_layers.split(",") if layer.strip()]:
        if layer in {"adapter_fc", "adapter+fc", "adapter-fc"}:
            layers.extend(["adapter", "fc"])
        elif layer in {"adapter", "fc"}:
            layers.append(layer)
        elif layer in {"none", "base"}:
            continue
        else:
            raise ValueError(f"Unsupported trainable layer: {layer}")

    return list(dict.fromkeys(layers)) or ["fc"]


def resolve_data_files(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.sequence_file is not None:
        return args.sequence_file, args.sequence_file

    return args.train_data_txt, args.test_data_txt


def load_dataloaders(
    train_data_txt: Path,
    test_data_txt: Path,
    block_size: int,
    batch_size: int,
    dataset_type: str,
    target: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return create_dataloaders(
        transform=None,
        batch_size=batch_size,
        block_size=block_size,
        target=target,
        dataset_type=dataset_type,
        num_workers=num_workers,
        train_data_txt=str(train_data_txt),
        test_data_txt=str(test_data_txt),
    )


def resolve_model_path(model_name: str, explicit_model_file: Path | None) -> Path:
    if explicit_model_file is not None:
        return explicit_model_file

    assets_dir = os.getenv("PYTORCH_MODEL_ASSETS_DIR")
    candidates = []
    if assets_dir:
        candidates.append(Path(assets_dir) / f"{model_name}.pt")

    candidates.extend(
        [
            Path("models/Pytorch/pretrained") / f"{model_name}.pt",
            Path("models/mobile_models") / f"{model_name}.pt",
            Path("../brp-app-main/android/app/src/main/assets") / f"{model_name}.pt",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a TorchScript model file. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_base_model(model_path: Path, device: torch.device) -> nn.Module:
    model = torch.jit.load(str(model_path), map_location=device)
    model.train()
    return model


def freeze_base_model(base_model: nn.Module) -> None:
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def configure_trainable_layers(
    model: FineTuningModel,
    trained_layers: list[str],
) -> tuple[list[nn.Parameter], dict[str, str]]:
    trainable_parameters: list[nn.Parameter] = []
    layer_sources: dict[str, str] = {}

    if model.input_adapter is not None:
        for parameter in model.input_adapter.parameters():
            parameter.requires_grad = False

    if "adapter" in trained_layers:
        adapter, adapter_source = get_adapter_layer_for_training(model)
        if adapter is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `adapter` layer to fine tune.")

        for parameter in adapter.parameters():
            parameter.requires_grad = True
            trainable_parameters.append(parameter)
        layer_sources["adapter"] = adapter_source

    if "fc" in trained_layers:
        fc = get_fc_layer(model.base_model)
        if fc is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `fc` layer to fine tune.")

        for parameter in fc.parameters():
            parameter.requires_grad = True
            trainable_parameters.append(parameter)
        layer_sources["fc"] = "model"

    if not trainable_parameters:
        raise ValueError("No trainable layers selected. Use adapter, fc, or adapter,fc.")

    return trainable_parameters, layer_sources


def get_adapter_layer(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "adapter"):
        layer = getattr(model, "adapter")
        if isinstance(layer, nn.Module):
            return layer

    for name, module in model.named_modules():
        if name == "adapter" or name.endswith(".adapter"):
            return module

    return None


def get_adapter_layer_for_training(model: FineTuningModel) -> tuple[nn.Module | None, str]:
    adapter = get_adapter_layer(model.base_model)
    if adapter is not None:
        return adapter, "model"

    if model.input_adapter is not None:
        return model.input_adapter, "input_wrapper"

    return None, "missing"


def get_fc_layer(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "fc"):
        layer = getattr(model, "fc")
        if isinstance(layer, nn.Module):
            return layer

    for name, module in model.named_modules():
        if name == "fc" or name.endswith(".fc"):
            return module

    return None


def train(
    model: FineTuningModel,
    trainable_parameters: list[nn.Parameter],
    train_dataloader,
    test_dataloader,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, float | int | list[float]]:
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        losses.append(epoch_loss / len(train_dataloader.dataset))

    accuracy = evaluate_accuracy(model, test_dataloader, device)
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_history": [round(loss, 6) for loss in losses],
        "test_accuracy": accuracy,
    }


def evaluate_accuracy(model: FineTuningModel, dataloader, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = torch.argmax(model(batch_x), dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.numel()

        if total_samples == 0:
            return 0.0

        return float(total_correct / total_samples)


def save_layer_artifacts(
    model: FineTuningModel,
    trained_layers: list[str],
    layers_dir: Path,
    run_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    layer_files: dict[str, str] = {}
    layer_sources: dict[str, str] = {}

    if "adapter" in trained_layers:
        adapter, adapter_source = get_adapter_layer_for_training(model)
        if adapter is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `adapter` layer to export.")

        adapter_path = layers_dir / f"{run_id}_adapter.pt"
        torch.save(adapter.state_dict(), adapter_path)
        layer_files["adapter"] = adapter_path.name
        layer_sources["adapter"] = adapter_source

    if "fc" in trained_layers:
        fc = get_fc_layer(model.base_model)
        if fc is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `fc` layer to export.")
        fc_path = layers_dir / f"{run_id}_fc.pt"
        torch.save(fc.state_dict(), fc_path)
        layer_files["fc"] = fc_path.name
        layer_sources["fc"] = "model"

    return layer_files, layer_sources


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.layers_dir.mkdir(parents=True, exist_ok=True)
    trained_layers = parse_layers(args.trained_layers)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_txt, test_data_txt = resolve_data_files(args)
    train_dataloader, test_dataloader = load_dataloaders(
        train_data_txt=train_data_txt,
        test_data_txt=test_data_txt,
        block_size=args.block_size,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        target=args.target,
        num_workers=args.num_workers,
    )
    first_batch_x, _ = next(iter(train_dataloader))
    feature_count = first_batch_x.shape[-1]
    model_path = resolve_model_path(args.model_name, args.model_file)
    base_model = load_base_model(model_path, device)
    freeze_base_model(base_model)

    has_model_adapter = get_adapter_layer(base_model) is not None
    input_adapter = None if has_model_adapter or "adapter" not in trained_layers else InputAdapter(feature_count)
    model = FineTuningModel(base_model=base_model, input_adapter=input_adapter)
    trainable_parameters, trainable_layer_sources = configure_trainable_layers(model, trained_layers)
    training_metrics = train(
        model,
        trainable_parameters,
        train_dataloader,
        test_dataloader,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    layer_files, artifact_layer_sources = save_layer_artifacts(model, trained_layers, args.layers_dir, run_id)

    manifest = {
        "format": "breathsense.fine_tuning_manifest.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_name": args.model_name,
        "runtime": args.runtime,
        "base_model_file": str(model_path),
        "trained_layers": trained_layers,
        "source_sequence_file": str(args.sequence_file),
        "train_data_txt": str(train_data_txt),
        "test_data_txt": str(test_data_txt),
        "training_samples": len(train_dataloader.dataset),
        "block_size": args.block_size,
        "feature_count": int(feature_count),
        "device": str(device),
        "trainable_layer_sources": trainable_layer_sources,
        "artifact_layer_sources": artifact_layer_sources,
        "training": training_metrics,
        "layer_files": layer_files,
        "ready_for_download": bool(layer_files),
    }
    manifest_path = args.layers_dir / f"{run_id}_manifest.json"
    latest_manifest_path = args.layers_dir / "latest_manifest.json"
    write_json(manifest_path, manifest)
    write_json(latest_manifest_path, manifest)
    print(json.dumps({**manifest, "manifest_file": manifest_path.name}))


if __name__ == "__main__":
    main()
