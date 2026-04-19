from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SEQUENCE_FILE = Path("data/NewData/sequence/concatenated.txt")
DEFAULT_LAYERS_DIR = Path("data/NewData/layers")
DEFAULT_BLOCK_SIZE = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 8
DEFAULT_LEARNING_RATE = 0.001


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
    def __init__(self, base_model: nn.Module, adapter: InputAdapter):
        super().__init__()
        self.adapter = adapter
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.adapter(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune selected layers for the mobile PyTorch breathing model. "
            "The base model is frozen; only adapter and/or fc parameters are updated."
        )
    )
    parser.add_argument("--sequence-file", default=DEFAULT_SEQUENCE_FILE, type=Path)
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
    layers = [layer.strip() for layer in trained_layers.split(",") if layer.strip()]
    return layers or ["fc"]


def load_sequence_rows(path: Path) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []

    if not path.exists():
        return features, labels

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        values = [float(value) for value in line.split(",")]
        if len(values) < 2:
            continue

        features.append(values[:-1])
        labels.append(int(values[-1]))

    return features, labels


def build_training_tensors(
    features: list[list[float]],
    labels: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not features:
        raise ValueError("No sequence rows available for fine tuning.")

    blocks: list[list[list[float]]] = []
    block_labels: list[int] = []
    if len(features) < block_size:
        padding = [features[0] for _ in range(block_size - len(features))]
        blocks.append([*padding, *features])
        block_labels.append(labels[-1])
    else:
        for index in range(block_size, len(features) + 1):
            blocks.append(features[index - block_size:index])
            block_labels.append(labels[index - 1])

    x = torch.tensor(blocks, dtype=torch.float32)
    y = torch.tensor(block_labels, dtype=torch.long)
    return x, y


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
) -> list[nn.Parameter]:
    trainable_parameters: list[nn.Parameter] = []
    for parameter in model.adapter.parameters():
        parameter.requires_grad = False

    if "adapter" in trained_layers:
        for parameter in model.adapter.parameters():
            parameter.requires_grad = True
            trainable_parameters.append(parameter)

    if "fc" in trained_layers:
        fc = get_fc_layer(model.base_model)
        if fc is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `fc` layer to fine tune.")

        for parameter in fc.parameters():
            parameter.requires_grad = True
            trainable_parameters.append(parameter)

    if not trainable_parameters:
        raise ValueError("No trainable layers selected. Use adapter, fc, or adapter,fc.")

    return trainable_parameters


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
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, float | int | list[float]]:
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        losses.append(epoch_loss / len(dataset))

    accuracy = evaluate_accuracy(model, x.to(device), y.to(device))
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_history": [round(loss, 6) for loss in losses],
        "training_accuracy": accuracy,
    }


def evaluate_accuracy(model: FineTuningModel, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(x), dim=1)
        return float((predictions == y).float().mean().item())


def save_layer_artifacts(
    model: FineTuningModel,
    trained_layers: list[str],
    layers_dir: Path,
    run_id: str,
) -> dict[str, str]:
    layer_files: dict[str, str] = {}

    if "adapter" in trained_layers:
        adapter_path = layers_dir / f"{run_id}_adapter.pt"
        torch.save(model.adapter.state_dict(), adapter_path)
        layer_files["adapter"] = adapter_path.name

    if "fc" in trained_layers:
        fc = get_fc_layer(model.base_model)
        if fc is None:
            raise RuntimeError("The loaded PyTorch model does not expose an `fc` layer to export.")
        fc_path = layers_dir / f"{run_id}_fc.pt"
        torch.save(fc.state_dict(), fc_path)
        layer_files["fc"] = fc_path.name

    return layer_files


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.layers_dir.mkdir(parents=True, exist_ok=True)
    trained_layers = parse_layers(args.trained_layers)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = load_sequence_rows(args.sequence_file)
    x, y = build_training_tensors(features, labels, args.block_size)
    feature_count = x.shape[-1]
    model_path = resolve_model_path(args.model_name, args.model_file)
    base_model = load_base_model(model_path, device)
    freeze_base_model(base_model)

    model = FineTuningModel(base_model=base_model, adapter=InputAdapter(feature_count))
    trainable_parameters = configure_trainable_layers(model, trained_layers)
    training_metrics = train(
        model,
        trainable_parameters,
        x,
        y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    layer_files = save_layer_artifacts(model, trained_layers, args.layers_dir, run_id)

    manifest = {
        "format": "breathsense.fine_tuning_manifest.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_name": args.model_name,
        "runtime": args.runtime,
        "base_model_file": str(model_path),
        "trained_layers": trained_layers,
        "source_sequence_file": str(args.sequence_file),
        "sequence_rows": len(features),
        "training_samples": int(x.shape[0]),
        "block_size": args.block_size,
        "feature_count": int(feature_count),
        "device": str(device),
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
