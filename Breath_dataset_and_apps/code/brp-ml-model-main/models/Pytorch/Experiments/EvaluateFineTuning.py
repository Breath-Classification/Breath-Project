from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from FineTuning import (
    FineTuningModel,
    InputAdapter,
    create_dataloaders,
    evaluate_accuracy,
    get_adapter_layer,
    get_adapter_layer_for_training,
    get_fc_layer,
    load_base_model,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the base mobile PyTorch model with a fine-tuned layer artifact."
    )
    parser.add_argument(
        "--manifest",
        default=Path("data/NewData/layers/latest_manifest.json"),
        type=Path,
        help="Fine-tuning manifest produced by FineTuning.py.",
    )
    parser.add_argument("--eval-data-txt", default=None, type=Path, help="Optional sequence file used for evaluation.")
    parser.add_argument("--model-file", default=None, type=Path, help="Optional explicit base TorchScript model path.")
    parser.add_argument("--dataset-type", default=None)
    parser.add_argument("--target", default=0, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--block-size", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--save-model-file", default=None, type=Path, help="Optional path for the model with loaded layers.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuning manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def path_from_manifest(path_value: str | None, *, ml_root: Path) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    if path.is_absolute() and path.parts[:2] == ("/", "ml"):
        local_path = ml_root.joinpath(*path.parts[2:])
        if local_path.exists():
            return local_path

    return path


def infer_ml_root(manifest_path: Path) -> Path:
    resolved = manifest_path.resolve()
    if resolved.parts[-3:] == ("NewData", "layers", manifest_path.name):
        return resolved.parents[3]
    return Path.cwd()


def resolve_eval_file(args: argparse.Namespace, manifest: dict, *, ml_root: Path) -> Path:
    if args.eval_data_txt is not None:
        return args.eval_data_txt

    manifest_eval_file = path_from_manifest(manifest.get("test_data_txt"), ml_root=ml_root)
    if manifest_eval_file is None:
        raise ValueError("Manifest does not contain test_data_txt. Pass --eval-data-txt explicitly.")
    return manifest_eval_file


def load_model_for_eval(
    *,
    manifest: dict,
    model_file: Path | None,
    feature_count: int,
    device: torch.device,
) -> FineTuningModel:
    explicit_model_file = model_file
    if explicit_model_file is None:
        manifest_model_file = path_from_manifest(manifest.get("base_model_file"), ml_root=Path.cwd())
        explicit_model_file = manifest_model_file if manifest_model_file is not None and manifest_model_file.exists() else None

    model_name = manifest.get("model_name", "LSTMBASE_tens")
    base_model_path = resolve_model_path(model_name, explicit_model_file)
    base_model = load_base_model(base_model_path, device)

    needs_input_adapter = (
        manifest.get("artifact_layer_sources", {}).get("adapter") == "input_wrapper"
        and get_adapter_layer(base_model) is None
    )
    input_adapter = InputAdapter(feature_count) if needs_input_adapter else None
    return FineTuningModel(base_model=base_model, input_adapter=input_adapter)


def apply_layer_artifacts(
    *,
    model: FineTuningModel,
    manifest: dict,
    manifest_path: Path,
    device: torch.device,
) -> list[str]:
    loaded_layers: list[str] = []
    layer_files = manifest.get("layer_files", {})

    if "adapter" in layer_files:
        adapter, _ = get_adapter_layer_for_training(model)
        if adapter is None:
            raise RuntimeError("The model does not expose an adapter layer for the adapter artifact.")

        adapter_path = manifest_path.parent / layer_files["adapter"]
        adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
        loaded_layers.append("adapter")

    if "fc" in layer_files:
        fc = get_fc_layer(model.base_model)
        if fc is None:
            raise RuntimeError("The model does not expose an fc layer for the fc artifact.")

        fc_path = manifest_path.parent / layer_files["fc"]
        fc.load_state_dict(torch.load(fc_path, map_location=device, weights_only=True))
        loaded_layers.append("fc")

    if not loaded_layers:
        raise ValueError("Manifest does not contain any layer artifacts to load.")

    return loaded_layers


def save_loaded_model(model: FineTuningModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model.input_adapter is not None:
        scripted = torch.jit.script(model)
        scripted.save(str(output_path))
        return

    torch.jit.save(model.base_model, str(output_path))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    ml_root = infer_ml_root(args.manifest)
    eval_file = resolve_eval_file(args, manifest, ml_root=ml_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_size = args.block_size or int(manifest.get("block_size", 30))
    batch_size = args.batch_size or int(manifest.get("training", {}).get("batch_size", 16))
    dataset_type = args.dataset_type or manifest.get("dataset_type", "BlockDataset")

    _, eval_dataloader = create_dataloaders(
        transform=None,
        batch_size=batch_size,
        block_size=block_size,
        target=args.target,
        dataset_type=dataset_type,
        num_workers=args.num_workers,
        train_data_txt=str(eval_file),
        test_data_txt=str(eval_file),
    )
    first_batch_x, _ = next(iter(eval_dataloader))
    feature_count = int(first_batch_x.shape[-1])

    base_model = load_model_for_eval(
        manifest=manifest,
        model_file=args.model_file,
        feature_count=feature_count,
        device=device,
    )
    tuned_model = load_model_for_eval(
        manifest=manifest,
        model_file=args.model_file,
        feature_count=feature_count,
        device=device,
    )
    loaded_layers = apply_layer_artifacts(
        model=tuned_model,
        manifest=manifest,
        manifest_path=args.manifest,
        device=device,
    )

    base_accuracy = evaluate_accuracy(base_model, eval_dataloader, device)
    tuned_accuracy = evaluate_accuracy(tuned_model, eval_dataloader, device)

    if args.save_model_file is not None:
        save_loaded_model(tuned_model, args.save_model_file)

    result = {
        "manifest_file": str(args.manifest),
        "eval_data_txt": str(eval_file),
        "model_name": manifest.get("model_name"),
        "loaded_layers": loaded_layers,
        "base_accuracy": base_accuracy,
        "fine_tuned_accuracy": tuned_accuracy,
        "accuracy_delta": tuned_accuracy - base_accuracy,
        "improved": tuned_accuracy > base_accuracy,
        "saved_model_file": str(args.save_model_file) if args.save_model_file is not None else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
