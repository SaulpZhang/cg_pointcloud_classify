from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from teacher import load_teacher


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_weights_path(weights_path: str, mode: str) -> str:
    path = Path(weights_path)
    if path.is_file():
        return str(path)

    if not path.exists():
        raise FileNotFoundError(f"Weights path not found: {path}")

    default_name = "clip_classifier_40cls.pth"
    candidate = path / default_name
    if candidate.exists():
        return str(candidate)

    pth_files = sorted(path.glob("*.pth"))
    if len(pth_files) == 1:
        return str(pth_files[0])

    raise FileNotFoundError(
        f"Cannot resolve checkpoint in directory {path}. Expected {default_name} or a single .pth file."
    )


def copy_misclassified_images(
    image_paths: List[str],
    predictions: List[int],
    label_idx: int,
    failed_root: str,
    sample_index: int,
    label_name: str,
) -> None:
    failed_dir = Path(failed_root) / f"sample_{sample_index:06d}_{label_name}"
    failed_dir.mkdir(parents=True, exist_ok=True)
    for image_path, prediction in zip(image_paths, predictions):
        if prediction == label_idx:
            continue
        shutil.copy2(image_path, failed_dir / Path(image_path).name)


def copy_failed_images(image_paths: List[str], failed_root: str, sample_index: int, label_name: str) -> None:
    failed_dir = Path(failed_root) / f"sample_{sample_index:06d}_{label_name}"
    failed_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, failed_dir / Path(image_path).name)


def parse_filename(image_path: Path) -> Tuple[int, int, int, str]:
    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename format invalid: {image_path.name}")

    label_name = "_".join(parts[:-3])
    label_index = int(parts[-3])
    object_index = int(parts[-2])
    view_id = int(parts[-1])
    return label_index, object_index, view_id, label_name


def collect_image_groups(image_dir: str) -> List[Tuple[int, int, str, List[str]]]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    groups = {}
    for image_path in sorted(root.rglob("*.png")):
        if not image_path.is_file():
            continue
        label_index, object_index, _, label_name = parse_filename(image_path)
        key = (label_index, object_index)
        if key not in groups:
            groups[key] = {"label_name": label_name, "image_paths": []}
        groups[key]["image_paths"].append(str(image_path))

    if not groups:
        raise ValueError(f"No PNG images found in: {image_dir}")

    grouped_items = []
    for (label_index, object_index), info in sorted(groups.items(), key=lambda item: item[0]):
        grouped_items.append((label_index, object_index, info["label_name"], info["image_paths"]))
    return grouped_items


@torch.no_grad()
def evaluate_teacher(
    image_dir: str,
    weights_path: str,
    failed_dir: str,
    device: torch.device,
    image_batch_size: int,
    clip_model_name: Optional[str],
    use_amp: bool,
) -> Tuple[float, float]:
    groups = collect_image_groups(image_dir)

    teacher = load_teacher(
        checkpoint_path=weights_path,
        device=str(device),
        clip_model_name=clip_model_name,
        use_amp=use_amp,
    )

    image_total = 0
    image_correct = 0
    point_total = 0
    point_correct = 0
    failed_count = 0
    for label_idx, object_index, label_name, image_paths in tqdm(groups, desc="Teacher inference"):
        sample_index = object_index

        try:
            prediction = teacher.predict_image_paths(image_paths=image_paths, batch_size=image_batch_size)
            image_total += len(prediction.view_predictions)
            image_correct += sum(int(pred == label_idx) for pred in prediction.view_predictions)
            point_total += 1
            point_correct += int(prediction.majority_label == label_idx)

            if failed_dir.strip():
                misclassified_predictions = [
                    pred for pred in prediction.view_predictions if pred != label_idx
                ]
                misclassified_image_paths = [
                    image_path for image_path, pred in zip(image_paths, prediction.view_predictions) if pred != label_idx
                ]
                if misclassified_image_paths:
                    # Only copy 2D images that are individually misclassified.
                    copy_misclassified_images(
                        image_paths=misclassified_image_paths,
                        predictions=misclassified_predictions,
                        label_idx=label_idx,
                        failed_root=failed_dir,
                        sample_index=sample_index,
                        label_name=label_name,
                    )
        except Exception as exc:
            failed_count += 1
            print(f"Teacher inference failed for sample {sample_index} ({label_name}): {exc}")
            if failed_dir.strip():
                copy_failed_images(image_paths, failed_dir, sample_index, label_name)
            image_total += len(image_paths)
            point_total += 1

    image_accuracy = image_correct / max(image_total, 1)
    point_accuracy = point_correct / max(point_total, 1)
    print(
        f"Teacher image accuracy: {image_accuracy:.4f} ({image_correct}/{image_total}), "
        f"point accuracy: {point_accuracy:.4f} ({point_correct}/{point_total}), failures: {failed_count}"
    )
    return image_accuracy, point_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher inference on a 2D multiview image directory")
    parser.add_argument("--data_dir", type=str, default="test_output_images", help="2D image directory")
    parser.add_argument("--weights_path", type=str, required=True, help="Checkpoint file path or directory")
    parser.add_argument("--device", type=str, default="", help="Optional device override, e.g. cpu or cuda:0")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--teacher_failed_dir", type=str, default="", help="If set, copy failed teacher inference images here")
    parser.add_argument("--teacher_clip_model", type=str, default="", help="Optional override for teacher CLIP model")
    parser.add_argument("--teacher_image_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.strip():
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = bool(args.use_amp and device.type == "cuda")
    weights_path = resolve_weights_path(args.weights_path, "teacher")

    print("Mode: teacher")
    print(f"Data dir: {args.data_dir}")
    print(f"Weights path: {weights_path}")
    print(f"Device: {device}")

    evaluate_teacher(
        image_dir=args.data_dir,
        weights_path=weights_path,
        failed_dir=args.teacher_failed_dir,
        device=device,
        image_batch_size=args.teacher_image_batch_size,
        clip_model_name=args.teacher_clip_model.strip() or None,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()