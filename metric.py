from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointnet2_model import PointNet2Student
from teacher import load_teacher


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pc_normalize(point_cloud: np.ndarray) -> np.ndarray:
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    scale = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    if scale > 0:
        point_cloud = point_cloud / scale
    return point_cloud


def _resolve_h5_path(modelnet_root: Path, line_path: str) -> Path:
    candidate = Path(line_path)
    if candidate.exists():
        return candidate

    fallback = modelnet_root / candidate.name
    if fallback.exists():
        return fallback

    if candidate.parts:
        fallback2 = modelnet_root / candidate.parts[-1]
        if fallback2.exists():
            return fallback2

    raise FileNotFoundError(f"Cannot resolve h5 path from line '{line_path}' under root {modelnet_root}")


def load_modelnet_h5_from_list(modelnet_root: str, split_file: str) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(modelnet_root)
    list_path = root / split_file
    if not list_path.exists():
        raise FileNotFoundError(f"File list not found: {list_path}")

    all_data: List[np.ndarray] = []
    all_label: List[np.ndarray] = []

    with open(list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        h5_path = _resolve_h5_path(root, line)
        with h5py.File(h5_path, "r") as hf:
            all_data.append(hf["data"][:])
            all_label.append(hf["label"][:])

    data_np = np.concatenate(all_data, axis=0)
    label_np = np.concatenate(all_label, axis=0).reshape(-1)
    if data_np.shape[-1] > 3:
        data_np = data_np[:, :, :3]

    return data_np.astype(np.float32), label_np.astype(np.int64)


def load_shape_names(modelnet_root: str) -> List[str]:
    path = Path(modelnet_root) / "shape_names.txt"
    if not path.exists():
        raise FileNotFoundError(f"shape_names.txt not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class PointCloudInferenceDataset(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray, num_points: int, normalize: bool = True) -> None:
        if points.shape[0] != labels.shape[0]:
            raise ValueError("points and labels must have the same sample size")

        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.int64).reshape(-1)
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        if self.num_points == n:
            return pts

        if self.num_points < n:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)

        return pts[idx]

    def __getitem__(self, idx: int):
        pts = self.points[idx]
        pts = self._sample_points(pts)
        if self.normalize:
            pts = pc_normalize(pts)

        return torch.from_numpy(pts).transpose(0, 1).contiguous(), int(self.labels[idx])


def load_student_checkpoint(checkpoint_path: str, device: torch.device) -> PointNet2Student:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("student_state_dict", checkpoint)
    meta = checkpoint.get("meta", {})
    num_classes = int(meta.get("num_classes", 40))

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "")] = value

    model = PointNet2Student(num_classes=num_classes).to(device)
    load_result = model.load_state_dict(cleaned_state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"Missing keys when loading student weights: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys when loading student weights: {load_result.unexpected_keys}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_weights_path(weights_path: str, mode: str) -> str:
    path = Path(weights_path)
    if path.is_file():
        return str(path)

    if not path.exists():
        raise FileNotFoundError(f"Weights path not found: {path}")

    default_name = "clip_classifier_40cls.pth" if mode == "teacher" else "student_pointnet2_distill.pth"
    candidate = path / default_name
    if candidate.exists():
        return str(candidate)

    pth_files = sorted(path.glob("*.pth"))
    if len(pth_files) == 1:
        return str(pth_files[0])

    raise FileNotFoundError(
        f"Cannot resolve checkpoint in directory {path}. Expected {default_name} or a single .pth file."
    )


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
) -> float:
    groups = collect_image_groups(image_dir)

    teacher = load_teacher(
        checkpoint_path=weights_path,
        device=str(device),
        clip_model_name=clip_model_name,
        use_amp=use_amp,
    )

    total = 0
    correct = 0
    failed_count = 0
    for label_idx, object_index, label_name, image_paths in tqdm(groups, desc="Teacher inference"):
        sample_index = object_index

        try:
            prediction = teacher.predict_image_paths(image_paths=image_paths, batch_size=image_batch_size)
            correct += int(prediction.majority_label == label_idx)
        except Exception as exc:
            failed_count += 1
            print(f"Teacher inference failed for sample {sample_index} ({label_name}): {exc}")
            if failed_dir.strip():
                copy_failed_images(image_paths, failed_dir, sample_index, label_name)
        total += 1

    accuracy = correct / max(total, 1)
    print(f"Teacher accuracy: {accuracy:.4f} ({correct}/{total}), failures: {failed_count}")
    return accuracy


@torch.no_grad()
def evaluate_student(
    modelnet_root: str,
    split_file: str,
    weights_path: str,
    device: torch.device,
    batch_size: int,
    num_points: int,
    use_amp: bool,
) -> float:
    points_np, labels_np = load_modelnet_h5_from_list(modelnet_root, split_file)
    dataset = PointCloudInferenceDataset(points_np, labels_np, num_points=num_points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_student_checkpoint(weights_path, device)

    total = 0
    correct = 0
    for points, labels in tqdm(loader, desc="Student inference"):
        points = points.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(points)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / max(total, 1)
    print(f"Student accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher or student inference on ModelNet40")
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="test_output_images",
        help="Teacher mode: 2D image directory. Student mode: ModelNet40 data root",
    )
    parser.add_argument("--split_file", type=str, default="test_files.txt", help="Student split list file under data_dir")
    parser.add_argument("--weights_path", type=str, required=True, help="Checkpoint file path or directory")
    parser.add_argument("--device", type=str, default="", help="Optional device override, e.g. cpu or cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=2048)
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
    weights_path = resolve_weights_path(args.weights_path, args.mode)

    print(f"Mode: {args.mode}")
    print(f"Data dir: {args.data_dir}")
    print(f"Split file: {args.split_file}")
    print(f"Weights path: {weights_path}")
    print(f"Device: {device}")

    if args.mode == "teacher":
        evaluate_teacher(
            image_dir=args.data_dir,
            weights_path=weights_path,
            failed_dir=args.teacher_failed_dir,
            device=device,
            image_batch_size=args.teacher_image_batch_size,
            clip_model_name=args.teacher_clip_model.strip() or None,
            use_amp=use_amp,
        )
    else:
        evaluate_student(
            modelnet_root=args.data_dir,
            split_file=args.split_file,
            weights_path=weights_path,
            device=device,
            batch_size=args.batch_size,
            num_points=args.num_points,
            use_amp=use_amp,
        )


if __name__ == "__main__":
    main()