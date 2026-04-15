from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointnet2_model import PointNet2Student


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


def resolve_weights_path(weights_path: str) -> str:
    path = Path(weights_path)
    if path.is_file():
        return str(path)

    if not path.exists():
        raise FileNotFoundError(f"Weights path not found: {path}")

    default_name = "student_pointnet2_distill.pth"
    candidate = path / default_name
    if candidate.exists():
        return str(candidate)

    pth_files = sorted(path.glob("*.pth"))
    if len(pth_files) == 1:
        return str(pth_files[0])

    raise FileNotFoundError(
        f"Cannot resolve checkpoint in directory {path}. Expected {default_name} or a single .pth file."
    )


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
    parser = argparse.ArgumentParser(description="Run student inference on ModelNet40 HDF5 data")
    parser.add_argument("--data_dir", type=str, default="modelnet40_ply_hdf5_2048", help="ModelNet40 data root")
    parser.add_argument("--split_file", type=str, default="test_files.txt", help="Split list file under data_dir")
    parser.add_argument("--weights_path", type=str, required=True, help="Checkpoint file path or directory")
    parser.add_argument("--device", type=str, default="", help="Optional device override, e.g. cpu or cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
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
    weights_path = resolve_weights_path(args.weights_path)

    print("Mode: student")
    print(f"Data dir: {args.data_dir}")
    print(f"Split file: {args.split_file}")
    print(f"Weights path: {weights_path}")
    print(f"Device: {device}")

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