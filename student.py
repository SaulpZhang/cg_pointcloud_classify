from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointnet2_model import PointNet2Student
from teacher import TeacherModel, load_teacher

from dotenv import load_dotenv
load_dotenv()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pc_normalize(point_cloud: np.ndarray) -> np.ndarray:
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    scale = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    if scale > 0:
        point_cloud = point_cloud / scale
    return point_cloud


@dataclass
class DistillBatchOutput:
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    kd_loss: torch.Tensor
    logits: torch.Tensor


def distillation_step(
    model: nn.Module,
    points: torch.Tensor,
    labels: torch.Tensor,
    teacher_probs: torch.Tensor,
    alpha: float,
    temperature: float,
) -> DistillBatchOutput:
    logits = model(points)

    ce_loss = F.cross_entropy(logits, labels)

    if alpha <= 0.0:
        kd_loss = torch.zeros_like(ce_loss)
        total_loss = ce_loss
    else:
        log_probs_student = F.log_softmax(logits / temperature, dim=1)
        # Ensure teacher_probs matches the dtype of logits (important for AMP)
        teacher_probs = teacher_probs.to(dtype=logits.dtype)
        kd_loss = F.kl_div(log_probs_student, teacher_probs, reduction="batchmean") * (temperature ** 2)
        total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss

    return DistillBatchOutput(
        total_loss=total_loss,
        ce_loss=ce_loss,
        kd_loss=kd_loss,
        logits=logits,
    )


class PointCloudSoftLabelDataset(Dataset):
    def __init__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        teacher_probs: torch.Tensor,
        num_points: int,
        normalize: bool = True,
    ) -> None:
        if points.shape[0] != labels.shape[0] or points.shape[0] != teacher_probs.shape[0]:
            raise ValueError("points, labels, teacher_probs must have the same sample size")

        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.int64).reshape(-1)
        self.teacher_probs = teacher_probs.float()
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
        label = int(self.labels[idx])
        soft = self.teacher_probs[idx]

        return (
            torch.from_numpy(pts).transpose(0, 1).contiguous(),
            torch.tensor(label, dtype=torch.long),
            soft,
        )


def build_dummy_soft_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    label_tensor = torch.from_numpy(labels.astype(np.int64).reshape(-1))
    return F.one_hot(label_tensor, num_classes=num_classes).float()


def _resolve_h5_path(modelnet_root: Path, line_path: str) -> Path:
    p = Path(line_path)
    if p.exists():
        return p

    candidate = modelnet_root / p.name
    if candidate.exists():
        return candidate

    # Handle txt entries such as data/modelnet40_ply_hdf5_2048/ply_data_train0.h5
    if p.parts:
        candidate2 = modelnet_root / p.parts[-1]
        if candidate2.exists():
            return candidate2

    raise FileNotFoundError(f"Cannot resolve h5 path from line '{line_path}' under root {modelnet_root}")


def load_modelnet_h5_from_list(modelnet_root: str, list_file: str) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(modelnet_root)
    txt_path = root / list_file
    if not txt_path.exists():
        raise FileNotFoundError(f"File list not found: {txt_path}")

    all_data: List[np.ndarray] = []
    all_label: List[np.ndarray] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        h5_path = _resolve_h5_path(root, line)
        with h5py.File(h5_path, "r") as hf:
            data = hf["data"][:]
            label = hf["label"][:]
            all_data.append(data)
            all_label.append(label)

    data_np = np.concatenate(all_data, axis=0)
    label_np = np.concatenate(all_label, axis=0).reshape(-1)
    if data_np.shape[-1] > 3:
        data_np = data_np[:, :, :3]

    return data_np.astype(np.float32), label_np.astype(np.int64)


def load_modelnet_h5_from_paths(modelnet_root: str, h5_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not h5_paths:
        raise ValueError("h5_paths must not be empty")

    root = Path(modelnet_root)
    all_data: List[np.ndarray] = []
    all_label: List[np.ndarray] = []

    for raw_path in h5_paths:
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        h5_path = _resolve_h5_path(root, raw_path)
        with h5py.File(h5_path, "r") as hf:
            data = hf["data"][:]
            label = hf["label"][:]
            all_data.append(data)
            all_label.append(label)

    if not all_data:
        raise ValueError("No valid h5 files provided")

    data_np = np.concatenate(all_data, axis=0)
    label_np = np.concatenate(all_label, axis=0).reshape(-1)
    if data_np.shape[-1] > 3:
        data_np = data_np[:, :, :3]

    return data_np.astype(np.float32), label_np.astype(np.int64)


def load_shape_names(modelnet_root: str, shape_names_path: str = "") -> List[str]:
    if shape_names_path.strip():
        candidate = Path(shape_names_path)
        path = candidate if candidate.exists() else Path(modelnet_root) / shape_names_path
    else:
        path = Path(modelnet_root) / "shape_names.txt"
    if not path.exists():
        raise FileNotFoundError(f"shape_names.txt not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_train_val(
    data: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    n = data.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_n = max(1, int(n * val_ratio))
    if val_n >= n:
        val_n = n - 1

    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    return data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]


def fuse_teacher_probabilities(
    mean_probabilities: torch.Tensor,
    view_probabilities: torch.Tensor,
    view_predictions: torch.Tensor,
    label_idx: int,
    distill_mode: int,
) -> torch.Tensor:
    if distill_mode == 0:
        probs = mean_probabilities.float()
        return probs / probs.sum().clamp(min=1e-8)

    if distill_mode == 1:
        view_probs = view_probabilities.float()
        view_preds = view_predictions.long()
        correct_mask = view_preds == int(label_idx)

        if int(correct_mask.sum().item()) > 0:
            probs = view_probs[correct_mask].mean(dim=0)
        else:
            # Fallback: if no view is correctly classified, keep training stable with the original mean fusion.
            probs = mean_probabilities.float()

        return probs / probs.sum().clamp(min=1e-8)

    if distill_mode == 2:
        view_probs = view_probabilities.float()
        if view_probs.ndim != 2 or view_probs.shape[0] == 0:
            probs = mean_probabilities.float()
            return probs / probs.sum().clamp(min=1e-8)

        # Confidence fusion: use each view's max class probability as fusion weight.
        view_confidence = view_probs.max(dim=1).values
        weight_sum = view_confidence.sum()
        if float(weight_sum.item()) <= 1e-8:
            probs = view_probs.mean(dim=0)
        else:
            weights = (view_confidence / weight_sum).unsqueeze(1)
            probs = (view_probs * weights).sum(dim=0)

        return probs / probs.sum().clamp(min=1e-8)

    raise ValueError(
        f"Unsupported distill_mode: {distill_mode}. "
        "Currently supported: 0 (mean), 1 (correct-only mean), 2 (confidence-weighted mean)."
    )


@torch.no_grad()
def compute_teacher_raw_outputs(
    teacher: TeacherModel,
    points_np: np.ndarray,
    labels_np: np.ndarray,
    shape_names: List[str],
    render_root: str,
    split_name: str,
    num_views: int,
    image_size: int,
    fov_degrees: float,
    point_radius: int,
    camera_distance: float,
    image_batch_size: int,
    split_desc: str,
) -> Dict[str, torch.Tensor]:
    split_render_dir = Path(render_root) / split_name
    split_render_dir.mkdir(parents=True, exist_ok=True)

    all_mean_probs: List[torch.Tensor] = []
    all_view_probs: List[torch.Tensor] = []
    all_view_preds: List[torch.Tensor] = []

    iterator = range(points_np.shape[0])
    for i in tqdm(iterator, desc=split_desc):
        points = points_np[i]
        label_idx = int(labels_np[i])
        label_name = shape_names[label_idx] if 0 <= label_idx < len(shape_names) else f"class_{label_idx}"

        pred = teacher.predict_point_cloud(
            points=points,
            index=i,
            label=label_name,
            output_dir=str(split_render_dir),
            label_index=label_idx,
            num_views=num_views,
            image_size=image_size,
            fov_degrees=fov_degrees,
            point_radius=point_radius,
            camera_distance=camera_distance,
            batch_size=image_batch_size,
        )
        all_mean_probs.append(torch.tensor(pred.mean_probabilities, dtype=torch.float32))
        all_view_probs.append(torch.tensor(pred.view_probabilities, dtype=torch.float32))
        all_view_preds.append(torch.tensor(pred.view_predictions, dtype=torch.long))

    return {
        "mean_probabilities": torch.stack(all_mean_probs, dim=0),
        "view_probabilities": torch.stack(all_view_probs, dim=0),
        "view_predictions": torch.stack(all_view_preds, dim=0),
    }


def load_or_compute_teacher_raw_outputs(
    teacher: TeacherModel,
    points_np: np.ndarray,
    labels_np: np.ndarray,
    shape_names: List[str],
    render_root: str,
    split_name: str,
    num_views: int,
    image_size: int,
    fov_degrees: float,
    point_radius: int,
    camera_distance: float,
    image_batch_size: int,
    split_desc: str,
    reextract: bool = False,
) -> Dict[str, torch.Tensor]:
    cache_path = Path(render_root) / f"teacher_raw_outputs_{split_name}.pt"
    cache_meta = {
        "split_name": split_name,
        "num_samples": int(points_np.shape[0]),
        "num_views": int(num_views),
        "image_size": int(image_size),
        "fov_degrees": float(fov_degrees),
        "point_radius": int(point_radius),
        "camera_distance": float(camera_distance),
        "teacher_checkpoint": getattr(teacher, "clip_model_name", None),
    }

    if (not reextract) and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        required_keys = {"mean_probabilities", "view_probabilities", "view_predictions"}
        if isinstance(cached, dict) and cached.get("meta") == cache_meta and required_keys.issubset(cached.keys()):
            print(f"Loaded cached teacher raw outputs: {cache_path}")
            return {
                "mean_probabilities": cached["mean_probabilities"].float(),
                "view_probabilities": cached["view_probabilities"].float(),
                "view_predictions": cached["view_predictions"].long(),
            }

    raw_outputs = compute_teacher_raw_outputs(
        teacher=teacher,
        points_np=points_np,
        labels_np=labels_np,
        shape_names=shape_names,
        render_root=render_root,
        split_name=split_name,
        num_views=num_views,
        image_size=image_size,
        fov_degrees=fov_degrees,
        point_radius=point_radius,
        camera_distance=camera_distance,
        image_batch_size=image_batch_size,
        split_desc=split_desc,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": cache_meta,
            "mean_probabilities": raw_outputs["mean_probabilities"].cpu(),
            "view_probabilities": raw_outputs["view_probabilities"].cpu(),
            "view_predictions": raw_outputs["view_predictions"].cpu(),
        },
        cache_path,
    )
    print(f"Saved teacher raw outputs cache: {cache_path}")
    return raw_outputs


def build_soft_labels_from_raw_outputs(
    raw_outputs: Dict[str, torch.Tensor],
    labels_np: np.ndarray,
    distill_mode: int,
) -> torch.Tensor:
    mean_probs_all = raw_outputs["mean_probabilities"]
    view_probs_all = raw_outputs["view_probabilities"]
    view_preds_all = raw_outputs["view_predictions"]

    all_probs: List[torch.Tensor] = []
    for i in range(mean_probs_all.shape[0]):
        probs = fuse_teacher_probabilities(
            mean_probabilities=mean_probs_all[i],
            view_probabilities=view_probs_all[i],
            view_predictions=view_preds_all[i],
            label_idx=int(labels_np[i]),
            distill_mode=distill_mode,
        )
        all_probs.append(probs)

    return torch.stack(all_probs, dim=0)


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    alpha: float,
    temperature: float,
) -> Tuple[float, float, float, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_correct = 0
    total = 0

    for points, labels, teacher_probs in tqdm(loader, desc="Student Train", leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_probs = teacher_probs.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = distillation_step(
                model=model,
                points=points,
                labels=labels,
                teacher_probs=teacher_probs,
                alpha=alpha,
                temperature=temperature,
            )

        if scaler is not None:
            scaler.scale(out.total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out.total_loss.backward()
            optimizer.step()

        pred = out.logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        bs = labels.size(0)
        total += bs

        total_loss += out.total_loss.item() * bs
        total_ce += out.ce_loss.item() * bs
        total_kd += out.kd_loss.item() * bs

    denom = max(total, 1)
    return total_loss / denom, total_ce / denom, total_kd / denom, total_correct / denom


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    alpha: float,
    temperature: float,
) -> Tuple[float, float, float, float]:
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_correct = 0
    total = 0

    for points, labels, teacher_probs in tqdm(loader, desc="Student Eval", leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_probs = teacher_probs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = distillation_step(
                model=model,
                points=points,
                labels=labels,
                teacher_probs=teacher_probs,
                alpha=alpha,
                temperature=temperature,
            )

        pred = out.logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        bs = labels.size(0)
        total += bs

        total_loss += out.total_loss.item() * bs
        total_ce += out.ce_loss.item() * bs
        total_kd += out.kd_loss.item() * bs

    denom = max(total, 1)
    return total_loss / denom, total_ce / denom, total_kd / denom, total_correct / denom


def build_student_and_device(gpu_ids_arg: str, num_classes: int) -> Tuple[nn.Module, torch.device, List[int], bool]:
    if torch.cuda.is_available():
        if gpu_ids_arg.strip():
            gpu_ids = [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))

        device = torch.device(f"cuda:{gpu_ids[0]}")
        use_dp = len(gpu_ids) > 1
        model = PointNet2Student(num_classes=num_classes).to(device)
        if use_dp:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        return model, device, gpu_ids, use_dp

    device = torch.device("cpu")
    model = PointNet2Student(num_classes=num_classes).to(device)
    return model, device, [], False


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    meta: dict,
) -> Tuple[str, str]:
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "student_state_dict": {k: v.cpu() for k, v in state_dict.items()},
        "meta": meta,
    }
    torch.save(checkpoint, save_path)

    meta_path = str(Path(save_path).with_suffix(".json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return save_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet++ student with CLIP teacher distillation")

    parser.add_argument("--modelnet_root", type=str, default="modelnet40_ply_hdf5_2048")
    parser.add_argument(
        "--shape_names_path",
        type=str,
        default="",
        help="Optional path to shape_names.txt. If unset, use <modelnet_root>/shape_names.txt",
    )
    parser.add_argument(
        "--train_h5_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional manual train h5 paths (support multiple paths)",
    )
    parser.add_argument(
        "--test_h5_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional manual test h5 paths (support multiple paths)",
    )
    parser.add_argument("--teacher_checkpoint", type=str, default="clip_classifier_40cls.pth")
    parser.add_argument("--teacher_clip_model", type=str, default="", help="Optional override for teacher CLIP model")
    parser.add_argument("--teacher_render_dir", type=str, default="teacher_render_cache")

    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reextract", action=argparse.BooleanOptionalAction, default=False, help="Recompute teacher soft labels and overwrite cache")

    parser.add_argument("--distill_alpha", type=float, default=0.7)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument(
        "--distill_mode",
        type=int,
        default=0,
        help=(
            "Distillation fusion mode. "
            "0: mean-probability fusion; "
            "1: mean over correctly-classified views only; "
            "2: confidence-weighted fusion"
        ),
    )

    parser.add_argument("--teacher_num_views", type=int, default=12)
    parser.add_argument("--teacher_image_size", type=int, default=256)
    parser.add_argument("--teacher_fov_degrees", type=float, default=45.0)
    parser.add_argument("--teacher_point_radius", type=int, default=2)
    parser.add_argument("--teacher_camera_distance", type=float, default=1.6)
    parser.add_argument("--teacher_image_batch_size", type=int, default=64)

    parser.add_argument("--save_path", type=str, default="student_pointnet2_distill.pth")
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_project", type=str, default="pointcloudclassify")
    parser.add_argument("--wandb_entity", type=str, default="group_cg")
    parser.add_argument("--wandb_run_name", type=str, default="student-pointnet2-distill")
    parser.add_argument("--wandb_log_weights_every", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wandb_run: Optional[object] = None

    try:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        shape_names = load_shape_names(args.modelnet_root, args.shape_names_path)

        if args.train_h5_paths:
            train_data_all, train_label_all = load_modelnet_h5_from_paths(args.modelnet_root, args.train_h5_paths)
        else:
            train_data_all, train_label_all = load_modelnet_h5_from_list(args.modelnet_root, "train_files.txt")

        if args.test_h5_paths:
            test_data, test_label = load_modelnet_h5_from_paths(args.modelnet_root, args.test_h5_paths)
        else:
            test_data, test_label = load_modelnet_h5_from_list(args.modelnet_root, "test_files.txt")

        train_data, train_label, val_data, val_label = split_train_val(
            train_data_all,
            train_label_all,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        use_distill = args.distill_alpha > 0.0
        teacher: Optional[TeacherModel] = None

        if use_distill:
            teacher_clip_override = args.teacher_clip_model.strip() or None
            teacher = load_teacher(
                checkpoint_path=args.teacher_checkpoint,
                clip_model_name=teacher_clip_override,
                num_classes=args.num_classes,
                use_amp=args.amp,
            )

            print(f"Teacher device: {teacher.device}")
            print("Computing/caching teacher raw outputs for train/val/test ...")

            train_raw = load_or_compute_teacher_raw_outputs(
                teacher=teacher,
                points_np=train_data,
                labels_np=train_label,
                shape_names=shape_names,
                render_root=args.teacher_render_dir,
                split_name="train",
                num_views=args.teacher_num_views,
                image_size=args.teacher_image_size,
                fov_degrees=args.teacher_fov_degrees,
                point_radius=args.teacher_point_radius,
                camera_distance=args.teacher_camera_distance,
                image_batch_size=args.teacher_image_batch_size,
                split_desc="Teacher raw outputs (train)",
                reextract=args.reextract,
            )
            val_raw = load_or_compute_teacher_raw_outputs(
                teacher=teacher,
                points_np=val_data,
                labels_np=val_label,
                shape_names=shape_names,
                render_root=args.teacher_render_dir,
                split_name="val",
                num_views=args.teacher_num_views,
                image_size=args.teacher_image_size,
                fov_degrees=args.teacher_fov_degrees,
                point_radius=args.teacher_point_radius,
                camera_distance=args.teacher_camera_distance,
                image_batch_size=args.teacher_image_batch_size,
                split_desc="Teacher raw outputs (val)",
                reextract=args.reextract,
            )
            test_raw = load_or_compute_teacher_raw_outputs(
                teacher=teacher,
                points_np=test_data,
                labels_np=test_label,
                shape_names=shape_names,
                render_root=args.teacher_render_dir,
                split_name="test",
                num_views=args.teacher_num_views,
                image_size=args.teacher_image_size,
                fov_degrees=args.teacher_fov_degrees,
                point_radius=args.teacher_point_radius,
                camera_distance=args.teacher_camera_distance,
                image_batch_size=args.teacher_image_batch_size,
                split_desc="Teacher raw outputs (test)",
                reextract=args.reextract,
            )

            train_soft = build_soft_labels_from_raw_outputs(train_raw, train_label, args.distill_mode)
            val_soft = build_soft_labels_from_raw_outputs(val_raw, val_label, args.distill_mode)
            test_soft = build_soft_labels_from_raw_outputs(test_raw, test_label, args.distill_mode)
        else:
            print("distill_alpha <= 0: skip teacher and run supervised PointNet++ training")
            train_soft = build_dummy_soft_labels(train_label, args.num_classes)
            val_soft = build_dummy_soft_labels(val_label, args.num_classes)
            test_soft = build_dummy_soft_labels(test_label, args.num_classes)

        train_dataset = PointCloudSoftLabelDataset(train_data, train_label, train_soft, num_points=args.num_points)
        val_dataset = PointCloudSoftLabelDataset(val_data, val_label, val_soft, num_points=args.num_points)
        test_dataset = PointCloudSoftLabelDataset(test_data, test_label, test_soft, num_points=args.num_points)

        model, device, gpu_ids, use_dp = build_student_and_device(args.gpu_ids, args.num_classes)
        use_amp = bool(args.amp and device.type == "cuda")

        print(f"Student device: {device}")
        print(f"Student GPUs: {gpu_ids if gpu_ids else 'CPU'}")
        print(f"AMP enabled: {use_amp}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if args.use_wandb:
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group='student_newdata',
                name=args.wandb_run_name,
                config={
                    "student_model": "PointNet++ SSG",
                    "teacher_checkpoint": args.teacher_checkpoint,
                    "teacher_clip_model": teacher.clip_model_name if teacher is not None else None,
                    "use_distillation": use_distill,
                    "num_classes": args.num_classes,
                    "num_points": args.num_points,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "val_ratio": args.val_ratio,
                    "shape_names_path": args.shape_names_path,
                    "train_h5_paths": args.train_h5_paths,
                    "test_h5_paths": args.test_h5_paths,
                    "distill_alpha": args.distill_alpha,
                    "distill_temperature": args.distill_temperature,
                    "distill_mode": args.distill_mode,
                    "teacher_num_views": args.teacher_num_views,
                    "teacher_image_size": args.teacher_image_size,
                    "teacher_fov_degrees": args.teacher_fov_degrees,
                    "teacher_point_radius": args.teacher_point_radius,
                    "teacher_camera_distance": args.teacher_camera_distance,
                    "reextract": args.reextract,
                    "gpu_ids": gpu_ids,
                    "use_data_parallel": use_dp,
                    "amp": use_amp,
                },
            )
            wandb_run.watch(model, log="all", log_freq=1000)

        best_val_acc = -1.0
        best_meta = None

        for epoch in tqdm(range(1, args.epochs + 1), desc="Student Epochs"):
            train_total, train_ce, train_kd, train_acc = run_epoch_train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                alpha=args.distill_alpha,
                temperature=args.distill_temperature,
            )
            val_total, val_ce, val_kd, val_acc = run_epoch_eval(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                alpha=args.distill_alpha,
                temperature=args.distill_temperature,
            )
            test_total, test_ce, test_kd, test_acc = run_epoch_eval(
                model=model,
                loader=test_loader,
                device=device,
                use_amp=use_amp,
                alpha=args.distill_alpha,
                temperature=args.distill_temperature,
            )

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_total:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_total:.4f} val_acc={val_acc:.4f} | "
                f"test_loss={test_total:.4f} test_acc={test_acc:.4f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/total_loss": train_total,
                        "train/ce_loss": train_ce,
                        "train/kd_loss": train_kd,
                        "train/acc": train_acc,
                        "val/total_loss": val_total,
                        "val/ce_loss": val_ce,
                        "val/kd_loss": val_kd,
                        "val/acc": val_acc,
                        "test/total_loss": test_total,
                        "test/ce_loss": test_ce,
                        "test/kd_loss": test_kd,
                        "test/acc": test_acc,
                    },
                    step=epoch,
                )

                if args.wandb_log_weights_every > 0 and epoch % args.wandb_log_weights_every == 0:
                    weight_logs = {
                        f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())
                        for name, param in model.named_parameters()
                    }
                    wandb.log(weight_logs, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_meta = {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "val_total_loss": val_total,
                }

        final_test_total, final_test_ce, final_test_kd, final_test_acc = run_epoch_eval(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
            alpha=args.distill_alpha,
            temperature=args.distill_temperature,
        )

        meta = {
            "student_model": "PointNet++ SSG",
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_clip_model": teacher.clip_model_name if teacher is not None else None,
            "use_distillation": use_distill,
            "num_classes": args.num_classes,
            "num_points": args.num_points,
            "shape_names_path": args.shape_names_path,
            "train_h5_paths": args.train_h5_paths,
            "test_h5_paths": args.test_h5_paths,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_val_acc": best_val_acc,
            "final_test_acc": final_test_acc,
            "final_test_total_loss": final_test_total,
            "final_test_ce_loss": final_test_ce,
            "final_test_kd_loss": final_test_kd,
            "best_snapshot": best_meta,
            "distill_alpha": args.distill_alpha,
            "distill_temperature": args.distill_temperature,
            "distill_mode": args.distill_mode,
        }

        save_path, meta_path = save_checkpoint(model, args.save_path, meta)
        print(f"Saved student model to: {save_path}")
        print(f"Saved metadata to: {meta_path}")

        if wandb_run is not None:
            wandb_run.summary["best_val_acc"] = best_val_acc
            wandb_run.summary["final_test_acc"] = final_test_acc

            artifact_name = f"student-pointnet2-{wandb_run.id}"
            model_artifact = wandb.Artifact(name=artifact_name, type="model", metadata=meta)
            model_artifact.add_file(save_path)
            model_artifact.add_file(meta_path)
            wandb_run.log_artifact(model_artifact)

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
