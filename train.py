import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


import wandb

from dotenv import load_dotenv
load_dotenv()


@dataclass
class Sample:
    image_path: Path
    label_name: str
    label_index: int
    object_index: int
    view_id: int


def parse_filename(image_path: Path) -> Sample:
    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename format invalid: {image_path.name}")

    label_name = "_".join(parts[:-3])
    label_index = int(parts[-3])
    object_index = int(parts[-2])
    view_id = int(parts[-1])

    return Sample(
        image_path=image_path,
        label_name=label_name,
        label_index=label_index,
        object_index=object_index,
        view_id=view_id,
    )


def collect_samples(image_dir: str) -> List[Sample]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted([p for p in root.rglob("*.png") if p.is_file()])
    if not image_paths:
        raise ValueError(f"No PNG images found in: {image_dir}")

    samples = [parse_filename(p) for p in image_paths]
    return samples


def split_by_object(
    samples: List[Sample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    grouped: Dict[Tuple[int, int], List[Sample]] = {}
    for s in samples:
        key = (s.label_index, s.object_index)
        grouped.setdefault(key, []).append(s)

    keys = list(grouped.keys())
    n_total = len(keys)
    if n_total < 2:
        raise ValueError("Need at least 2 unique objects to split into train/val")

    if val_ratio < 0 or val_ratio >= 1.0:
        raise ValueError("Require 0 <= val_ratio < 1")

    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = max(1, int(n_total * val_ratio))

    if n_val >= n_total:
        n_val = n_total - 1

    val_keys = set(keys[:n_val])

    train_samples, val_samples = [], []
    for key, group in grouped.items():
        if key in val_keys:
            val_samples.extend(group)
        else:
            train_samples.extend(group)

    return train_samples, val_samples


class ImagePathDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return image, s.label_index


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class CLIPClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 40, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def image_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def build_samples_digest(samples: List[Sample]) -> str:
    hasher = hashlib.sha1()
    for sample in samples:
        hasher.update(str(sample.image_path).encode("utf-8"))
        hasher.update(str(sample.label_index).encode("utf-8"))
        hasher.update(str(sample.object_index).encode("utf-8"))
        hasher.update(str(sample.view_id).encode("utf-8"))
    return hasher.hexdigest()


def build_feature_cache_path(cache_dir: str, split_name: str, image_dir: str, clip_model_name: str) -> Path:
    safe_image_dir = Path(image_dir).name.replace("/", "_")
    safe_clip_model = clip_model_name.replace("/", "_").replace(":", "_")
    return Path(cache_dir) / f"{split_name}_{safe_image_dir}_{safe_clip_model}.pt"


def load_or_extract_clip_features(
    split_name: str,
    samples: List[Sample],
    dataloader: DataLoader,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    use_amp: bool,
    image_dir: str,
    clip_model_name: str,
    reextract: bool,
    cache_dir: str = "feature_cache",
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_path = build_feature_cache_path(cache_dir, split_name, image_dir, clip_model_name)
    cache_meta = {
        "cache_version": 1,
        "split_name": split_name,
        "image_dir": image_dir,
        "clip_model": clip_model_name,
        "num_samples": len(samples),
        "samples_digest": build_samples_digest(samples),
    }

    if not reextract and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, dict) and cached.get("meta") == cache_meta and "features" in cached and "labels" in cached:
            print(f"Loaded cached CLIP features for {split_name}: {cache_path}")
            return cached["features"].float(), cached["labels"].long()

    print(f"Extracting CLIP {split_name} features...")
    features, labels = extract_clip_features(clip_model, processor, dataloader, device, use_amp)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": cache_meta,
            "features": features.cpu(),
            "labels": labels.cpu(),
        },
        cache_path,
    )
    print(f"Saved CLIP {split_name} feature cache: {cache_path}")
    return features, labels


def extract_clip_features(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_features, all_labels = [], []

    clip_model.eval()
    
    # Get vision model and projection; handle DataParallel wrapper
    if isinstance(clip_model, nn.DataParallel):
        vision_model = clip_model.module.vision_model
        visual_projection = clip_model.module.visual_projection
    else:
        vision_model = clip_model.vision_model
        visual_projection = clip_model.visual_projection
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features", leave=False):
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Extract vision features using vision encoder only
                vision_outputs = vision_model(pixel_values)
                image_features = vision_outputs.last_hidden_state[:, 0, :]  # [CLS] token
                
                # Apply visual projection to get CLIP embedding space (512-dim)
                image_features = visual_projection(image_features)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_features.append(image_features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="Train", leave=False):
        features = features.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="Eval", leave=False):
        features = features.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a 40-class classifier on CLIP image features")
    parser.add_argument("--image_dir", type=str, default="output_images", help="Directory containing rendered PNG images")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model name")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_image_dir", type=str, default="test_output_images", help="Directory containing test PNG images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma separated GPU ids, e.g. '0,1,2,3,4,5,6,7'. Empty means all visible GPUs.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision (recommended on V100)")
    parser.add_argument("--reextract", action=argparse.BooleanOptionalAction, default=False, help="Re-extract CLIP features and overwrite cache")
    parser.add_argument("--save_path", type=str, default="clip_classifier_40cls.pth")
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_project", type=str, default="pointcloudclassify")
    parser.add_argument("--wandb_entity", type=str, default="group_cg")
    parser.add_argument("--wandb_run_name", type=str, default="try")
    parser.add_argument("--wandb_log_weights_every", type=int, default=1)
    args = parser.parse_args()
    wandb_run: Optional[object] = None

    try:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            if args.gpu_ids.strip():
                gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
            else:
                gpu_ids = list(range(torch.cuda.device_count()))
            device = torch.device(f"cuda:{gpu_ids[0]}")
            use_data_parallel = len(gpu_ids) > 1
            print(f"Using CUDA GPUs: {gpu_ids}")
        else:
            gpu_ids = []
            device = torch.device("cpu")
            use_data_parallel = False
            print("Using CPU")

        use_amp = bool(args.amp and device.type == "cuda")
        print(f"AMP enabled: {use_amp}")

        samples = collect_samples(args.image_dir)
        test_samples = collect_samples(args.test_image_dir)

        train_samples, val_samples = split_by_object(
            samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        print(f"Total images: {len(samples)}")
        print(f"Train images: {len(train_samples)}")
        print(f"Val images: {len(val_samples)}")
        print(f"Test images: {len(test_samples)}")

        train_dataset_img = ImagePathDataset(train_samples)
        val_dataset_img = ImagePathDataset(val_samples)
        test_dataset_img = ImagePathDataset(test_samples)

        train_loader_img = DataLoader(
            train_dataset_img,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=image_collate,
        )
        val_loader_img = DataLoader(
            val_dataset_img,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=image_collate,
        )
        test_loader_img = DataLoader(
            test_dataset_img,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=image_collate,
        )

        processor = CLIPProcessor.from_pretrained(args.clip_model, use_fast=True)
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
        if use_data_parallel:
            clip_model = nn.DataParallel(clip_model, device_ids=gpu_ids)

        # Freeze CLIP parameters: only train the classifier.
        for p in clip_model.parameters():
            p.requires_grad = False

        train_features, train_labels = load_or_extract_clip_features(
            split_name="train",
            samples=train_samples,
            dataloader=train_loader_img,
            clip_model=clip_model,
            processor=processor,
            device=device,
            use_amp=use_amp,
            image_dir=args.image_dir,
            clip_model_name=args.clip_model,
            reextract=args.reextract,
        )
        val_features, val_labels = load_or_extract_clip_features(
            split_name="val",
            samples=val_samples,
            dataloader=val_loader_img,
            clip_model=clip_model,
            processor=processor,
            device=device,
            use_amp=use_amp,
            image_dir=args.image_dir,
            clip_model_name=args.clip_model,
            reextract=args.reextract,
        )
        test_features, test_labels = load_or_extract_clip_features(
            split_name="test",
            samples=test_samples,
            dataloader=test_loader_img,
            clip_model=clip_model,
            processor=processor,
            device=device,
            use_amp=use_amp,
            image_dir=args.test_image_dir,
            clip_model_name=args.clip_model,
            reextract=args.reextract,
        )

        feature_dim = train_features.size(1)
        classifier = CLIPClassifier(feature_dim=feature_dim, num_classes=40).to(device)
        if use_data_parallel:
            classifier = nn.DataParallel(classifier, device_ids=gpu_ids)

        train_feat_loader = DataLoader(
            FeatureDataset(train_features, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        val_feat_loader = DataLoader(
            FeatureDataset(val_features, val_labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        test_feat_loader = DataLoader(
            FeatureDataset(test_features, test_labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if args.use_wandb:
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group='teacher',
                name=args.wandb_run_name,
                config={
                    "image_dir": args.image_dir,
                    "test_image_dir": args.test_image_dir,
                    "clip_model": args.clip_model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "val_ratio": args.val_ratio,
                    "seed": args.seed,
                    "feature_dim": feature_dim,
                    "num_classes": 40,
                    "gpu_ids": gpu_ids,
                    "use_data_parallel": use_data_parallel,
                    "amp": use_amp,
                    "reextract": args.reextract,
                },
            )
            wandb_run.watch(classifier, log="all", log_freq=1000)

        best_val_acc = -1.0
        best_state = None

        for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
            train_loss, train_acc = train_one_epoch(classifier, train_feat_loader, optimizer, criterion, device, scaler, use_amp)
            val_loss, val_acc = evaluate(classifier, val_feat_loader, criterion, device, use_amp)
            test_loss, test_acc = evaluate(classifier, test_feat_loader, criterion, device, use_amp)

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "test/loss": test_loss,
                        "test/acc": test_acc,
                    },
                    step=epoch,
                )

                if args.wandb_log_weights_every > 0 and epoch % args.wandb_log_weights_every == 0:
                    weight_logs = {
                        f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())
                        for name, param in classifier.named_parameters()
                    }
                    wandb.log(weight_logs, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                state_dict = classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict()
                best_state = {k: v.cpu() for k, v in state_dict.items()}

        if best_state is None:
            state_dict = classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict()
            best_state = {k: v.cpu() for k, v in state_dict.items()}

        final_test_loss, final_test_acc = evaluate(classifier, test_feat_loader, criterion, device, use_amp)

        meta = {
            "clip_model": args.clip_model,
            "feature_dim": feature_dim,
            "num_classes": 40,
            "best_val_acc": best_val_acc,
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
            "image_dir": args.image_dir,
            "test_image_dir": args.test_image_dir,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
        }

        checkpoint = {
            "classifier_state_dict": best_state,
            "meta": meta,
        }
        torch.save(checkpoint, args.save_path)

        meta_path = str(Path(args.save_path).with_suffix(".json"))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved model to: {args.save_path}")
        print(f"Saved metadata to: {meta_path}")

        if wandb_run is not None:
            wandb_run.summary["best_val_acc"] = best_val_acc
            wandb_run.summary["final_test_acc"] = final_test_acc

            artifact_name = f"clip-classifier-{wandb_run.id}"
            model_artifact = wandb.Artifact(name=artifact_name, type="model", metadata=meta)
            model_artifact.add_file(args.save_path)
            model_artifact.add_file(meta_path)
            wandb_run.log_artifact(model_artifact)

    finally:
        # Always close wandb cleanly, even if training is interrupted.
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
