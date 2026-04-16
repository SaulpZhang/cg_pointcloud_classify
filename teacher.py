from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pointcloud_multiview import save_point_cloud_multiview_images
from train import CLIPClassifier


@dataclass
class TeacherPrediction:
    majority_label: int
    view_predictions: List[int]
    view_probabilities: List[List[float]]
    vote_counts: Dict[int, int]
    vote_distribution: Dict[int, float]
    mean_probabilities: List[float]
    image_paths: List[str]


class TeacherModel:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        clip_model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        use_amp: bool = True,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "classifier_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing key: classifier_state_dict")

        meta = checkpoint.get("meta", {})
        self.clip_model_name = clip_model_name or meta.get("clip_model", "openai/clip-vit-base-patch32")

        classifier_state = checkpoint["classifier_state_dict"]
        linear_weight = classifier_state.get("classifier.4.weight")
        if linear_weight is None:
            linear_weight = classifier_state.get("classifier.2.weight")
        if linear_weight is None:
            classifier_weight_keys = [
                k for k in classifier_state.keys() if k.startswith("classifier.") and k.endswith(".weight")
            ]
            if not classifier_weight_keys:
                raise KeyError("classifier_state_dict missing classifier.*.weight keys")
            last_key = sorted(classifier_weight_keys, key=lambda x: int(x.split(".")[1]))[-1]
            linear_weight = classifier_state[last_key]

        feature_dim = int(meta.get("feature_dim", linear_weight.shape[1]))
        checkpoint_num_classes = int(meta.get("num_classes", linear_weight.shape[0]))
        if num_classes is None:
            num_classes = checkpoint_num_classes
        elif int(num_classes) != checkpoint_num_classes:
            print(
                f"Warning: requested num_classes={num_classes} differs from checkpoint num_classes={checkpoint_num_classes}. "
                "Using the requested value for classifier construction."
            )
        num_classes = int(num_classes)

        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name, use_fast=True)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.classifier = CLIPClassifier(feature_dim=feature_dim, num_classes=num_classes).to(self.device)

        load_result = self.classifier.load_state_dict(classifier_state, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys when loading classifier weights: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: unexpected keys when loading classifier weights: {load_result.unexpected_keys}")

        self.clip_model.eval()
        self.classifier.eval()

        for p in self.clip_model.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _extract_features_from_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            vision_outputs = self.clip_model.vision_model(pixel_values)
            image_features = vision_outputs.last_hidden_state[:, 0, :]
            image_features = self.clip_model.visual_projection(image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def predict_image_paths(self, image_paths: List[str], batch_size: int = 64) -> TeacherPrediction:
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        all_probs: List[torch.Tensor] = []
        all_preds: List[int] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            features = self._extract_features_from_images(batch_images)
            # Ensure features are in float32 before passing to classifier (important for AMP)
            features = features.float()
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.extend(preds.cpu().tolist())

        probs_all = torch.cat(all_probs, dim=0)
        mean_probs = probs_all.mean(dim=0)

        num_classes = int(mean_probs.shape[0])
        pred_tensor = torch.tensor(all_preds, dtype=torch.long)
        vote_tensor = torch.bincount(pred_tensor, minlength=num_classes)

        majority_label = int(vote_tensor.argmax().item())
        total_votes = max(int(vote_tensor.sum().item()), 1)

        vote_counts = {i: int(v) for i, v in enumerate(vote_tensor.tolist()) if v > 0}
        vote_distribution = {i: float(v / total_votes) for i, v in enumerate(vote_tensor.tolist()) if v > 0}

        return TeacherPrediction(
            majority_label=majority_label,
            view_predictions=all_preds,
            view_probabilities=probs_all.tolist(),
            vote_counts=vote_counts,
            vote_distribution=vote_distribution,
            mean_probabilities=mean_probs.tolist(),
            image_paths=image_paths,
        )

    def predict_point_cloud(
        self,
        points: np.ndarray,
        index: int,
        label: str,
        output_dir: str,
        label_index: int = -1,
        num_views: int = 12,
        image_size: int = 256,
        fov_degrees: float = 45.0,
        point_radius: int = 2,
        camera_distance: float = 1.6,
        batch_size: int = 64,
    ) -> TeacherPrediction:
        image_paths = save_point_cloud_multiview_images(
            points=points,
            label_index=label_index,
            index=index,
            label=label,
            output_dir=output_dir,
            num_views=num_views,
            image_size=image_size,
            fov_degrees=fov_degrees,
            point_radius=point_radius,
            camera_distance=camera_distance,
        )
        return self.predict_image_paths(image_paths=image_paths, batch_size=batch_size)


def load_teacher(
    checkpoint_path: str,
    device: Optional[str] = None,
    clip_model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    use_amp: bool = True,
) -> TeacherModel:
    return TeacherModel(
        checkpoint_path=checkpoint_path,
        device=device,
        clip_model_name=clip_model_name,
        num_classes=num_classes,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a teacher classifier checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="clip_classifier_40cls.pth")
    parser.add_argument("--clip_model_name", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=40, help="Number of classes expected by the classifier")
    args = parser.parse_args()

    ckpt = args.checkpoint_path
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    teacher = load_teacher(ckpt, clip_model_name=args.clip_model_name, num_classes=args.num_classes)
    print(f"Teacher ready on {teacher.device}, CLIP model: {teacher.clip_model_name}")
