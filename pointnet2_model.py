from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    b = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    b, n, c = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, c)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    b, n, _ = xyz.shape
    s = new_xyz.shape[1]

    group_idx = torch.arange(n, dtype=torch.long, device=device).view(1, 1, n).repeat([b, s, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius * radius] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(b, s, 1).repeat([1, 1, nsample])
    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(new_xyz.shape[0], npoint, 1, 3)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    device = xyz.device
    b, n, _ = xyz.shape
    new_xyz = torch.zeros(b, 1, 3, device=device)
    grouped_xyz = xyz.view(b, 1, n, 3)

    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(b, 1, n, -1)], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: List[int],
        group_all: bool,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp_conv = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if xyz.dim() != 3:
            raise ValueError(f"xyz must be a 3D tensor, got shape {tuple(xyz.shape)}")

        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if self.npoint is None or self.radius is None or self.nsample is None:
                raise ValueError("npoint/radius/nsample must be set when group_all=False")
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, dim=2)[0]

        return new_xyz.permute(0, 2, 1).contiguous(), new_points


class PointNet2Student(nn.Module):
    def __init__(self, num_classes: int = 40, dropout: float = 0.4):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=131,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[-1] == 3 and xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1).contiguous()

        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.squeeze(-1)
        return self.classifier(global_feat)
