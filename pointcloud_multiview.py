"""Multi-view 3D point cloud to 2D image renderer.

This module provides a single main function that takes a point cloud with
shape (N, 3), renders multiple perspective views from cameras sampled on a
sphere, and saves the resulting images to the given output directory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center the point cloud at the origin and scale it into a unit cube.

    The cloud is first shifted by the center of its axis-aligned bounding box,
    then uniformly scaled by the longest box side so that it fits inside a cube
    with side length 1, centered at the origin.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    scale = float(np.max(maximum - minimum))
    if scale <= 0.0:
        scale = 1.0

    return (points - center) / scale


def fibonacci_sphere(num_views: int) -> np.ndarray:
    """Generate approximately uniform directions on a sphere."""
    if num_views <= 0:
        raise ValueError("num_views must be positive")

    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    if num_views == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    for i in range(num_views):
        y = 1.0 - 2.0 * (i / (num_views - 1))
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        directions.append([x, y, z])

    return np.asarray(directions, dtype=np.float32)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a world-to-camera rotation matrix using a standard look-at setup."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(forward, fallback_up)) > 0.99:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
    right = right / (right_norm + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    # World to camera transform for a camera looking along +Z in camera space.
    return np.stack([right, true_up, forward], axis=0)


def _project_points(
    points: np.ndarray,
    camera_position: np.ndarray,
    target: np.ndarray,
    image_size: int,
    fov_degrees: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates with a z-buffer friendly depth."""
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rotation = _look_at(camera_position, target, up)
    camera_space = (points - camera_position) @ rotation.T

    focal = 0.5 * image_size / math.tan(math.radians(fov_degrees) * 0.5)
    cx = (image_size - 1) * 0.5
    cy = (image_size - 1) * 0.5

    z = camera_space[:, 2]
    valid = z > 1e-6
    x = np.full(points.shape[0], np.nan, dtype=np.float32)
    y = np.full(points.shape[0], np.nan, dtype=np.float32)

    x[valid] = focal * (camera_space[valid, 0] / z[valid]) + cx
    y[valid] = focal * (-camera_space[valid, 1] / z[valid]) + cy
    return np.stack([x, y], axis=1), z


def _render_single_view(
    points: np.ndarray,
    camera_position: np.ndarray,
    target: np.ndarray,
    image_size: int,
    fov_degrees: float,
    point_radius: int,
) -> np.ndarray:
    """Render one view using a simple z-buffer rasterizer."""
    projected, depth = _project_points(points, camera_position, target, image_size, fov_degrees)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    depth_buffer = np.full((image_size, image_size), np.inf, dtype=np.float32)

    valid = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    if not np.any(valid):
        return image

    valid_indices = np.where(valid)[0]
    depth_valid = depth[valid_indices]

    # Sort by depth so nearer points overwrite farther ones consistently.
    order = np.argsort(depth_valid)[::-1]
    valid_indices = valid_indices[order]

    max_depth = float(np.max(depth[valid]))
    min_depth = float(np.min(depth[valid]))
    depth_span = max(max_depth - min_depth, 1e-6)

    for idx in valid_indices:
        px, py = projected[idx]
        z = depth[idx]
        if not np.isfinite(px) or not np.isfinite(py):
            continue

        x0 = int(round(px))
        y0 = int(round(py))
        if x0 < -point_radius or x0 >= image_size + point_radius:
            continue
        if y0 < -point_radius or y0 >= image_size + point_radius:
            continue

        brightness = 1.0 - (z - min_depth) / depth_span
        brightness = float(np.clip(brightness, 0.2, 1.0))
        color = np.array([brightness * 255.0] * 3, dtype=np.uint8)

        for dy in range(-point_radius, point_radius + 1):
            yy = y0 + dy
            if yy < 0 or yy >= image_size:
                continue
            for dx in range(-point_radius, point_radius + 1):
                xx = x0 + dx
                if xx < 0 or xx >= image_size:
                    continue
                if dx * dx + dy * dy > point_radius * point_radius:
                    continue
                if z < depth_buffer[yy, xx]:
                    depth_buffer[yy, xx] = z
                    image[yy, xx] = color

    return image


def save_point_cloud_multiview_images(
    points: np.ndarray,
    label_index: int,
    index: int,
    label: str,
    output_dir: str,
    num_views: int = 12,
    image_size: int = 256,
    fov_degrees: float = 45.0,
    point_radius: int = 2,
    camera_distance: Optional[float] = 1.6,
) -> List[str]:
    """Render a point cloud from multiple spherical views and save PNG images.

    Args:
        points: numpy array with shape (N, 3).
        label_index: numeric label index used in the output file name.
        index: sample index used in the output file name.
        label: class label used in the output file name.
        output_dir: directory where images will be saved.
        num_views: number of camera views sampled on the sphere.
        image_size: output image resolution (square image).
        fov_degrees: camera field of view in degrees.
        point_radius: rasterized point radius in pixels.
        camera_distance: distance from camera to object center. Smaller values make
            objects occupy more of the image. If None, defaults to 1.6.

    Returns:
        A list of saved image file paths.
    """
    normalized_points = normalize_point_cloud(points)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    directions = fibonacci_sphere(num_views)
    file_paths: List[str] = []
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Place cameras closer to increase object occupancy in the rendered image.
    if camera_distance is None:
        camera_distance = 1.6
    camera_distance = float(max(camera_distance, 1.1))

    for view_id, direction in enumerate(directions):
        camera_position = target + direction * camera_distance
        image = _render_single_view(
            normalized_points,
            camera_position=camera_position,
            target=target,
            image_size=image_size,
            fov_degrees=fov_degrees,
            point_radius=point_radius,
        )

        file_name = f"{label}_{label_index}_{index}_{view_id}.png"
        file_path = output_path / file_name
        plt.imsave(file_path, image)
        file_paths.append(str(file_path))

    return file_paths
