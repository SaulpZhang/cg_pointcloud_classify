from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

import pointcloud_multiview
import visualize_demo


def load_shape_names(shape_names_path: str) -> List[str]:
    path = Path(shape_names_path)
    if not path.exists():
        raise FileNotFoundError(f"shape_names.txt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_one_sample(h5_path: str, sample_index: int) -> Tuple[np.ndarray, int]:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f or "label" not in f:
            raise KeyError("H5 file must contain 'data' and 'label' datasets")

        data = cast(h5py.Dataset, f["data"])
        labels = cast(h5py.Dataset, f["label"])

        total = int(data.shape[0])
        if sample_index < 0 or sample_index >= total:
            raise IndexError(f"sample_index out of range: {sample_index}, total={total}")

        points = np.asarray(data[sample_index], dtype=np.float32)
        label_raw = labels[sample_index]
        if np.ndim(label_raw) > 0:
            label_idx = int(np.asarray(label_raw).reshape(-1)[0])
        else:
            label_idx = int(label_raw)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shape (N, 3+) but got {points.shape}")

    return points[:, :3], label_idx


def find_first_sample_index_by_label(
    h5_path: str,
    target_label_name: str,
    shape_names_path: str,
) -> Optional[int]:
    shape_names = load_shape_names(shape_names_path)
    target = target_label_name.strip().lower()
    if not target:
        raise ValueError("target_label_name must not be empty")

    label_to_id = {name.lower(): idx for idx, name in enumerate(shape_names)}
    if target not in label_to_id:
        return None
    target_label_idx = label_to_id[target]

    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        labels = cast(h5py.Dataset, f["label"])
        labels_np = np.asarray(labels)

    labels_flat = labels_np.reshape(-1)
    matched = np.where(labels_flat == target_label_idx)[0]
    if matched.size == 0:
        return None
    return int(matched[0])


def show_multiview_grid(image_paths: List[str], rows: int = 4, cols: int = 3) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.5))
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < len(image_paths):
            img = plt.imread(image_paths[i])
            ax.imshow(img)
        ax.axis("off")

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one point cloud from H5, render in 3D and generate/show 4x3 multiview 2D images"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="modelnet40_ply_hdf5_2048/ply_data_train0.h5",
        help="Path to a source H5 file",
    )
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index inside the H5 file")
    parser.add_argument(
        "--target_label_name",
        type=str,
        default="airplane",
        help="If --find_by_label is enabled, pick the first sample with this label name",
    )
    parser.add_argument(
        "--find_by_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically find first sample index by target label name",
    )
    parser.add_argument(
        "--shape_names_path",
        type=str,
        default="modelnet40_ply_hdf5_2048/shape_names.txt",
        help="Path to shape_names.txt for label name lookup",
    )
    parser.add_argument("--output_dir", type=str, default="demo_multiview_output", help="Directory to save multiview PNG images")
    parser.add_argument("--num_views", type=int, default=12, help="Number of rendered views (default 12 for 4x3 grid)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fov_degrees", type=float, default=45.0)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--camera_distance", type=float, default=1.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.find_by_label:
        found = find_first_sample_index_by_label(
            h5_path=args.h5_path,
            target_label_name=args.target_label_name,
            shape_names_path=args.shape_names_path,
        )
        if found is None:
            raise ValueError(
                f"Could not find label '{args.target_label_name}' in {args.h5_path} using {args.shape_names_path}"
            )
        args.sample_index = found

    points, label_idx = load_one_sample(args.h5_path, args.sample_index)
    shape_names = load_shape_names(args.shape_names_path)
    label_name = shape_names[label_idx] if 0 <= label_idx < len(shape_names) else f"class_{label_idx}"

    print(f"Loaded sample: index={args.sample_index}, label_idx={label_idx}, label_name={label_name}, points={points.shape}")

    # 1) 3D render via visualize_demo (Open3D)
    visualize_demo.viz_open3d(points, title=f"3D Point Cloud: {label_name} ({label_idx})")

    # 2) Multiview processing via pointcloud_multiview
    image_paths = pointcloud_multiview.save_point_cloud_multiview_images(
        points=points,
        label_index=label_idx,
        index=args.sample_index,
        label=label_name,
        output_dir=args.output_dir,
        num_views=args.num_views,
        image_size=args.image_size,
        fov_degrees=args.fov_degrees,
        point_radius=args.point_radius,
        camera_distance=args.camera_distance,
    )

    print(f"Saved {len(image_paths)} images to: {args.output_dir}")

    # 3) Display 2D multiview images in 4x3 layout
    show_multiview_grid(image_paths, rows=4, cols=3)


if __name__ == "__main__":
    main()
