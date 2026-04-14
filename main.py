from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

import visualize_demo
import pointcloud_multiview


def _render_one_sample(task: Tuple[np.ndarray, int, int, str, str]) -> None:
    points, label_idx, whole_index, label_name, output_dir = task
    pointcloud_multiview.save_point_cloud_multiview_images(
        points=points,
        label_index=label_idx,
        index=whole_index,
        label=label_name,
        output_dir=output_dir,
        num_views=12,
        image_size=256,
        fov_degrees=45.0,
        point_radius=2,
        camera_distance=1.6,
    )


def transform_point_cloud_to_2d(h5_file_list: list, labels: dict, output_dir: str, num_workers: int = 4) -> None:
    """从 HDF5 文件中读取点云数据并可视化为 2D 图像"""
    tasks: list[Tuple[np.ndarray, int, int, str, str]] = []
    whole_index = 0

    for h5_file in h5_file_list:
        with h5py.File(h5_file, 'r') as f:
            print(f"文件: {h5_file}")
            print(list(f.keys()))    # 列出顶级数据集名

            for i in range(len(f['data'])):
                points = f['data'][i]
                label_idx = int(f['label'][i][0]) if f['label'].ndim > 1 else int(f['label'][i])
                label_name = labels.get(label_idx, "Unknown")

                whole_index += 1
                print(f"样本 {whole_index}: 标签索引={label_idx}, 标签名称={label_name}")

                # whole_index is assigned before parallel work starts,
                # so each sample keeps a stable globally unique index.
                tasks.append((points, label_idx, whole_index, label_name, output_dir))

    if num_workers <= 1:
        for task in tasks:
            _render_one_sample(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_render_one_sample, tasks))
                


if __name__ == "__main__": 
    labels = defaultdict(str)

    with open('modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as f:
        file_list = f.read().splitlines()
        for i, label in enumerate(file_list):
            labels[i] = label

    h5_file_list = [
        'modelnet40_ply_hdf5_2048/ply_data_train0.h5',
        'modelnet40_ply_hdf5_2048/ply_data_train1.h5',
        'modelnet40_ply_hdf5_2048/ply_data_train2.h5',
        'modelnet40_ply_hdf5_2048/ply_data_train3.h5',
        'modelnet40_ply_hdf5_2048/ply_data_train4.h5'
    ]

    transform_point_cloud_to_2d(h5_file_list, labels, 'output_images')

    test_h5_file_list = [
        'modelnet40_ply_hdf5_2048/ply_data_test0.h5',
        'modelnet40_ply_hdf5_2048/ply_data_test1.h5'
    ]
    transform_point_cloud_to_2d(test_h5_file_list, labels, 'test_output_images')