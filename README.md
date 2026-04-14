# 环境准备

pip install -r requirements.txt

# 数据准备

下载 ModelNet40 数据集并解压到 `modelnet40_ply_hdf5_2048` 目录下

运行：
``` bash
python main.py
```
output_images 目录下会生成训练集的2D图像
test_output_images 目录下会生成测试集的2D图像

# Teacher 模型训练
运行：
``` bash
python train.py \
  --image_dir output_images \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 200 \
  --save_path clip_classifier_40cls.pth \
  --clip_model openai/clip-vit-large-patch14 \
  --wandb_run_name clip-vit-large-patch14
```
# Student 模型训练
运行：
``` bash
python student.py \
  --pointcloud_dir pointcloud_data \
  --teacher_checkpoint clip_classifier_40cls.pth \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 200 \
  --save_path student_pointnet2_distill.pth \
  --wandb_run_name pointnet_student_training
```
