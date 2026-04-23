# 环境准备

pip install -r requirements.txt

创建.env，并添加以下内容：

HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here

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
  --test_image_dir test_output_images \
  --gpu_ids 0 \
  --num_classes 15 \
  --batch_size 256 \
  --num_workers 4 \
  --epochs 100 \
  --save_path clip_classifier_40cls.pth \
  --clip_model openai/clip-vit-large-patch14 \
  --wandb_run_name clip_teacher_training_newdata_classifier
```
# Student 模型训练
运行：
``` bash
python student.py \
  --modelnet_root scanobjectnn/main_split_nobg \
  --train_h5_paths \
    scanobjectnn/main_split_nobg/training_objectdataset.h5 \
    scanobjectnn/main_split_nobg/training_objectdataset_augmented25_norot.h5 \
  --test_h5_paths \
    scanobjectnn/main_split_nobg/test_objectdataset.h5 \
    scanobjectnn/main_split_nobg/test_objectdataset_augmented25_norot.h5 \
  --shape_names_path scanobjectnn/shape_names.txt \
  --teacher_checkpoint clip_classifier_40cls.pth \
  --teacher_clip_model openai/clip-vit-large-patch14 \
  --num_classes 15 \
  --batch_size 64 \
  --num_workers 4 \
  --epochs 100 \
  --save_path student_pointnet2_distill.pth \
  --teacher_image_batch_size 256 \
  --wandb_run_name pointnet_student_training_newdata_distill_mode_0 \
  --distill_mode 0
```

distill_mode 0: 特征平均融合， 1: 去除预测失败的样本，剩余样本特征平均融合， 2: 置信度特征融合
