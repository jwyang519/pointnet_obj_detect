# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Update
**2021/03/27:** 

(1) Release pre-trained models for semantic segmentation, where PointNet++ can achieve **53.5\%** mIoU.

(2) Release pre-trained models for classification and part segmentation in `log/`.

**2021/03/20:** Update codes for classification, including:

(1) Add codes for training **ModelNet10** dataset. Using setting of ``--num_category 10``. 

(2) Add codes for running on CPU only. Using setting of ``--use_cpu``. 

(3) Add codes for offline data preprocessing to accelerate training. Using setting of ``--process_data``. 

(4) Add codes for training with uniform sampling. Using setting of ``--use_uniform_sample``. 

**2019/11/26:**

(1) Fixed some errors in previous codes and added data augmentation tricks. Now classification by only 1024 points can achieve **92.8\%**! 

(2) Added testing codes, including classification and segmentation, and semantic segmentation with visualization. 

(3) Organized all models into `./models` files for easy using.

## Table Classification Pipeline

This section describes the updated table classification pipeline for point cloud binary classification using the SUN3D and UCL datasets.

### Dataset Generation and Processing

The pipeline includes tools for generating and inspecting table classification datasets:

```bash
# Generate fixed datasets with improved point cloud processing
python regenerate_datasets.py

# Check dataset statistics without regenerating
python regenerate_datasets.py --check_only

# Only regenerate SUN3D datasets
python regenerate_datasets.py --sun3d_only

# Only regenerate UCL dataset
python regenerate_datasets.py --ucl_only
```

### Training the Table Classifier

The `pipeline_a/train.py` script provides a comprehensive set of options for training a binary table classifier:

```bash
# Basic training with fixed datasets
python pipeline_a/train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir table_classifier_fixed --epoch 15

# Training with lower learning rate and class weighting
python pipeline_a/train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir table_classifier_weighted \
    --learning_rate 0.0001 --non_table_weight 10.0 --table_weight 1.0 --epoch 20

# Training with cosine annealing learning rate schedule
python pipeline_a/train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir table_classifier_cosine \
    --learning_rate 0.0001 --scheduler cosine --epoch 30 --non_table_weight 5.0
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_file` | Path to the training dataset | `../dataset/sun3d_train.h5` |
| `--test_file` | Path to separate test file (not used in training) | `../dataset/sun3d_test.h5` |
| `--val_split` | Fraction of training data to use for validation | `0.2` |
| `--epoch` | Number of epochs to train | `100` |
| `--batch_size` | Batch size | `24` |
| `--num_point` | Number of points per point cloud | `1024` |
| `--learning_rate` | Initial learning rate | `0.0001` |
| `--scheduler` | Learning rate scheduler type (step, cosine, plateau, none) | `step` |
| `--lr_decay` | Decay rate for scheduler | `0.7` |
| `--step_size` | Step size for StepLR scheduler | `20` |
| `--min_lr` | Minimum learning rate | `1e-6` |
| `--non_table_weight` | Weight for non-table class (class 0) | `1.0` |
| `--table_weight` | Weight for table class (class 1) | `1.0` |
| `--use_weights` | Use class weights | `True` |
| `--log_dir` | Experiment directory | `None` (auto-generated timestamp) |

### Learning Rate Schedulers

The training script supports multiple learning rate schedulers:

1. **Step Scheduler** (`--scheduler step`): Reduces the learning rate by a gamma factor every `step_size` epochs
2. **Cosine Annealing** (`--scheduler cosine`): Gradually reduces learning rate following a cosine curve
3. **Reduce on Plateau** (`--scheduler plateau`): Reduces learning rate when validation loss stops improving
4. **None** (`--scheduler none`): Uses constant learning rate throughout training

### Performance Visualization

After training, the script automatically generates visualization plots:

1. **Training Curves**: A comprehensive plot showing training and validation metrics:
   - Loss vs. Epochs
   - Overall Accuracy vs. Epochs
   - Table Class Accuracy vs. Epochs
   - Non-Table Class Accuracy vs. Epochs

2. **Learning Rate Curve**: Shows how the learning rate changed over training epochs

These plots are saved in the experiment directory specified by `--log_dir`.

## Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (ModelNet10/40)
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
You can run different modes with following codes. 
* If you want to use offline processing of data, you can use `--process_data` in the first run. You can download pre-processd data [here](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing) and save it in `data/modelnet40_normal_resampled/`.
* If you want to train on ModelNet10, you can use `--num_category 10`.
```shell
# ModelNet40
## Select different models in ./models 

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps

# ModelNet10
## Similar setting like ModelNet40, just using --num_category 10

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### Performance
| Model | Accuracy |
|--|--|
| PointNet (Official) |  89.2|
| PointNet2 (Official) | 91.9 |
| PointNet (Pytorch without normal) |  90.6|
| PointNet (Pytorch with normal) |  91.4|
| PointNet2_SSG (Pytorch without normal) |  92.2|
| PointNet2_SSG (Pytorch with normal) |  92.4|
| PointNet2_MSG (Pytorch with normal) |  **92.8**|

## Part Segmentation (ShapeNet)
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```
### Performance
| Model | Inctance avg IoU| Class avg IoU 
|--|--|--|
|PointNet (Official)	|83.7|80.4	
|PointNet2 (Official)|85.1	|81.9	
|PointNet (Pytorch)|	84.3	|81.1|	
|PointNet2_SSG (Pytorch)|	84.9|	81.8	
|PointNet2_MSG (Pytorch)|	**85.4**|	**82.5**	


## Semantic Segmentation (S3DIS)
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```
Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).

### Performance
|Model  | Overall Acc |Class avg IoU | Checkpoint 
|--|--|--|--|
| PointNet (Pytorch) | 78.9 | 43.7| [40.7MB](log/sem_seg/pointnet_sem_seg) |
| PointNet2_ssg (Pytorch) | **83.0** | **53.5**| [11.2MB](log/sem_seg/pointnet2_sem_seg) |

## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
```
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```
## Selected Projects using This Codebase
* [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
* [PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud](https://github.com/lly007/PointStructuringNet)
* [Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22](https://github.com/dvlab-research/stratified-transformer)

# PointNet++ for Binary Segmentation using SUN3D Dataset

This repository contains an implementation of PointNet++ for binary segmentation on the SUN3D dataset. The goal is to classify each point in a 3D point cloud as either "table" (label 1) or "background" (label 0).

## Overview

We've implemented a binary point cloud segmentation pipeline using the [PointNet++](https://arxiv.org/abs/1706.02413) architecture. The pipeline includes:

1. **Dataset preparation** script that converts SUN3D RGB-D images and polygon annotations into labeled point clouds
2. **Custom PyTorch dataset loader** for the SUN3D dataset that handles point clouds with binary labels
3. **PointNet++ segmentation model** adapted for binary segmentation
4. **Training and testing scripts** for the binary segmentation task

## Dataset Preparation

The SUN3D dataset consists of RGB-D images with polygon annotations for tables. We provide a script to convert these into labeled point clouds with binary segmentation labels.

Input data should have the following structure:

```
data_dir/
├── building_name/           # e.g., mit_32_d507
│   └── scene_name/         # e.g., d507_2
│       ├── depthTSDF/      # Depth images
│       │   └── *.png       # 16-bit depth images
│       ├── image/          # RGB images (not used in processing)
│       ├── labels/         # Table annotations
│       │   └── tabletop_labels.dat  # Pickle file with polygon coordinates
│       └── intrinsics.txt  # Camera parameters
└── ...                     # Other buildings/scenes
```

To prepare the dataset:

```bash
python data_utils/prepare_sun3d_dataset.py \
    --data_dir CW2-Dataset/data \
    --output_dir CW2-Dataset \
    --max_points 1024
```

This will create HDF5 files containing point clouds with binary labels:
- `sun3d_train.h5`: Training data from MIT scenes
- `sun3d_test.h5`: Testing data from Harvard scenes

## Training

To train the model:

```bash
python train_sun3d.py \
    --model pointnet2_sem_seg \
    --h5_file CW2-Dataset/sun3d_train.h5 \
    --log_dir sun3d_binary_seg \
    --batch_size 16 \
    --npoint 1024 \
    --epoch 100
```

Training parameters:
- `--model`: Model architecture to use (default: `pointnet2_sem_seg`)
- `--h5_file`: Path to the training HDF5 file
- `--log_dir`: Directory name for saving logs and checkpoints
- `--batch_size`: Batch size for training (default: 16)
- `--npoint`: Number of points per point cloud (default: 1024)
- `--epoch`: Number of training epochs (default: 32)

## Testing

To evaluate the trained model:

```bash
python test_sun3d.py \
    --model pointnet2_sem_seg \
    --h5_file CW2-Dataset/sun3d_test.h5 \
    --log_dir sun3d_binary_seg \
    --visual
```

Testing parameters:
- `--model`: Model architecture to use (should match training)
- `--h5_file`: Path to the testing HDF5 file
- `--log_dir`: Directory name where the trained model is saved
- `--visual`: Enable visualization output (saves point clouds and predictions)

## Results Visualization

When running testing with the `--visual` flag, the script will save point clouds and their predictions in the `log/sun3d_seg/[log_dir]/visual` directory. These files can be visualized using any 3D point cloud viewer.

## Acknowledgements

This repository is based on [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by yanx27. The original implementation has been adapted for binary segmentation on the SUN3D dataset.

## License
MIT License