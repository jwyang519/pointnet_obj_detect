# PointNet++ for Table Detection and Segmentation

This repository implements two distinct pipelines for 3D point cloud analysis focused on tables:

1. **Pipeline A**: Binary classification of entire point clouds (table present vs. not present)
2. **Pipeline C**: Binary semantic segmentation of point clouds (labeling individual points as table or background)

Both pipelines use PointNet++ architectures and are trained on the SUN3D dataset.

## 🏠 Project Structure

- **pipeline_a/**: Binary classification models and tools
- **pipeline_c/**: Semantic segmentation models and tools
- **data_utils/**: Data preparation and processing utilities
- **models/**: PointNet++ model implementations
- **visualizer/**: Point cloud visualization tools
- **weights/**: Saved model weights
- **log/**: Training logs and checkpoints

## 📦 Dataset

The project uses the SUN3D dataset, processed into HDF5 (.h5) files. The dataset is prepared using scripts in the `data_utils/` directory:

```
CW2-Dataset/
  ├── sun3d_train_fixed.h5  # Training dataset (MIT scenes)
  ├── sun3d_test_fixed.h5   # Testing dataset (Harvard scenes)
  └── ucl_data_fixed.h5     # Additional UCL dataset
```

## 🧠 Models

Both pipelines use variants of the PointNet++ architecture:

- **Pipeline A**: Uses PointNet++ SSG (Single Scale Grouping) for classification (`pointnet2_cls_ssg.py`)
- **Pipeline C**: Uses PointNet++ for semantic segmentation (`pointnet2_sem_seg.py`)

Other available models include:
- PointNet++ MSG (Multi-Scale Grouping) variants
- PointNet models for classification and segmentation

## 🔍 Pipeline A: Table Classification

Pipeline A focuses on classifying entire point clouds as either containing a table (1) or not (0).

### Training

Basic usage:
```bash
python pipeline_a/train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir table_classification
```

For K-fold cross-validation:
```bash
python pipeline_a/kfold_train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir kfold_experiment
```

Advanced parameters:
```bash
python pipeline_a/kfold_train.py \
  --train_file CW2-Dataset/sun3d_train_fixed.h5 \
  --log_dir kfold_experiment \
  --n_folds 5 \
  --epoch 100 \
  --batch_size 24 \
  --learning_rate 0.0001 \
  --scheduler cosine \
  --lr_decay 0.7 \
  --step_size 20 \
  --num_point 1024 \
  --use_weights
```

### Testing

```bash
python pipeline_a/test.py --log_dir table_classification --test_file CW2-Dataset/sun3d_test_fixed.h5 --ucl_file CW2-Dataset/ucl_data_fixed.h5
```

### Key Parameters

- `--train_file`: Path to training H5 file
- `--test_file`: Path to testing H5 file
- `--log_dir`: Directory name for saving logs and models
- `--batch_size`: Batch size for training (default: 24)
- `--learning_rate`: Initial learning rate
- `--num_point`: Number of points per point cloud (default: 1024)
- `--use_weights`: Whether to use class weights (helps with imbalanced datasets)

### Output

- Model weights and checkpoints in `log/table_classification/<log_dir>/checkpoints/`
- Training curves and metrics in `log/table_classification/<log_dir>/`
- K-fold cross-validation summaries in `log/table_classification/kfold_<log_dir>/`

## 🧩 Pipeline C: Semantic Segmentation

Pipeline C performs point-level semantic segmentation to identify which points in a point cloud belong to tables.

### Training

```bash
python pipeline_c/train_sun3d.py --h5_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir semantic_segmentation --batch_size 16 --epoch 50 --learning_rate 0.001 --use_augmentation
```

### Testing

```bash
python pipeline_c/test_sun3d.py --log_dir semantic_segmentation_TIMESTAMP --h5_file CW2-Dataset/sun3d_test_fixed.h5 --batch_size 16
```

For visualization:
```bash
python pipeline_c/test_sun3d.py --log_dir semantic_segmentation_TIMESTAMP --h5_file CW2-Dataset/sun3d_test_fixed.h5 --batch_size 16 --visual
```

### Key Parameters

- `--h5_file`: Path to H5 file with point clouds and labels
- `--batch_size`: Batch size for training (default: 16)
- `--epoch`: Number of epochs (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--npoint`: Number of points (default: 1024)
- `--use_augmentation`: Enable data augmentation (recommended)

### Output

- Model weights and checkpoints in `log/sun3d_binary_seg/<log_dir>/checkpoints/`
- Training curves and metrics in `log/sun3d_binary_seg/<log_dir>/`
- Visualization data (if `--visual` is used) in `log/sun3d_binary_seg/<log_dir>/visual/`

## 🧰 Data Utilities

### Dataset Preparation

The dataset is prepared from the SUN3D dataset using the following process:

1. Process depth maps from SUN3D using DepthTSDF format
2. Generate point clouds from the depth data
3. Combine point clouds and labels into H5 files

If raw depth maps need to be converted to DepthTSDF format:

```bash
# Convert depth to TSDF format for specific building
python data_utils/depth_to_tsdf.py --building harvard_tea_2

# Generate the dataset
python data_utils/prepare_sun3d_dataset.py --prefix mit --output CW2-Dataset/sun3d_train.h5
python data_utils/prepare_sun3d_dataset.py --prefix harvard --output CW2-Dataset/sun3d_test.h5
```

## 🎨 Visualization

The repository includes tools for visualizing point clouds, ground truth, and predictions:

1. Run testing with visualization enabled:
```bash
python pipeline_c/test_sun3d.py --log_dir semantic_segmentation_TIMESTAMP --h5_file CW2-Dataset/sun3d_test_fixed.h5 --visual
```

2. Convert to interactive HTML visualizations:
```bash
python visualizer/convert_txt_to_html.py --log_dir semantic_segmentation_TIMESTAMP --output_dir point_cloud_visualizations
```

3. Package visualizations for download:
```bash
python visualizer/package_visualizations.py --log_dir semantic_segmentation_TIMESTAMP
```

## 🏆 Pretrained Models

Pretrained models are available in the `weights/` directory:

- `pipeline_a_best_model.pth`: Binary classification model (table vs. no-table)
- `pipeline_c_best_model.pth`: Semantic segmentation model (table vs. background)

To use these models:

```bash
# For pipeline A
python pipeline_a/test.py --log_dir custom --test_file CW2-Dataset/sun3d_test_fixed.h5

# For pipeline C
python pipeline_c/test_sun3d.py --log_dir custom --h5_file CW2-Dataset/sun3d_test_fixed.h5
```

## 🛠️ Implementation Details

### PointNet++ Architecture

Both pipelines use variants of the PointNet++ architecture:

- **PointNet++ SSG** (Single Scale Grouping): Used in pipeline A for classification
- **PointNet++ Semantic Segmentation**: Used in pipeline C for point-level segmentation

The models share a core architecture that uses:
1. Set abstraction layers to hierarchically capture features
2. Feature propagation layers to upsample features for segmentation
3. MLP layers for final classification/segmentation

### Loss Functions

- **Pipeline A**: Uses Cross-Entropy loss with optional class weighting
- **Pipeline C**: Uses Cross-Entropy loss with optional class weighting on point-level predictions

### Data Augmentation

Both pipelines support data augmentation during training:
- Random scaling
- Random translation
- Random point dropout
- Random jittering

## 📊 Performance

### Pipeline A (Classification)
- **Accuracy**: ~85% overall accuracy on test dataset
- **Table Class Accuracy**: ~97% (high recall for tables)
- **Non-Table Class Accuracy**: ~50% (challenges with false positives)

### Pipeline C (Segmentation)
- **Overall Accuracy**: ~85% point-level accuracy
- **Mean IoU**: ~0.85 across classes
- **Table IoU**: ~0.74
- **Background IoU**: ~0.97

## 🔧 Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA compatible GPU (recommended)
- Additional requirements:
  - numpy
  - h5py
  - matplotlib
  - sklearn
  - tqdm
  - plotly (for visualization) 