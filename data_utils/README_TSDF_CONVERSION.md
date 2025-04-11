# Depth to TSDF Conversion for SUN3D Dataset

This document explains how to handle the inconsistency in the SUN3D dataset where some scenes (like `harvard_tea_2`) use raw depth maps instead of the DepthTSDF format used by other scenes. This inconsistency can cause poor performance on certain test scenes.

## The Problem

The dataset includes two different types of depth data:
- Most scenes use `depthTSDF` format
- Some scenes (notably `harvard_tea_2`) only have raw `depth` images

This inconsistency leads to:
1. Different depth value ranges and meaning
2. Errors during dataset preparation
3. Poor performance during testing

## The Solution

We've created two scripts to solve this problem:

1. `depth_to_tsdf.py`: Converts raw depth maps to DepthTSDF format
2. Updated `prepare_sun3d_dataset.py`: Now checks for and prefers DepthTSDF data, with fallback to raw depth

## How to Use

### Step 1: Convert Depth to TSDF

First, check which scenes need conversion:

```bash
python data_utils/prepare_sun3d_dataset.py --check_only
```

If you see warnings about scenes with depth but no depthTSDF, run the conversion script:

```bash
# Convert all scenes that need conversion
python data_utils/depth_to_tsdf.py

# Convert a specific building (recommended for harvard_tea_2)
python data_utils/depth_to_tsdf.py --building harvard_tea_2
```

### Step 2: Regenerate the Dataset

After converting the depth data, regenerate the dataset:

```bash
# Generate training dataset (MIT scenes)
python data_utils/prepare_sun3d_dataset.py --prefix mit --output CW2-Dataset/sun3d_train.h5

# Generate testing dataset (Harvard scenes)
python data_utils/prepare_sun3d_dataset.py --prefix harvard --output CW2-Dataset/sun3d_test.h5
```

### Step 3: Retrain and Test

Train your model with the improved dataset:

```bash
python train_sun3d.py --model pointnet2_sem_seg --h5_file CW2-Dataset/sun3d_train.h5 --log_dir sun3d_improved --batch_size 4 --npoint 1024 --use_augmentation
```

Test your model:

```bash
python test_sun3d.py --model pointnet2_sem_seg --h5_file CW2-Dataset/sun3d_test.h5 --log_dir sun3d_improved_TIMESTAMP --batch_size 4 --npoint 1024
```

## Advanced Options

### Creating Empty Labels

If a scene is missing labels, you can create empty placeholder labels:

```bash
python data_utils/depth_to_tsdf.py --building harvard_tea_2 --create_labels
```

Note: These will be all-background labels. You'll need to manually annotate tables for accurate results.

### Processing a Specific Scene

To process just one scene:

```bash
python data_utils/prepare_sun3d_dataset.py --scene mit_76_studyroom/76-1studyroom2 --output CW2-Dataset/mit_76_studyroom_scene2.h5
```

## Technical Details

### DepthTSDF Format

The DepthTSDF format is a representation of depth that provides better information for 3D reconstruction:

- Raw depth: Direct distance measurements from the camera
- TSDF: Truncated Signed Distance Function, representing distances to surfaces in 3D space

Our conversion uses a simplified approach that ensures consistency with existing DepthTSDF files while maintaining all necessary information for point cloud generation. 