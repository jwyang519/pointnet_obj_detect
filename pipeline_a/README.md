# Pipeline A: Binary Point Cloud Classification

This pipeline implements a binary classifier for point clouds to determine whether a scene contains a table (label 1) or not (label 0).

## Overview

- Uses PointNet++ (SSG version) for classification
- Takes point clouds of shape (B, N, 3) as input
- Outputs a binary label (0 or 1) per point cloud
- Uses cross-entropy loss for training

## Dataset Format

The pipeline uses .h5 files with the following structure:
- 'data': Point cloud data of shape (num_samples, num_points, 3)
- 'label': Binary labels of shape (num_samples,) where:
  - 0: No table in the scene
  - 1: Table present in the scene

## Files

- `data_loader.py`: Custom data loader for .h5 files
- `train.py`: Script for training the binary classifier
- `test.py`: Script for testing the trained model
- `README.md`: Documentation (this file)

## Usage

### Training

```bash
cd Pointnet2_pytorch
python pipeline_a/train.py --train_file path/to/sun3d_train.h5 --test_file path/to/sun3d_test.h5 --log_dir binary_cls
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 24)
- `--num_point`: Number of points per point cloud (default: 1024)
- `--epoch`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer (default: 'Adam')
- `--model`: Model to use (default: 'pointnet2_cls_ssg')
- `--gpu`: GPU to use (default: '0')
- `--use_cpu`: Use CPU instead of GPU

### Testing

```bash
cd Pointnet2_pytorch
python pipeline_a/test.py --log_dir binary_cls --test_file path/to/sun3d_test.h5 --ucl_file path/to/ucl_data.h5
```

Optional arguments:
- `--batch_size`: Batch size for testing (default: 24)
- `--num_point`: Number of points per point cloud (default: 1024)
- `--model`: Model to use (default: 'pointnet2_cls_ssg')
- `--gpu`: GPU to use (default: '0')
- `--use_cpu`: Use CPU instead of GPU

## Expected Output

The training script will save the model with the best accuracy in `./log/table_classification/[log_dir]/checkpoints/best_model.pth`.

The testing script will evaluate the model on both the test dataset and the UCL dataset, providing:
- Overall accuracy
- Per-class accuracy (table vs. no table)
- Confusion matrix

## Notes

- The UCL dataset is assumed to contain only scenes with tables (label 1)
- The model uses only XYZ coordinates (not normals)
- Data augmentation is applied during training 