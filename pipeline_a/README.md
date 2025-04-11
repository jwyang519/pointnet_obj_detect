# Pipeline A: K-fold Cross-Validation for Table Classification

This directory contains scripts for training and evaluating a binary classification model for table detection in point clouds using K-fold cross-validation.

## K-fold Cross-Validation

The `kfold_train.py` script implements K-fold cross-validation for model training and evaluation. It splits the dataset into K folds and trains a separate model on each fold, then reports the average performance and variance across folds.

### Usage

Basic usage:
```
python kfold_train.py --train_file CW2-Dataset/sun3d_train_fixed.h5 --log_dir my_kfold_experiment
```

Full usage with all parameters:
```
python kfold_train.py \
  --train_file CW2-Dataset/sun3d_train_fixed.h5 \
  --log_dir my_kfold_experiment \
  --n_folds 5 \
  --epoch 100 \
  --batch_size 24 \
  --learning_rate 0.0001 \
  --scheduler step \
  --lr_decay 0.7 \
  --step_size 20 \
  --num_point 1024 \
  --use_weights
```

### Parameters

- `--train_file`: Path to the H5 file containing training data
- `--log_dir`: Directory name for saving logs and models
- `--n_folds`: Number of folds for cross-validation (default: 5)
- `--epoch`: Number of training epochs per fold (default: 100)
- `--batch_size`: Batch size for training (default: 24)
- `--learning_rate`: Initial learning rate (default: 0.0001)
- `--scheduler`: Learning rate scheduler, options: 'step', 'cosine', 'plateau', 'none' (default: 'step')
- `--lr_decay`: Decay rate for the learning rate scheduler (default: 0.7)
- `--step_size`: Step size for the StepLR scheduler (default: 20)
- `--min_lr`: Minimum learning rate for schedulers (default: 1e-6)
- `--num_point`: Number of points in each point cloud (default: 1024)
- `--use_weights`: Whether to use class weights to handle class imbalance (default: True)
- `--random_state`: Random seed for reproducibility (default: 42)

### Output

The script outputs:
- A summary of each fold's performance 
- Average metrics across all folds with standard deviation
- Learning curves for each fold
- Comparison plots of metrics across folds
- Trained model checkpoints for each fold

### Logs and Model Checkpoints

All logs and models are saved under:
```
log/table_classification/kfold_<log_dir>
```

For each fold, there is a separate subdirectory:
```
log/table_classification/kfold_<log_dir>/fold_<n>/
```

Each fold directory contains:
- `checkpoints/best_model.pth`: Best model weights for that fold
- `logs/<model>.txt`: Training logs
- `training_curves.png`: Plot of metrics during training
- `lr_curve.png`: Plot of learning rate schedule

The root directory also contains:
- `kfold_cv.txt`: Full cross-validation logs
- `fold_comparison.png`: Comparison of metrics across all folds 