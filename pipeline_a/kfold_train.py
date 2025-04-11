"""
Pipeline A: K-fold cross-validation for binary classification of point clouds (table vs. no table)
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

# Add necessary paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Import the provider module from the root directory
sys.path.append(ROOT_DIR)  # Add root directory again to be sure
import provider

# Import the dataset
from data_loader import TableClassificationDataset

# Import functions from train.py
from train import inplace_relu, test, plot_training_curves, get_lr_scheduler, calculate_class_weights, calculate_metrics

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet++ K-Fold Cross-Validation Training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    
    # Learning rate parameters
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate in training')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau', 'none'], 
                       help='learning rate scheduler type')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='decay rate for scheduler')
    parser.add_argument('--step_size', type=int, default=20, help='step size for StepLR scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate for schedulers')
    
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--train_file', type=str, default='CW2-Dataset/sun3d_train_fixed.h5', help='path to training file')
    parser.add_argument('--test_file', type=str, default='CW2-Dataset/sun3d_test_fixed.h5', help='path to separate test file (not used during training)')
    parser.add_argument('--table_weight', type=float, default=1.0, help='weight for table class (class 1)')
    parser.add_argument('--non_table_weight', type=float, default=1.0, help='weight for non-table class (class 0)')
    parser.add_argument('--use_weights', action='store_true', default=True, help='use class weights')
    parser.add_argument('--n_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=42, help='random state for k-fold split')
    return parser.parse_args()

def train_one_fold(train_dataset, val_dataset, args, fold_idx, exp_dir):
    """
    Train and evaluate the model on one fold
    
    Args:
        train_dataset: Training dataset for this fold
        val_dataset: Validation dataset for this fold
        args: Command-line arguments
        fold_idx: Index of the current fold
        exp_dir: Directory to save logs and checkpoints
        
    Returns:
        Dictionary of metrics for this fold
    """
    def log_string(str):
        logger.info(str)
        print(str)
    
    # Create fold-specific directories
    fold_dir = exp_dir.joinpath(f'fold_{fold_idx}')
    fold_dir.mkdir(exist_ok=True)
    checkpoints_dir = fold_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = fold_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger(f"Model_Fold_{fold_idx}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string(f'Starting training for fold {fold_idx}/{args.n_folds}')
    log_string(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')
    
    # Create data loaders
    trainDataLoader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    valDataLoader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Calculate class weights
    train_labels = np.array([train_dataset.dataset.cloud_labels[i] for i in train_dataset.indices])
    class_weights = calculate_class_weights(train_labels)
    
    # Log class distribution and weights
    num_table = np.sum(train_labels == 1)
    num_non_table = np.sum(train_labels == 0)
    log_string(f'Training set composition: {num_table} tables, {num_non_table} non-tables')
    log_string(f'Original class ratio (non-table:table): {num_non_table/num_table:.3f}')
    log_string(f'Calculated soft class weights: {class_weights.tolist()}')
    
    if args.use_weights:
        class_weights = class_weights.cuda() if not args.use_cpu else class_weights
        log_string('Using calculated class weights for loss function')
    else:
        class_weights = None
        log_string('Not using class weights')
    
    # Model loading
    num_class = 2
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, normal_channel=False)
    criterion = model.get_loss(alpha=class_weights)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    # Setup optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Set up learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, args)
    log_string(f'Using learning rate scheduler: {args.scheduler}')
    log_string(f'Initial learning rate: {args.learning_rate}')
    
    # Training variables
    global_epoch = 0
    best_val_f1 = 0.0
    patience = 30  # Increased patience for early stopping
    min_epochs = 20  # Minimum number of epochs before allowing early stopping
    patience_counter = 0
    best_epoch = 0
    
    # Lists to store metrics for plotting
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_table_acc_list = []
    test_table_acc_list = []
    train_non_table_acc_list = []
    test_non_table_acc_list = []
    lr_list = []
    
    # Training loop
    log_string('Start training...')
    for epoch in range(args.epoch):
        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')
        
        # Training
        train_total_loss = 0.0
        train_total_correct = 0
        train_total = 0
        train_table_acc_sum = 0.0
        train_non_table_acc_sum = 0.0
        train_f1_sum = 0.0
        train_samples = 0
        
        classifier.train()
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            points = points.cpu().numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long())
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            
            # Calculate batch metrics
            correct, table_acc, non_table_acc, f1_score = calculate_metrics(pred, target)
            train_total_correct += correct
            train_total += points.size(0)
            
            # Keep track of metrics for averaging
            train_table_acc_sum += table_acc * points.size(0)
            train_non_table_acc_sum += non_table_acc * points.size(0)
            train_f1_sum += f1_score * points.size(0)
            train_samples += points.size(0)
        
        # Calculate training metrics
        train_loss = train_total_loss / len(trainDataLoader)
        train_acc = train_total_correct / float(train_total)
        train_table_acc = train_table_acc_sum / train_samples
        train_non_table_acc = train_non_table_acc_sum / train_samples
        train_f1 = train_f1_sum / train_samples
        
        log_string(f'Train Loss: {train_loss:.4f}')
        log_string(f'Train Accuracy: {train_acc:.4f}')
        log_string(f'Train Table Accuracy: {train_table_acc:.4f}')
        log_string(f'Train Non-Table Accuracy: {train_non_table_acc:.4f}')
        log_string(f'Train F1 Score: {train_f1:.4f}')
        
        # Validation
        classifier.eval()
        val_total_loss = 0.0
        val_total_correct = 0
        val_total = 0
        val_table_acc_sum = 0.0
        val_non_table_acc_sum = 0.0
        val_f1_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for points, target in valDataLoader:
                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda()
                
                points = points.transpose(2, 1)
                pred, _ = classifier(points)
                loss = criterion(pred, target.long())
                
                val_total_loss += loss.item()
                
                # Calculate batch metrics
                correct, table_acc, non_table_acc, f1_score = calculate_metrics(pred, target)
                val_total_correct += correct
                val_total += points.size(0)
                
                # Keep track of metrics for averaging
                val_table_acc_sum += table_acc * points.size(0)
                val_non_table_acc_sum += non_table_acc * points.size(0)
                val_f1_sum += f1_score * points.size(0)
                val_samples += points.size(0)
        
        # Calculate validation metrics
        val_loss = val_total_loss / len(valDataLoader)
        val_acc = val_total_correct / float(val_total)
        val_table_acc = val_table_acc_sum / val_samples
        val_non_table_acc = val_non_table_acc_sum / val_samples
        val_f1 = val_f1_sum / val_samples
        
        log_string(f'Validation Loss: {val_loss:.4f}')
        log_string(f'Validation Accuracy: {val_acc:.4f}')
        log_string(f'Validation Table Accuracy: {val_table_acc:.4f}')
        log_string(f'Validation Non-Table Accuracy: {val_non_table_acc:.4f}')
        log_string(f'Validation F1 Score: {val_f1:.4f}')
        
        # Store metrics for plotting
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        test_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(val_acc)
        train_table_acc_list.append(train_table_acc)
        test_table_acc_list.append(val_table_acc)
        train_non_table_acc_list.append(train_non_table_acc)
        test_non_table_acc_list.append(val_non_table_acc)
        lr_list.append(optimizer.param_groups[0]['lr'])
        
        # Early stopping check
        if val_f1 > best_val_f1 and epoch >= min_epochs:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save the best model
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'table_acc': val_table_acc,
                'non_table_acc': val_non_table_acc,
                'f1_score': val_f1,
            }
            torch.save(save_dict, str(checkpoints_dir) + '/best_model.pth')
            log_string('Saved new best model')
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                log_string(f'Early stopping triggered. Best F1: {best_val_f1:.4f} at epoch {best_epoch}')
                break
        
        # Step the scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        global_epoch += 1
    
    # Plot the training curves at the end of training
    log_string('Generating training curves plot...')
    plot_training_curves(
        epoch_list, train_loss_list, test_loss_list, 
        train_acc_list, test_acc_list, 
        train_table_acc_list, test_table_acc_list, 
        train_non_table_acc_list, test_non_table_acc_list, 
        str(fold_dir)
    )
    
    # Additionally plot learning rate curve
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, lr_list, 'g-')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(str(fold_dir), 'lr_curve.png'))
    plt.close()
    
    log_string('Training curves saved to %s/training_curves.png' % str(fold_dir))
    log_string('Learning rate curve saved to %s/lr_curve.png' % str(fold_dir))
    
    # Return the best metrics for this fold
    fold_results = {
        'fold': fold_idx,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'val_acc': val_acc,
        'val_table_acc': val_table_acc,
        'val_non_table_acc': val_non_table_acc
    }
    
    return fold_results

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('table_classification')
    exp_dir.mkdir(exist_ok=True)
    
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(f'kfold_{timestr}')
    else:
        exp_dir = exp_dir.joinpath(f'kfold_{args.log_dir}')
    
    exp_dir.mkdir(exist_ok=True)
    
    '''LOGGING'''
    logger = logging.getLogger("KFold_CV")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/kfold_cv.txt' % exp_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    def log_string(str):
        logger.info(str)
        print(str)
    
    log_string('PARAMETER ...')
    log_string(args)
    
    '''DATA LOADING'''
    log_string('Loading dataset ...')
    
    # Load the full dataset
    full_dataset = TableClassificationDataset(args.train_file, args.num_point)
    
    # Extract all labels for stratification
    all_labels = full_dataset.cloud_labels
    
    log_string(f'Full dataset size: {len(full_dataset)} samples')
    log_string(f'Table samples: {np.sum(all_labels == 1)}')
    log_string(f'Non-table samples: {np.sum(all_labels == 0)}')
    
    # Set up K-Fold cross-validation
    n_splits = args.n_folds
    log_string(f'Performing {n_splits}-fold cross-validation')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    
    # Store results from each fold
    all_results = []
    
    # Train on each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        log_string(f'Starting fold {fold_idx+1}/{n_splits}')
        
        # Create train and validation datasets for this fold
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        # Train on this fold
        fold_results = train_one_fold(train_dataset, val_dataset, args, fold_idx+1, exp_dir)
        all_results.append(fold_results)
        
        log_string(f'Completed fold {fold_idx+1}/{n_splits}')
        log_string(f'Best validation F1: {fold_results["best_val_f1"]:.4f} at epoch {fold_results["best_epoch"]}')
        log_string(f'Validation Accuracy: {fold_results["val_acc"]:.4f}')
        log_string(f'Validation Table Accuracy: {fold_results["val_table_acc"]:.4f}')
        log_string(f'Validation Non-Table Accuracy: {fold_results["val_non_table_acc"]:.4f}')
        log_string('----------------------------------------')
    
    # Calculate and print summary statistics
    f1_scores = [fold['best_val_f1'] for fold in all_results]
    acc_scores = [fold['val_acc'] for fold in all_results]
    table_acc_scores = [fold['val_table_acc'] for fold in all_results]
    non_table_acc_scores = [fold['val_non_table_acc'] for fold in all_results]
    
    log_string('\nCross-Validation Results Summary:')
    log_string(f'F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
    log_string(f'Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}')
    log_string(f'Table Accuracy: {np.mean(table_acc_scores):.4f} ± {np.std(table_acc_scores):.4f}')
    log_string(f'Non-Table Accuracy: {np.mean(non_table_acc_scores):.4f} ± {np.std(non_table_acc_scores):.4f}')
    
    # Plot fold comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(range(1, n_splits+1), f1_scores)
    plt.title('F1 Score by Fold')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, n_splits+1))
    
    plt.subplot(2, 2, 2)
    plt.bar(range(1, n_splits+1), acc_scores)
    plt.title('Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, n_splits+1))
    
    plt.subplot(2, 2, 3)
    plt.bar(range(1, n_splits+1), table_acc_scores)
    plt.title('Table Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, n_splits+1))
    
    plt.subplot(2, 2, 4)
    plt.bar(range(1, n_splits+1), non_table_acc_scores)
    plt.title('Non-Table Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, n_splits+1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(str(exp_dir), 'fold_comparison.png'))
    plt.close()
    
    log_string('Fold comparison plot saved to %s/fold_comparison.png' % str(exp_dir))
    log_string('K-fold cross-validation completed.')

if __name__ == '__main__':
    args = parse_args()
    main(args) 