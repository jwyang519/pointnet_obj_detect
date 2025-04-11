"""
Pipeline A: Binary classification of point clouds (table vs. no table)
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import shutil
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, Subset

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

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet++ Binary Classification Training')
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
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of training data to use for validation (0-1)')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader):
    with torch.no_grad():
        mean_correct = []
        table_correct = []
        non_table_correct = []
        total_tables = 0
        total_non_tables = 0
        test_loss = 0.0
        
        classifier = model.eval()

        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            
            # Calculate loss
            if class_weights is not None:
                batch_loss = torch.nn.functional.nll_loss(pred, target.long(), weight=class_weights)
            else:
                batch_loss = torch.nn.functional.nll_loss(pred, target.long())
            
            test_loss += batch_loss.item() * points.size(0)
            
            pred_choice = pred.data.max(1)[1]

            # Overall accuracy
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            
            # Per-class accuracy
            for i in range(len(target)):
                if target[i] == 1:  # Table
                    if pred_choice[i] == 1:
                        table_correct.append(1.0)
                    else:
                        table_correct.append(0.0)
                    total_tables += 1
                else:  # Non-table
                    if pred_choice[i] == 0:
                        non_table_correct.append(1.0)
                    else:
                        non_table_correct.append(0.0)
                    total_non_tables += 1

        instance_acc = np.mean(mean_correct)
        table_acc = np.mean(table_correct) if total_tables > 0 else 0.0
        non_table_acc = np.mean(non_table_correct) if total_non_tables > 0 else 0.0
        avg_loss = test_loss / (total_tables + total_non_tables) if (total_tables + total_non_tables) > 0 else 0.0
        
        return instance_acc, table_acc, non_table_acc, total_tables, total_non_tables, avg_loss

def plot_training_curves(epochs, train_losses, test_losses, train_accs, test_accs, 
                         train_table_accs, test_table_accs, train_non_table_accs, 
                         test_non_table_accs, save_dir):
    """Plot training and test metrics over epochs"""
    plt.figure(figsize=(20, 15))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Overall Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Validation Accuracy')
    plt.title('Overall Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Table Class Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_table_accs, 'b-', label='Training Table Accuracy')
    plt.plot(epochs, test_table_accs, 'r-', label='Validation Table Accuracy')
    plt.title('Table Class Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Non-Table Class Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_non_table_accs, 'b-', label='Training Non-Table Accuracy')
    plt.plot(epochs, test_non_table_accs, 'r-', label='Validation Non-Table Accuracy')
    plt.title('Non-Table Class Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def get_lr_scheduler(optimizer, args):
    """Create a learning rate scheduler based on arguments"""
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.lr_decay
        )
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_decay,
            patience=5,
            min_lr=args.min_lr
        )
    else:  # 'none'
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

def calculate_class_weights(train_labels):
    """Calculate balanced class weights with softer weighting."""
    num_table = np.sum(train_labels == 1)
    num_non_table = np.sum(train_labels == 0)
    total = num_table + num_non_table
    
    # Calculate basic inverse frequency weights
    raw_weights = np.array([total / (2 * num_non_table), total / (2 * num_table)])
    
    # Apply softening: bring weights closer to 1.0
    softened_weights = np.sqrt(raw_weights)  # Square root to reduce extremes
    
    # Normalize weights to maintain relative proportions while being closer to 1
    normalized_weights = softened_weights / softened_weights.mean()
    
    # Convert to tensor
    weights = torch.FloatTensor(normalized_weights)
    return weights

def calculate_metrics(pred, target):
    """Calculate per-class accuracy and F1 score."""
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    
    # Per-class accuracy
    table_mask = target == 1
    non_table_mask = target == 0
    
    table_acc = pred_choice[table_mask].eq(target[table_mask]).float().mean().item() if table_mask.any() else 0
    non_table_acc = pred_choice[non_table_mask].eq(target[non_table_mask]).float().mean().item() if non_table_mask.any() else 0
    
    # Calculate F1 score
    tp = pred_choice[table_mask].eq(target[table_mask]).float().sum().item()
    fp = pred_choice[non_table_mask].ne(target[non_table_mask]).float().sum().item()
    fn = pred_choice[table_mask].ne(target[table_mask]).float().sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return correct.item(), table_acc, non_table_acc, f1

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('table_classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    
    # Load the full dataset from the training file
    full_dataset = TableClassificationDataset(args.train_file, args.num_point)
    
    # Calculate the size of the validation set
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    log_string(f'Dataset split: {train_size} training samples, {val_size} validation samples')
    
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

    # Calculate class weights with softer weighting
    train_labels = []
    for idx in train_dataset.indices:
        train_labels.append(full_dataset.cloud_labels[idx])
    
    train_labels = np.array(train_labels)
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

    '''MODEL LOADING'''
    num_class = 2
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, normal_channel=False)
    criterion = model.get_loss(alpha=class_weights)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

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
    
    global_epoch = 0
    global_step = 0
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

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # Training
        train_total_loss = 0.0
        classifier = classifier.train()
        
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
            
        # Calculate training metrics
        train_loss = train_total_loss / len(trainDataLoader)
        log_string(f'Train Loss: {train_loss:.4f}')
        
        # Validation
        classifier = classifier.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        val_table_correct = []
        val_non_table_correct = []
        
        with torch.no_grad():
            for points, target in valDataLoader:
                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda()
                
                points = points.transpose(2, 1)
                pred, _ = classifier(points)
                loss = criterion(pred, target.long())
                
                val_total_loss += loss.item()
                correct, table_acc, non_table_acc, f1_score = calculate_metrics(pred, target)
                
                val_correct += correct
                val_total += points.size(0)
                val_table_correct.append(table_acc)
                val_non_table_correct.append(non_table_acc)
        
        val_loss = val_total_loss / len(valDataLoader)
        val_acc = val_correct / float(val_total)
        val_table_acc = np.mean(val_table_correct)
        val_non_table_acc = np.mean(val_non_table_correct)
        
        log_string(f'Validation Loss: {val_loss:.4f}')
        log_string(f'Validation Accuracy: {val_acc:.4f}')
        log_string(f'Table Accuracy: {val_table_acc:.4f}')
        log_string(f'Non-Table Accuracy: {val_non_table_acc:.4f}')
        log_string(f'F1 Score: {f1_score:.4f}')
        
        # Early stopping check
        if f1_score > best_val_f1:
            best_val_f1 = f1_score
            best_epoch = epoch
            patience_counter = 0
            
            # Save the best model
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'f1_score': f1_score,
            }
            torch.save(save_dict, str(checkpoints_dir) + '/best_model.pth')
            log_string('Saved new best model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_string(f'Early stopping triggered. Best F1: {best_val_f1:.4f} at epoch {best_epoch}')
                break
        
        # Step the scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Update learning rate lists for plotting
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        
        global_epoch += 1
    
    # Plot the training curves at the end of training
    log_string('Generating training curves plot...')
    plot_training_curves(
        epoch_list, train_loss_list, test_loss_list, 
        train_acc_list, test_acc_list, 
        train_table_acc_list, test_table_acc_list, 
        train_non_table_acc_list, test_non_table_acc_list, 
        str(exp_dir)
    )
    
    # Additionally plot learning rate curve
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, lr_list, 'g-')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(str(exp_dir), 'lr_curve.png'))
    plt.close()
    
    log_string('Training curves saved to %s/training_curves.png' % str(exp_dir))
    log_string('Learning rate curve saved to %s/lr_curve.png' % str(exp_dir))

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args) 