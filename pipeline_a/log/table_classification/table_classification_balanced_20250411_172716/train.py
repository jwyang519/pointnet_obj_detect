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
import torch.nn.functional as F

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
    parser.add_argument('--train_file', type=str, default='../dataset/sun3d_train.h5', help='path to training file')
    parser.add_argument('--test_file', type=str, default='../dataset/sun3d_test.h5', help='path to separate test file (not used during training)')
    parser.add_argument('--table_weight', type=float, default=1.0, help='weight for table class (class 1)')
    parser.add_argument('--non_table_weight', type=float, default=1.0, help='weight for non-table class (class 0)')
    parser.add_argument('--use_weights', action='store_true', default=True, help='use class weights')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of training data to use for validation (0-1)')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, focal_loss=None):
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
            
            # Calculate loss using focal loss if available
            if focal_loss is not None:
                batch_loss = focal_loss(pred, target.long())
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

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        if weight is not None:
            self.weight = weight.cuda()

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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

    # Calculate class weights for loss function
    # Since random_split doesn't provide access to attributes of the full dataset,
    # we need to determine the class distribution using indices
    train_labels = []
    for idx in train_dataset.indices:
        train_labels.append(full_dataset.cloud_labels[idx])
    
    train_labels = np.array(train_labels)
    num_table = np.sum(train_labels == 1)
    num_non_table = np.sum(train_labels == 0)
    total = num_table + num_non_table
    
    log_string(f'Training set composition: {num_table} tables, {num_non_table} non-tables')
    
    # Calculate even more aggressive weights
    if args.use_weights:
        ratio = num_table / num_non_table
        non_table_weight = min(20.0, ratio * 15)  # Cap at 20, but make it more aggressive
        table_weight = 1.0
        class_weights = torch.FloatTensor([non_table_weight, table_weight])
        focal_loss = FocalLoss(gamma=2.0, weight=class_weights)
        log_string(f'Using focal loss with class weights: [non-table: {non_table_weight}, table: {table_weight}]')
    else:
        class_weights = None
        focal_loss = FocalLoss(gamma=2.0)
        log_string('Using focal loss without class weights')

    '''MODEL LOADING'''
    num_class = 2  # Binary classification: table or no table
    model = importlib.import_module(args.model)
    
    # Copy model files for logging
    try:
        shutil.copy(os.path.join(ROOT_DIR, 'models', f'{args.model}.py'), str(exp_dir))
        shutil.copy(os.path.join(ROOT_DIR, 'models', 'pointnet2_utils.py'), str(exp_dir))
        shutil.copy(os.path.join(BASE_DIR, 'train.py'), str(exp_dir))
    except Exception as e:
        log_string(f'Error copying files: {e}')

    classifier = model.get_model(num_class, normal_channel=False)  # Using only XYZ coordinates, no normals
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        if class_weights is not None:
            class_weights = class_weights.cuda()

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
    best_instance_acc = 0.0
    best_table_acc = 0.0
    
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
        mean_correct = []
        table_correct = []
        non_table_correct = []
        total_tables = 0
        total_non_tables = 0
        total_loss = 0.0
        
        classifier = classifier.train()

        # Step the scheduler at the beginning of each epoch
        # For ReduceLROnPlateau, we'll step it after validation
        if args.scheduler != 'plateau':
            scheduler.step()
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        log_string(f'Current learning rate: {current_lr}')
        lr_list.append(current_lr)
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Apply data augmentation
            points = points.cpu().numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            
            # Transpose points for PointNet++ input format
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            
            # Apply class weights to loss
            if class_weights is not None:
                loss = focal_loss(pred, target.long())
            else:
                loss = focal_loss(pred, target.long())
            
            total_loss += loss.item() * points.size(0)
                
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
            
            loss.backward()
            optimizer.step()
            global_step += 1
        
        # Calculate average training loss and accuracies
        train_loss = total_loss / (total_tables + total_non_tables) if (total_tables + total_non_tables) > 0 else 0.0
        train_instance_acc = np.mean(mean_correct)
        train_table_acc = np.mean(table_correct) if total_tables > 0 else 0.0
        train_non_table_acc = np.mean(non_table_correct) if total_non_tables > 0 else 0.0
        
        log_string('Train Loss: %f' % train_loss)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string(f'Train Table Accuracy: {train_table_acc:.4f} ({total_tables} samples)')
        log_string(f'Train Non-Table Accuracy: {train_non_table_acc:.4f} ({total_non_tables} samples)')

        # Validate the model for this epoch
        with torch.no_grad():
            instance_acc, table_acc, non_table_acc, val_tables, val_non_tables, val_loss = test(classifier.eval(), valDataLoader, focal_loss)

            log_string('Validation Loss: %f' % val_loss)
            log_string('Validation Instance Accuracy: %f' % instance_acc)
            log_string(f'Validation Table Accuracy: {table_acc:.4f} ({val_tables} samples)')
            log_string(f'Validation Non-Table Accuracy: {non_table_acc:.4f} ({val_non_tables} samples)')
            
            # Step the scheduler if using ReduceLROnPlateau
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            
            # Store metrics for plotting
            epoch_list.append(epoch + 1)
            train_loss_list.append(train_loss)
            test_loss_list.append(val_loss)
            train_acc_list.append(train_instance_acc)
            test_acc_list.append(instance_acc)
            train_table_acc_list.append(train_table_acc)
            test_table_acc_list.append(table_acc)
            train_non_table_acc_list.append(train_non_table_acc)
            test_non_table_acc_list.append(non_table_acc)
            
            # Save best model based on table accuracy if there are table samples
            if val_tables > 0 and (table_acc >= best_table_acc):
                best_table_acc = table_acc
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'table_acc': table_acc,
                    'non_table_acc': non_table_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            # If no table samples in validation, save based on instance accuracy
            elif val_tables == 0 and (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'table_acc': table_acc,
                    'non_table_acc': non_table_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                
            log_string('Best Instance Accuracy: %f' % best_instance_acc)
            log_string('Best Table Accuracy: %f' % best_table_acc)
            
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