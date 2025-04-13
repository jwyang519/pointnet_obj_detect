"""
Author: Based on Benny's train_semseg.py
Date: Apr 2024
Description: Training script for SUN3D binary segmentation (table vs background)
"""
import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # Get the parent directory (project root)
sys.path.append(ROOT_DIR)  # Add the project root to the path
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))  # Add data_utils directly

# Now import from data_utils
from data_utils.sun3d_dataset_pytorch import SUN3DDataset, get_data_loaders
import provider

# Binary segmentation: 0 = background, 1 = table
classes = ['background', 'table']
NUM_CLASSES = len(classes)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to h5 file with point clouds and labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation [default: False]')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # For tracking metrics over training
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []
    mean_ious = []
    class_ious = {i: [] for i in range(NUM_CLASSES)}
    
    # For tracking augmentation statistics
    aug_stats = {
        'avg_table_ratio': [],
        'jitter_strength': [],
        'scale_range': [],
        'shift_range': [],
        'dropout_chance': [],
        'dropout_ratio': []
    }

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    
    # Create a more structured directory naming
    experiment_dir = experiment_dir.joinpath('sun3d_binary_seg')
    experiment_dir.mkdir(exist_ok=True)
    
    if args.log_dir is None:
        # Use default naming with timestamp only
        log_name = f"run_{timestr}"
    else:
        # Append timestamp to user-provided name to avoid overwriting
        log_name = f"{args.log_dir}_{timestr}"
    
    # Create the final experiment directory
    experiment_dir = experiment_dir.joinpath(log_name)
    experiment_dir.mkdir(exist_ok=True)
    
    # Log the directory path for user reference
    print(f"Results will be saved to: {experiment_dir}")
    
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Create plots directory
    plots_dir = experiment_dir.joinpath('plots/')
    plots_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("Loading SUN3D training data...")
    train_loader, test_loader = get_data_loaders(
        args.h5_file,
        batch_size=BATCH_SIZE,
        num_points=NUM_POINT,
        num_workers=4
    )

    # Get dataset sizes
    train_dataset_size = len(train_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    
    log_string("The number of training samples is: %d" % train_dataset_size)
    log_string("The number of testing samples is: %d" % test_dataset_size)

    # Class distribution (for setting up weights)
    # In binary segmentation, we might need to weight classes
    # if they are highly imbalanced
    class_counts = np.zeros(NUM_CLASSES)
    for batch in train_loader:
        _, labels = batch
        # Flatten labels to count all instances
        flat_labels = labels.view(-1).numpy()
        unique_labels, counts = np.unique(flat_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label < NUM_CLASSES:  # Ensure the label is within range
                class_counts[label] += count
    
    if np.min(class_counts) == 0:
        log_string("Warning: Some classes have no samples!")
        # Avoid division by zero
        class_counts[class_counts == 0] = 1
    
    # The smaller the count, the higher the weight
    # But ensure weights are more balanced to avoid numerical issues
    class_weights = 1.0 / np.sqrt(class_counts + 1)  # Using sqrt to smooth extreme differences
    
    # Ensure table class (class 1) gets significantly more weight - multiply by additional factor
    if NUM_CLASSES > 1:
        table_boost_factor = 2.0  # Boost the table class weight by this factor
        class_weights[1] *= table_boost_factor
        
    # Normalize weights to sum to 1
    class_weights = class_weights / np.sum(class_weights)
    log_string(f"Class weights: {class_weights}")
    weights = torch.FloatTensor(class_weights).cuda()

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

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

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on the dataset'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad()

            # Convert to numpy for data augmentation
            points_np = points.numpy()  # [B, C, N]
            target_np = target.numpy()  # [B, N]
            
            # Only apply augmentation if enabled
            if args.use_augmentation:
                # Calculate percentage of table points in each sample
                table_percentage = np.mean(target_np == 1, axis=1)  # [B]
                
                # Transpose for augmentation functions which expect [B, N, C]
                points_np = points_np.transpose(0, 2, 1)  # [B, N, C]
                
                # Apply data augmentation (more aggressive for samples with more table points)
                batch_aug_stats = {
                    'table_ratios': [],
                    'jitter_strengths': [],
                    'scale_ranges': [],
                    'shift_ranges': [],
                    'dropout_applied': [],
                    'dropout_ratios': []
                }
                
                for b in range(points_np.shape[0]):
                    # Get table percentage for this batch item
                    table_ratio = table_percentage[b]
                    batch_aug_stats['table_ratios'].append(float(table_ratio))
                    sample_points = points_np[b:b+1]
                    
                    # Apply stronger augmentation based on table ratio
                    # More rotation for samples with more tables - but less aggressive
                    sample_points = provider.rotate_point_cloud(sample_points)
                    
                    # Jitter strength scales with table percentage - reduced by half
                    jitter_sigma = 0.005 * (1.0 + table_ratio)  # Was 0.01, now 0.005
                    batch_aug_stats['jitter_strengths'].append(float(jitter_sigma))
                    sample_points = provider.jitter_point_cloud(sample_points, sigma=jitter_sigma, clip=0.03)  # Was 0.05, now 0.03
                    
                    # More scaling variation for samples with more tables - made less extreme
                    scale_low = 0.9 - 0.05 * table_ratio  # Was 0.8-0.1, now 0.9-0.05
                    scale_high = 1.1 + 0.1 * table_ratio   # Was 1.25+0.15, now 1.1+0.1
                    batch_aug_stats['scale_ranges'].append((float(scale_low), float(scale_high)))
                    sample_points = provider.random_scale_point_cloud(sample_points, scale_low, scale_high)
                    
                    # More shifting for samples with more tables - reduced range
                    shift_range = 0.05 + 0.05 * table_ratio  # Was 0.1+0.1, now 0.05+0.05
                    batch_aug_stats['shift_ranges'].append(float(shift_range))
                    sample_points = provider.shift_point_cloud(sample_points, shift_range)
                    
                    # Point dropout has higher chance with more table points - reduced probability and ratio
                    dropout_chance = 0.2 + 0.2 * table_ratio  # Was 0.3+0.4, now 0.2+0.2
                    dropout_applied = False
                    dropout_ratio = 0.05 + 0.05 * table_ratio  # Was 0.1+0.1, now 0.05+0.05
                    
                    if np.random.random() < dropout_chance:
                        dropout_applied = True
                        sample_points = provider.random_point_dropout(sample_points, max_dropout_ratio=dropout_ratio)
                    
                    batch_aug_stats['dropout_applied'].append(dropout_applied)
                    batch_aug_stats['dropout_ratios'].append(float(dropout_ratio))
                    
                    # Update the batch with augmented points
                    points_np[b:b+1] = sample_points
                
                # Update global augmentation statistics (every 10 batches)
                if i % 10 == 0:
                    aug_stats['avg_table_ratio'].append(np.mean(batch_aug_stats['table_ratios']))
                    aug_stats['jitter_strength'].append(np.mean(batch_aug_stats['jitter_strengths']))
                    avg_scale_range = np.mean([high - low for low, high in batch_aug_stats['scale_ranges']])
                    aug_stats['scale_range'].append(avg_scale_range)
                    aug_stats['shift_range'].append(np.mean(batch_aug_stats['shift_ranges']))
                    aug_stats['dropout_chance'].append(np.mean([0.2 + 0.2 * tr for tr in batch_aug_stats['table_ratios']]))
                    aug_stats['dropout_ratio'].append(np.mean(batch_aug_stats['dropout_ratios']))
                    
                    # Log augmentation statistics occasionally
                    if i % 100 == 0:
                        aug_log = 'Augmentation stats | '
                        aug_log += f"Avg table ratio: {aug_stats['avg_table_ratio'][-1]:.3f} | "
                        aug_log += f"Jitter: {aug_stats['jitter_strength'][-1]:.4f} | "
                        aug_log += f"Scale range: {aug_stats['scale_range'][-1]:.3f} | "
                        aug_log += f"Shift: {aug_stats['shift_range'][-1]:.3f} | "
                        aug_log += f"Dropout chance: {aug_stats['dropout_chance'][-1]:.3f} | "
                        aug_log += f"Dropout ratio: {aug_stats['dropout_ratio'][-1]:.3f}"
                        log_string(aug_log)
                
                # Transpose back to [B, C, N] format
                points_np = points_np.transpose(0, 2, 1)
            
            # Convert to CUDA tensors
            points = torch.from_numpy(points_np).float().cuda()
            target = torch.from_numpy(target_np).long().cuda()
            
            # Forward pass
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()  # Convert to Python float using .item()
        
        # Track training metrics for this epoch
        epoch_train_loss = loss_sum / num_batches
        epoch_train_accuracy = total_correct / float(total_seen)
        train_losses.append(float(epoch_train_loss))  # Convert to Python float
        train_accuracies.append(float(epoch_train_accuracy))  # Convert to Python float
        
        log_string('Training mean loss: %f' % (epoch_train_loss))
        log_string('Training accuracy: %f' % (epoch_train_accuracy))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on testing data'''
        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                # No need for data augmentation in evaluation
                # Just convert to tensors and move to cuda
                points = points.float().cuda()
                target = target.long().cuda()
                
                # Forward pass
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()  # Convert to Python float using .item()
                pred_val = np.argmax(pred_val, 2)
                
                # Reshape batch_label to match pred_val shape for proper comparison
                batch_label = batch_label.reshape(pred_val.shape)
                
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6))
            
            # Store metrics for plotting
            eval_loss = loss_sum / float(num_batches)
            eval_accuracy = total_correct / float(total_seen)
            eval_losses.append(float(eval_loss))  # Convert to Python float
            eval_accuracies.append(float(eval_accuracy))  # Convert to Python float
            mean_ious.append(float(mIoU))  # Convert to Python float
            for l in range(NUM_CLASSES):
                iou = total_correct_class[l] / (float(total_iou_deno_class[l]) + 1e-6)
                class_ious[l].append(float(iou))  # Convert to Python float
            
            log_string('eval mean loss: %f' % (eval_loss))
            log_string('eval point accuracy: %f' % (eval_accuracy))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class) + 1e-6))))
            
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s IoU: %.3f \n' % (
                    classes[l] + ' ' * (14 - len(classes[l])), total_correct_class[l] / (float(total_iou_deno_class[l]) + 1e-6))
            log_string(iou_per_class_str)
            
            acc_per_class_str = '------- Acc --------\n'
            for l in range(NUM_CLASSES):
                acc_per_class_str += 'class %s Acc: %.3f \n' % (
                    classes[l] + ' ' * (14 - len(classes[l])), total_correct_class[l] / (float(total_seen_class[l]) + 1e-6))
            log_string(acc_per_class_str)
            
            # Create and save plots
            if epoch == args.epoch - 1:
                log_string('Creating final plots...')
                create_plots(train_losses, eval_losses, train_accuracies, eval_accuracies, 
                             mean_ious, class_ious, plots_dir, epoch, classes)
                
                # Plot augmentation statistics only if augmentation was used
                if args.use_augmentation and aug_stats['avg_table_ratio']:
                    plot_augmentation_stats(aug_stats, plots_dir, epoch)

            if mIoU >= best_iou:
                best_iou = mIoU
                
                # Save best model with timestamp
                model_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                best_model_path = str(checkpoints_dir) + f'/best_model_{model_timestamp}.pth'
                log_string('Saving at %s' % best_model_path)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, best_model_path)
                
                # Also save as the generic best_model.pth for compatibility
                torch.save(state, str(checkpoints_dir) + '/best_model.pth')
                log_string('Best model saved!')
            
            global_epoch += 1

def create_plots(train_losses, eval_losses, train_accuracies, eval_accuracies, 
                mean_ious, class_ious, plots_dir, epoch, classes):
    """
    Create and save plots of training metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Loss per epoch
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(eval_losses)+1), eval_losses, marker='x', label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy per epoch
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, len(eval_accuracies)+1), eval_accuracies, marker='x', label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Mean IoU per epoch
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(mean_ious)+1), mean_ious, marker='s', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU per Epoch')
    plt.grid(True)
    
    # Plot 4: Class-wise IoU per epoch
    plt.subplot(2, 2, 4)
    for class_idx, class_name in enumerate(classes):
        plt.plot(range(1, len(class_ious[class_idx])+1), class_ious[class_idx], 
                 marker='o', label=f'Class {class_name}')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Class-wise IoU per Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(str(plots_dir) + f'/metrics_epoch_{epoch+1}_{timestamp}.png')
    plt.savefig(str(plots_dir) + f'/metrics_latest.png')  # Always overwrite latest
    plt.close()

def plot_augmentation_stats(aug_stats, plots_dir, epoch):
    """
    Create and save plots of augmentation statistics
    """
    if not aug_stats['avg_table_ratio']:  # Skip if no augmentation stats collected
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Average table ratio
    plt.subplot(2, 3, 1)
    plt.plot(aug_stats['avg_table_ratio'], marker='o', color='blue')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Ratio')
    plt.title('Average Table Ratio')
    plt.grid(True)
    
    # Plot 2: Jitter strength
    plt.subplot(2, 3, 2)
    plt.plot(aug_stats['jitter_strength'], marker='o', color='green')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Strength')
    plt.title('Jitter Strength')
    plt.grid(True)
    
    # Plot 3: Scale range
    plt.subplot(2, 3, 3)
    plt.plot(aug_stats['scale_range'], marker='o', color='red')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Range')
    plt.title('Scale Range')
    plt.grid(True)
    
    # Plot 4: Shift range
    plt.subplot(2, 3, 4)
    plt.plot(aug_stats['shift_range'], marker='o', color='purple')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Range')
    plt.title('Shift Range')
    plt.grid(True)
    
    # Plot 5: Dropout chance
    plt.subplot(2, 3, 5)
    plt.plot(aug_stats['dropout_chance'], marker='o', color='brown')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Probability')
    plt.title('Dropout Chance')
    plt.grid(True)
    
    # Plot 6: Dropout ratio
    plt.subplot(2, 3, 6)
    plt.plot(aug_stats['dropout_ratio'], marker='o', color='orange')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Ratio')
    plt.title('Dropout Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(str(plots_dir) + f'/augmentation_stats_epoch_{epoch+1}_{timestamp}.png')
    plt.savefig(str(plots_dir) + f'/augmentation_stats_latest.png')  # Always overwrite latest
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    main(args) 