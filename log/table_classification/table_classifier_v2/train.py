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
from pathlib import Path
from tqdm import tqdm

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
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--train_file', type=str, default='../dataset/sun3d_train.h5', help='path to training file')
    parser.add_argument('--test_file', type=str, default='../dataset/sun3d_test.h5', help='path to test file')
    parser.add_argument('--class_weight', type=float, default=20.0, help='weight for table class')
    parser.add_argument('--use_weights', action='store_true', default=True, help='use class weights')
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
        
        classifier = model.eval()

        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            pred, _ = classifier(points)
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
        
        return instance_acc, table_acc, non_table_acc, total_tables, total_non_tables

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
    
    train_dataset = TableClassificationDataset(args.train_file, args.num_point)
    test_dataset = TableClassificationDataset(args.test_file, args.num_point)
    
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Calculate class weights for loss function
    train_labels = train_dataset.cloud_labels
    num_table = np.sum(train_labels == 1)
    num_non_table = np.sum(train_labels == 0)
    total = num_table + num_non_table
    
    if args.use_weights and num_table > 0:
        # Inverse frequency weighting
        class_weights = torch.FloatTensor([1.0, args.class_weight])
        log_string(f'Using class weights: {class_weights}')
    else:
        class_weights = None
        log_string('Not using class weights')

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
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_table_acc = 0.0

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        table_correct = []
        non_table_correct = []
        total_tables = 0
        total_non_tables = 0
        
        classifier = classifier.train()

        scheduler.step()
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
                loss = torch.nn.functional.nll_loss(pred, target.long(), weight=class_weights)
            else:
                loss = criterion(pred, target.long(), trans_feat)
                
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

        train_instance_acc = np.mean(mean_correct)
        train_table_acc = np.mean(table_correct) if total_tables > 0 else 0.0
        train_non_table_acc = np.mean(non_table_correct) if total_non_tables > 0 else 0.0
        
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string(f'Train Table Accuracy: {train_table_acc:.4f} ({total_tables} samples)')
        log_string(f'Train Non-Table Accuracy: {train_non_table_acc:.4f} ({total_non_tables} samples)')

        with torch.no_grad():
            instance_acc, table_acc, non_table_acc, test_tables, test_non_tables = test(classifier.eval(), testDataLoader)

            log_string('Test Instance Accuracy: %f' % instance_acc)
            log_string(f'Test Table Accuracy: {table_acc:.4f} ({test_tables} samples)')
            log_string(f'Test Non-Table Accuracy: {non_table_acc:.4f} ({test_non_tables} samples)')
            
            # Save best model based on table accuracy if there are table samples
            if test_tables > 0 and (table_acc >= best_table_acc):
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
            # If no table samples in test, save based on instance accuracy
            elif test_tables == 0 and (instance_acc >= best_instance_acc):
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

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args) 