"""
Author: Based on Benny's test_semseg.py
Date: Apr 2024
Description: Testing script for SUN3D binary segmentation (table vs background)
"""
import argparse
import os
import torch
import numpy as np
import importlib
import sys
from tqdm import tqdm

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # Get the parent directory (project root)
sys.path.append(ROOT_DIR)  # Add the project root to the path
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))  # Add data_utils directly

# Now import from data_utils
from data_utils.sun3d_dataset_pytorch import SUN3DDataset, get_data_loaders

# Binary segmentation: 0 = background, 1 = table
classes = ['background', 'table']
NUM_CLASSES = len(classes)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to test h5 file with point clouds and labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root [default: None]')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize result [default: False]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        print(str)

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sun3d_binary_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    if args.visual and not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    # Create test dataset and loader
    _, test_loader = get_data_loaders(
        args.h5_file,
        batch_size=args.batch_size,
        num_points=args.npoint,
        num_workers=4
    )
    
    log_string("The number of test data is: %d" % len(test_loader.dataset))

    # Load model
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        classifier = classifier.eval()
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION ----')
        
        for i, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            # Points are already in [B, C, N] format from the dataset loader
            points, target = points.float().cuda(), target.long().cuda()
            
            # No need for additional transposition since data is already in correct format
            seg_pred, _ = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (args.batch_size * args.npoint)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # Visualization
            if args.visual and i < 10:
                for b in range(points.shape[0]):
                    pts = points[b, :, :].transpose(1, 0).contiguous().cpu().data.numpy()
                    pred_label = pred_val[b]
                    gt_label = batch_label[b]
                    
                    # Save to file for visualization
                    np.savetxt(os.path.join(visual_dir, f'pts_{i}_{b}.txt'), pts[:, :3])
                    np.savetxt(os.path.join(visual_dir, f'pred_{i}_{b}.txt'), pred_label)
                    np.savetxt(os.path.join(visual_dir, f'gt_{i}_{b}.txt'), gt_label)

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6))
        log_string('Test point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('Test point avg class IoU: %f' % (mIoU))
        log_string('Test point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class) + 1e-6))))
        
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s IoU: %.3f \n' % (
                classes[l], total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))
        log_string(iou_per_class_str)
        
        acc_per_class_str = '------- Acc --------\n'
        for l in range(NUM_CLASSES):
            acc_per_class_str += 'class %s Acc: %.3f \n' % (
                classes[l], total_correct_class[l] / float(total_seen_class[l] + 1e-6))
        log_string(acc_per_class_str)


if __name__ == '__main__':
    args = parse_args()
    main(args) 