import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SUN3DDataset(Dataset):
    def __init__(self, h5_file, num_points=1024, split='train', transform=None):
        """
        Args:
            h5_file (string): Path to the h5 file with point clouds and labels
            num_points (int): Number of points to sample from each point cloud
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.h5_file = h5_file
        self.num_points = num_points
        self.split = split
        self.transform = transform
        
        # Load data from h5 file
        with h5py.File(h5_file, 'r') as f:
            # Load all data at once (may be memory intensive for large datasets)
            self.data = f['data'][:]
            self.label = f['label'][:]
        
        # Normalize point clouds
        self.normalize_data()
        
        # Split data into train and test
        total_frames = self.data.shape[0] // num_points
        self.frames = []
        
        # Group points into frames (scenes) of num_points each
        for i in range(total_frames):
            start_idx = i * num_points
            end_idx = start_idx + num_points
            if end_idx <= self.data.shape[0]:
                self.frames.append((start_idx, end_idx))
        
        # Shuffle and split into train/test
        random.shuffle(self.frames)
        split_idx = int(len(self.frames) * 0.8)  # 80% for training
        
        if split == 'train':
            self.frame_indices = self.frames[:split_idx]
        else:
            self.frame_indices = self.frames[split_idx:]
    
    def normalize_data(self):
        """Normalize point clouds to have zero mean and unit variance"""
        # Center each point cloud
        centroid = np.mean(self.data, axis=0)
        self.data = self.data - centroid
        
        # Scale to unit sphere
        m = np.max(np.sqrt(np.sum(self.data**2, axis=1)))
        self.data = self.data / m
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.frame_indices[idx]
        
        # Get points and labels for this frame
        points = self.data[start_idx:end_idx]
        labels = self.label[start_idx:end_idx]
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        # Reshape points to [C, N] format expected by PointNet++
        # C = 3 (xyz coordinates), N = num_points
        points = points.transpose(0, 1)  # Shape: [3, num_points]
        
        # For PointNet++, return only XYZ and set points to None
        return points, labels

def get_data_loaders(h5_file, batch_size=32, num_points=1024, num_workers=4):
    """
    Get train and test data loaders for the SUN3D dataset
    
    Args:
        h5_file (string): Path to the h5 file with point clouds and labels
        batch_size (int): Batch size for data loaders
        num_points (int): Number of points to sample from each point cloud
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
    """
    # Create datasets
    train_dataset = SUN3DDataset(h5_file, num_points=num_points, split='train')
    test_dataset = SUN3DDataset(h5_file, num_points=num_points, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 