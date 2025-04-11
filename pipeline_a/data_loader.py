import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class TableClassificationDataset(Dataset):
    """
    Dataset for table classification
    Takes an h5 file with point clouds and labels to create a dataset
    that categorizes point clouds as either containing a table (1) or not (0)
    """
    def __init__(self, h5_file, num_points=1024, normalize=True):
        """
        Initialize the dataset
        
        Args:
            h5_file (str): Path to the h5 file
            num_points (int): Number of points per point cloud
            normalize (bool): Whether to normalize point clouds
        """
        self.num_points = num_points
        self.normalize = normalize
        
        # Load the h5 file
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['label'][:]
            
            # Check if sample_indices exist (to identify point cloud boundaries)
            if 'sample_indices' in f:
                self.sample_indices = f['sample_indices'][:]
                self.cloud_data = []
                self.cloud_labels = []
                
                # Process each point cloud using sample_indices
                for i in range(len(self.sample_indices) - 1):
                    start_idx = self.sample_indices[i]
                    end_idx = self.sample_indices[i+1]
                    
                    # Get points for this point cloud
                    cloud = self.data[start_idx:end_idx]
                    cloud_label = self.labels[start_idx:end_idx]
                    
                    # If we have more points than needed, randomly sample
                    if len(cloud) > self.num_points:
                        idx = np.random.choice(len(cloud), self.num_points, replace=False)
                        cloud = cloud[idx]
                        cloud_label = cloud_label[idx]
                    # If we have fewer points, duplicate some points
                    elif len(cloud) < self.num_points:
                        idx = np.random.choice(len(cloud), self.num_points - len(cloud))
                        cloud = np.vstack((cloud, cloud[idx]))
                        cloud_label = np.append(cloud_label, cloud_label[idx])
                    
                    # Store the point cloud and its label (1 if any point is labeled as table)
                    self.cloud_data.append(cloud)
                    self.cloud_labels.append(1 if np.sum(cloud_label) > 0 else 0)
                
                # Convert to numpy arrays
                self.cloud_data = np.array(self.cloud_data)
                self.cloud_labels = np.array(self.cloud_labels)
            else:
                # If no sample_indices, assume each num_points consecutive points form a point cloud
                total_points = len(self.labels)
                num_clouds = total_points // self.num_points
                
                # Reshape data and labels to form point clouds
                self.cloud_data = self.data[:num_clouds * self.num_points].reshape(num_clouds, self.num_points, -1)
                point_labels = self.labels[:num_clouds * self.num_points].reshape(num_clouds, self.num_points)
                
                # Classify each point cloud (1 if any point is labeled as table)
                self.cloud_labels = np.zeros(num_clouds)
                for i in range(num_clouds):
                    if np.sum(point_labels[i]) > 0:
                        self.cloud_labels[i] = 1
        
        # Normalize point clouds if requested
        if self.normalize:
            self.normalize_point_clouds()
    
    def normalize_point_clouds(self):
        """Normalize point clouds to zero mean and unit variance"""
        for i in range(len(self.cloud_data)):
            # Get XYZ coordinates
            xyz = self.cloud_data[i, :, 0:3]
            
            # Calculate centroid and scale
            centroid = np.mean(xyz, axis=0)
            xyz = xyz - centroid
            
            # Scale to unit sphere
            max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
            if max_dist > 0:
                xyz = xyz / max_dist
            
            # Update the point cloud
            self.cloud_data[i, :, 0:3] = xyz
    
    def __len__(self):
        """Return the number of point clouds in the dataset"""
        return len(self.cloud_labels)
    
    def __getitem__(self, idx):
        """Get a point cloud and its label by index"""
        point_cloud = self.cloud_data[idx]
        label = self.cloud_labels[idx]
        
        # Convert to PyTorch tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        label = torch.tensor(label).long()
        
        return point_cloud, label 