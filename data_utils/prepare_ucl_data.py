import os
import numpy as np
import h5py
import glob
import cv2
from tqdm import tqdm
import pickle

def load_camera_intrinsics(intrinsics_path):
    """
    Load camera intrinsics for the UCL data.
    
    Args:
        intrinsics_path: Path to the intrinsics.txt file
    
    Returns:
        dict with camera parameters (fx, fy, cx, cy)
    """
    try:
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            
            # Parse the depth intrinsics section
            depth_section = False
            for line in lines:
                if "Depth Intrinsics:" in line:
                    depth_section = True
                    continue
                
                if depth_section:
                    if "PPX:" in line:
                        cx = float(line.split(":")[1].strip())
                    elif "PPY:" in line:
                        cy = float(line.split(":")[1].strip())
                    elif "Fx:" in line:
                        fx = float(line.split(":")[1].strip())
                    elif "Fy:" in line:
                        fy = float(line.split(":")[1].strip())
                    elif "Extrinsics" in line:
                        # We've moved past the depth section
                        break
            
            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
    except Exception as e:
        print(f"Error loading intrinsics from {intrinsics_path}: {e}")
        # Return default values if file can't be loaded
        return {
            'fx': 390.7360534667969,
            'fy': 390.7360534667969,
            'cx': 320.08819580078125,
            'cy': 244.1026153564453
        }

def depth_to_pointcloud(depth_img, intrinsics):
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth_img: HxW depth image
        intrinsics: dict with camera parameters (fx, fy, cx, cy)
    
    Returns:
        points: Nx3 array of 3D points
    """
    rows, cols = depth_img.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # Convert to 3D coordinates
    z = depth_img.astype(float) / 1000.0  # Convert from mm to meters
    x = (c - intrinsics['cx']) * z / intrinsics['fx']
    y = (r - intrinsics['cy']) * z / intrinsics['fy']
    
    # Stack coordinates and remove invalid points (where depth is 0)
    points = np.stack([x, y, z], axis=-1)
    valid_mask = depth_img > 0
    points = points[valid_mask]
    
    return points

def prepare_ucl_data(data_dir, output_file, max_points=1024):
    """
    Process UCL data and save to H5 file.
    
    Args:
        data_dir: Path to UCL data directory
        output_file: Path to output H5 file
        max_points: Maximum number of points per frame
    """
    print(f"Processing UCL data from {data_dir}")
    
    # Load camera intrinsics
    intrinsics_path = os.path.join(data_dir, 'intrinsics.txt')
    intrinsics = load_camera_intrinsics(intrinsics_path)
    print(f"Loaded camera intrinsics: {intrinsics}")
    
    # Define paths
    tsdf_dir = os.path.join(data_dir, 'depthTSDF')
    
    # Get list of TSDF files
    tsdf_files = sorted(glob.glob(os.path.join(tsdf_dir, '*.png')))
    print(f"Found {len(tsdf_files)} TSDF files")
    
    # Process each frame
    all_points = []
    all_labels = []
    
    # Tracking statistics
    total_frames = 0
    total_table_points = 0
    total_points = 0
    
    for tsdf_file in tqdm(tsdf_files, desc="Processing frames"):
        # Load depth image
        depth_img = cv2.imread(tsdf_file, cv2.IMREAD_ANYDEPTH)
        
        if depth_img is None:
            print(f"Warning: Could not load TSDF image {tsdf_file}")
            continue
        
        # Convert to point cloud
        points = depth_to_pointcloud(depth_img, intrinsics)
        
        # Skip if no valid points
        if len(points) == 0:
            print(f"Warning: No valid points in frame {tsdf_file}")
            continue
        
        # IMPORTANT: UCL data contains tables, so label ALL points as tables (label 1)
        labels = np.ones(len(points), dtype=np.int32)
        
        # Update statistics
        total_frames += 1
        frame_points = len(points)
        total_points += frame_points
        total_table_points += frame_points  # All points are labeled as tables
        
        # Randomly sample points if max_points is specified
        if max_points is not None and len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            labels = labels[indices]  # Update labels too
            
            # Update statistics after sampling
            frame_points = len(points)
            
        # Print frame statistics
        print(f"Frame {os.path.basename(tsdf_file)}: {frame_points} points (all labeled as tables)")
        
        all_points.append(points)
        all_labels.append(labels)
    
    # Concatenate all points and labels
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=all_points)
        f.create_dataset('label', data=all_labels)
    
    # Print summary statistics
    print(f"\n=== UCL DATASET SUMMARY ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total points: {len(all_points)}")
    print(f"Table points: {np.sum(all_labels)}/{len(all_labels)} ({np.sum(all_labels)/len(all_labels)*100:.2f}%)")
    print(f"All points are labeled as tables (1), since UCL data contains tables")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    data_dir = 'dataset/ucl_data'
    output_file = 'CW2-Dataset/ucl_data.h5'
    
    # Process the UCL data
    prepare_ucl_data(data_dir, output_file, max_points=1024) 