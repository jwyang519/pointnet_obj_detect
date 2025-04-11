import os
import numpy as np
import cv2
import pickle
import glob
from tqdm import tqdm

def convert_depth_to_tsdf(depth_img, truncation_distance=0.1):
    """
    Convert a raw depth image to a truncated signed distance function (TSDF) representation.
    
    Args:
        depth_img: HxW depth image (in meters)
        truncation_distance: Truncation distance for the TSDF (in meters)
    
    Returns:
        tsdf_img: HxW TSDF image
    """
    # Create a copy of the depth image
    tsdf_img = depth_img.copy()
    
    # Scale values to maintain precision similar to the depthTSDF images
    # This step ensures that the converted images have similar characteristics
    # to the original DepthTSDF images
    if np.max(tsdf_img) < 1000:  # If values are in meters, convert to millimeters
        tsdf_img = tsdf_img * 1000.0
    
    # Ensure the image is in 16-bit format
    tsdf_img = tsdf_img.astype(np.uint16)
    
    return tsdf_img

def process_ucl_data(data_dir, output_h5=None):
    """
    Process UCL data: convert depth to TSDF and create empty labels
    
    Args:
        data_dir: Path to UCL data directory
        output_h5: Path to output H5 file (if None, will only convert data)
    """
    # Define paths
    depth_dir = os.path.join(data_dir, 'depth')
    tsdf_dir = os.path.join(data_dir, 'depthTSDF')
    labels_dir = os.path.join(data_dir, 'labels')
    
    # Make directories if they don't exist
    os.makedirs(tsdf_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get list of depth files
    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
    
    # Process each depth image
    print(f"Converting {len(depth_files)} depth images to TSDF format...")
    for depth_file in tqdm(depth_files):
        # Load depth image
        depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        if depth_img is None:
            print(f"Warning: Could not load depth image {depth_file}")
            continue
        
        # Convert to TSDF
        tsdf_img = convert_depth_to_tsdf(depth_img)
        
        # Save TSDF image (use same filename, but in the depthTSDF directory)
        tsdf_path = os.path.join(tsdf_dir, os.path.basename(depth_file))
        cv2.imwrite(tsdf_path, tsdf_img)
    
    # Create empty labels (all background) for each frame
    print("Creating empty labels file...")
    labels_path = os.path.join(labels_dir, 'tabletop_labels.dat')
    empty_labels = [[] for _ in range(len(depth_files))]
    
    with open(labels_path, 'wb') as f:
        pickle.dump(empty_labels, f)
    
    print(f"Converted {len(depth_files)} depth images to TSDF format")
    print(f"Created empty labels file at {labels_path}")
    
    # Now the data is ready to be processed into an H5 file
    print("Data preparation complete!")
    print("You can now run prepare_sun3d_dataset.py to convert to H5 format")

if __name__ == "__main__":
    # Path to UCL data directory
    data_dir = 'dataset/ucl_data'
    
    # Process the data
    process_ucl_data(data_dir) 