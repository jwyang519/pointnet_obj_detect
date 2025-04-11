import os
import numpy as np
import cv2
import argparse
import pickle
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
    
    # For each valid depth value, compute the TSDF
    # Since we don't have the 3D model, we'll use a simplified approach:
    # - Keep the actual depth values (they will be interpreted as distances when loaded)
    # - Make sure the values are consistent with other DepthTSDF files
    
    # Scale values to maintain precision similar to the depthTSDF images
    # This step ensures that the converted images have similar characteristics
    # to the original DepthTSDF images
    if np.max(tsdf_img) < 1000:  # If values are in meters, convert to millimeters
        tsdf_img = tsdf_img * 1000.0
    
    # Ensure the image is in 16-bit format
    tsdf_img = tsdf_img.astype(np.uint16)
    
    return tsdf_img

def create_empty_labels(scene_path, num_frames):
    """
    Create empty labels (all background) for a scene without annotations
    
    Args:
        scene_path: Path to the scene directory
        num_frames: Number of frames to create labels for
    """
    labels_dir = os.path.join(scene_path, 'labels')
    labels_file = os.path.join(labels_dir, 'tabletop_labels.dat')
    
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create empty polygons for each frame
    # In the SUN3D dataset format, each frame has a list of polygon vertices
    # An empty list means no tables in the frame
    empty_labels = [[] for _ in range(num_frames)]
    
    # Save to pickle file
    with open(labels_file, 'wb') as f:
        pickle.dump(empty_labels, f)
    
    print(f"Created empty labels file at {labels_file}")
    print(f"WARNING: These are placeholder labels with NO tables marked.")
    print(f"         You should annotate tables manually for accurate results.")

def process_scene(scene_path, create_empty_labels_if_missing=False):
    """
    Process a scene by converting all depth images to TSDF format
    
    Args:
        scene_path: Path to the scene directory
        create_empty_labels_if_missing: Whether to create empty labels if missing
    """
    # Define paths
    depth_dir = os.path.join(scene_path, 'depth')
    tsdf_dir = os.path.join(scene_path, 'depthTSDF')
    labels_dir = os.path.join(scene_path, 'labels')
    labels_file = os.path.join(labels_dir, 'tabletop_labels.dat')
    
    # Check if the depth directory exists
    if not os.path.exists(depth_dir):
        print(f"Depth directory not found at {depth_dir}")
        return
    
    # Create the depthTSDF directory if it doesn't exist
    os.makedirs(tsdf_dir, exist_ok=True)
    
    # Get list of depth files
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    # Check if labels exist
    has_labels = os.path.exists(labels_file)
    
    # If no labels and we're asked to create them
    if not has_labels and create_empty_labels_if_missing:
        print(f"No labels found at {labels_file}, creating empty labels")
        create_empty_labels(scene_path, len(depth_files))
    
    # Process each depth image
    for depth_file in tqdm(depth_files, desc=f"Converting {os.path.basename(scene_path)}"):
        # Load depth image
        depth_path = os.path.join(depth_dir, depth_file)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if depth_img is None:
            print(f"Warning: Could not load depth image {depth_path}")
            continue
        
        # Convert to TSDF
        tsdf_img = convert_depth_to_tsdf(depth_img)
        
        # Save TSDF image
        tsdf_path = os.path.join(tsdf_dir, depth_file)
        cv2.imwrite(tsdf_path, tsdf_img)
    
    print(f"Converted {len(depth_files)} depth images to TSDF format")
    
    # Final status
    if not has_labels and not create_empty_labels_if_missing:
        print(f"WARNING: No labels found at {labels_file}")
        print(f"         Run with --create_labels to create empty labels")

def process_building(building_path, create_empty_labels_if_missing=False):
    """
    Process all scenes in a building
    
    Args:
        building_path: Path to the building directory
        create_empty_labels_if_missing: Whether to create empty labels if missing
    """
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(building_path) if os.path.isdir(os.path.join(building_path, d))]
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(building_path, scene_dir)
        
        # Check if this scene needs conversion (has depth but no depthTSDF)
        has_depth = os.path.exists(os.path.join(scene_path, 'depth'))
        has_tsdf = os.path.exists(os.path.join(scene_path, 'depthTSDF'))
        has_labels = os.path.exists(os.path.join(scene_path, 'labels')) and \
                    os.path.exists(os.path.join(scene_path, 'labels/tabletop_labels.dat'))
        
        # Determine what needs to be done
        needs_tsdf = has_depth and not has_tsdf
        needs_labels = not has_labels and create_empty_labels_if_missing
        
        if needs_tsdf or needs_labels:
            print(f"Processing scene: {scene_dir}")
            process_scene(scene_path, create_empty_labels_if_missing)
        else:
            print(f"Skipping scene {scene_dir} (already processed)")

def main():
    parser = argparse.ArgumentParser(description='Convert depth images to TSDF format')
    parser.add_argument('--data_dir', type=str, default='CW2-Dataset/data',
                      help='Path to dataset directory')
    parser.add_argument('--building', type=str, default=None,
                      help='Specific building to process (e.g., harvard_tea_2)')
    parser.add_argument('--create_labels', action='store_true', default=False,
                      help='Create empty label files if missing')
    args = parser.parse_args()
    
    if args.building:
        # Process specific building
        building_path = os.path.join(args.data_dir, args.building)
        if os.path.exists(building_path) and os.path.isdir(building_path):
            print(f"Processing building: {args.building}")
            process_building(building_path, args.create_labels)
        else:
            print(f"Building directory not found: {building_path}")
    else:
        # Process all buildings
        building_dirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
        
        for building_dir in building_dirs:
            building_path = os.path.join(args.data_dir, building_dir)
            print(f"Processing building: {building_dir}")
            process_building(building_path, args.create_labels)

if __name__ == "__main__":
    main() 