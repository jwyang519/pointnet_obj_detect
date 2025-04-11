import os
import numpy as np
import h5py
import pickle
import cv2
from tqdm import tqdm
from skimage.draw import polygon
import argparse
import glob

def load_camera_intrinsics(intrinsics_path):
    """
    Load camera intrinsics from the intrinsics.txt file.
    
    Args:
        intrinsics_path: Path to the intrinsics.txt file
    
    Returns:
        dict with camera parameters (fx, fy, cx, cy)
    """
    try:
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            # Parse the intrinsics matrix
            fx = float(lines[0].split()[0])
            fy = float(lines[1].split()[1])
            cx = float(lines[0].split()[2])
            cy = float(lines[1].split()[2])
            
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
            'fx': 570.34,
            'fy': 570.34,
            'cx': 320.0,
            'cy': 240.0
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
    z = depth_img
    x = (c - intrinsics['cx']) * z / intrinsics['fx']
    y = (r - intrinsics['cy']) * z / intrinsics['fy']
    
    # Stack coordinates and remove invalid points (where depth is 0)
    points = np.stack([x, y, z], axis=-1)
    valid_mask = depth_img > 0
    points = points[valid_mask]
    
    return points

def create_polygon_mask(polygon_points, image_shape):
    """
    Create a binary mask from polygon vertices.
    
    Args:
        polygon_points: List of [x, y] coordinates
        image_shape: (height, width) of the output mask
    
    Returns:
        mask: Binary mask where 1 indicates points inside polygon
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(polygon_points) < 3:  # Need at least 3 points for a polygon
        return mask
    
    # Convert polygon points to the format expected by skimage.draw.polygon
    rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0])
    mask[rr, cc] = 1
    return mask

def process_frame(depth_img, polygon_list, intrinsics):
    """
    Process a single frame to generate labeled point cloud.
    
    Args:
        depth_img: HxW depth image
        polygon_list: List of polygon vertices for tables
        intrinsics: Camera parameters
    
    Returns:
        points: Nx3 array of 3D points
        labels: Nx1 array of binary labels
    """
    # Get image dimensions for debugging
    height, width = depth_img.shape
    print(f"Depth image dimensions: {width}x{height}")
    
    # Create binary mask for all polygons
    mask = np.zeros(depth_img.shape, dtype=np.uint8)
    
    # Track polygon processing
    polygons_processed = 0
    valid_polygons = 0
    invalid_formats = 0
    no_points = 0
    
    print(f"Processing {len(polygon_list)} polygons")
    
    # First, create a mask of valid depth pixels
    valid_depth_mask = depth_img > 0
    valid_depth_pixels = np.sum(valid_depth_mask)
    print(f"Valid depth pixels: {valid_depth_pixels}/{width*height} ({valid_depth_pixels/(width*height)*100:.2f}%)")
    
    # Create coordinate maps to match 3D points back to 2D pixels
    y_indices, x_indices = np.nonzero(valid_depth_mask)
    
    # Process each polygon and add to mask
    for poly_idx, poly in enumerate(polygon_list):
        try:
            # Check if we have a valid polygon format
            poly_array = np.array(poly)
            
            if len(poly_array.shape) == 2 and poly_array.shape[0] == 2:
                # Format: [x_coordinates, y_coordinates]
                x_coords, y_coords = poly_array
                
                # Print some debug info about the polygon
                print(f"Polygon {poly_idx}: {len(x_coords)} points")
                print(f"  X range: {np.min(x_coords):.1f} to {np.max(x_coords):.1f}")
                print(f"  Y range: {np.min(y_coords):.1f} to {np.max(y_coords):.1f}")
                
                # Skip if not enough points for a polygon (need at least 3)
                if len(x_coords) < 3:
                    print(f"  Skipping: Not enough points")
                    no_points += 1
                    continue
                
                # Check if coordinates are within image bounds
                out_of_bounds_x = np.sum((x_coords < 0) | (x_coords >= width))
                out_of_bounds_y = np.sum((y_coords < 0) | (y_coords >= height))
                
                if out_of_bounds_x > 0 or out_of_bounds_y > 0:
                    print(f"  Warning: {out_of_bounds_x + out_of_bounds_y} points out of bounds")
                
                try:
                    # Convert to integer coordinates for polygon function
                    x_coords_int = np.round(x_coords).astype(np.int32)
                    y_coords_int = np.round(y_coords).astype(np.int32)
                    
                    # Create polygon mask - y_coords (rows) first, then x_coords (cols)
                    rr, cc = polygon(y_coords_int, x_coords_int, shape=depth_img.shape)
                    
                    # Add to mask
                    if len(rr) > 0:
                        mask[rr, cc] = 1
                        valid_polygons += 1
                        print(f"  Success: Filled {len(rr)} pixels")
                    else:
                        print(f"  Warning: No pixels in polygon")
                    
                except Exception as e:
                    print(f"  Error creating polygon: {e}")
            else:
                print(f"  Invalid polygon format: shape {poly_array.shape}")
                invalid_formats += 1
                
            polygons_processed += 1
                
        except Exception as e:
            print(f"  Error processing polygon {poly_idx}: {e}")
    
    # Convert depth to 3D points - this only returns points for valid depth values
    points = depth_to_pointcloud(depth_img, intrinsics)
    
    # CRITICAL FIX: We need to apply the mask ONLY to pixels with valid depth
    # Use the valid_depth_mask to extract table labels
    valid_points_mask = mask[valid_depth_mask]
    labels = valid_points_mask.astype(np.int32)
    
    # Verify lengths
    if len(points) != len(labels):
        print(f"WARNING: Mismatch between points ({len(points)}) and labels ({len(labels)})")
    
    # Print statistics
    table_points = np.sum(labels)
    total_points = len(labels)
    table_pixels = np.sum(mask)
    total_pixels = width * height
    
    print(f"Polygon processing summary:")
    print(f"  Total polygons: {len(polygon_list)}")
    print(f"  Processed: {polygons_processed}")
    print(f"  Valid: {valid_polygons}")
    print(f"  Invalid format: {invalid_formats}")
    print(f"  Too few points: {no_points}")
    print(f"Mask statistics:")
    print(f"  Table pixels in mask: {table_pixels}/{total_pixels} ({table_pixels/total_pixels*100:.2f}%)")
    print(f"  Table points in cloud: {table_points}/{total_points} ({table_points/max(1, total_points)*100:.2f}%)")
    
    return points, labels

def find_scenes_by_prefix(base_dir, prefix):
    """
    Find all scene directories that start with the given prefix.
    
    Args:
        base_dir: Base directory of the dataset
        prefix: Prefix to match (e.g., 'mit' or 'harvard')
    
    Returns:
        List of scene directories
    """
    scenes = []
    for building in os.listdir(base_dir):
        if building.startswith(prefix):
            building_path = os.path.join(base_dir, building)
            if os.path.isdir(building_path):
                for scene in os.listdir(building_path):
                    scene_path = os.path.join(building_path, scene)
                    if os.path.isdir(scene_path):
                        # Check if this is a valid scene directory with necessary components
                        has_depth = os.path.exists(os.path.join(scene_path, 'depth'))
                        has_tsdf = os.path.exists(os.path.join(scene_path, 'depthTSDF'))
                        has_intrinsics = os.path.exists(os.path.join(scene_path, 'intrinsics.txt'))
                        has_labels = os.path.exists(os.path.join(scene_path, 'labels')) and \
                                     os.path.exists(os.path.join(scene_path, 'labels/tabletop_labels.dat'))
                        
                        # Scene is valid if it has either depth or depthTSDF, and has intrinsics and labels
                        if (has_depth or has_tsdf) and has_intrinsics and has_labels:
                            scenes.append(scene_path)
    return scenes

def process_scenes(scenes, output_file, max_points=None):
    """
    Process a list of scenes and save to HDF5 file.
    
    Args:
        scenes: List of scene directories to process
        output_file: Path to output HDF5 file
        max_points: Maximum number of points per frame (None for no limit)
    """
    all_points = []
    all_labels = []
    
    # Tracking variables for dataset statistics
    total_frames_processed = 0
    frames_with_tables = 0
    total_table_points = 0
    total_points = 0
    
    # Process each scene
    for scene_path in tqdm(scenes, desc="Processing scenes"):
        scene_name = os.path.basename(os.path.dirname(scene_path)) + '/' + os.path.basename(scene_path)
        print(f"\n\n==== Processing scene: {scene_name} ====")
        
        # Scene statistics
        scene_frames_processed = 0
        scene_frames_with_tables = 0
        scene_table_points = 0
        scene_total_points = 0
        
        # Load camera intrinsics
        intrinsics_path = os.path.join(scene_path, 'intrinsics.txt')
        intrinsics = load_camera_intrinsics(intrinsics_path)
        
        # Load polygon labels
        labels_path = os.path.join(scene_path, 'labels/tabletop_labels.dat')
        try:
            with open(labels_path, 'rb') as f_labels:
                tabletop_labels = pickle.load(f_labels)
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            continue
        
        # Count frames with table polygons
        frames_with_polygons = sum(1 for polygons in tabletop_labels if len(polygons) > 0)
        print(f"Found {frames_with_polygons}/{len(tabletop_labels)} frames with table polygons")
        
        # Get list of depth files, preferring depthTSDF if available
        has_tsdf = os.path.exists(os.path.join(scene_path, 'depthTSDF'))
        
        if has_tsdf:
            print(f"Using depthTSDF for scene {scene_name}")
            depth_dir = os.path.join(scene_path, 'depthTSDF')
        else:
            print(f"WARNING: No depthTSDF found, using raw depth for scene {scene_name}")
            print(f"         Consider running depth_to_tsdf.py to convert depth to TSDF")
            depth_dir = os.path.join(scene_path, 'depth')
        
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        
        # Ensure we have matching labels and depth files
        min_count = min(len(depth_files), len(tabletop_labels))
        if min_count < len(depth_files) or min_count < len(tabletop_labels):
            print(f"WARNING: Mismatch between depth files ({len(depth_files)}) and labels ({len(tabletop_labels)})")
            print(f"Processing only the first {min_count} frames")
        
        # Process each frame
        for idx, (depth_file, polygon_list) in enumerate(tqdm(list(zip(depth_files[:min_count], tabletop_labels[:min_count])), desc=f"Processing {scene_name}")):
            print(f"\n--- Frame {idx}: {depth_file} ---")
            
            # Check if frame has table polygons
            has_table_polygons = len(polygon_list) > 0
            if has_table_polygons:
                print(f"Frame has {len(polygon_list)} table polygons")
            else:
                print(f"Frame has no table polygons")
            
            # Load depth image
            depth_path = os.path.join(depth_dir, depth_file)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                print(f"Warning: Could not load depth image {depth_path}")
                continue
            
            # Process frame
            points, labels = process_frame(depth_img, polygon_list, intrinsics)
            
            # Skip if no valid points
            if len(points) == 0:
                print(f"Warning: No valid points in frame {depth_file}")
                continue
            
            # Update statistics
            table_points_in_frame = np.sum(labels)
            total_points_in_frame = len(labels)
            has_table_points = table_points_in_frame > 0
            
            scene_frames_processed += 1
            scene_total_points += total_points_in_frame
            scene_table_points += table_points_in_frame
            if has_table_points:
                scene_frames_with_tables += 1
            
            # Randomly sample points if max_points is specified
            if max_points is not None and len(points) > max_points:
                # We need to preserve the ratio of table to non-table points
                table_indices = np.where(labels == 1)[0]
                non_table_indices = np.where(labels == 0)[0]
                
                # Calculate how many points to sample from each class
                table_count = len(table_indices)
                non_table_count = len(non_table_indices)
                
                if table_count > 0:
                    # Ensure we keep some table points
                    table_ratio = table_count / (table_count + non_table_count)
                    table_samples = min(max(int(max_points * table_ratio), 1), table_count)
                    non_table_samples = max_points - table_samples
                    
                    # Sample from each class
                    sampled_table_indices = np.random.choice(table_indices, table_samples, replace=False)
                    sampled_non_table_indices = np.random.choice(non_table_indices, non_table_samples, replace=False)
                    
                    # Combine indices
                    sampled_indices = np.concatenate([sampled_table_indices, sampled_non_table_indices])
                else:
                    # No table points, just sample randomly
                    sampled_indices = np.random.choice(len(points), max_points, replace=False)
                
                # Apply sampling
                points = points[sampled_indices]
                labels = labels[sampled_indices]
                
                # Update statistics after sampling
                table_points_in_frame = np.sum(labels)
                total_points_in_frame = len(labels)
                
                print(f"Sampled {len(points)} points from original {len(labels)}")
                print(f"After sampling: {table_points_in_frame}/{total_points_in_frame} table points ({table_points_in_frame/total_points_in_frame*100:.2f}%)")
            
            all_points.append(points)
            all_labels.append(labels)
        
        # Update global statistics
        total_frames_processed += scene_frames_processed
        frames_with_tables += scene_frames_with_tables
        total_table_points += scene_table_points
        total_points += scene_total_points
        
        # Print scene summary
        print(f"\n=== Scene Summary: {scene_name} ===")
        print(f"Processed {scene_frames_processed} frames")
        print(f"Frames with table points: {scene_frames_with_tables}/{scene_frames_processed} ({scene_frames_with_tables/max(1, scene_frames_processed)*100:.2f}%)")
        print(f"Total points: {scene_total_points}")
        print(f"Table points: {scene_table_points}/{scene_total_points} ({scene_table_points/max(1, scene_total_points)*100:.2f}%)")
    
    # Concatenate all points and labels
    if not all_points:
        print("ERROR: No valid points were processed!")
        return
        
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)
    
    # Calculate final statistics
    final_table_points = np.sum(all_labels)
    final_total_points = len(all_labels)
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=all_points)
        f.create_dataset('label', data=all_labels)
    
    # Print overall summary
    print(f"\n\n====== OVERALL DATASET SUMMARY ======")
    print(f"Total scenes processed: {len(scenes)}")
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Frames with table points: {frames_with_tables}/{total_frames_processed} ({frames_with_tables/max(1, total_frames_processed)*100:.2f}%)")
    print(f"Total points: {final_total_points}")
    print(f"Table points: {final_table_points}/{final_total_points} ({final_table_points/max(1, final_total_points)*100:.2f}%)")
    print(f"Dataset saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='CW2-Dataset/data', 
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to output HDF5 file (default: CW2-Dataset/sun3d_{train|test}.h5)')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Prefix to filter buildings (e.g., mit, harvard)')
    parser.add_argument('--scene', type=str, default=None,
                        help='Specific scene path to process (e.g., mit_76_studyroom/76-1studyroom2)')
    parser.add_argument('--max_points', type=int, default=1024, 
                        help='Maximum number of points per frame')
    parser.add_argument('--check_only', action='store_true', default=False,
                        help='Only check which scenes would be processed')
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output) if args.output else 'CW2-Dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if depth to TSDF conversion is needed
    needs_conversion = []
    
    if args.scene:
        # Process a specific scene
        scene_parts = args.scene.split('/')
        if len(scene_parts) != 2:
            print(f"Scene should be in format 'building/scene', got {args.scene}")
            return
        
        building, scene = scene_parts
        scene_path = os.path.join(args.data_dir, building, scene)
        
        if not os.path.exists(scene_path):
            print(f"Scene not found: {scene_path}")
            return
        
        # Check if conversion is needed
        has_depth = os.path.exists(os.path.join(scene_path, 'depth'))
        has_tsdf = os.path.exists(os.path.join(scene_path, 'depthTSDF'))
        
        if has_depth and not has_tsdf:
            needs_conversion.append(scene_path)
            print(f"WARNING: Scene {args.scene} has depth but no depthTSDF")
            print(f"         Consider running depth_to_tsdf.py to convert depth to TSDF")
        
        # Set output path if not specified
        if not args.output:
            args.output = os.path.join(output_dir, f"{building}_{scene}.h5")
        
        if args.check_only:
            print(f"Would process scene: {scene_path}")
            print(f"Output would go to: {args.output}")
            return
        
        # Process the scene
        process_scenes([scene_path], args.output, args.max_points)
    else:
        # Find scenes based on prefix
        prefix = args.prefix or ""
        scenes = find_scenes_by_prefix(args.data_dir, prefix)
        
        if not scenes:
            print(f"No scenes found with prefix '{prefix}'")
            return
        
        print(f"Found {len(scenes)} scenes with prefix '{prefix}'")
        
        # Check which scenes need conversion
        for scene_path in scenes:
            has_depth = os.path.exists(os.path.join(scene_path, 'depth'))
            has_tsdf = os.path.exists(os.path.join(scene_path, 'depthTSDF'))
            
            if has_depth and not has_tsdf:
                needs_conversion.append(scene_path)
        
        if needs_conversion:
            print(f"WARNING: {len(needs_conversion)} scenes have depth but no depthTSDF")
            for scene in needs_conversion:
                scene_name = os.path.basename(os.path.dirname(scene)) + '/' + os.path.basename(scene)
                print(f"  - {scene_name}")
            print(f"Consider running depth_to_tsdf.py to convert depth to TSDF")
        
        if args.check_only:
            print("Would process the following scenes:")
            for scene in scenes:
                scene_name = os.path.basename(os.path.dirname(scene)) + '/' + os.path.basename(scene)
                print(f"  - {scene_name}")
            print(f"Output would go to: {args.output or 'CW2-Dataset/sun3d_[train|test].h5'}")
            return
        
        # Process scenes based on prefix for train/test split
        if prefix.lower() == "mit":
            if not args.output:
                args.output = os.path.join(output_dir, 'sun3d_train.h5')
            print(f"Processing {len(scenes)} MIT scenes as training data")
            process_scenes(scenes, args.output, args.max_points)
        elif prefix.lower() == "harvard":
            if not args.output:
                args.output = os.path.join(output_dir, 'sun3d_test.h5')
            print(f"Processing {len(scenes)} Harvard scenes as testing data")
            process_scenes(scenes, args.output, args.max_points)
        else:
            # Use a generic output name if not specified
            if not args.output:
                args.output = os.path.join(output_dir, f"sun3d_{prefix or 'combined'}.h5")
            print(f"Processing {len(scenes)} scenes with prefix '{prefix}'")
            process_scenes(scenes, args.output, args.max_points)

if __name__ == "__main__":
    main() 