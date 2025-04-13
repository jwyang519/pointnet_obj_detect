"""
Check table labels in SUN3D dataset to count how many images contain tables.
"""

import os
import pickle
import numpy as np
import glob
from tqdm import tqdm

def check_table_labels(base_dir):
    """
    Check all label files in the dataset and count images with tables.
    
    Args:
        base_dir: Base directory of SUN3D dataset
    
    Returns:
        Dictionary with statistics
    """
    results = {
        'harvard': {
            'scenes': 0,
            'total_images': 0,
            'images_with_tables': 0,
            'scene_details': {}
        },
        'mit': {
            'scenes': 0,
            'total_images': 0,
            'images_with_tables': 0,
            'scene_details': {}
        }
    }
    
    # List all directories
    for building_dir in os.listdir(base_dir):
        building_path = os.path.join(base_dir, building_dir)
        if not os.path.isdir(building_path):
            continue
        
        # Determine if this is Harvard or MIT
        if building_dir.startswith('harvard'):
            prefix = 'harvard'
        elif building_dir.startswith('mit'):
            prefix = 'mit'
        else:
            print(f"Unknown building prefix: {building_dir}")
            continue
        
        # Count scenes in this building
        for scene_dir in os.listdir(building_path):
            scene_path = os.path.join(building_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            
            # Check if this scene has labels
            labels_path = os.path.join(scene_path, 'labels', 'tabletop_labels.dat')
            if not os.path.exists(labels_path):
                print(f"No labels found for scene: {os.path.join(building_dir, scene_dir)}")
                continue
            
            # Load label file
            try:
                with open(labels_path, 'rb') as f:
                    tabletop_labels = pickle.load(f)
            except Exception as e:
                print(f"Error loading labels from {labels_path}: {e}")
                continue
            
            # Count images and images with tables
            total_images = len(tabletop_labels)
            images_with_tables = sum(1 for polygons in tabletop_labels if len(polygons) > 0)
            
            # Add to statistics
            results[prefix]['scenes'] += 1
            results[prefix]['total_images'] += total_images
            results[prefix]['images_with_tables'] += images_with_tables
            
            # Add scene details
            scene_name = f"{building_dir}/{scene_dir}"
            results[prefix]['scene_details'][scene_name] = {
                'total_images': total_images,
                'images_with_tables': images_with_tables,
                'percentage_with_tables': (images_with_tables / total_images * 100) if total_images > 0 else 0
            }
    
    return results

def main():
    base_dir = 'dataset/sun3d_data'
    print(f"Checking table labels in {base_dir}...")
    
    results = check_table_labels(base_dir)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for prefix in ['harvard', 'mit']:
        total_images = results[prefix]['total_images']
        images_with_tables = results[prefix]['images_with_tables']
        percentage = (images_with_tables / total_images * 100) if total_images > 0 else 0
        
        print(f"\n{prefix.upper()} SCENES:")
        print(f"  Number of scenes: {results[prefix]['scenes']}")
        print(f"  Total images: {total_images}")
        print(f"  Images with tables: {images_with_tables} ({percentage:.2f}%)")
        
        print("\n  SCENE DETAILS:")
        for scene_name, details in results[prefix]['scene_details'].items():
            print(f"    {scene_name}:")
            print(f"      Images: {details['total_images']}")
            print(f"      With tables: {details['images_with_tables']} ({details['percentage_with_tables']:.2f}%)")
    
    print("\n=== OVERALL ===")
    total_harvard = results['harvard']['total_images']
    harvard_with_tables = results['harvard']['images_with_tables']
    total_mit = results['mit']['total_images']
    mit_with_tables = results['mit']['images_with_tables']
    
    total_images = total_harvard + total_mit
    total_with_tables = harvard_with_tables + mit_with_tables
    total_percentage = (total_with_tables / total_images * 100) if total_images > 0 else 0
    
    print(f"Total scenes: {results['harvard']['scenes'] + results['mit']['scenes']}")
    print(f"Total images: {total_images}")
    print(f"Images with tables: {total_with_tables} ({total_percentage:.2f}%)")

if __name__ == "__main__":
    main() 