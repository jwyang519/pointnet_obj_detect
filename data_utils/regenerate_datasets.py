"""
Regenerate the SUN3D and UCL datasets with fixed label processing.

This script:
1. Regenerates sun3d_train.h5 (MIT data with corrected polygon processing)
2. Regenerates sun3d_test.h5 (Harvard data with corrected polygon processing)
3. Regenerates ucl_data.h5 (UCL data with all points labeled as tables)
"""

import os
import sys
import subprocess
import argparse
import h5py
import numpy as np

def print_h5_info(h5_file, max_points_per_cloud=1024):
    """Print information about an h5 file, including how many point clouds have tables"""
    try:
        with h5py.File(h5_file, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]
            
            # Calculate basic stats
            table_points = np.sum(labels)
            total_points = len(labels)
            table_percentage = (table_points / total_points * 100) if total_points > 0 else 0
            
            # Estimate the number of point clouds based on max_points
            # This assumes each point cloud has approximately max_points points
            estimated_num_clouds = total_points // max_points_per_cloud
            if total_points % max_points_per_cloud > 0:
                estimated_num_clouds += 1
                
            # Check if dataset has 'sample_indices' attribute to identify point cloud boundaries
            # If not, we'll use our estimate based on max_points
            if 'sample_indices' in f:
                indices = f['sample_indices'][:]
                clouds_with_tables = 0
                
                for i in range(len(indices)-1):
                    start_idx = indices[i]
                    end_idx = indices[i+1]
                    cloud_labels = labels[start_idx:end_idx]
                    if np.sum(cloud_labels) > 0:
                        clouds_with_tables += 1
                
                total_clouds = len(indices) - 1
            else:
                # If we don't have indices, try to estimate based on max_points
                clouds_with_tables = 0
                total_clouds = 0
                
                # Process data in chunks of max_points
                for i in range(0, total_points, max_points_per_cloud):
                    end_idx = min(i + max_points_per_cloud, total_points)
                    cloud_labels = labels[i:end_idx]
                    if np.sum(cloud_labels) > 0:
                        clouds_with_tables += 1
                    total_clouds += 1
            
            # Print information
            print(f"\nDataset information for {h5_file}:")
            print(f"  Total points: {total_points}")
            print(f"  Points labeled as tables: {table_points} ({table_percentage:.2f}%)")
            print(f"  Estimated total point clouds: {total_clouds}")
            print(f"  Point clouds with tables: {clouds_with_tables} ({clouds_with_tables/total_clouds*100:.2f}%)")
            
            return {
                'total_points': total_points,
                'table_points': table_points,
                'total_clouds': total_clouds,
                'clouds_with_tables': clouds_with_tables
            }
    except Exception as e:
        print(f"Error reading {h5_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sun3d_only', action='store_true', default=False, 
                        help='Only regenerate SUN3D datasets (skip UCL)')
    parser.add_argument('--ucl_only', action='store_true', default=False, 
                        help='Only regenerate UCL dataset (skip SUN3D)')
    parser.add_argument('--check_only', action='store_true', default=False,
                        help='Only check existing datasets without regenerating')
    parser.add_argument('--max_points', type=int, default=1024,
                        help='Maximum number of points per frame')
    args = parser.parse_args()
    
    # Directory for output files
    output_dir = 'CW2-Dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define dataset files
    sun3d_train_file = f'{output_dir}/sun3d_train_fixed.h5'
    sun3d_test_file = f'{output_dir}/sun3d_test_fixed.h5'
    ucl_data_file = f'{output_dir}/ucl_data_fixed.h5'
    
    # Check if we should just analyze existing files
    if args.check_only:
        print("=== Checking existing datasets ===")
        # Check original files
        original_train = f'{output_dir}/sun3d_train.h5'
        original_test = f'{output_dir}/sun3d_test.h5'
        original_ucl = f'{output_dir}/ucl_data.h5'
        
        print("\n=== ORIGINAL DATASETS ===")
        train_stats = None
        test_stats = None
        ucl_stats = None
        
        if os.path.exists(original_train):
            train_stats = print_h5_info(original_train, args.max_points)
        else:
            print(f"{original_train} does not exist")
        
        if os.path.exists(original_test):
            test_stats = print_h5_info(original_test, args.max_points)
        else:
            print(f"{original_test} does not exist")
            
        if os.path.exists(original_ucl):
            ucl_stats = print_h5_info(original_ucl, args.max_points)
        else:
            print(f"{original_ucl} does not exist")
        
        # Check fixed files
        print("\n=== FIXED DATASETS ===")
        fixed_train_stats = None
        fixed_test_stats = None
        fixed_ucl_stats = None
        
        if os.path.exists(sun3d_train_file):
            fixed_train_stats = print_h5_info(sun3d_train_file, args.max_points)
        else:
            print(f"{sun3d_train_file} does not exist")
            
        if os.path.exists(sun3d_test_file):
            fixed_test_stats = print_h5_info(sun3d_test_file, args.max_points)
        else:
            print(f"{sun3d_test_file} does not exist")
            
        if os.path.exists(ucl_data_file):
            fixed_ucl_stats = print_h5_info(ucl_data_file, args.max_points)
        else:
            print(f"{ucl_data_file} does not exist")
        
        # Print summary
        print("\n=== SUMMARY: POINT CLOUDS WITH TABLES ===")
        if train_stats and fixed_train_stats:
            print(f"SUN3D Train: Original {train_stats['clouds_with_tables']}/{train_stats['total_clouds']} " + 
                  f"({train_stats['clouds_with_tables']/train_stats['total_clouds']*100:.2f}%), " +
                  f"Fixed {fixed_train_stats['clouds_with_tables']}/{fixed_train_stats['total_clouds']} " +
                  f"({fixed_train_stats['clouds_with_tables']/fixed_train_stats['total_clouds']*100:.2f}%)")
        
        if test_stats and fixed_test_stats:
            print(f"SUN3D Test: Original {test_stats['clouds_with_tables']}/{test_stats['total_clouds']} " + 
                  f"({test_stats['clouds_with_tables']/test_stats['total_clouds']*100:.2f}%), " +
                  f"Fixed {fixed_test_stats['clouds_with_tables']}/{fixed_test_stats['total_clouds']} " +
                  f"({fixed_test_stats['clouds_with_tables']/fixed_test_stats['total_clouds']*100:.2f}%)")
        
        if ucl_stats and fixed_ucl_stats:
            print(f"UCL Data: Original {ucl_stats['clouds_with_tables']}/{ucl_stats['total_clouds']} " + 
                  f"({ucl_stats['clouds_with_tables']/ucl_stats['total_clouds']*100:.2f}%), " +
                  f"Fixed {fixed_ucl_stats['clouds_with_tables']}/{fixed_ucl_stats['total_clouds']} " +
                  f"({fixed_ucl_stats['clouds_with_tables']/fixed_ucl_stats['total_clouds']*100:.2f}%)")
            
        return
    
    print("=== Regenerating datasets with fixed processing ===")
    print("This process will fix the issues with table labeling in the datasets.")
    
    # Regenerate SUN3D datasets if not ucl_only
    if not args.ucl_only:
        print("\n=== Regenerating SUN3D training dataset (MIT scenes) ===")
        print("This will correctly process polygon labels for tables in MIT scenes.")
        
        sun3d_train_cmd = [
            'python', 'data_utils/prepare_sun3d_dataset.py',
            '--data_dir', 'dataset/sun3d_data',
            '--prefix', 'mit',
            '--output', sun3d_train_file,
            '--max_points', str(args.max_points)
        ]
        subprocess.run(sun3d_train_cmd)
        
        if os.path.exists(sun3d_train_file):
            print_h5_info(sun3d_train_file, args.max_points)
        
        print("\n=== Regenerating SUN3D test dataset (Harvard scenes) ===")
        print("This will correctly process polygon labels for tables in Harvard scenes.")
        
        sun3d_test_cmd = [
            'python', 'data_utils/prepare_sun3d_dataset.py',
            '--data_dir', 'dataset/sun3d_data',
            '--prefix', 'harvard',
            '--output', sun3d_test_file,
            '--max_points', str(args.max_points)
        ]
        subprocess.run(sun3d_test_cmd)
        
        if os.path.exists(sun3d_test_file):
            print_h5_info(sun3d_test_file, args.max_points)
    
    # Regenerate UCL dataset if not sun3d_only
    if not args.sun3d_only:
        print("\n=== Regenerating UCL dataset (all labeled as tables) ===")
        print("This will label ALL points in UCL data as tables (label 1).")
        
        # Import prepare_ucl_data directly and run it
        sys.path.append('data_utils')
        from prepare_ucl_data import prepare_ucl_data
        
        prepare_ucl_data('dataset/ucl_data', ucl_data_file, max_points=args.max_points)
        
        if os.path.exists(ucl_data_file):
            print_h5_info(ucl_data_file, args.max_points)
    
    print("\n=== Dataset regeneration complete! ===")
    print("\nNew datasets:")
    if not args.ucl_only:
        print(f"- {sun3d_train_file}")
        print(f"- {sun3d_test_file}")
    if not args.sun3d_only:
        print(f"- {ucl_data_file}")
    
    print("\nTo use these datasets with the classification pipeline, update the file paths in your commands:")
    if not args.ucl_only:
        print(f"--train_file {sun3d_train_file} --test_file {sun3d_test_file}")
    if not args.sun3d_only:
        print(f"--ucl_file {ucl_data_file}")

if __name__ == "__main__":
    main() 