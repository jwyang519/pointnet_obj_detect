import os
import argparse
import zipfile
import glob
from pathlib import Path

def package_visualizations(visualization_dir, output_zip):
    """
    Package all HTML visualization files in a directory into a zip file
    
    Args:
        visualization_dir: Directory containing HTML visualizations
        output_zip: Path to output zip file
    """
    # Check if directory exists
    if not os.path.exists(visualization_dir):
        print(f"Visualization directory {visualization_dir} does not exist")
        return
    
    # Create zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all HTML files in the directory
        html_files = glob.glob(os.path.join(visualization_dir, '*.html'))
        if not html_files:
            print(f"No HTML files found in {visualization_dir}")
            return
        
        # Add index.html first (if it exists)
        index_file = os.path.join(visualization_dir, 'index.html')
        if os.path.exists(index_file):
            zipf.write(index_file, os.path.basename(index_file))
            html_files.remove(index_file)
        
        # Add all other HTML files
        for html_file in sorted(html_files):
            zipf.write(html_file, os.path.basename(html_file))
    
    print(f"Successfully packaged {len(html_files) + (1 if os.path.exists(index_file) else 0)} HTML files into {output_zip}")
    print(f"You can now download this file and extract it to view the visualizations")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Package HTML visualizations into a zip file')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory name (e.g., sun3d_balanced_mild_aug_2025-04-07_17-27-34)')
    parser.add_argument('--visualization_dir', type=str, default='point_cloud_visualizations', help='Directory containing HTML visualizations')
    parser.add_argument('--output_dir', type=str, default='downloads', help='Directory to save zip file')
    args = parser.parse_args()
    
    # Construct visualization directory path
    vis_dir = os.path.join(args.visualization_dir, args.log_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output zip file path
    output_zip = os.path.join(args.output_dir, f'{args.log_dir}_visualizations.zip')
    
    # Package visualizations
    package_visualizations(vis_dir, output_zip)

if __name__ == '__main__':
    main() 