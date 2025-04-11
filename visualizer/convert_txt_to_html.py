import os
import numpy as np
import plotly.graph_objects as go
import argparse
from pathlib import Path
import glob

def load_point_cloud(pts_file, pred_file=None, gt_file=None):
    """
    Load point cloud data and labels from txt files
    
    Args:
        pts_file: Path to the points file
        pred_file: Path to the prediction file
        gt_file: Path to the ground truth file
        
    Returns:
        points: Nx3 array of 3D points
        predictions: N array of prediction labels
        ground_truth: N array of ground truth labels
    """
    # Load points
    points = np.loadtxt(pts_file)
    
    # Load predictions if available
    predictions = None
    if pred_file and os.path.exists(pred_file):
        predictions = np.loadtxt(pred_file)
        
    # Load ground truth if available
    ground_truth = None
    if gt_file and os.path.exists(gt_file):
        ground_truth = np.loadtxt(gt_file)
        
    return points, predictions, ground_truth

def create_visualization(points, predictions=None, ground_truth=None, title="Point Cloud Visualization"):
    """
    Create an interactive 3D visualization of a point cloud
    
    Args:
        points: Nx3 array of 3D points
        predictions: N array of prediction labels
        ground_truth: N array of ground truth labels
        title: Title for the visualization
        
    Returns:
        fig: Plotly figure object
    """
    # Create a new figure
    fig = go.Figure()
    
    # Define colors for background (0) and table (1)
    colors = ['#1f77b4', '#ff7f0e']  # Blue for background, orange for table
    
    # If both predictions and ground truth are available, create a 2x2 subplot
    if predictions is not None and ground_truth is not None:
        # Create a grid of subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Original Point Cloud', 'Ground Truth', 'Predictions', 'Errors')
        )
        
        # Add original point cloud (all blue)
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='#1f77b4',
                    opacity=0.7
                ),
                name='Points'
            ),
            row=1, col=1
        )
        
        # Add ground truth
        colors_gt = [colors[int(label)] for label in ground_truth]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors_gt,
                    opacity=0.7
                ),
                name='Ground Truth'
            ),
            row=1, col=2
        )
        
        # Add predictions
        colors_pred = [colors[int(label)] for label in predictions]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors_pred,
                    opacity=0.7
                ),
                name='Predictions'
            ),
            row=2, col=1
        )
        
        # Add errors (red for incorrect, green for correct)
        errors = predictions != ground_truth
        error_colors = ['#2ca02c' if not err else '#d62728' for err in errors]  # Green for correct, red for errors
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=error_colors,
                    opacity=0.7
                ),
                name='Errors'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene3=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene4=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            width=1200,
            height=1000
        )
    
    # If only predictions are available (no ground truth)
    elif predictions is not None:
        # Create a subplot with 2 plots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Original Point Cloud', 'Predictions')
        )
        
        # Add original point cloud (all blue)
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='#1f77b4',
                    opacity=0.7
                ),
                name='Points'
            ),
            row=1, col=1
        )
        
        # Add predictions
        colors_pred = [colors[int(label)] for label in predictions]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors_pred,
                    opacity=0.7
                ),
                name='Predictions'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            width=1000,
            height=500
        )
    
    # If just point cloud without labels
    else:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='#1f77b4',
                    opacity=0.7
                ),
                name='Points'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            width=800,
            height=600
        )
    
    return fig

def save_visualization(fig, output_file):
    """
    Save visualization to HTML file
    
    Args:
        fig: Plotly figure object
        output_file: Path to output HTML file
    """
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Saved visualization to {output_file}")

def process_directory(input_dir, output_dir):
    """
    Process all point cloud files in a directory
    
    Args:
        input_dir: Directory containing point cloud txt files
        output_dir: Directory to save HTML visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all point cloud files
    pts_files = sorted(glob.glob(os.path.join(input_dir, 'pts_*.txt')))
    
    # Process each file
    for pts_file in pts_files:
        base_name = os.path.basename(pts_file)[4:-4]  # Remove 'pts_' prefix and '.txt' suffix
        
        # Construct paths to prediction and ground truth files
        pred_file = os.path.join(input_dir, f'pred_{base_name}.txt')
        gt_file = os.path.join(input_dir, f'gt_{base_name}.txt')
        
        # Skip if prediction file doesn't exist
        if not os.path.exists(pred_file):
            print(f"Skipping {pts_file} (no prediction file)")
            continue
        
        # Load data
        points, predictions, ground_truth = load_point_cloud(pts_file, pred_file, gt_file)
        
        # Create visualization
        fig = create_visualization(
            points, predictions, ground_truth,
            title=f"Point Cloud {base_name}"
        )
        
        # Save visualization
        output_file = os.path.join(output_dir, f'visualization_{base_name}.html')
        save_visualization(fig, output_file)
        
        print(f"Processed {base_name}")
    
    print(f"Processed {len(pts_files)} point clouds")

def create_index_html(output_dir, visualizations):
    """
    Create an index.html file that links to all visualizations
    
    Args:
        output_dir: Directory to save index.html
        visualizations: List of visualization filenames
    """
    # Create index.html
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('    <title>Point Cloud Visualizations</title>\n')
        f.write('    <style>\n')
        f.write('        body { font-family: Arial, sans-serif; margin: 20px; }\n')
        f.write('        h1 { color: #333; }\n')
        f.write('        ul { list-style-type: none; padding: 0; }\n')
        f.write('        li { margin: 10px 0; }\n')
        f.write('        a { color: #0066cc; text-decoration: none; }\n')
        f.write('        a:hover { text-decoration: underline; }\n')
        f.write('    </style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write('    <h1>Point Cloud Visualizations</h1>\n')
        f.write('    <ul>\n')
        
        for vis in visualizations:
            name = os.path.basename(vis)
            f.write(f'        <li><a href="{name}" target="_blank">{name}</a></li>\n')
        
        f.write('    </ul>\n')
        f.write('</body>\n')
        f.write('</html>\n')
    
    print(f"Created index.html at {index_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert point cloud txt files to HTML visualizations')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory of the experiment (e.g., sun3d_balanced_mild_aug_2025-04-07_17-27-34)')
    parser.add_argument('--output_dir', type=str, default='html_visualizations', help='Directory to save HTML visualizations')
    args = parser.parse_args()
    
    # Construct input directory
    input_dir = os.path.join('log', 'sun3d_binary_seg', args.log_dir, 'visual')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process point cloud files
    output_dir = os.path.join(args.output_dir, args.log_dir)
    process_directory(input_dir, output_dir)
    
    # Create index.html
    vis_files = sorted(glob.glob(os.path.join(output_dir, '*.html')))
    create_index_html(output_dir, vis_files)
    
    print(f"Done! Open {os.path.join(output_dir, 'index.html')} to view visualizations")

if __name__ == '__main__':
    main() 