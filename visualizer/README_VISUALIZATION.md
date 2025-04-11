# Point Cloud Visualization Tools

This directory contains tools for visualizing point cloud data from the SUN3D binary segmentation model. The visualizations are interactive 3D HTML files that can be viewed in any modern web browser.

## Visualization Files Format

When running `test_sun3d.py` with the `--visual` flag, the model saves visualization data in the following formats:

1. `pts_{batch}_{index}.txt`: The point cloud coordinates (x, y, z)
2. `pred_{batch}_{index}.txt`: The model's class predictions (0 for background, 1 for table)
3. `gt_{batch}_{index}.txt`: The ground truth labels (0 for background, 1 for table)

These files are saved in the experiment's visual directory: `log/sun3d_binary_seg/[experiment_name]/visual/`

## Converting to HTML Visualizations

To convert these text files to interactive HTML visualizations, use the `convert_txt_to_html.py` script:

```bash
python visualizer/convert_txt_to_html.py --log_dir [experiment_name] --output_dir point_cloud_visualizations
```

For example:
```bash
python visualizer/convert_txt_to_html.py --log_dir sun3d_balanced_mild_aug_2025-04-07_17-27-34 --output_dir point_cloud_visualizations
```

This will create interactive HTML files in the specified output directory, along with an index.html file that provides links to all visualizations.

## Creating a Download Package

To package the visualizations for download to your local machine, use the `package_visualizations.py` script:

```bash
python visualizer/package_visualizations.py --log_dir [experiment_name]
```

For example:
```bash
python visualizer/package_visualizations.py --log_dir sun3d_balanced_mild_aug_2025-04-07_17-27-34
```

This will create a zip file in the `downloads` directory that contains all the HTML visualizations.

## Viewing the Visualizations

After downloading and extracting the zip file to your local machine:

1. Open the `index.html` file in your web browser
2. Use the links to navigate to the different point cloud visualizations
3. Each visualization provides:
   - Original point cloud view
   - Ground truth labels (orange = table, blue = background)
   - Model predictions (orange = table, blue = background)
   - Error visualization (red = incorrect predictions, green = correct predictions)

## Interacting with the Visualizations

The visualization interface allows you to:

- Rotate the point cloud by clicking and dragging
- Zoom in/out using the scroll wheel
- Pan by holding shift while dragging
- Reset the view using the home button
- Download the current view as a PNG
- Toggle the visibility of different elements

## Requirements

To generate these visualizations, you need:
- Python 3.6+
- Plotly library: `pip install plotly` 