import h5py
import numpy as np

# Path to the h5 file
h5_file = 'dataset/sun3d_train.h5'

# Open the h5 file
with h5py.File(h5_file, 'r') as f:
    # List all the keys (datasets) in the file
    print(f"Keys in the h5 file: {list(f.keys())}")
    
    # Check the shape of each dataset
    for key in f.keys():
        data = f[key]
        print(f"Dataset '{key}' has shape {data.shape} and dtype {data.dtype}")
        
        # Print a sample of the data (first 5 elements)
        if len(data.shape) == 1:
            print(f"Sample data: {data[:5]}")
        elif len(data.shape) == 2:
            print(f"Sample data (first row): {data[0]}")
            print(f"Sample data (second row): {data[1]}")
        else:
            print(f"Sample data (first item): {data[0][0]}") 