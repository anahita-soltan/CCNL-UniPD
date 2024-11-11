import numpy as np

# Define the file path
file_path = r'C:\Users\Admin\Documents\Mistry\data\activity\cornet_adam\epoch49\condition.npz'

# Load the data
data = np.load(file_path)

# Print the shapes and summaries of the arrays in the file
for key in data.keys():
    array = data[key]
    print(f"Array '{key}' shape: {array.shape}")
    
    # Check if the array is one-dimensional or contains more than one dimension
    if array.ndim == 1:
        # Print basic statistics if one-dimensional
        unique, counts = np.unique(array, return_counts=True)
        print(f"Unique values in '{key}': {dict(zip(unique, counts))}")
    elif array.ndim > 1:
        # If multidimensional, handle appropriately, here just print unique values per column if 2D
        if array.shape[1] == 1:
            unique, counts = np.unique(array.flatten(), return_counts=True)
            print(f"Unique values in '{key}': {dict(zip(unique, counts))}")
        else:
            for i in range(array.shape[1]):
                unique, counts = np.unique(array[:, i], return_counts=True)
                print(f"Column {i} of '{key}' unique values: {dict(zip(unique, counts))}")

    # Include additional summary statistics as needed
    print(f"Mean value in '{key}': {np.mean(array)}")
    print(f"Standard Deviation in '{key}': {np.std(array)}")
    print(f"Max value in '{key}': {np.max(array)}")
    print(f"Min value in '{key}': {np.min(array)}\n")
