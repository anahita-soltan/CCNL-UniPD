import numpy as np

# Define the file path using a raw string literal
CAV1 = r'C:\Users\Admin\Documents\Mistry\data\activity\cornet_adam\epoch49\param.npz'

# Load the NPZ file
cornet_adam_V1 = np.load(CAV1)

# Access the array stored under the key 'V1'
array_CAV1 = cornet_adam_V1['param']

# Print the array
print(array_CAV1)
