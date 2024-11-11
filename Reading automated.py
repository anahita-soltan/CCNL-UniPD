import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory containing the data
data_dir = r'C:\Users\Admin\Documents\Mistry\data'

# Define a function to load data from npz files
def load_data(file_path):
    data = np.load(file_path)
    v1_data = data['V1']
    v2_data = data['V2'] if 'V2' in data else None
    v4_data = data['V4'] if 'V4' in data else None
    return v1_data, v2_data, v4_data

# Initialize a dictionary to store the loaded data
model_data = {}

# Iterate through each model folder
for model_folder in os.listdir(os.path.join(data_dir, 'activity')):
    model_path = os.path.join(data_dir, 'activity', model_folder, 'epoch49')  # Assuming only epoch49 for all models
    
    # Check if the item in the directory is a folder
    if os.path.isdir(model_path):
        print(f"Loading data for model: {model_folder}")
        
        # Load data for layers V1, V2, and V4
        v1_data, v2_data, v4_data = load_data(os.path.join(model_path, 'V1.npz'))
        
        # Store the loaded data in the model_data dictionary
        model_data[model_folder] = {'V1': v1_data, 'V2': v2_data, 'V4': v4_data}
        
        # Perform any necessary operations with the loaded data
        # For example, you can process or analyze the data
        
        print("-------------------------------------------------")

# Now you can access the data for each model and layer using model_data

import matplotlib.pyplot as plt

# Extract the data for the cornet_adam model
v1_data = model_data['cornet_adam']['V1']
v2_data = model_data['cornet_adam']['V2']

# Plot the first image in the V1 data
plt.imshow(v2_data[0, 0, 0], cmap='viridis')
plt.title('Cornet Adam - V1 Layer Activity')
plt.colorbar()
plt.show()
