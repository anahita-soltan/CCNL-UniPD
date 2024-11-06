import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, zscore

# Define the base directory containing the data
base_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'

# List of model directories
model_directories = ['cornet_pretrained/epoch49', 'cornet_adam/epoch49']

# Define layer names
layers = ['V1', 'V2', 'V4', 'IT']

# Directory for saving results
save_path = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Normalized\New'
if not os.path.exists(save_path):
    os.makedirs(save_path)  # Create the directory if it does not exist

# Collect all significant neurons across models and layers
for model_dir in model_directories:
    for layer in layers:
        # Construct the full path to the layer and label data files
        layer_file_path = os.path.join(base_dir, model_dir, f'{layer}.npz')
        label_file_path = os.path.join(base_dir, model_dir, 'label.npz')

        # Load layer and label data
        try:
            layer_data = np.load(layer_file_path)[layer]
            label_data = np.load(label_file_path)['label'].flatten()
        except FileNotFoundError:
            print(f"File not found: {layer_file_path} or {label_file_path}")
            continue

        # Dimensions of neurons in the dataset for the current layer
        num_channels = layer_data.shape[1]
        num_kernels = layer_data.shape[2]
        num_pixels_i = layer_data.shape[3]
        num_pixels_j = layer_data.shape[4]

        # Prepare to collect significant neurons for this model and layer
        significant_neurons = []

        # Perform ANOVA for each neuron
        for channel in range(num_channels):
            for kernel in range(num_kernels):
                for i in range(num_pixels_i):
                    for j in range(num_pixels_j):
                        # Collect activations for all numbers for this neuron
                        activations = layer_data[:, channel, kernel, i, j]
                        normalized_activations = zscore(activations)

                        # Perform ANOVA test
                        if np.isnan(normalized_activations).any():
                            continue  # Skip if zscore calculation results in NaN
                        activations_by_number = [
                            normalized_activations[label_data == num] for num in range(9)
                        ]
                        f_statistic, p_value = f_oneway(*activations_by_number)
                        
                        if p_value < 0.01:
                            significant_neurons.append({
                                'Channel': channel,
                                'Kernel': kernel,
                                'Neuron_i': i,
                                'Neuron_j': j,
                                'ANOVA_P-value': p_value
                            })

        # Save results for each model and layer to an Excel file
        if significant_neurons:
            df_significant_neurons = pd.DataFrame(significant_neurons)
            file_name = f'{model_dir.replace("/", "_")}_{layer}_ANOVA1.xlsx'
            excel_file_path = os.path.join(save_path, file_name)
            df_significant_neurons.to_excel(excel_file_path, index=False)
            print(f'Results saved to {excel_file_path}')
