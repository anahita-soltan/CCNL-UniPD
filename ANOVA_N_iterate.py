import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind

# Define the base directory containing the data
base_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'

# List of model directories
model_directories = ['cornet_pretrained/epoch49', 'cornet_adam/epoch49']

# Define layer names
layers = ['V1', 'V2', 'V4', 'IT']

# Collect all significant neurons across models and layers
all_significant_neurons = []

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

        # Define the range of numbers (0 to 8)
        num_range = range(9)

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
                        activations_by_number = [
                            layer_data[label_data == num, channel, kernel, i, j].flatten()
                            for num in num_range
                        ]

                        # Normalize each group of activations using min-max scaling
                        activations_by_number = [
                            (activations - np.min(activations)) / (np.max(activations) - np.min(activations)) 
                            if np.max(activations) != np.min(activations) else activations
                            for activations in activations_by_number
                        ]

                        # Check if all values in each group are the same
                        if not any(np.var(activations) > 0 for activations in activations_by_number):
                            continue  # Skip ANOVA if there's no variance among the groups

                        # Perform ANOVA test
                        f_statistic, p_value = f_oneway(*activations_by_number)
                        
                        # If ANOVA is significant, find the specific number the neuron is tuned to
                        if p_value < 0.01:
                            mean_activations = [np.mean(activations) for activations in activations_by_number]
                            max_mean_activation = max(mean_activations)
                            max_mean_index = mean_activations.index(max_mean_activation)
                            
                            # Check for significant differences with other numbers
                            significant_diff = True
                            for idx, activations in enumerate(activations_by_number):
                                if idx != max_mean_index:
                                    t_stat, posthoc_p = ttest_ind(activations_by_number[max_mean_index], activations)
                                    if posthoc_p >= 0.01/len(num_range):  # Bonferroni correction
                                        significant_diff = False
                                        break
                            
                            if significant_diff:
                                significant_neurons.append({
                                    'Channel': channel,
                                    'Kernel': kernel,
                                    'Neuron_i': i,
                                    'Neuron_j': j,
                                    'Tuned_Number': max_mean_index,
                                    'Mean_Activation': max_mean_activation,
                                    'ANOVA_P-value': p_value
                                })

        # Save results for each model and layer to an Excel file
        if significant_neurons:
            df_significant_neurons = pd.DataFrame(significant_neurons)
            save_path = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Normalized\New'
            file_name = f'{model_dir.replace("/", "_")}_{layer}_ANOVA1.xlsx'
            excel_file_path = os.path.join(save_path, file_name)
            df_significant_neurons.to_excel(excel_file_path, index=False)
            print(f'Results saved to {excel_file_path}')
