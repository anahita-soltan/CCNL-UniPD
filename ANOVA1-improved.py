import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve command-line arguments for directories
base_dir = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Admin\Documents\Mistry\data\activity'
save_path = sys.argv[2] if len(sys.argv) > 2 else r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Normalized\New'

model_directories = ['cornet_pretrained/epoch49', 'cornet_adam/epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

def load_data(base_dir, model_dir, layer):
    layer_file_path = os.path.join(base_dir, model_dir, f'{layer}.npz')
    label_file_path = os.path.join(base_dir, model_dir, 'label.npz')
    try:
        layer_data = np.load(layer_file_path)[layer]
        label_data = np.load(label_file_path)['label'].flatten()
        return layer_data, label_data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None, None

def normalize_activations(data):
    min_val = np.min(data, axis=(0, 1))  # Normalizing across the first two axes if needed
    max_val = np.max(data, axis=(0, 1))
    safe_denom = np.where(max_val > min_val, max_val - min_val, 1)
    return (data - min_val) / safe_denom

def perform_anova(layer_data, label_data, num_range):
    if layer_data is None or label_data is None:
        logging.error("Data not loaded properly")
        return []
    if layer_data.ndim != 5:
        logging.error(f"Unexpected data dimensions: {layer_data.shape}")
        return []
    samples, num_channels, num_kernels, num_pixels_i, num_pixels_j = layer_data.shape
    significant_neurons = []
    # Vectorized normalization
    layer_data = normalize_activations(layer_data)
    for sample in range(samples):
        for channel in range(num_channels):
            for kernel in range(num_kernels):
                for i in range(num_pixels_i):
                    for j in range(num_pixels_j):
                        activations_by_number = [
                            layer_data[sample, channel, kernel, i, j][label_data == num].flatten()
                            for num in num_range
                        ]
                        if not any(np.var(activations) > 0 for activations in activations_by_number):
                            continue
                        f_statistic, p_value = f_oneway(*activations_by_number)
                        if p_value < 0.01:
                            posthoc_results = posthoc_tests(activations_by_number, p_value)
                            if posthoc_results:
                                significant_neurons.append(posthoc_results)
    return significant_neurons

def posthoc_tests(activations_by_number, p_value):
    mean_activations = np.array([np.mean(activations) for activations in activations_by_number])
    max_mean_index = np.argmax(mean_activations)
    comparisons = [ttest_ind(activations_by_number[max_mean_index], activations)
                   for idx, activations in enumerate(activations_by_number) if idx != max_mean_index]
    if all(p < 0.01 / len(activations_by_number) for _, p in comparisons):
        return {
            'Sample': sample,
            'Channel': channel,
            'Kernel': kernel,
            'Neuron_i': i,
            'Neuron_j': j,
            'Tuned_Number': max_mean_index,
            'Mean_Activation': mean_activations[max_mean_index],
            'ANOVA_P-value': p_value
        }

def save_results(significant_neurons, model_dir, layer):
    if significant_neurons:
        df = pd.DataFrame(significant_neurons)
        file_name = f'{model_dir.replace("/", "_")}_{layer}_ANOVA1.xlsx'
        excel_file_path = os.path.join(save_path, file_name)
        df.to_excel(excel_file_path, index=False)
        logging.info(f'Results saved to {excel_file_path}')

for model_dir in model_directories:
    for layer in layers:
        layer_data, label_data = load_data(base_dir, model_dir, layer)
        num_range = range(9)
        significant_neurons = perform_anova(layer_data, label_data, num_range)
        save_results(significant_neurons, model_dir, layer)
