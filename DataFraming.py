import os
import numpy as np
import pandas as pd

def generate_activation_data(layer_data, label_data, num_range):
    num_channels, num_kernels, num_pixels_i, num_pixels_j = layer_data.shape[1:]
    for channel in range(num_channels):
        for kernel in range(num_kernels):
            for i in range(num_pixels_i):
                for j in range(num_pixels_j):
                    activations_by_number = [
                        layer_data[label_data == num, channel, kernel, i, j].flatten()
                        for num in num_range
                    ]
                    for num, activations in zip(num_range, activations_by_number):
                        for activation in activations:
                            yield {
                                'Channel': channel,
                                'Kernel': kernel,
                                'Pixel_i': i,
                                'Pixel_j': j,
                                'Number': num,
                                'Activation': activation
                            }

# Define the directory containing the data
data_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'

# List of model directories and layers
model_directories = ['cornet_adam/epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

# Iterate over each model and layer
for model_dir in model_directories:
    for layer in layers:
        layer_data = np.load(os.path.join(data_dir, model_dir, f'{layer}.npz'))[layer]
        label_data = np.load(os.path.join(data_dir, model_dir, 'label.npz'))['label'].flatten()
        num_range = range(9)

        # Prepare generator
        data_generator = generate_activation_data(layer_data, label_data, num_range)

        # Setup CSV path
        network_path = r'\\147.162.145.100\Public\Anahita'  # Raw string for network path
        directory = os.path.join(network_path, f'activations_data_{model_dir}_{layer}')
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        csv_file_path = os.path.join(directory, 'activations.csv')

        batch_size = 100000  # Adjust batch size as needed
        data_list = []

        # Iterate over generator and write to CSV in batches
        for data_point in data_generator:
            data_list.append(data_point)
            if len(data_list) >= batch_size:
                df_batch = pd.DataFrame(data_list)
                df_batch.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
                data_list = []  # Reset for the next batch

        # Save any remaining data
        if data_list:
            df_batch = pd.DataFrame(data_list)
            df_batch.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))

        print(f"Data saved for {model_dir} {layer} to {csv_file_path}")
