import os
import pandas as pd
import numpy as np

# Define directories
input_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA2'
data_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\Pooling&Normalizing1'

os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

for model_name in models:
    for epoch in epochs:
        for layer in layers:
            # Construct paths and check existence
            file_path = os.path.join(input_directory, f'{model_name}_{epoch}_{layer}_ANOVA2.xlsx')
            activity_path = os.path.join(data_dir, model_name, epoch, f'{layer}.npz')
            label_path = os.path.join(data_dir, model_name, epoch, 'label.npz')

            if not all(os.path.exists(path) for path in [file_path, activity_path, label_path]):
                print(f"Files missing for {model_name} {epoch} {layer}")
                continue

            # Load data
            df_neurons = pd.read_excel(file_path)
            activity_data = np.load(activity_path)[layer]
            label_data = np.load(label_path)['label'].flatten()

            # Create DataFrame for activation curves
            unique_labels = np.unique(label_data)
            df_curves = pd.DataFrame(index=unique_labels)

            # Function to calculate activations
            def calculate_activations(neurons, activity_data, label_data, unique_labels):
                activations = np.zeros((len(neurons), len(unique_labels)))
                for idx, neuron in enumerate(neurons.itertuples()):
                    channel, kernel, neuron_i, neuron_j = int(neuron.Channel), int(neuron.Kernel), int(neuron.Neuron_i), int(neuron.Neuron_j)
                    neuron_activations = activity_data[:, channel, kernel, neuron_i, neuron_j]
                    for label_idx, label in enumerate(unique_labels):
                        activations[idx][label_idx] = neuron_activations[label_data == label].mean()
                return np.mean(activations, axis=0)

            # Normalize and pool activations
            def normalize_and_pool(activations):
                min_val = np.min(activations)
                max_val = np.max(activations)
                return (activations - min_val) / (max_val - min_val) if max_val != min_val else activations

            # Process each preferred numerosity
            for pn in df_neurons['Tuned_Number'].unique():
                neurons_pn = df_neurons[df_neurons['Tuned_Number'] == pn]
                activations = calculate_activations(neurons_pn, activity_data, label_data, unique_labels)
                normalized_activations = normalize_and_pool(activations)  # Normalize pooled activations
                df_curves[f'PN_{pn}'] = normalized_activations

            # Save results
            output_file = os.path.join(output_directory, f'{model_name}_{epoch}_{layer}_NormalizedPools.csv')
            df_curves.to_csv(output_file, index_label='Number')
            print(f'Saved activation curves for {model_name} {epoch} {layer} in {output_file}')
