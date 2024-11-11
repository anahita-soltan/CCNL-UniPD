import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
input_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\all_layers\ANOVA2'
data_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\all_layers\Tuning_Curves'

os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

for model_name in models:
    for epoch in epochs:
        for layer in layers:
            # Construct paths and check existence
            file_path = os.path.join(input_directory, f'{model_name}_{epoch}_{layer}_ANOVA2.parquet')
            activity_path = os.path.join(data_dir, model_name, epoch, f'{layer}.npz')
            label_path = os.path.join(data_dir, model_name, epoch, 'label.npz')

            if not all(os.path.exists(path) for path in [file_path, activity_path, label_path]):
                print(f"Files missing for {model_name} {epoch} {layer}")
                continue

            # Load data
            df_neurons = pd.read_parquet(file_path)
            activity_data = np.load(activity_path)[layer]
            label_data = np.load(label_path)['label'].flatten()

            # Create and process DataFrame for activation curves
            unique_labels = np.unique(label_data)
            df_curves = pd.DataFrame(index=unique_labels)

            def calculate_activations(neurons, activity_data, label_data, unique_labels):
                activations = np.zeros((len(neurons), len(unique_labels)))
                for idx, neuron in enumerate(neurons.itertuples()):
                    channel, kernel, neuron_i, neuron_j = int(neuron.Channel), int(neuron.Kernel), int(neuron.Neuron_i), int(neuron.Neuron_j)
                    neuron_activations = activity_data[:, channel, kernel, neuron_i, neuron_j]
                    for label_idx, label in enumerate(unique_labels):
                        activations[idx][label_idx] = neuron_activations[label_data == label].mean()
                return np.mean(activations, axis=0)

            def normalize_activations(mean_activations, preferred_number):
                # Normalize activations by the activation at the preferred number
                normalization_factor = mean_activations[preferred_number]  # Use preferred_number directly for zero-based labels
                return mean_activations / normalization_factor if normalization_factor != 0 else mean_activations

            for pn in df_neurons['Preferred_Number'].unique():  # Assuming 'Preferred_Number' replaces 'Tuned_Number'
                neurons_pn = df_neurons[df_neurons['Preferred_Number'] == pn]
                activations = calculate_activations(neurons_pn, activity_data, label_data, unique_labels)
                normalized_activations = normalize_activations(activations, pn)  # Pass pn directly
                df_curves[f'PN_{pn}'] = normalized_activations

            # Save results
            output_file = os.path.join(output_directory, f'{model_name}_{epoch}_{layer}_activation_curves.csv')
            df_curves.to_csv(output_file, index_label='Number')
            print(f'Saved activation curves for {model_name} {epoch} {layer} in {output_file}')

# Now, let's generate the plots

# Lists to store filenames based on model type
pretrained_files = []
trained_files = []

# Custom sorting key to order the plots by V1, V2, V4, IT
def custom_sort(filename):
    order = ['V1', 'V2', 'V4', 'IT']
    parts = filename.split('_')
    for idx, part in enumerate(order):
        if part in parts:
            return idx
    return len(order)

# Classify files based on an assumed keyword in the filenames
for file in os.listdir(output_directory):
    if file.endswith(".csv"):
        if 'pretrained' in file:
            pretrained_files.append(file)
        else:
            trained_files.append(file)

# Sort files according to the custom logic
pretrained_files.sort(key=custom_sort)
trained_files.sort(key=custom_sort)

def plot_files(files, title_suffix, color):
    # Prepare a figure with subplots in one row
    fig, axs = plt.subplots(1, len(files), figsize=(17, 5), sharey=True)  # Larger y-dimension for better detail visibility

    # Loop through every file and subplot axis
    for ax, filename in zip(axs, files):
        file_path = os.path.join(output_directory, filename)  # Full file path
        df = pd.read_csv(file_path)  # Load the CSV file into a DataFrame

        x = df['Number'] + 1  # Adjusting x-axis to start from 1 instead of 0
        
        sorted_columns = sorted(df.columns[1:], key=lambda x: int(x.split('_')[1]))
        
        for column in sorted_columns:
            ax.plot(x, df[column], color=color)  # Use one color per model type

        model_name = filename.split('_')[1].capitalize()
        layer_name = filename.split('_')[3]
        ax.set_title(f"{model_name} - {layer_name}")
        ax.set_ylim(0, 1)  # Fix y-axis between 0 and 1
        ax.set_xlabel('')  # Remove x-label
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

    # General settings
    # Setting y-label on the figure, adjusting its position
    fig.text(0.04, 0.5, 'Avg Activation Values', va='center', rotation='vertical', fontsize=12)  # Positioning it more centrally

    fig.suptitle(f'{title_suffix} Normalized Tuning Curves', fontsize=14)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Adjust the left margin
    plt.show()

# Plot files for each model type
plot_files(pretrained_files, 'Pre-trained', 'blue')
plot_files(trained_files, 'Trained', 'red')
