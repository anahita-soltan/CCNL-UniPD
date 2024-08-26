import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Define directories
input_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA2'
data_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\No_Monotonic'

os.makedirs(output_directory, exist_ok=True)

models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

for model_name in models:
    for epoch in epochs:
        for layer in layers:
            file_path = os.path.join(input_directory, f'{model_name}_{epoch}_{layer}_ANOVA2.xlsx')
            activity_path = os.path.join(data_dir, f'{model_name}', f'{epoch}', f'{layer}.npz')
            label_path = os.path.join(data_dir, f'{model_name}', f'{epoch}', 'label.npz')

            if not all(os.path.exists(path) for path in [file_path, activity_path, label_path]):
                print(f"Files missing for {model_name} {epoch} {layer}")
                continue

            df_neurons = pd.read_excel(file_path)
            activity_data = np.load(activity_path)[layer]
            label_data = np.load(label_path)['label'].flatten()

            neurons_to_remove = []

            for _, neuron in df_neurons.iterrows():
                if neuron['Tuned_Number'] not in [0, 8]:
                    continue
                neuron_index = neuron.name
                neuron_activity = activity_data[:, int(neuron['Channel']), int(neuron['Kernel']), int(neuron['Neuron_i']), int(neuron['Neuron_j'])]
                mean_activations = np.array([neuron_activity[label_data == i].mean() for i in np.unique(label_data)])

                slope, intercept, r_value, _, _ = linregress(np.unique(label_data), mean_activations)
                r_squared = r_value**2

                if r_squared > 0.5:
                    neurons_to_remove.append(neuron_index)

            # Filter out the neurons to remove
            df_filtered = df_neurons.drop(neurons_to_remove)

            # Save the updated DataFrame
            output_file = os.path.join(output_directory, f'{model_name}_{epoch}_{layer}_No_Monotonic.xlsx')
            df_filtered.to_excel(output_file, index=False)
            print(f"Updated ANOVA file saved for {model_name} {epoch} {layer} excluding summation neurons.")

